"""Provider adapters: one conversation interface over Anthropic, OpenAI and a scripted fake.

Each adapter keeps the conversation in its provider's native format (append-only, so
Anthropic thinking blocks are passed back unchanged) and reports token usage, so every
agent role is measured the same way regardless of provider.

API keys are read from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY). Never put
keys in code or files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    parse_error: Optional[str] = None  # set when the model produced arguments that are not JSON


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop: str                          # "tool_use" | "end" | "max_tokens" | "refusal"
    served_by: Optional[str] = None    # model that actually answered (differs on a fallback)
    fallback_used: bool = False


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    refusals: int = 0
    fallbacks: int = 0

    def add(self, other: "Usage") -> None:
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) + getattr(other, k))


class Conversation:
    """One agent's conversation. Subclasses implement _call()."""

    def __init__(self, system: str, tools: list[dict]) -> None:
        self.system = system
        self.tools = tools
        self.usage = Usage()

    def send_user(self, text: str) -> Turn:
        raise NotImplementedError

    def send_tool_results(self, results: list[tuple[str, str, bool]]) -> Turn:
        """results: (tool_call_id, content, is_error), all from the previous turn."""
        raise NotImplementedError


# =========================================================================== Anthropic

class AnthropicConversation(Conversation):
    """Manual tool-use loop on the Messages API.

    - Thinking is adaptive (always on for Claude Opus 5.5); depth is set with effort.
    - The whole response.content is appended back each turn (append-only history).
    - Stable system prompt + tool list are cached.
    - Server-side refusal fallback is optional and OFF for the study (every result comes from
      exactly one model); a refused turn is recorded as a refusal. If it is ever turned on, every
      turn records which model served it.
    """

    def __init__(self, system: str, tools: list[dict], model: str, effort: str = "medium",
                 fallbacks: bool = False, client=None) -> None:
        super().__init__(system, tools)
        import anthropic
        # More patient than the SDK default (2) so a brief network drop does not end a run.
        self.client = client or anthropic.Anthropic(max_retries=6)
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.messages: list[dict] = []

    def _call(self) -> Turn:
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=16000, system=self.system, tools=self.tools,
            messages=self.messages, output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        response = self.client.beta.messages.create(**kwargs)
        self.messages.append({"role": "assistant", "content": response.content})

        u = response.usage
        usage = Usage(calls=1, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
        fallback_used = any(getattr(it, "type", None) == "fallback_message"
                            for it in (getattr(u, "iterations", None) or []))
        usage.fallbacks = int(fallback_used)

        if response.stop_reason == "refusal":
            usage.refusals = 1
            self.usage.add(usage)
            return Turn("", [], "refusal", response.model, fallback_used)
        self.usage.add(usage)

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        stop = {"tool_use": "tool_use", "max_tokens": "max_tokens"}.get(response.stop_reason, "end")
        return Turn(text, calls, stop, response.model, fallback_used)

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": content, "is_error": is_error}
            for cid, content, is_error in results]})
        return self._call()


# =========================================================================== OpenAI

class OpenAIConversation(Conversation):
    """Responses API function calling with reasoning on (current OpenAI reasoning models do not
    allow function tools with reasoning on Chat Completions). Turns are chained server-side with
    previous_response_id so the model's reasoning items carry across tool calls. The model ID must
    be given explicitly.
    """

    def __init__(self, system: str, tools: list[dict], model: str, reasoning_effort: Optional[str] = None,
                 client=None) -> None:
        super().__init__(system, tools)
        import openai
        self.client = client or openai.OpenAI(max_retries=6)  # patient retries, as above
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.previous_id: Optional[str] = None
        # strict=False: our schemas have optional fields and patterns; the environment validates them.
        self.oa_tools = [{"type": "function", "name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"], "strict": False} for t in tools]

    def _call(self, items: list[dict]) -> Turn:
        kwargs: dict[str, Any] = dict(model=self.model, instructions=self.system, input=items,
                                      tools=self.oa_tools, max_output_tokens=16000)
        if self.previous_id:
            kwargs["previous_response_id"] = self.previous_id
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**kwargs)
        self.previous_id = response.id

        u = response.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        # input_tokens includes the cached ones; count them once, as cache reads.
        self.usage.add(Usage(calls=1, input_tokens=u.input_tokens - cached, output_tokens=u.output_tokens,
                             cache_read_tokens=cached))

        calls, texts, refused = [], [], False
        for item in response.output:
            if item.type == "function_call":
                try:
                    calls.append(ToolCall(item.call_id, item.name, json.loads(item.arguments or "{}")))
                except json.JSONDecodeError as e:
                    calls.append(ToolCall(item.call_id, item.name, {}, parse_error=str(e)))
            elif item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        texts.append(part.text)
                    elif part.type == "refusal":
                        refused = True
        if refused and not calls:
            self.usage.refusals += 1
            return Turn("", [], "refusal", response.model)
        if calls:
            stop = "tool_use"
        elif response.status == "incomplete":
            stop = "max_tokens"
        else:
            stop = "end"
        return Turn("".join(texts), calls, stop, response.model)

    def send_user(self, text: str) -> Turn:
        return self._call([{"role": "user", "content": text}])

    def send_tool_results(self, results):
        return self._call([{"type": "function_call_output", "call_id": cid, "output": content}
                           for cid, content, _ in results])


# =========================================================================== Scripted (tests, no API)

class ScriptedConversation(Conversation):
    """Replays a fixed list of turns. Used to test the harness without an API key."""

    def __init__(self, system: str, tools: list[dict], script: list[Turn]) -> None:
        super().__init__(system, tools)
        self.script = list(script)
        self.received: list[Any] = []

    def _next(self) -> Turn:
        self.usage.calls += 1
        return self.script.pop(0) if self.script else Turn("done", [], "end")

    def send_user(self, text: str) -> Turn:
        self.received.append(text)
        return self._next()

    def send_tool_results(self, results):
        self.received.append(results)
        return self._next()


# =========================================================================== factory

@dataclass
class ModelConfig:
    provider: str                  # "anthropic" | "openai" | "scripted"
    model: str = ""
    effort: str = "medium"         # Anthropic effort / OpenAI reasoning_effort
    fallbacks: bool = False        # Anthropic server-side refusal fallback (off for the study)
    scripts: dict = field(default_factory=dict)  # role -> list[list[Turn]] for "scripted"

    def label(self) -> str:
        return f"{self.provider}:{self.model}:{self.effort}"


def new_conversation(cfg: ModelConfig, role: str, system: str, tools: list[dict]) -> Conversation:
    if cfg.provider == "anthropic":
        return AnthropicConversation(system, tools, cfg.model or "claude-opus-5-5", cfg.effort, cfg.fallbacks)
    if cfg.provider == "openai":
        if not cfg.model:
            raise ValueError("Give the OpenAI model ID explicitly (e.g. --openai-model <id>)")
        return OpenAIConversation(system, tools, cfg.model, cfg.effort)
    if cfg.provider == "scripted":
        queue = cfg.scripts.get(role, [])
        return ScriptedConversation(system, tools, queue.pop(0) if queue else [])
    raise ValueError(f"unknown provider {cfg.provider}")
