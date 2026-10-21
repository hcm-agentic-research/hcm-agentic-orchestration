"""Core of the mock: clock, tool registry, schema validation, fault injection, audit log.

Every call from an agent (or from the fixed-flow baseline) goes through
Environment.call(), so all strategies are measured the same way.
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Optional, Union

from .systems import ITSM, Finance, Identity, Legal, OrgPolicy, Workday

# Business clock: tick 0 is the start of BASE_DATE, and one tick is one minute.
# Every tool call costs one tick, so how long a strategy takes is measured too.
BASE_DATE = date(2026, 11, 2)
TICKS_PER_HOUR = 60
TICKS_PER_DAY = 24 * TICKS_PER_HOUR


def ok(data: Any = None) -> dict:
    return {"ok": True, "data": data}


def err(code: int, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


# --------------------------------------------------------------------------- tool registry

@dataclass
class ToolSpec:
    name: str
    system: str
    description: str
    schema: dict
    handler: Callable[..., dict]
    tier: Union[str, Callable[[dict], str]]

    def tier_for(self, args: dict) -> str:
        return self.tier(args) if callable(self.tier) else self.tier


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, system: str, tier, description: str,
         properties: dict, required: tuple | list = ()):
    """Register a tool. `tier` is a tier string or a function of the arguments."""
    def deco(fn):
        schema = {"type": "object", "properties": properties,
                  "required": list(required), "additionalProperties": False}
        REGISTRY[name] = ToolSpec(name, system, description, schema, fn, tier)
        return fn
    return deco


def tool_specs(fmt: str = "anthropic", names: Optional[list[str]] = None) -> list[dict]:
    """Export tool definitions in the format a provider's API expects."""
    specs = [REGISTRY[n] for n in (names or REGISTRY)]
    if fmt == "anthropic":
        return [{"name": s.name, "description": s.description, "input_schema": s.schema} for s in specs]
    if fmt == "openai":
        return [{"type": "function",
                 "function": {"name": s.name, "description": s.description, "parameters": s.schema}}
                for s in specs]
    raise ValueError(f"unknown format {fmt}")


# --------------------------------------------------------------------------- validation

_PY_TYPES = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict}


def validate(schema: dict, value: Any, path: str = "args") -> list[str]:
    """Minimal JSON-schema check: type, enum, pattern, minLength, min/max, required, extra keys, items."""
    t = schema.get("type")
    if t:
        is_bool = isinstance(value, bool)
        if not isinstance(value, _PY_TYPES[t]) or (t in ("integer", "number") and is_bool):
            return [f"{path}: expected {t}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    if "pattern" in schema and isinstance(value, str) and not re.fullmatch(schema["pattern"], value):
        errors.append(f"{path}: '{value}' does not match required format {schema['pattern']}")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        errors.append(f"{path}: must be at least {schema['minLength']} characters")
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: must be >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: must be <= {schema['maximum']}")
    if t == "object":
        props = schema.get("properties", {})
        errors += [f"{path}.{r}: required" for r in schema.get("required", []) if r not in value]
        for key, sub in value.items():
            if key in props:
                errors += validate(props[key], sub, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: unknown parameter")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(schema["items"], item, f"{path}[{i}]")
    return errors


# --------------------------------------------------------------------------- faults

FAULT_ERRORS = {
    "transient": (429, "Rate limit exceeded; retry later"),
    "permission": (403, "Integration user lacks permission for this operation"),
    "outage": (503, "Service unavailable"),
}


@dataclass
class FaultRule:
    """Injects a fault into matching tool calls.

    kinds: transient | permission | outage (no side effect, error returned)
           timeout_after_commit (side effect happens, caller sees 504)
           silent_drop (caller sees success, the change never lands)
    """
    tool: str                         # exact tool name, or a prefix ending in "*"
    kind: str
    probability: float = 1.0
    nth_call: Optional[int] = None    # fire only on the nth matching call (1-based)
    max_fires: Optional[int] = None
    start_tick: int = 0
    end_tick: Optional[int] = None    # active while start_tick <= tick < end_tick
    calls: int = 0
    fires: int = 0

    def matches(self, name: str) -> bool:
        return name == self.tool or (self.tool.endswith("*") and name.startswith(self.tool[:-1]))


# --------------------------------------------------------------------------- environment

class Environment:
    def __init__(self, seed: int = 0, faults: list[FaultRule] | tuple = (),
                 policy: Optional[OrgPolicy] = None) -> None:
        self.rng = random.Random(seed)
        self.tick = 0
        self.policy = policy or OrgPolicy()
        self.workday = Workday()
        self.identity = Identity()
        self.finance = Finance()
        self.itsm = ITSM()
        self.legal = Legal()
        self.faults = list(faults)
        self.audit: list[dict] = []
        self.escalations: list[dict] = []
        self.fix_requests: dict[str, Any] = {}     # request id -> owners.FixRequest
        self.owner_delays: dict[str, int] = {}     # issue -> ticks for the owning team to act
        self.identity_truth: dict[str, Optional[str]] = {}  # new worker wid -> former worker wid HR would confirm
        self.scheduled: list[tuple[int, int, str, Callable]] = []
        self.current_fault: Optional[str] = None  # visible to handlers during a call

    def new_wid(self) -> str:
        return "%032x" % self.rng.getrandbits(128)

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{self.rng.getrandbits(32):08x}"

    # ---- business dates
    def today(self) -> date:
        return BASE_DATE + timedelta(days=self.tick // TICKS_PER_DAY)

    def date_tick(self, iso_date: str) -> int:
        """Tick at which a business date starts (negative for dates before BASE_DATE)."""
        return (date.fromisoformat(iso_date) - BASE_DATE).days * TICKS_PER_DAY

    def date_str(self, days_from_today: int = 0) -> str:
        return (self.today() + timedelta(days=days_from_today)).isoformat()

    # ---- scheduled things that happen while a strategy is running (e.g. a manager change)
    def schedule(self, at_tick: int, fn: Callable[[], None], label: str = "") -> None:
        self.scheduled.append((at_tick, len(self.scheduled), label, fn))

    def advance(self, ticks: int = 1) -> None:
        from .owners import process_fix_requests  # local import: owners registers tools
        for _ in range(ticks):
            self.tick += 1
            due = sorted(s for s in self.scheduled if s[0] <= self.tick)
            self.scheduled = [s for s in self.scheduled if s[0] > self.tick]
            for _, _, _, fn in due:
                fn()
            self.workday.process_due(self.tick)
            self.finance.process_due(self.tick, self.workday)
            process_fix_requests(self)

    def _fault_for(self, name: str) -> Optional[str]:
        for rule in self.faults:
            if not rule.matches(name):
                continue
            rule.calls += 1
            if self.tick < rule.start_tick or (rule.end_tick is not None and self.tick >= rule.end_tick):
                continue
            if rule.nth_call is not None and rule.calls != rule.nth_call:
                continue
            if rule.max_fires is not None and rule.fires >= rule.max_fires:
                continue
            if self.rng.random() < rule.probability:
                rule.fires += 1
                return rule.kind
        return None

    def call(self, name: str, args: dict, actor: str = "agent") -> dict:
        """Execute one tool call and record it in the audit log."""
        self.advance()
        spec = REGISTRY.get(name)
        if spec is None:
            return self._log(name, None, None, args, err(404, f"Unknown tool '{name}'"), actor)
        errors = validate(spec.schema, args)
        if errors:
            return self._log(name, spec.system, None, args, err(400, "; ".join(errors)), actor)

        fault = self._fault_for(name)
        if fault in FAULT_ERRORS:
            return self._log(name, spec.system, spec.tier_for(args), args,
                             err(*FAULT_ERRORS[fault]), actor, fault)

        self.current_fault = fault
        try:
            result = spec.handler(self, **args)
        finally:
            self.current_fault = None
        committed = result.get("ok", False)
        if fault == "timeout_after_commit" and committed:
            result = err(504, "Gateway timeout; the request may or may not have been processed")
        return self._log(name, spec.system, spec.tier_for(args), args, result, actor, fault, committed)

    def _log(self, name, system, tier, args, result, actor, fault=None, committed=False) -> dict:
        self.audit.append({
            "tick": self.tick, "actor": actor, "tool": name, "system": system, "tier": tier,
            "args": copy.deepcopy(args), "ok": result.get("ok", False),
            "committed": committed, "fault": fault, "result": copy.deepcopy(result),
        })
        return result
