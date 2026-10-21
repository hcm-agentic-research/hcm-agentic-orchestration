"""The strategies compared in Paper A.

- FixedFlow: a conventional event-driven integration, built well for the KNOWN cases
  (retries with idempotency keys, dead-letter to a human on errors). No LLM.
- SingleAgent: one LLM agent with every tool, no guardrails.
- MultiAgent: Investigator -> Planner <-> Critic -> per-system Executors -> Reconciler,
  with bounded autonomy (gate) and untrusted-text isolation. Each mechanism can be
  switched off for the unguarded variant and for ablations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from wdsim.env import REGISTRY

from . import prompts
from .harness import Gate, extract_json, run_agent
from .llm import ModelConfig, Usage

READ_TOOLS = sorted(n for n, s in REGISTRY.items() if s.tier == "T0"
                    and n not in ("hitl_escalate", "owner_request_fix", "sys_wait"))
COMMON_TOOLS = ["sys_wait", "sys_get_time", "owner_request_fix", "owner_get_request", "hitl_escalate",
                "wd_get_worker", "wd_search_workers", "org_get_policy", "legal_get_holds"]
SYSTEM_PREFIX = {"identity": "idm_", "workday": "wd_", "finance": "fin_", "itsm": "itsm_"}
SYSTEM_ORDER = ["identity", "workday", "finance", "itsm"]


def tool_catalog() -> str:
    """Every tool with its system, risk tier and description, so roles that plan or review
    (but cannot change anything) still name the exact tools executors will use."""
    lines = []
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        tier = spec.tier if isinstance(spec.tier, str) else "tier depends on arguments"
        lines.append(f"- {name} [{spec.system}, {tier}]: {spec.description}")
    return "\n".join(lines)


TOOL_CATALOG_NOTE = ("Complete tool catalog (executors use these exact names; higher-risk tools run only if the "
                     "approved plan names them exactly):\n")


def plan_segments(steps: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split an ordered plan into consecutive same-system blocks, keeping the plan's order, so an
    interleaved plan (identity -> workday writeback -> identity ...) runs in the order planned."""
    segments: list[tuple[str, list[dict]]] = []
    for s in steps:
        system = s.get("system")
        if system not in SYSTEM_ORDER and system != "owner":
            continue  # "human" steps are escalated by the coordinator after execution
        if segments and segments[-1][0] == system:
            segments[-1][1].append(s)
        else:
            segments.append((system, [s]))
    return segments


def system_tools(system: str) -> list[str]:
    """Tools for one executor. The 'owner' executor (requests to owning teams, e.g. HR identity
    confirmation) gets only the shared coordination and read tools."""
    own = [n for n in REGISTRY if n.startswith(SYSTEM_PREFIX[system])] if system in SYSTEM_PREFIX else []
    return sorted(set(own) | set(COMMON_TOOLS))


@dataclass
class RunInfo:
    strategy: str
    usage: Usage = field(default_factory=Usage)
    llm_turns: dict = field(default_factory=dict)   # role -> number of agent runs
    blocked: int = 0
    step_limits: int = 0
    notes: list = field(default_factory=list)
    transcript: list = field(default_factory=list)  # one entry per agent run, saved in the run trace
    conversations: list = field(default_factory=list, repr=False)

    def finish(self) -> "RunInfo":
        """Sum token usage over every conversation exactly once (continued ones included)."""
        for conv in self.conversations:
            self.usage.add(conv.usage)
        self.conversations = []
        return self


# =========================================================================== fixed flow

class FixedFlow:
    """Point-to-point integration as typically built: one handler per event type."""
    name = "fixed_flow"
    actor = "fixed_flow"

    def __init__(self, quiet_minutes: int = 45) -> None:
        self.quiet_minutes = quiet_minutes

    def call(self, env, tool, args, retries=3):
        for attempt in range(retries + 1):
            r = env.call(tool, args, actor=self.actor)
            if r["ok"] or r["error"]["code"] not in (429, 503, 504) or attempt == retries:
                return r
            env.call("sys_wait", {"minutes": 2 ** attempt}, actor=self.actor)
        return r

    def dead_letter(self, env, case, tool, r):
        env.call("hitl_escalate", {"case_id": case, "reason": f"{tool} failed: {r['error']['message']}",
                                   "proposed_action": f"Investigate and complete {tool} manually"}, actor=self.actor)

    def must(self, env, case, tool, args) -> Optional[dict]:
        r = self.call(env, tool, args)
        if not r["ok"]:
            self.dead_letter(env, case, tool, r)
            return None
        return r["data"]

    def run(self, env) -> RunInfo:
        seq, quiet = 0, 0
        while quiet * 15 < self.quiet_minutes:
            events = self.call(env, "wd_get_events", {"since_seq": seq})["data"]
            if events:
                seq = max(e["seq"] for e in events)
            work = [e for e in events if e["type"] != "WORKER_CONTACT_CHANGED"]
            if not work:
                quiet += 1
                env.call("sys_wait", {"minutes": 15}, actor=self.actor)
                continue
            quiet = 0
            for e in work:
                handler = getattr(self, "on_" + e["type"].lower(), None)
                if handler:
                    handler(env, e)
        return RunInfo(self.name)

    def on_hire(self, env, e):
        case = e["correlation_id"]
        w = self.must(env, case, "wd_get_worker", {"worker": e["worker_wid"]})
        if not w:
            return
        first, last = w["name"].lower().split(" ", 1)
        upn = f"{first}.{last.replace(' ', '')}@example.com"
        acct = self.must(env, case, "idm_create_account", {
            "worker_wid": w["worker_wid"], "upn": upn, "display_name": w["name"],
            "idempotency_key": f"{case}-account", "employee_type": w["worker_type"]})
        if acct:
            for g in self.must(env, case, "idm_get_birthright", {"job_profile": w["job_profile"]}) or []:
                self.must(env, case, "idm_add_group", {"account_id": acct["account_id"], "group": g["group"]})
            self.must(env, case, "wd_update_work_email", {"worker": w["worker_wid"], "email": upn,
                                                          "idempotency_key": f"{case}-wb", "correlation_id": case})
        mapping = self.must(env, case, "fin_map_cost_center", {"workday_cost_center_wid": w["cost_center"]["wid"]})
        if mapping and w["manager"]:
            self.must(env, case, "fin_create_profile", {
                "worker_wid": w["worker_wid"], "cost_center_code": mapping["cost_center_code"],
                "approver_worker_wid": w["manager"]["worker_wid"], "idempotency_key": f"{case}-fin"})
        bundle = "engineering" if "Engineer" in w["job_profile"] else "standard"
        self.must(env, case, "itsm_order_equipment", {"worker_wid": w["worker_wid"], "bundle": bundle,
                                                      "idempotency_key": f"{case}-eq"})
        for t in self.must(env, case, "wd_get_onboarding_tasks", {"worker": w["worker_wid"]}) or []:
            self.must(env, case, "wd_update_onboarding_task", {"task_wid": t["task_wid"], "status": "Complete"})

    def on_termination(self, env, e):
        case = e["correlation_id"]
        wid = e["worker_wid"]
        for a in self.must(env, case, "idm_search_accounts", {"query": wid}) or []:
            aid = a["account_id"]
            self.must(env, case, "idm_disable_account", {"account_id": aid})
            self.must(env, case, "idm_revoke_sessions", {"account_id": aid})
            for g in a["groups"]:
                self.must(env, case, "idm_remove_group", {"account_id": aid, "group": g})
            self.must(env, case, "idm_remove_licenses", {"account_id": aid})
            self.must(env, case, "idm_set_mailbox", {"account_id": aid, "mode": "shared"})
        profile = self.call(env, "fin_get_profile", {"worker_wid": wid})
        if profile["ok"]:
            if profile["data"]["card_active"]:
                self.must(env, case, "fin_deactivate_card", {"worker_wid": wid})
            self.must(env, case, "fin_close_expense_reports", {"worker_wid": wid})
            self.must(env, case, "fin_request_final_pay", {"worker_wid": wid})
        self.must(env, case, "itsm_create_asset_recovery", {"worker_wid": wid, "idempotency_key": f"{case}-asset"})

    def on_job_change(self, env, e):
        case = e["correlation_id"]
        w = self.must(env, case, "wd_get_worker", {"worker": e["worker_wid"]})
        if not w:
            return
        new = {g["group"] for g in self.must(env, case, "idm_get_birthright", {"job_profile": w["job_profile"]}) or []}
        old_job = e.get("before", {}).get("job_profile")
        old = {g["group"] for g in (self.must(env, case, "idm_get_birthright", {"job_profile": old_job}) or [])} \
            if old_job else set()
        for a in self.must(env, case, "idm_search_accounts", {"query": w["worker_wid"]}) or []:
            for g in sorted(new - set(a["groups"])):        # grant new access first
                self.must(env, case, "idm_add_group", {"account_id": a["account_id"], "group": g})
            for g in sorted((old - new) & set(a["groups"])):  # then remove the old role's access
                self.must(env, case, "idm_remove_group", {"account_id": a["account_id"], "group": g})
        mapping = self.must(env, case, "fin_map_cost_center", {"workday_cost_center_wid": w["cost_center"]["wid"]})
        if mapping:
            self.must(env, case, "fin_update_cost_center", {"worker_wid": w["worker_wid"],
                                                            "cost_center_code": mapping["cost_center_code"]})
        if w["manager"]:
            self.must(env, case, "fin_set_approver", {"worker_wid": w["worker_wid"],
                                                      "approver_worker_wid": w["manager"]["worker_wid"]})

    def on_manager_change(self, env, e):
        case = e["correlation_id"]
        self.must(env, case, "fin_set_approver", {"worker_wid": e["worker_wid"],
                                                  "approver_worker_wid": e["new_manager_wid"]})


# =========================================================================== single agent

class SingleAgent:
    name = "single_agent"

    def __init__(self, cfg: ModelConfig, max_steps: int = 250) -> None:
        self.cfg = cfg
        self.max_steps = max_steps

    def run(self, env) -> RunInfo:
        info = RunInfo(self.name)
        run, _ = run_agent(env, self.cfg, "single_agent", prompts.SINGLE_AGENT, sorted(REGISTRY),
                           "Start now: read the event feed and handle every change.", max_steps=self.max_steps)
        info.usage.add(run.usage)
        info.transcript.append({"role": "single_agent", "tick": env.tick, "stop": run.stop,
                                "tool_calls": run.tool_calls, "blocked": run.blocked, "final_text": run.final_text})
        info.llm_turns["single_agent"] = 1
        info.step_limits += run.stop == "step_limit"
        return info


# =========================================================================== multi-agent

@dataclass
class MultiAgentConfig:
    investigator: bool = True
    critic: bool = True
    reconciler: bool = True
    gating: bool = True
    isolation: bool = True
    max_revisions: int = 2
    repair_cycles: int = 1
    quiet_minutes: int = 45

    @classmethod
    def unguarded(cls) -> "MultiAgentConfig":
        return cls(investigator=False, critic=False, reconciler=False, gating=False, isolation=False)


class MultiAgent:
    def __init__(self, cfg: ModelConfig, mcfg: Optional[MultiAgentConfig] = None, name: str = "multi_agent") -> None:
        self.cfg = cfg
        self.m = mcfg or MultiAgentConfig()
        self.name = name

    # ---------------------------------------------------------------- helpers
    def _agent(self, env, info, role, system, tools, task, gate=None, conversation=None, max_steps=60):
        run, conv = run_agent(env, self.cfg, role, system, tools, task, gate=gate,
                              isolate=self.m.isolation, max_steps=max_steps, conversation=conversation)
        if conversation is None:
            info.conversations.append(conv)
        info.transcript.append({"role": role, "tick": env.tick, "stop": run.stop, "tool_calls": run.tool_calls,
                                "blocked": run.blocked, "final_text": run.final_text})
        info.llm_turns[role] = info.llm_turns.get(role, 0) + 1
        info.blocked += run.blocked
        info.step_limits += run.stop == "step_limit"
        return run, conv

    def run(self, env) -> RunInfo:
        info = RunInfo(self.name)
        seq, quiet, case_log = 0, 0, []
        while quiet * 15 < self.m.quiet_minutes:
            events = env.call("wd_get_events", {"since_seq": seq}, actor="coordinator")["data"]
            if events:
                seq = max(e["seq"] for e in events)
            work = [e for e in events if e["type"] != "WORKER_CONTACT_CHANGED"]
            if not work:
                quiet += 1
                env.call("sys_wait", {"minutes": 15}, actor="coordinator")
                continue
            quiet = 0
            self.handle_batch(env, info, work, case_log)
        return info.finish()

    def handle_batch(self, env, info: RunInfo, events: list[dict], case_log: list[str]) -> None:
        events_text = json.dumps(events, indent=1, default=str)
        history = "\n".join(case_log[-8:]) or "(nothing handled yet)"
        context = f"New Workday events:\n{events_text}\n\nWhat the team has already done earlier:\n{history}"

        findings = "(no separate investigation)"
        if self.m.investigator:
            run, _ = self._agent(env, info, "investigator", prompts.INVESTIGATOR, READ_TOOLS, context)
            findings = run.final_text

        planner_tools = READ_TOOLS
        catalog = TOOL_CATALOG_NOTE + tool_catalog()
        plan_run, planner_conv = self._agent(env, info, "planner", prompts.PLANNER, planner_tools,
                                             f"{context}\n\nInvestigator findings:\n{findings}\n\n{catalog}\n\n"
                                             "Produce the plan.")
        plan = extract_json(plan_run.final_text) or {}

        for cycle in range(self.m.repair_cycles + 1):
            if self.m.critic:
                for _ in range(self.m.max_revisions):
                    crit, _ = self._agent(env, info, "critic", prompts.CRITIC, READ_TOOLS,
                                          f"{context}\n\nFindings:\n{findings}\n\n{catalog}\n\nProposed plan:\n"
                                          f"{json.dumps(plan, indent=1)}")
                    verdict = extract_json(crit.final_text) or {}
                    if verdict.get("verdict") == "approve":
                        break
                    plan_run, _ = self._agent(env, info, "planner", prompts.PLANNER, planner_tools,
                                              "The Critic asked for revisions:\n"
                                              f"{json.dumps(verdict.get('issues', crit.final_text), indent=1)}\n"
                                              "Return the full revised plan.", conversation=planner_conv)
                    plan = extract_json(plan_run.final_text) or plan
                else:
                    info.notes.append("plan not approved after revisions; executing latest plan under the gate")

            reports = self.execute(env, info, plan, context)
            case_log.append(f"Plan summary: {json.dumps(plan.get('cases', []), default=str)}\n" + "\n".join(reports))

            if not self.m.reconciler or cycle == self.m.repair_cycles:
                break
            rec, _ = self._agent(env, info, "reconciler", prompts.RECONCILER, READ_TOOLS,
                                 f"{context}\n\nExecution reports:\n" + "\n".join(reports))
            result = extract_json(rec.final_text) or {}
            if result.get("consistent", True) or not result.get("discrepancies"):
                break
            plan_run, _ = self._agent(env, info, "planner", prompts.PLANNER, planner_tools,
                                      "The Reconciler found these discrepancies after execution:\n"
                                      f"{json.dumps(result['discrepancies'], indent=1)}\n"
                                      "Plan only the repairs needed.", conversation=planner_conv)
            plan = extract_json(plan_run.final_text) or {}

    def execute(self, env, info: RunInfo, plan: dict, context: str) -> list[str]:
        steps = sorted(plan.get("steps", []), key=lambda s: s.get("order", 0))
        gate = None
        if self.m.gating:
            gate = Gate(env, {t for s in steps for t in s.get("tools", [])})
        reports = []
        for system, block in plan_segments(steps):
            task = (f"{context}\n\nYour steps (system: {system}):\n{json.dumps(block, indent=1)}\n\n"
                    f"Reports from executors that ran before you:\n" + ("\n".join(reports) or "(none)"))
            run, _ = self._agent(env, info, f"executor_{system}", prompts.EXECUTOR.format(system=system),
                                 system_tools(system), task, gate=gate, max_steps=80)
            reports.append(f"[{system}] {run.final_text}")
        for s in steps:
            if s.get("system") == "human":
                env.call("hitl_escalate", {"case_id": s.get("case_id", "unknown"), "reason": s.get("action", ""),
                                           "proposed_action": s.get("action", "")}, actor="planner")
        return reports


def make_strategy(name: str, cfg: Optional[ModelConfig]):
    if name == "fixed_flow":
        return FixedFlow()
    if name == "single_agent":
        return SingleAgent(cfg)
    if name == "multi_agent":
        return MultiAgent(cfg)
    if name == "multi_agent_unguarded":
        return MultiAgent(cfg, MultiAgentConfig.unguarded(), name)
    if name.startswith("multi_agent_no_"):  # ablation, e.g. multi_agent_no_critic
        mechanism = name[len("multi_agent_no_"):]
        mcfg = MultiAgentConfig(**{mechanism: False})
        return MultiAgent(cfg, mcfg, name)
    raise ValueError(f"unknown strategy {name}")
