"""Simulated owning teams (Finance, Identity, HR) that act on fix requests.

When a downstream system rejects part of a change, a strategy can ask the owning team
for a fix. The team acts after a delay. A precise request (right issue, right record)
gets resolved; a vague or wrong one gets declined. Every request is a human touch.
This is what makes "reapply after the fix" and "reroute when declined" measurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .env import TICKS_PER_HOUR, err, ok, tool

DEFAULT_DELAY = 4 * TICKS_PER_HOUR  # a team acts on a precise request within half a working day

ISSUES = ["missing_cost_center_mapping", "permission_grant", "sod_exception", "identity_confirmation", "other"]


@dataclass
class FixRequest:
    request_id: str
    case_id: str
    team: str
    issue: str
    subject_ref: str
    details: str
    created_tick: int
    resolve_at: int
    status: str = "Open"           # "Open" | "Resolved" | "Declined"
    resolution: Optional[str] = None

    def view(self) -> dict:
        return dict(self.__dict__)


def _resolve(env, req: FixRequest) -> tuple[str, str]:
    if req.issue == "missing_cost_center_mapping":
        if req.team != "finance" or req.subject_ref not in env.finance.unmapped:
            return "Declined", "No unmapped Workday cost center matches the reference given"
        code = env.finance.unmapped.pop(req.subject_ref)
        env.finance.cc_map[req.subject_ref] = code
        return "Resolved", f"Mapped Workday cost center to {code}"
    if req.issue == "permission_grant":
        rules = [r for r in env.faults if r.kind == "permission" and r.matches(req.subject_ref)]
        if not rules:
            return "Declined", "The integration user already has that permission"
        for r in rules:
            r.max_fires = r.fires  # permission granted: the rule stops firing
        return "Resolved", f"Granted the integration user permission for {req.subject_ref}"
    if req.issue == "identity_confirmation":
        # HR confirms whether a new worker record is a returning person (subject = new worker WID).
        if req.team != "hr" or req.subject_ref not in env.identity_truth:
            return "Declined", "HR cannot confirm: give the new worker's WID as the reference"
        former = env.identity_truth[req.subject_ref]
        if former is None:
            return "Resolved", f"HR confirms worker {req.subject_ref} is not a returning worker; treat as a new hire."
        w = env.workday.workers[former]
        return "Resolved", (f"HR confirms worker {req.subject_ref} is the same person as former worker "
                            f"{former} (employee ID {w.employee_id}).")
    if req.issue == "sod_exception":
        return "Declined", "Segregation-of-duties exceptions are not granted for this combination"
    return "Declined", "Request not actionable: name the issue type and the exact record"


def process_fix_requests(env) -> None:
    for req in env.fix_requests.values():
        if req.status == "Open" and req.resolve_at <= env.tick:
            req.status, req.resolution = _resolve(env, req)


@tool("owner_request_fix", "system", "T0",
      "Ask the team that owns a downstream system to fix a data or policy problem that blocks a change "
      "(e.g. a Workday cost center with no Finance mapping, a missing integration permission). The team "
      "acts after a delay; poll owner_get_request. HR can also confirm whether a new worker record belongs to "
      "a returning person (issue identity_confirmation, reference = the new worker's WID). Be precise: give the "
      "exact issue and the exact record reference (e.g. the Workday cost center WID, or the tool name for a "
      "permission). Each request costs human effort.",
      {"case_id": {"type": "string"},
       "team": {"type": "string", "enum": ["finance", "identity", "hr"]},
       "issue": {"type": "string", "enum": ISSUES},
       "subject_ref": {"type": "string", "description": "Exact reference of the record or permission to fix."},
       "details": {"type": "string"}},
      ["case_id", "team", "issue", "subject_ref", "details"])
def owner_request_fix(env, case_id, team, issue, subject_ref, details):
    req = FixRequest(env.new_id("req"), case_id, team, issue, subject_ref, details, env.tick,
                     env.tick + env.owner_delays.get(issue, DEFAULT_DELAY))
    env.fix_requests[req.request_id] = req
    return ok({"request_id": req.request_id, "status": req.status})


@tool("owner_get_request", "system", "T0", "Get the status and resolution of a fix request.",
      {"request_id": {"type": "string"}}, ["request_id"])
def owner_get_request(env, request_id):
    req = env.fix_requests.get(request_id)
    return ok(req.view()) if req else err(404, f"No fix request {request_id}")
