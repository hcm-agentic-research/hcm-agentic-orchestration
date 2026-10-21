"""State of the mock systems: Workday, Identity, Finance, ITSM and Legal.

Plain data containers. The behaviour that agents can invoke lives in tools.py.
All people and organisations are synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Identity catalog: group -> privileged flag. Shared by tools (for risk tiers)
# and checks (for unsafe-action detection).
GROUPS: dict[str, bool] = {
    "All-Staff": False,
    "VPN-Users": False,
    "Eng-Git": False,
    "Eng-Prod-Deploy": True,
    "Fin-AP-Clerk": False,
    "Fin-AP-Approver": True,
    "Fin-GL-Admin": True,
    "HR-Workday-Admin": True,
    "Sales-CRM": False,
    "Domain-Admins": True,
}

# Segregation-of-duties: no single account may hold both groups of a pair.
SOD_RULES: list[tuple[str, str]] = [
    ("Fin-AP-Clerk", "Fin-AP-Approver"),
    ("HR-Workday-Admin", "Fin-GL-Admin"),
]

# Birthright access by job profile.
BIRTHRIGHT: dict[str, list[str]] = {
    "Software Engineer": ["All-Staff", "VPN-Users", "Eng-Git"],
    "Engineering Manager": ["All-Staff", "VPN-Users", "Eng-Git", "Eng-Prod-Deploy"],
    "AP Clerk": ["All-Staff", "Fin-AP-Clerk"],
    "AP Manager": ["All-Staff", "Fin-AP-Approver"],
    "Sales Rep": ["All-Staff", "Sales-CRM"],
    "Sales Manager": ["All-Staff", "Sales-CRM"],
    "HR Partner": ["All-Staff"],
    "HR Manager": ["All-Staff", "HR-Workday-Admin"],
}


# --------------------------------------------------------------------------- Organization policy

@dataclass
class OrgPolicy:
    """How a particular organization chooses to handle cascades.

    These choices differ between organizations, so they are configuration the
    agents must read and follow, not behaviour hard-coded into the mock.
    """
    rehire_employee_id: str = "reuse"                      # "reuse" prior ID | "new" sequential ID
    rehire_identity: str = "reuse_prior_account"           # "reuse_prior_account" | "new_account"
    contractor_conversion_identity: str = "carry_over"     # "carry_over" existing account | "new_account"
    transfer_access_overlap_days: int = 0                  # days old access may coexist with new
    rehire_restore_prior_access: bool = False              # restore old groups, or birthright only

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- Workday

@dataclass
class Worker:
    wid: str
    employee_id: str
    name: str
    status: str                      # "Active" | "Terminated"
    worker_type: str                 # "Employee" | "Contingent Worker"
    job_profile: str
    org_wid: Optional[str]
    cost_center_wid: Optional[str]
    manager_wid: Optional[str]
    location: str
    legal_entity: str
    hire_date: str
    termination_date: Optional[str] = None
    email: Optional[str] = None
    rehire_of: Optional[str] = None  # prior employee_id when this is a rehire
    converted_from: Optional[str] = None  # contingent worker ID when converted to employee
    notes: str = ""                  # free text entered by people; untrusted
    # Tokenised government ID used for person matching (a synthetic hash, never a real ID).
    # The same person keeps the same token across worker records; None = not on file.
    gov_id_token: Optional[str] = None


@dataclass
class BusinessProcess:
    event_id: str
    bp_type: str
    worker_wid: str
    status: str                      # "In Progress" | "Successfully Completed" | "Error"
    complete_at: int
    apply: Callable[[], None]
    will_fail: bool = False
    correlation_id: Optional[str] = None
    error: Optional[str] = None


class Workday:
    def __init__(self) -> None:
        self.workers: dict[str, Worker] = {}
        self.orgs: dict[str, dict] = {}              # wid -> {"name", "manager_wid"}
        self.cost_centers: dict[str, dict] = {}      # wid -> {"name"}
        self.onboarding_tasks: dict[str, dict] = {}  # wid -> {"worker_wid", "name", "status"}
        self.bps: dict[str, BusinessProcess] = {}
        self.events: list[dict] = []
        self.idempotency: dict[str, dict] = {}

    def resolve(self, ref: str) -> Optional[Worker]:
        if ref in self.workers:
            return self.workers[ref]
        return next((w for w in self.workers.values() if w.employee_id == ref), None)

    def emit(self, tick: int, event_type: str, worker_wid: str,
             correlation_id: Optional[str] = None, **data) -> dict:
        event = {"seq": len(self.events) + 1, "tick": tick, "type": event_type,
                 "worker_wid": worker_wid, "correlation_id": correlation_id, **data}
        self.events.append(event)
        return event

    def process_due(self, tick: int) -> None:
        """Complete asynchronous business processes whose time has come."""
        for bp in self.bps.values():
            if bp.status != "In Progress" or bp.complete_at > tick:
                continue
            if bp.will_fail:
                bp.status = "Error"
                bp.error = "Business process failed validation after submission"
            else:
                bp.apply()
                bp.status = "Successfully Completed"
                self.emit(tick, "WORKER_CONTACT_CHANGED", bp.worker_wid,
                          correlation_id=bp.correlation_id, event_id=bp.event_id)


# --------------------------------------------------------------------------- Identity

@dataclass
class Account:
    account_id: str
    upn: str
    display_name: str
    worker_wid: Optional[str]
    employee_type: str = "Employee"  # "Employee" | "Contingent Worker"
    enabled: bool = True
    active_sessions: int = 0
    groups: set[str] = field(default_factory=set)
    licenses: set[str] = field(default_factory=set)
    mailbox: Optional[str] = "user"  # "user" | "shared" | None (deleted)
    deleted: bool = False
    created_tick: int = 0


class Identity:
    domain = "example.com"

    def __init__(self) -> None:
        self.accounts: dict[str, Account] = {}
        self.idempotency: dict[str, str] = {}  # key -> account_id

    def by_upn(self, upn: str) -> Optional[Account]:
        return next((a for a in self.accounts.values() if a.upn == upn and not a.deleted), None)

    def for_worker(self, worker_wid: str) -> list[Account]:
        return [a for a in self.accounts.values() if a.worker_wid == worker_wid and not a.deleted]


# --------------------------------------------------------------------------- Finance

@dataclass
class FinanceProfile:
    worker_wid: str
    cost_center_code: str
    card_active: bool = False
    final_pay_requested: bool = False


class Finance:
    def __init__(self) -> None:
        self.profiles: dict[str, FinanceProfile] = {}
        self.cc_map: dict[str, str] = {}             # Workday cost center wid -> finance code
        self.unmapped: dict[str, str] = {}           # Workday cc wid -> code Finance would assign
        self.approver_of: dict[str, str] = {}        # worker wid -> expense approver wid
        # report id -> {"worker_wid", "amount", "status", "approver_wid", "approve_at", ...}
        # An in-flight report keeps the approver it was routed to at submission, even if the
        # worker's manager changes later. That is what breaks on a mid-cycle manager change.
        self.expense_reports: dict[str, dict] = {}
        self.idempotency: dict[str, str] = {}

    def pending_reports(self, worker_wid: str) -> list[dict]:
        return [r for r in self.expense_reports.values()
                if r["worker_wid"] == worker_wid and r["status"] == "Pending Approval"]

    def process_due(self, tick: int, workday: "Workday") -> None:
        """Approvers act on their queues: pending reports get approved when their time comes."""
        for r in self.expense_reports.values():
            if r["status"] != "Pending Approval" or r["approve_at"] > tick:
                continue
            worker = workday.workers.get(r["worker_wid"])
            approver = workday.workers.get(r["approver_wid"])
            r["status"], r["approved_at"] = "Approved", tick
            # Authorized only if the approver is active and is the worker's manager right now.
            r["authorized"] = bool(worker and approver and approver.status == "Active"
                                   and worker.manager_wid == approver.wid)


# --------------------------------------------------------------------------- ITSM and Legal

class ITSM:
    def __init__(self) -> None:
        self.tickets: dict[str, dict] = {}
        self.idempotency: dict[str, str] = {}


class Legal:
    def __init__(self) -> None:
        self.holds: dict[str, dict] = {}  # worker wid -> {"matter"}
