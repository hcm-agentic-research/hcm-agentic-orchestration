"""Builds a synthetic baseline company and raises Workday events for cascades.

No real people, employers or tenant data. Everything is generated from a seed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .systems import BIRTHRIGHT, Account, FinanceProfile, Worker


def gov_token(person_key: str) -> str:
    """Synthetic tokenised government ID for a person (derived without using the seeded RNG,
    so adding it does not change the generated company)."""
    return "gid-" + hashlib.sha256(person_key.encode()).hexdigest()[:12]

FIRST_NAMES = ["Avery", "Jordan", "Priya", "Mateo", "Lena", "Kofi", "Hana", "Diego", "Mira", "Tomas",
               "Aisha", "Noah", "Ines", "Ravi", "Sofia", "Kenji", "Leila", "Owen", "Zara", "Felix"]
LAST_NAMES = ["Carter", "Nguyen", "Okafor", "Silva", "Fischer", "Patel", "Kowalski", "Haddad",
              "Larsen", "Moreau", "Tanaka", "Alvarez", "Brennan", "Iyer", "Novak", "Mensah"]

# org name -> (manager job profile, staff job profiles, cost center code, finance-mapped?)
ORGS = {
    "Engineering": ("Engineering Manager", ["Software Engineer"], "CC-1000", True),
    "Finance": ("AP Manager", ["AP Clerk"], "CC-2000", True),
    "Sales": ("Sales Manager", ["Sales Rep"], "CC-3000", True),
    "Human Resources": ("HR Manager", ["HR Partner"], "CC-4000", True),
    # Exists in Workday but was never mapped in Finance: a realistic integration gap.
    "Data Platform": ("Engineering Manager", ["Software Engineer"], "CC-5000", False),
}


@dataclass
class Company:
    org_wids: dict[str, str] = field(default_factory=dict)         # org name -> wid
    cc_wids: dict[str, str] = field(default_factory=dict)          # org name -> cost center wid
    managers: dict[str, str] = field(default_factory=dict)         # org name -> manager worker wid
    used_names: set[str] = field(default_factory=set)
    next_employee_number: int = 1001
    next_contingent_number: int = 5001

    def next_employee_id(self) -> str:
        eid = f"E{self.next_employee_number}"
        self.next_employee_number += 1
        return eid

    def next_contingent_id(self) -> str:
        cid = f"C{self.next_contingent_number}"
        self.next_contingent_number += 1
        return cid


def _upn(name: str) -> str:
    first, last = name.lower().split(" ", 1)
    return f"{first}.{last.replace(' ', '')}@example.com"


def _add_worker(env, company: Company, name: str, job: str, org: str, manager_wid: Optional[str],
                status: str = "Active", hire_date: str = "2023-03-01",
                worker_type: str = "Employee") -> Worker:
    """Create a worker whose systems are already consistent (baseline population)."""
    contingent = worker_type == "Contingent Worker"
    worker_id = company.next_contingent_id() if contingent else company.next_employee_id()
    w = Worker(wid=env.new_wid(), employee_id=worker_id, name=name, status=status,
               worker_type=worker_type, job_profile=job, org_wid=company.org_wids[org],
               cost_center_wid=company.cc_wids[org], manager_wid=manager_wid,
               location="Austin, TX", legal_entity="Example Corp US", hire_date=hire_date,
               gov_id_token=gov_token(f"person:{name}:{worker_id}"))
    env.workday.workers[w.wid] = w
    w.email = _upn(name)
    if status == "Active":
        acct = Account(env.new_id("acct"), w.email, name, w.wid, employee_type=worker_type,
                       active_sessions=1, groups=set(BIRTHRIGHT[job]), licenses={"Office-E3"})
    else:  # historical leaver: account disabled and stripped
        w.termination_date = "2024-06-30"
        acct = Account(env.new_id("acct"), w.email, name, w.wid, employee_type=worker_type,
                       enabled=False, mailbox="shared")
    env.identity.accounts[acct.account_id] = acct
    if status == "Active":
        # Every current worker has equipment on record (fulfilled earlier). The ticket ID is derived
        # from the worker ID, not the seeded RNG, so the generated company is otherwise unchanged.
        ticket_id = f"tkt-eq-{worker_id}"
        env.itsm.tickets[ticket_id] = {"ticket_id": ticket_id, "worker_wid": w.wid, "type": "equipment_order",
                                       "status": "Fulfilled", "created_tick": -1,
                                       "bundle": "engineering" if "Engineer" in job else "standard"}
    code = ORGS[org][2]
    env.finance.profiles[w.wid] = FinanceProfile(w.wid, code, card_active=(status == "Active" and
                                                                          job.endswith("Manager")))
    if manager_wid:
        env.finance.approver_of[w.wid] = manager_wid
        if status == "Active" and env.rng.random() < 0.5:
            # An expense report already in the manager's approval queue. Managers work
            # through their queue within 1-5 days (one tick = one minute).
            report_id = env.new_id("exp")
            env.finance.expense_reports[report_id] = {
                "report_id": report_id, "worker_wid": w.wid, "amount": env.rng.randint(40, 2400),
                "status": "Pending Approval", "approver_wid": manager_wid,
                "approve_at": env.rng.randint(1, 5) * 1440 + env.rng.randint(0, 600)}
    return w


def build_company(env, staff_per_org: int = 3, leavers: int = 2, contractors: int = 2) -> Company:
    """A small, internally consistent company to run cascades against."""
    company = Company()
    for org, (_, _, code, mapped) in ORGS.items():
        company.org_wids[org] = env.new_wid()
        company.cc_wids[org] = env.new_wid()
        env.workday.orgs[company.org_wids[org]] = {"name": org, "manager_wid": None}
        env.workday.cost_centers[company.cc_wids[org]] = {"name": f"{code} {org}"}
        if mapped:
            env.finance.cc_map[company.cc_wids[org]] = code
        else:
            env.finance.unmapped[company.cc_wids[org]] = code

    def fresh_name() -> str:
        while True:
            name = f"{env.rng.choice(FIRST_NAMES)} {env.rng.choice(LAST_NAMES)}"
            if name not in company.used_names:
                company.used_names.add(name)
                return name

    for org, (mgr_job, staff_jobs, _, _) in ORGS.items():
        mgr = _add_worker(env, company, fresh_name(), mgr_job, org, None)
        company.managers[org] = mgr.wid
        env.workday.orgs[company.org_wids[org]]["manager_wid"] = mgr.wid
        for _ in range(staff_per_org):
            _add_worker(env, company, fresh_name(), env.rng.choice(staff_jobs), org, mgr.wid)
    for _ in range(leavers):
        org = env.rng.choice(list(ORGS))
        _add_worker(env, company, fresh_name(), ORGS[org][1][0], org, company.managers[org],
                    status="Terminated")
    for _ in range(contractors):
        _add_worker(env, company, fresh_name(), "Software Engineer", "Engineering",
                    company.managers["Engineering"], worker_type="Contingent Worker")
    return company


def add_former_worker(env, company: Company, name: str, org: str, upn: Optional[str] = None) -> Worker:
    """A historical leaver with a disabled account, e.g. to create name ambiguity on a rehire."""
    w = _add_worker(env, company, name, ORGS[org][1][0], org, company.managers[org], status="Terminated")
    if upn:
        w.email = upn
        env.identity.for_worker(w.wid)[0].upn = upn
    return w


# --------------------------------------------------------------------------- cascade triggers

def raise_hire(env, company: Company, name: str, job: str, org: str, hire_date: str = "2026-11-02",
               worker_type: str = "Employee", notes: str = "", rehire_of: Optional[str] = None,
               manager_wid: Optional[str] = "default", case_id: Optional[str] = None,
               gov_id: Optional[str] = "new") -> tuple[Worker, dict]:
    """A hire completed in Workday: worker exists with no email yet; HIRE event emitted.

    gov_id: "new" gives a new person's token; a token string marks an existing person;
    None means the government ID is not on file.
    """
    employee_id = company.next_employee_id()
    w = Worker(wid=env.new_wid(), employee_id=employee_id, name=name, status="Active",
               worker_type=worker_type, job_profile=job, org_wid=company.org_wids[org],
               cost_center_wid=company.cc_wids[org],
               manager_wid=company.managers[org] if manager_wid == "default" else manager_wid,
               location="Austin, TX", legal_entity="Example Corp US", hire_date=hire_date,
               notes=notes, rehire_of=rehire_of,
               gov_id_token=gov_token(f"person:{name}:{employee_id}") if gov_id == "new" else gov_id)
    env.workday.workers[w.wid] = w
    for task in ("Create accounts and email", "Provision equipment"):
        env.workday.onboarding_tasks[env.new_wid()] = {"worker_wid": w.wid, "name": task,
                                                       "status": "Not Started"}
    event = env.workday.emit(env.tick, "HIRE", w.wid, correlation_id=case_id or env.new_id("case"),
                             effective_date=hire_date)
    return w, event


def raise_rehire(env, company: Company, former: Worker, job: str, org: str,
                 hire_date: str = "2026-11-02", case_id: Optional[str] = None) -> tuple[Worker, dict]:
    """Rehire a former worker, following the organization's employee-ID policy.

    "reuse": the former worker record is reactivated with its prior ID (and its old email).
    "new":   a new worker record with a new sequential ID, linked back via rehire_of.
    Either way the person's old, disabled identity account still exists downstream.
    """
    assert former.status == "Terminated"
    if env.policy.rehire_employee_id == "reuse":
        w = former
        w.status, w.hire_date, w.termination_date = "Active", hire_date, None
        w.job_profile, w.org_wid = job, company.org_wids[org]
        w.cost_center_wid, w.manager_wid = company.cc_wids[org], company.managers[org]
        w.rehire_of = former.employee_id
    else:
        w = Worker(wid=env.new_wid(), employee_id=company.next_employee_id(), name=former.name,
                   status="Active", worker_type="Employee", job_profile=job, org_wid=company.org_wids[org],
                   cost_center_wid=company.cc_wids[org], manager_wid=company.managers[org],
                   location=former.location, legal_entity=former.legal_entity, hire_date=hire_date,
                   rehire_of=former.employee_id, gov_id_token=former.gov_id_token)
        env.workday.workers[w.wid] = w
    for task in ("Create accounts and email", "Provision equipment"):
        env.workday.onboarding_tasks[env.new_wid()] = {"worker_wid": w.wid, "name": task,
                                                       "status": "Not Started"}
    event = env.workday.emit(env.tick, "HIRE", w.wid, correlation_id=case_id or env.new_id("case"),
                             effective_date=hire_date, rehire=True)
    return w, event


def raise_contractor_conversion(env, company: Company, contractor: Worker, job: str, org: str,
                                effective_date: str = "2026-11-02",
                                hire_event_delay: int = 0) -> tuple[Worker, list[dict]]:
    """Convert a contingent worker to an employee the way Workday typically does it.

    The contingent assignment ends (a TERMINATION event) and the person is hired as an
    employee on a new worker record (a HIRE event). The two events arrive separately with
    different correlation IDs; nothing in them says they are one person. A naive cascade
    disables the person's account and creates a second one. With hire_event_delay > 0 the
    HIRE event arrives that many ticks later (the new record already exists in Workday).
    """
    assert contractor.worker_type == "Contingent Worker" and contractor.status == "Active"
    contractor.status, contractor.termination_date = "Terminated", effective_date
    end_event = env.workday.emit(env.tick, "TERMINATION", contractor.wid, correlation_id=env.new_id("case"),
                                 effective_date=effective_date, reason="End Contingent Assignment")
    w = Worker(wid=env.new_wid(), employee_id=company.next_employee_id(), name=contractor.name,
               status="Active", worker_type="Employee", job_profile=job, org_wid=company.org_wids[org],
               cost_center_wid=company.cc_wids[org], manager_wid=company.managers[org],
               location=contractor.location, legal_entity=contractor.legal_entity,
               hire_date=effective_date, converted_from=contractor.employee_id,
               gov_id_token=contractor.gov_id_token)
    env.workday.workers[w.wid] = w
    for task in ("Create accounts and email", "Provision equipment"):
        env.workday.onboarding_tasks[env.new_wid()] = {"worker_wid": w.wid, "name": task,
                                                       "status": "Not Started"}
    case_id = env.new_id("case")

    def emit_hire():
        return env.workday.emit(env.tick, "HIRE", w.wid, correlation_id=case_id, effective_date=effective_date)

    if hire_event_delay:
        env.schedule(env.tick + hire_event_delay, emit_hire, "conversion HIRE event")
        return w, [end_event]
    return w, [end_event, emit_hire()]


def replay_event(env, event: dict) -> dict:
    """Deliver the same event a second time (same correlation ID), as integrations sometimes do."""
    data = {k: v for k, v in event.items() if k not in ("seq", "tick", "type", "worker_wid", "correlation_id")}
    return env.workday.emit(env.tick, event["type"], event["worker_wid"],
                            correlation_id=event["correlation_id"], **data)


def raise_manager_change(env, company: Company, org: str, new_manager: Worker,
                         effective_date: str = "2026-11-01") -> list[dict]:
    """Mid-cycle manager change: an org gets a new manager while expense reports are in flight.

    Workday moves every report to the new manager and emits one MANAGER_CHANGE event per
    affected worker. The new manager (typically promoted from within) keeps reporting to the
    old manager, who moves up. Finance still routes their pending (and future) expenses to the old
    manager until someone updates it, so reimbursements get approved by the wrong person.
    """
    org_wid = company.org_wids[org]
    old_manager_wid = env.workday.orgs[org_wid]["manager_wid"]
    env.workday.orgs[org_wid]["manager_wid"] = new_manager.wid
    company.managers[org] = new_manager.wid
    events = []
    for w in env.workday.workers.values():
        if w.status == "Active" and w.manager_wid == old_manager_wid and w.wid != new_manager.wid:
            w.manager_wid = new_manager.wid
            events.append(env.workday.emit(env.tick, "MANAGER_CHANGE", w.wid,
                                           correlation_id=env.new_id("case"), effective_date=effective_date,
                                           old_manager_wid=old_manager_wid, new_manager_wid=new_manager.wid))
    return events


def raise_termination(env, worker: Worker, termination_date: Optional[str] = None,
                      case_id: Optional[str] = None) -> dict:
    """Termination effective at the end of termination_date (default: today)."""
    termination_date = termination_date or env.today().isoformat()
    worker.status = "Terminated"
    worker.termination_date = termination_date
    return env.workday.emit(env.tick, "TERMINATION", worker.wid, correlation_id=case_id or env.new_id("case"),
                            effective_date=termination_date)


def raise_transfer(env, company: Company, worker: Worker, new_job: str, new_org: str,
                   effective_date: str = "2026-11-01", case_id: Optional[str] = None) -> dict:
    before = {"job_profile": worker.job_profile, "org_wid": worker.org_wid,
              "cost_center_wid": worker.cost_center_wid, "manager_wid": worker.manager_wid}
    worker.job_profile = new_job
    worker.org_wid = company.org_wids[new_org]
    worker.cost_center_wid = company.cc_wids[new_org]
    worker.manager_wid = company.managers[new_org]
    return env.workday.emit(env.tick, "JOB_CHANGE", worker.wid, correlation_id=case_id or env.new_id("case"),
                            effective_date=effective_date, before=before)
