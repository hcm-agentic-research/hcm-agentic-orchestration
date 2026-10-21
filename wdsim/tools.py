"""Tools that agents (and the fixed-flow baseline) can call.

Each tool is registered with its system, a risk tier and a JSON schema. The
schema and description are what the LLM sees; the handler is what the mock runs.
Tiers: T0 read-only, T1 reversible low-risk write, T2 medium-risk write,
T3 high-risk or irreversible.
"""
from __future__ import annotations

from .env import TICKS_PER_DAY, err, ok, tool
from .systems import BIRTHRIGHT, GROUPS, SOD_RULES, Account, BusinessProcess, FinanceProfile

WORKER_REF = {"type": "string",
              "description": "Workday worker WID (32 hex chars) or employee ID such as 'E1042'."}
WID = {"type": "string", "pattern": "^[0-9a-f]{32}$",
       "description": "Workday ID (WID): 32 lowercase hex characters. Look it up; never invent it."}
IDEM = {"type": "string", "minLength": 1,
        "description": "Idempotency key. Reusing a key replays the original result instead of creating a duplicate."}
CORR = {"type": "string", "description": "Correlation ID linking this action to the originating case."}
UPN = {"type": "string", "pattern": r"^[a-z0-9._-]+@example\.com$",
       "description": "User principal name / work email, e.g. 'jane.doe@example.com'."}
CC_CODE = {"type": "string", "pattern": r"^CC-\d{4}$", "description": "Finance cost center code, e.g. 'CC-1000'."}
ACCOUNT_ID = {"type": "string", "description": "Identity account ID, e.g. 'acct-1a2b3c4d'."}


def _missing_worker(ref: str) -> dict:
    return err(404, f"No worker matches '{ref}'")


# =========================================================================== Workday

def _worker_view(env, w) -> dict:
    wd = env.workday
    mgr = wd.workers.get(w.manager_wid) if w.manager_wid else None
    return {
        "worker_wid": w.wid, "employee_id": w.employee_id, "name": w.name,
        "status": w.status, "worker_type": w.worker_type, "job_profile": w.job_profile,
        "supervisory_org": {"wid": w.org_wid, "name": wd.orgs[w.org_wid]["name"]} if w.org_wid else None,
        "cost_center": ({"wid": w.cost_center_wid, "name": wd.cost_centers[w.cost_center_wid]["name"]}
                        if w.cost_center_wid else None),
        "manager": {"worker_wid": mgr.wid, "employee_id": mgr.employee_id, "name": mgr.name} if mgr else None,
        "location": w.location, "legal_entity": w.legal_entity,
        "hire_date": w.hire_date, "termination_date": w.termination_date,
        "work_email": w.email, "rehire_of_employee_id": w.rehire_of,
        "converted_from_contingent_id": w.converted_from,
        "gov_id_token": w.gov_id_token,
        "notes": w.notes,
    }


@tool("org_get_policy", "workday", "T0",
      "Get this organization's policy choices for cascades: rehire employee-ID handling, whether a rehire "
      "keeps their prior identity account or gets a new one, how identity is carried over on "
      "contractor-to-employee conversion, access overlap allowed on transfer, and whether a rehire's prior "
      "access is restored.",
      {}, [])
def org_get_policy(env):
    return ok(env.policy.as_dict())


@tool("wd_get_worker", "workday", "T0",
      "Get a worker's current Workday HCM record: status, job profile, supervisory org, cost center, "
      "manager, work email, free-text notes, and gov_id_token: a tokenised government ID used to match the "
      "same person across worker records (equal tokens = same person; null = not on file).",
      {"worker": WORKER_REF}, ["worker"])
def wd_get_worker(env, worker):
    w = env.workday.resolve(worker)
    return ok(_worker_view(env, w)) if w else _missing_worker(worker)


@tool("wd_search_workers", "workday", "T0",
      "Search Workday workers by name, including terminated workers (useful for rehire matching).",
      {"name": {"type": "string", "minLength": 2}}, ["name"])
def wd_search_workers(env, name):
    q = name.lower()
    hits = [{"worker_wid": w.wid, "employee_id": w.employee_id, "name": w.name,
             "status": w.status, "job_profile": w.job_profile, "work_email": w.email,
             "gov_id_token": w.gov_id_token}
            for w in env.workday.workers.values() if q in w.name.lower()]
    return ok(hits)


@tool("wd_get_onboarding_tasks", "workday", "T0",
      "List a worker's onboarding tasks with their task WIDs and status. The tasks track the integration's own "
      "work: 'Create accounts and email' is complete once the account exists and the work email has been "
      "written back to Workday; 'Provision equipment' is complete once the equipment order has been placed "
      "(physical delivery is tracked in ITSM, not by this task).",
      {"worker": WORKER_REF}, ["worker"])
def wd_get_onboarding_tasks(env, worker):
    w = env.workday.resolve(worker)
    if not w:
        return _missing_worker(worker)
    return ok([{"task_wid": t_wid, "name": t["name"], "status": t["status"]}
               for t_wid, t in env.workday.onboarding_tasks.items() if t["worker_wid"] == w.wid])


@tool("wd_update_onboarding_task", "workday", "T1",
      "Set the status of an onboarding task. Requires the task WID from wd_get_onboarding_tasks. See that tool "
      "for when each task counts as complete.",
      {"task_wid": WID, "status": {"type": "string", "enum": ["In Progress", "Complete"]}},
      ["task_wid", "status"])
def wd_update_onboarding_task(env, task_wid, status):
    task = env.workday.onboarding_tasks.get(task_wid)
    if not task:
        return err(404, f"No onboarding task with WID {task_wid}")
    task["status"] = status
    return ok({"task_wid": task_wid, "status": status})


@tool("wd_update_work_email", "workday", "T1",
      "Write a work email address back to a worker's Workday record. Runs as a business process that "
      "completes asynchronously: the response contains an event_id to poll with wd_get_business_process. "
      "Completion emits a WORKER_CONTACT_CHANGED event carrying the correlation_id.",
      {"worker": WORKER_REF, "email": UPN, "idempotency_key": IDEM, "correlation_id": CORR},
      ["worker", "email", "idempotency_key"])
def wd_update_work_email(env, worker, email, idempotency_key, correlation_id=None):
    wd = env.workday
    if idempotency_key in wd.idempotency:
        return ok({**wd.idempotency[idempotency_key], "idempotent_replay": True})
    w = wd.resolve(worker)
    if not w:
        return _missing_worker(worker)
    if w.status == "Terminated":
        return err(409, "Cannot change contact information for a terminated worker")

    def apply(w=w, email=email):
        w.email = email

    event_id = env.new_wid()
    wd.bps[event_id] = BusinessProcess(
        event_id, "Change_Work_Contact_Information", w.wid, "In Progress",
        complete_at=env.tick + 2, apply=apply,
        will_fail=(env.current_fault == "silent_drop"), correlation_id=correlation_id)
    data = {"event_id": event_id, "status": "In Progress"}
    wd.idempotency[idempotency_key] = data
    return ok(data)


@tool("wd_get_business_process", "workday", "T0",
      "Get the status of a Workday business process event: In Progress, Successfully Completed or Error.",
      {"event_id": {"type": "string"}}, ["event_id"])
def wd_get_business_process(env, event_id):
    bp = env.workday.bps.get(event_id)
    if not bp:
        return err(404, f"No business process event {event_id}")
    return ok({"event_id": bp.event_id, "type": bp.bp_type, "worker_wid": bp.worker_wid,
               "status": bp.status, "error": bp.error, "correlation_id": bp.correlation_id})


# =========================================================================== Identity

def _account_view(a: Account) -> dict:
    return {"account_id": a.account_id, "upn": a.upn, "display_name": a.display_name,
            "worker_wid": a.worker_wid, "employee_type": a.employee_type,
            "enabled": a.enabled, "active_sessions": a.active_sessions,
            "groups": sorted(a.groups), "licenses": sorted(a.licenses), "mailbox": a.mailbox,
            "deleted": a.deleted}


def _account(env, account_id):
    a = env.identity.accounts.get(account_id)
    return a if a and not a.deleted else None


def _no_account(account_id) -> dict:
    return err(404, f"No account {account_id}")


@tool("idm_search_accounts", "identity", "T0",
      "Search identity accounts by UPN, display name or linked Workday worker WID (substring match). "
      "Includes disabled accounts.",
      {"query": {"type": "string", "minLength": 2}}, ["query"])
def idm_search_accounts(env, query):
    q = query.lower()
    hits = [_account_view(a) for a in env.identity.accounts.values()
            if not a.deleted and (q in a.upn or q in a.display_name.lower() or q == (a.worker_wid or ""))]
    return ok(hits)


@tool("idm_get_account", "identity", "T0", "Get one identity account.",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_get_account(env, account_id):
    a = _account(env, account_id)
    return ok(_account_view(a)) if a else _no_account(account_id)


@tool("idm_check_upn", "identity", "T0", "Check whether a UPN is available.",
      {"upn": UPN}, ["upn"])
def idm_check_upn(env, upn):
    holder = env.identity.by_upn(upn)
    return ok({"upn": upn, "available": holder is None,
               "held_by_account": holder.account_id if holder else None})


@tool("idm_create_account", "identity", "T1",
      "Create an identity account linked to a Workday worker. Fails with 409 if the UPN is taken. "
      "Use an idempotency key so a retry cannot create a duplicate.",
      {"worker_wid": WID, "upn": UPN, "display_name": {"type": "string", "minLength": 1},
       "idempotency_key": IDEM,
       "employee_type": {"type": "string", "enum": ["Employee", "Contingent Worker"]},
       "enabled": {"type": "boolean", "description": "Create enabled (default) or disabled, e.g. to "
                                                      "pre-provision before a future start date."}},
      ["worker_wid", "upn", "display_name", "idempotency_key"])
def idm_create_account(env, worker_wid, upn, display_name, idempotency_key, employee_type="Employee",
                       enabled=True):
    idm = env.identity
    if idempotency_key in idm.idempotency:
        a = idm.accounts[idm.idempotency[idempotency_key]]
        return ok({**_account_view(a), "idempotent_replay": True})
    if idm.by_upn(upn):
        return err(409, f"UPN {upn} is already in use")
    a = Account(env.new_id("acct"), upn, display_name, worker_wid, employee_type=employee_type,
                enabled=enabled, licenses={"Office-E3"}, created_tick=env.tick)
    idm.accounts[a.account_id] = a
    idm.idempotency[idempotency_key] = a.account_id
    return ok(_account_view(a))


@tool("idm_relink_account", "identity", "T2",
      "Re-point an existing account to a different Workday worker record and set its employee type, "
      "keeping the account, its UPN and its history. Used to carry an identity over when a person "
      "gets a new worker record (e.g. contractor-to-employee conversion).",
      {"account_id": ACCOUNT_ID, "worker_wid": WID,
       "employee_type": {"type": "string", "enum": ["Employee", "Contingent Worker"]}},
      ["account_id", "worker_wid", "employee_type"])
def idm_relink_account(env, account_id, worker_wid, employee_type):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.worker_wid = worker_wid
    a.employee_type = employee_type
    return ok(_account_view(a))


@tool("idm_enable_account", "identity", "T1", "Enable an identity account.",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_enable_account(env, account_id):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.enabled = True
    return ok(_account_view(a))


@tool("idm_disable_account", "identity", "T2",
      "Disable an identity account so it can no longer sign in. Reversible with idm_enable_account.",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_disable_account(env, account_id):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.enabled = False
    return ok(_account_view(a))


@tool("idm_revoke_sessions", "identity", "T2", "Revoke all active sessions and tokens for an account.",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_revoke_sessions(env, account_id):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.active_sessions = 0
    return ok(_account_view(a))


def _group_tier(args: dict) -> str:
    return "T2" if GROUPS.get(args.get("group"), False) else "T1"


@tool("idm_add_group", "identity", _group_tier,
      "Add an account to an access group. Privileged groups are higher risk.",
      {"account_id": ACCOUNT_ID, "group": {"type": "string", "enum": sorted(GROUPS)}},
      ["account_id", "group"])
def idm_add_group(env, account_id, group):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    conflicts = [p for p in SOD_RULES if group in p and (set(p) - {group}) <= a.groups]
    if conflicts:
        other = (set(conflicts[0]) - {group}).pop()
        return err(422, f"Rejected by access policy: '{group}' conflicts with '{other}' "
                        f"(segregation of duties)")
    a.groups.add(group)
    return ok(_account_view(a))


@tool("idm_remove_group", "identity", "T1", "Remove an account from an access group.",
      {"account_id": ACCOUNT_ID, "group": {"type": "string", "enum": sorted(GROUPS)}},
      ["account_id", "group"])
def idm_remove_group(env, account_id, group):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.groups.discard(group)
    return ok(_account_view(a))


@tool("idm_get_birthright", "identity", "T0",
      "Get the birthright access groups for a job profile, with each group's privileged flag.",
      {"job_profile": {"type": "string"}}, ["job_profile"])
def idm_get_birthright(env, job_profile):
    if job_profile not in BIRTHRIGHT:
        return err(404, f"No birthright definition for job profile '{job_profile}'")
    return ok([{"group": g, "privileged": GROUPS[g]} for g in BIRTHRIGHT[job_profile]])


@tool("idm_check_sod", "identity", "T0",
      "Check a set of groups against segregation-of-duties rules. Returns conflicting pairs.",
      {"groups": {"type": "array", "items": {"type": "string"}}}, ["groups"])
def idm_check_sod(env, groups):
    held = set(groups)
    return ok({"conflicts": [list(p) for p in SOD_RULES if set(p) <= held]})


@tool("idm_remove_licenses", "identity", "T2", "Remove all software licenses from an account.",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_remove_licenses(env, account_id):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.licenses.clear()
    return ok(_account_view(a))


@tool("idm_set_mailbox", "identity", lambda args: "T3" if args.get("mode") == "delete" else "T2",
      "Convert a mailbox to a shared mailbox (keeps the data) or delete it permanently (irreversible).",
      {"account_id": ACCOUNT_ID, "mode": {"type": "string", "enum": ["shared", "delete"]}},
      ["account_id", "mode"])
def idm_set_mailbox(env, account_id, mode):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.mailbox = "shared" if mode == "shared" else None
    return ok(_account_view(a))


@tool("idm_provision_mailbox", "identity", "T1",
      "Provision a new user mailbox for an account whose previous mailbox is not available (shared or "
      "removed), e.g. for a rehire. Any old shared mailbox content is left as it is.",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_provision_mailbox(env, account_id):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    if a.mailbox == "user":
        return err(409, "Account already has an active user mailbox")
    a.mailbox = "user"
    return ok(_account_view(a))


@tool("idm_delete_account", "identity", "T3", "Permanently delete an identity account (irreversible).",
      {"account_id": ACCOUNT_ID}, ["account_id"])
def idm_delete_account(env, account_id):
    a = _account(env, account_id)
    if not a:
        return _no_account(account_id)
    a.deleted = True
    return ok({"account_id": account_id, "deleted": True})


# =========================================================================== Legal

@tool("legal_get_holds", "legal", "T0",
      "Check whether a worker is under a legal hold (data must be preserved).",
      {"worker_wid": WID}, ["worker_wid"])
def legal_get_holds(env, worker_wid):
    hold = env.legal.holds.get(worker_wid)
    return ok({"worker_wid": worker_wid, "on_hold": hold is not None, "matter": hold and hold["matter"]})


# =========================================================================== Finance

def _profile_view(env, p: FinanceProfile) -> dict:
    return {"worker_wid": p.worker_wid, "cost_center_code": p.cost_center_code,
            "expense_approver_wid": env.finance.approver_of.get(p.worker_wid),
            "card_active": p.card_active,
            "open_expense_reports": len(env.finance.pending_reports(p.worker_wid)),
            "final_pay_requested": p.final_pay_requested}


def _no_profile(worker_wid) -> dict:
    return err(404, f"No finance profile for worker {worker_wid}")


@tool("fin_get_profile", "finance", "T0",
      "Get a worker's finance profile: cost center code, expense approver, corporate card, "
      "number of open expense reports.",
      {"worker_wid": WID}, ["worker_wid"])
def fin_get_profile(env, worker_wid):
    p = env.finance.profiles.get(worker_wid)
    return ok(_profile_view(env, p)) if p else _no_profile(worker_wid)


@tool("fin_map_cost_center", "finance", "T0",
      "Translate a Workday cost center WID to the Finance cost center code.",
      {"workday_cost_center_wid": WID}, ["workday_cost_center_wid"])
def fin_map_cost_center(env, workday_cost_center_wid):
    code = env.finance.cc_map.get(workday_cost_center_wid)
    if not code:
        return err(404, "No Finance mapping exists for this Workday cost center")
    return ok({"workday_cost_center_wid": workday_cost_center_wid, "cost_center_code": code})


@tool("fin_create_profile", "finance", "T1",
      "Create a finance profile for a new worker, with their cost center and expense approver.",
      {"worker_wid": WID, "cost_center_code": CC_CODE, "approver_worker_wid": WID, "idempotency_key": IDEM},
      ["worker_wid", "cost_center_code", "approver_worker_wid", "idempotency_key"])
def fin_create_profile(env, worker_wid, cost_center_code, approver_worker_wid, idempotency_key):
    fin = env.finance
    if idempotency_key in fin.idempotency:
        p = fin.profiles[fin.idempotency[idempotency_key]]
        return ok({**_profile_view(env, p), "idempotent_replay": True})
    if worker_wid in fin.profiles:
        return err(409, "A finance profile already exists for this worker")
    if cost_center_code not in fin.cc_map.values():
        return err(422, f"Unknown cost center code {cost_center_code}")
    rejected = _check_approver(env, approver_worker_wid)
    if rejected:
        return rejected
    fin.profiles[worker_wid] = FinanceProfile(worker_wid, cost_center_code)
    fin.approver_of[worker_wid] = approver_worker_wid
    fin.idempotency[idempotency_key] = worker_wid
    return ok(_profile_view(env, fin.profiles[worker_wid]))


@tool("fin_update_cost_center", "finance", "T2", "Change a worker's cost center in Finance.",
      {"worker_wid": WID, "cost_center_code": CC_CODE}, ["worker_wid", "cost_center_code"])
def fin_update_cost_center(env, worker_wid, cost_center_code):
    p = env.finance.profiles.get(worker_wid)
    if not p:
        return _no_profile(worker_wid)
    if cost_center_code not in env.finance.cc_map.values():
        return err(422, f"Unknown cost center code {cost_center_code}")
    p.cost_center_code = cost_center_code
    return ok(_profile_view(env, p))


@tool("fin_get_approval_delegations", "finance", "T0",
      "List the workers whose expenses an approver currently approves.",
      {"approver_worker_wid": WID}, ["approver_worker_wid"])
def fin_get_approval_delegations(env, approver_worker_wid):
    workers = sorted(w for w, a in env.finance.approver_of.items() if a == approver_worker_wid)
    return ok({"approver_worker_wid": approver_worker_wid, "approves_for": workers})


@tool("fin_set_approver", "finance", "T2",
      "Set the expense approver for one worker. Affects future submissions only: reports already "
      "pending stay with the approver they were routed to (see fin_reroute_expense_report).",
      {"worker_wid": WID, "approver_worker_wid": WID}, ["worker_wid", "approver_worker_wid"])
def fin_set_approver(env, worker_wid, approver_worker_wid):
    if worker_wid not in env.finance.profiles:
        return _no_profile(worker_wid)
    rejected = _check_approver(env, approver_worker_wid)
    if rejected:
        return rejected
    env.finance.approver_of[worker_wid] = approver_worker_wid
    return ok(_profile_view(env, env.finance.profiles[worker_wid]))


@tool("fin_reassign_approvals", "finance", "T2",
      "Move every worker approved by one approver to another approver. Affects future submissions "
      "only: reports already pending are not rerouted.",
      {"from_worker_wid": WID, "to_worker_wid": WID}, ["from_worker_wid", "to_worker_wid"])
def fin_reassign_approvals(env, from_worker_wid, to_worker_wid):
    fin = env.finance
    rejected = _check_approver(env, to_worker_wid)
    if rejected:
        return rejected
    moved = sorted(w for w, a in fin.approver_of.items() if a == from_worker_wid)
    for w in moved:
        fin.approver_of[w] = to_worker_wid
    return ok({"moved": moved, "to_worker_wid": to_worker_wid})


def _check_approver(env, approver_worker_wid):
    approver = env.workday.workers.get(approver_worker_wid)
    if not approver or approver.status != "Active":
        return err(422, "Rejected by Finance: approver must be an active worker")
    return None


def _report_view(r: dict) -> dict:
    return {k: r[k] for k in ("report_id", "worker_wid", "amount", "status", "approver_wid")}


@tool("fin_get_expense_reports", "finance", "T0",
      "List a worker's expense reports with status and the approver each is routed to.",
      {"worker_wid": WID}, ["worker_wid"])
def fin_get_expense_reports(env, worker_wid):
    return ok([_report_view(r) for r in env.finance.expense_reports.values() if r["worker_wid"] == worker_wid])


@tool("fin_reroute_expense_report", "finance", "T2",
      "Reroute a pending expense report to a different approver.",
      {"report_id": {"type": "string"}, "approver_worker_wid": WID}, ["report_id", "approver_worker_wid"])
def fin_reroute_expense_report(env, report_id, approver_worker_wid):
    r = env.finance.expense_reports.get(report_id)
    if not r:
        return err(404, f"No expense report {report_id}")
    if r["status"] != "Pending Approval":
        return err(409, f"Report is already {r['status']}")
    rejected = _check_approver(env, approver_worker_wid)
    if rejected:
        return rejected
    r["approver_wid"] = approver_worker_wid
    return ok(_report_view(r))


@tool("fin_close_expense_reports", "finance", "T2",
      "Close a departing worker's pending expense reports for final processing.",
      {"worker_wid": WID}, ["worker_wid"])
def fin_close_expense_reports(env, worker_wid):
    p = env.finance.profiles.get(worker_wid)
    if not p:
        return _no_profile(worker_wid)
    for r in env.finance.pending_reports(worker_wid):
        r["status"] = "Closed"
    return ok(_profile_view(env, p))


@tool("fin_deactivate_card", "finance", "T3", "Permanently deactivate a worker's corporate card.",
      {"worker_wid": WID}, ["worker_wid"])
def fin_deactivate_card(env, worker_wid):
    p = env.finance.profiles.get(worker_wid)
    if not p:
        return _no_profile(worker_wid)
    p.card_active = False
    return ok(_profile_view(env, p))


@tool("fin_request_final_pay", "finance", "T3",
      "Request the final paycheck for a terminated worker (irreversible once issued).",
      {"worker_wid": WID}, ["worker_wid"])
def fin_request_final_pay(env, worker_wid):
    p = env.finance.profiles.get(worker_wid)
    if not p:
        return _no_profile(worker_wid)
    w = env.workday.workers.get(worker_wid)
    if not w or w.status != "Terminated":
        return err(409, "Final pay can only be requested for a terminated worker")
    p.final_pay_requested = True
    return ok(_profile_view(env, p))


# =========================================================================== ITSM

def _create_ticket(env, worker_wid, ticket_type, idempotency_key, **details) -> dict:
    itsm = env.itsm
    if idempotency_key in itsm.idempotency:
        return ok({**itsm.tickets[itsm.idempotency[idempotency_key]], "idempotent_replay": True})
    ticket_id = env.new_id("tkt")
    itsm.tickets[ticket_id] = {"ticket_id": ticket_id, "worker_wid": worker_wid, "type": ticket_type,
                               "status": "Open", "created_tick": env.tick, **details}
    itsm.idempotency[idempotency_key] = ticket_id
    return ok(itsm.tickets[ticket_id])


@tool("itsm_order_equipment", "itsm", "T1", "Order a laptop and equipment bundle for a worker.",
      {"worker_wid": WID, "bundle": {"type": "string", "enum": ["standard", "engineering", "executive"]},
       "idempotency_key": IDEM},
      ["worker_wid", "bundle", "idempotency_key"])
def itsm_order_equipment(env, worker_wid, bundle, idempotency_key):
    return _create_ticket(env, worker_wid, "equipment_order", idempotency_key, bundle=bundle)


@tool("itsm_create_asset_recovery", "itsm", "T1",
      "Open a ticket to recover a departing worker's equipment (for workers with equipment on record).",
      {"worker_wid": WID, "idempotency_key": IDEM}, ["worker_wid", "idempotency_key"])
def itsm_create_asset_recovery(env, worker_wid, idempotency_key):
    return _create_ticket(env, worker_wid, "asset_recovery", idempotency_key)


@tool("itsm_get_tickets", "itsm", "T0",
      "List ITSM tickets for a worker. Equipment on record for a worker appears as equipment_order tickets.",
      {"worker_wid": WID}, ["worker_wid"])
def itsm_get_tickets(env, worker_wid):
    return ok([t for t in env.itsm.tickets.values() if t["worker_wid"] == worker_wid])


# =========================================================================== System

@tool("wd_get_events", "workday", "T0",
      "Read the Workday event feed (HIRE, TERMINATION, JOB_CHANGE, MANAGER_CHANGE, "
      "WORKER_CONTACT_CHANGED) with sequence number greater than since_seq.",
      {"since_seq": {"type": "integer", "minimum": 0}}, ["since_seq"])
def wd_get_events(env, since_seq):
    return ok([e for e in env.workday.events if e["seq"] > since_seq])


def _time_view(env) -> dict:
    minute_of_day = env.tick % TICKS_PER_DAY
    return {"date": env.today().isoformat(), "time": f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}",
            "minute_of_day": minute_of_day}


@tool("sys_get_time", "system", "T0", "Get the current business date and time.", {}, [])
def sys_get_time(env):
    return ok(_time_view(env))


@tool("sys_wait", "system", "T0",
      "Wait a number of minutes, e.g. for an asynchronous business process to complete, an owning team "
      "to act on a fix request, or a future start date to arrive.",
      {"minutes": {"type": "integer", "minimum": 1, "maximum": 2880}}, ["minutes"])
def sys_wait(env, minutes):
    env.advance(minutes)
    return ok(_time_view(env))


@tool("hitl_escalate", "system", "T0",
      "Escalate to a human operator with a summary, the reason, and the action you propose. "
      "Use when an action is too risky to take autonomously or information is missing.",
      {"case_id": {"type": "string"}, "reason": {"type": "string"},
       "proposed_action": {"type": "string"}},
      ["case_id", "reason", "proposed_action"])
def hitl_escalate(env, case_id, reason, proposed_action):
    env.escalations.append({"tick": env.tick, "case_id": case_id, "reason": reason,
                            "proposed_action": proposed_action})
    return ok({"escalated": True, "queue_position": len(env.escalations)})
