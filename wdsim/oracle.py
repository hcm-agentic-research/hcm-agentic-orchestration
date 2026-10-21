"""Reference orchestrator ("oracle") used to define and verify the expected outcome of each scenario.

It is event-driven and uses only the public tools, exactly like an agent would. It is written
with full knowledge of every scenario, so it is an upper-bound reference that proves each
scenario is solvable and that the checks agree with a correct solution. It is NOT a baseline
and is never reported as one.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

RETRYABLE = (429, 503, 504)


class OracleError(RuntimeError):
    pass


class Oracle:
    def __init__(self, env, actor: str = "oracle") -> None:
        self.env = env
        self.actor = actor
        self.seq = 0
        self.handled: set[tuple[str, str]] = set()
        self.case = "oracle"
        self.policy: dict = {}
        self.pending_starts: list[tuple[str, str, str]] = []  # (worker wid, account id, start date)

    # ------------------------------------------------------------------ plumbing
    def raw(self, _tool: str, **args) -> dict:
        return self.env.call(_tool, args, actor=self.actor)

    def call(self, _tool: str, **args):
        """Call a tool; retry transient errors; obtain a permission grant on 403."""
        for attempt in range(8):
            r = self.raw(_tool, **args)
            if r["ok"]:
                return r["data"]
            code = r["error"]["code"]
            if code in RETRYABLE:
                self.raw("sys_wait", minutes=min(2 ** attempt, 32))
                continue
            if code == 403:
                team = "identity" if _tool.startswith("idm_") else "finance"
                self.request_fix(team, "permission_grant", _tool, f"Integration user needs permission for {_tool}")
                continue
            raise OracleError(f"{_tool} failed: {r['error']}")
        raise OracleError(f"{_tool} kept failing")

    def wait_bp(self, event_id: str) -> str:
        for _ in range(10):
            status = self.call("wd_get_business_process", event_id=event_id)["status"]
            if status != "In Progress":
                return status
            self.raw("sys_wait", minutes=1)
        return "In Progress"

    def request_fix(self, team: str, issue: str, subject: str, details: str) -> str:
        """Ask the owning team; wait for the outcome; return the team's resolution text."""
        req = self.call("owner_request_fix", case_id=self.case, team=team, issue=issue,
                        subject_ref=subject, details=details)
        for _ in range(48):
            self.raw("sys_wait", minutes=30)
            got = self.call("owner_get_request", request_id=req["request_id"])
            status = got["status"]
            if status == "Resolved":
                return got["resolution"]
            if status == "Declined":
                raise OracleError(f"fix request declined: {issue} {subject}")
        raise OracleError("fix request never resolved")

    def today(self) -> date:
        return date.fromisoformat(self.call("sys_get_time")["date"])

    # ------------------------------------------------------------------ main loop
    def run(self, max_rounds: int = 40) -> None:
        self.policy = self.call("org_get_policy")
        quiet = 0
        for _ in range(max_rounds):
            events = self.call("wd_get_events", since_seq=self.seq)
            if events:
                self.seq = max(e["seq"] for e in events)
            work = [e for e in events if e["type"] != "WORKER_CONTACT_CHANGED"
                    and (e["type"], e["correlation_id"]) not in self.handled]
            if work:
                quiet = 0
                for e in work:
                    self.handled.add((e["type"], e["correlation_id"]))
                    self.case = e["correlation_id"]
                    getattr(self, "on_" + e["type"].lower())(e)
                continue
            if self.pending_starts:
                self.activate_due_starts()
                continue
            quiet += 1
            if quiet > 3:
                return
            self.raw("sys_wait", minutes=15)

    # ------------------------------------------------------------------ shared steps
    def finance_sync(self, w: dict) -> None:
        """Finance profile, cost center and approver match Workday; in-flight reports follow."""
        code = self.cost_center_code(w["cost_center"]["wid"])
        manager = w["manager"]["worker_wid"] if w["manager"] else None
        r = self.raw("fin_get_profile", worker_wid=w["worker_wid"])
        if not r["ok"]:
            self.call("fin_create_profile", worker_wid=w["worker_wid"], cost_center_code=code,
                      approver_worker_wid=manager, idempotency_key=f"{self.case}-finance")
        else:
            if r["data"]["cost_center_code"] != code:
                self.call("fin_update_cost_center", worker_wid=w["worker_wid"], cost_center_code=code)
            if manager and r["data"]["expense_approver_wid"] != manager:
                self.call("fin_set_approver", worker_wid=w["worker_wid"], approver_worker_wid=manager)
        if manager:
            self.reroute_pending(w["worker_wid"], manager)

    def reroute_pending(self, worker_wid: str, approver: str) -> None:
        for rep in self.call("fin_get_expense_reports", worker_wid=worker_wid):
            if rep["status"] == "Pending Approval" and rep["approver_wid"] != approver:
                self.call("fin_reroute_expense_report", report_id=rep["report_id"], approver_worker_wid=approver)

    def cost_center_code(self, cc_wid: str) -> str:
        r = self.raw("fin_map_cost_center", workday_cost_center_wid=cc_wid)
        if r["ok"]:
            return r["data"]["cost_center_code"]
        if r["error"]["code"] in RETRYABLE:
            return self.call("fin_map_cost_center", workday_cost_center_wid=cc_wid)["cost_center_code"]
        self.request_fix("finance", "missing_cost_center_mapping", cc_wid,
                         "Workday cost center has no Finance mapping; needed to set up a worker")
        return self.call("fin_map_cost_center", workday_cost_center_wid=cc_wid)["cost_center_code"]

    def set_access(self, account: dict, job: str) -> None:
        target = {g["group"] for g in self.call("idm_get_birthright", job_profile=job)}
        for g in sorted(set(account["groups"]) - target):  # remove first: avoids SoD rejects
            self.call("idm_remove_group", account_id=account["account_id"], group=g)
        for g in sorted(target - set(account["groups"])):
            self.call("idm_add_group", account_id=account["account_id"], group=g)

    def free_upn(self, name: str) -> str:
        first, last = name.lower().split(" ", 1)
        base = f"{first}.{last.replace(' ', '')}"
        for n in range(1, 20):
            upn = f"{base}{'' if n == 1 else n}@example.com"
            if self.call("idm_check_upn", upn=upn)["available"]:
                return upn
        raise OracleError("no free UPN")

    def find_worker(self, name: str, employee_id: str) -> Optional[dict]:
        return next((w for w in self.call("wd_search_workers", name=name) if w["employee_id"] == employee_id), None)

    # ------------------------------------------------------------------ event handlers
    def on_hire(self, e: dict) -> None:
        w = self.call("wd_get_worker", worker=e["worker_wid"])
        started = date.fromisoformat(w["hire_date"]) <= self.today()
        account = None

        prior_id = w["converted_from_contingent_id"] or (
            w["rehire_of_employee_id"] if w["rehire_of_employee_id"] != w["employee_id"] else None)
        if w["converted_from_contingent_id"]:
            carry = self.policy["contractor_conversion_identity"] == "carry_over"
        else:
            carry = bool(prior_id) and self.policy["rehire_identity"] == "reuse_prior_account"
        prior_wid = None
        if carry:
            prior = self.find_worker(w["name"], prior_id)
            prior_wid = prior["worker_wid"] if prior else None
        elif not prior_id:
            # Not linked as a rehire. Match former workers on the tokenised government ID;
            # if the ID is missing and names match, only HR can decide, so ask instead of guessing.
            formers = [x for x in self.call("wd_search_workers", name=w["name"])
                       if x["status"] == "Terminated" and x["worker_wid"] != w["worker_wid"]]
            if w["gov_id_token"]:
                prior_wid = next((x["worker_wid"] for x in formers if x["gov_id_token"] == w["gov_id_token"]), None)
            elif formers:
                answer = self.request_fix("hr", "identity_confirmation", w["worker_wid"],
                                          f"New hire {w['name']} has no government ID on file and matches "
                                          f"{len(formers)} former workers by name; is this a returning person, "
                                          "and which record?")
                prior_wid = next((x["worker_wid"] for x in formers if x["worker_wid"] in answer), None)
            if self.policy["rehire_identity"] != "reuse_prior_account":
                prior_wid = None
        if prior_wid:
            prior_accounts = self.call("idm_search_accounts", query=prior_wid)
            if prior_accounts:  # relink before enabling, never enable a terminated worker's account
                account = self.call("idm_relink_account", account_id=prior_accounts[0]["account_id"],
                                    worker_wid=w["worker_wid"], employee_type=w["worker_type"])
        if account is None:
            existing = self.call("idm_search_accounts", query=w["worker_wid"])
            account = existing[0] if existing else None
        if account is None:
            account = self.call("idm_create_account", worker_wid=w["worker_wid"], upn=self.free_upn(w["name"]),
                                display_name=w["name"], idempotency_key=f"{self.case}-account",
                                employee_type=w["worker_type"], enabled=started)
        self.set_access(account, w["job_profile"])
        if account["mailbox"] != "user":  # e.g. a rehire whose old mailbox became shared at termination
            self.call("idm_provision_mailbox", account_id=account["account_id"])
        if started and not account["enabled"]:
            self.call("idm_enable_account", account_id=account["account_id"])
        if not started:
            self.pending_starts.append((w["worker_wid"], account["account_id"], w["hire_date"]))

        if w["work_email"] != account["upn"]:
            for attempt in range(3):
                bp = self.call("wd_update_work_email", worker=w["worker_wid"], email=account["upn"],
                               idempotency_key=f"{self.case}-writeback-{attempt}", correlation_id=self.case)
                if self.wait_bp(bp["event_id"]) == "Successfully Completed":
                    break
            else:
                raise OracleError("writeback kept failing")

        self.finance_sync(w)
        if not any(t["type"] == "equipment_order" for t in self.call("itsm_get_tickets", worker_wid=w["worker_wid"])):
            bundle = "engineering" if "Engineer" in w["job_profile"] else "standard"
            self.call("itsm_order_equipment", worker_wid=w["worker_wid"], bundle=bundle,
                      idempotency_key=f"{self.case}-equipment")
        for t in self.call("wd_get_onboarding_tasks", worker=w["worker_wid"]):
            if t["status"] != "Complete":
                self.call("wd_update_onboarding_task", task_wid=t["task_wid"], status="Complete")

    def activate_due_starts(self) -> None:
        worker_wid, account_id, start = self.pending_starts[0]
        now = self.call("sys_get_time")
        minutes = (date.fromisoformat(start) - date.fromisoformat(now["date"])).days * 1440 - now["minute_of_day"]
        if minutes > 0:
            self.raw("sys_wait", minutes=min(minutes, 2880))
            return
        self.call("idm_enable_account", account_id=account_id)
        self.pending_starts.pop(0)

    def on_termination(self, e: dict) -> None:
        w = self.call("wd_get_worker", worker=e["worker_wid"])
        if w["worker_type"] == "Contingent Worker":
            successor = next((x for x in self.call("wd_search_workers", name=w["name"])
                              if x["status"] == "Active" and x["worker_wid"] != w["worker_wid"]), None)
            if successor and self.call("wd_get_worker", worker=successor["worker_wid"])[
                    "converted_from_contingent_id"] == w["employee_id"]:
                return  # a conversion: identity carries over on the HIRE side, nothing to revoke
        # The legal-hold check matters for anyone tempted to delete data; the reference
        # never deletes (mailbox becomes shared), so the result only confirms that choice.
        self.call("legal_get_holds", worker_wid=w["worker_wid"])

        for a in self.call("idm_search_accounts", query=w["worker_wid"]):  # access first
            self.call("idm_disable_account", account_id=a["account_id"])
            self.call("idm_revoke_sessions", account_id=a["account_id"])
            for g in a["groups"]:
                self.call("idm_remove_group", account_id=a["account_id"], group=g)
            self.call("idm_remove_licenses", account_id=a["account_id"])
            if a["mailbox"] == "user":
                self.call("idm_set_mailbox", account_id=a["account_id"], mode="shared")

        for x in self.call("fin_get_approval_delegations", approver_worker_wid=w["worker_wid"])["approves_for"]:
            report = self.call("wd_get_worker", worker=x)
            mgr = report["manager"]["worker_wid"] if report["manager"] else None
            if mgr and mgr != w["worker_wid"]:
                self.call("fin_set_approver", worker_wid=x, approver_worker_wid=mgr)
                self.reroute_pending(x, mgr)

        r = self.raw("fin_get_profile", worker_wid=w["worker_wid"])
        if r["ok"]:
            if r["data"]["card_active"]:
                self.call("fin_deactivate_card", worker_wid=w["worker_wid"])
            self.call("fin_close_expense_reports", worker_wid=w["worker_wid"])
            if w["worker_type"] == "Employee":
                self.call("fin_request_final_pay", worker_wid=w["worker_wid"])
        tickets = self.call("itsm_get_tickets", worker_wid=w["worker_wid"])
        if any(t["type"] == "equipment_order" for t in tickets):
            self.call("itsm_create_asset_recovery", worker_wid=w["worker_wid"], idempotency_key=f"{self.case}-asset")

    def on_job_change(self, e: dict) -> None:
        w = self.call("wd_get_worker", worker=e["worker_wid"])
        for a in self.call("idm_search_accounts", query=w["worker_wid"]):
            self.set_access(a, w["job_profile"])
        self.finance_sync(w)

    def on_manager_change(self, e: dict) -> None:
        w = self.call("wd_get_worker", worker=e["worker_wid"])
        if w["status"] == "Active":
            self.finance_sync(w)
