"""Re-score saved runs by replaying their traces (no LLM calls, no cost).

The environment is deterministic given the seed and the sequence of tool calls, so replaying every audited
call into a freshly built scenario reproduces the end state exactly. Each replay is first checked against the
recorded run (same tool outcomes); then the current scoring code is applied.

  python rescore.py --traces results/traces --out results/stage2-rescored.jsonl

Written 2026-10-15 for the conversion equipment-check correction described in the paper (Section VI-D).
"""
import argparse
import copy
import glob
import json
import os

from wdsim.scenarios import BY_ID, build, evaluate


def replay(trace: dict):
    rec = trace["record"]
    scenario = BY_ID[rec["scenario"]]
    env, ctx = build(scenario, rec["seed"])
    mismatches = 0
    for a in trace["audit"]:
        if a["fault"] == "gated":  # blocked by the Gate: logged, never executed, no clock tick
            env.audit.append({"tick": env.tick, "actor": a["actor"], "tool": a["tool"], "system": None, "tier": None,
                              "args": copy.deepcopy(a["args"]), "ok": False, "committed": False, "fault": "gated",
                              "result": {"ok": False}})
            continue
        result = env.call(a["tool"], copy.deepcopy(a["args"]), actor=a["actor"])
        last = env.audit[-1]
        if result.get("ok", False) != a["ok"] or last["fault"] != a["fault"] or last["tick"] != a["tick"]:
            mismatches += 1
    return scenario, env, ctx, mismatches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default=os.path.join("results", "traces"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rows = []
    for path in sorted(glob.glob(os.path.join(a.traces, "*.json"))):
        trace = json.load(open(path, encoding="utf-8"))
        rec = trace["record"]
        scenario, env, ctx, mismatches = replay(trace)
        res = evaluate(scenario, env, ctx)
        new = {
            "success": res["success"] and rec["error"] is None,
            "violations": [v["kind"] for v in res["violations"]],
            "unsafe": [u["kind"] for u in res["unsafe"]],
            "failed_expectations": res["failed_expectations"],
            "human_touches": res["effort"]["human_touches"],
        }
        rows.append({**rec, "raw": {k: rec[k] for k in new}, "rescored": new, "replay_mismatches": mismatches,
                     "trace": os.path.basename(path)})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    bad = [r["trace"] for r in rows if r["replay_mismatches"]]
    changed = [r for r in rows if r["raw"] != r["rescored"]]
    print(f"replayed {len(rows)} traces; replay mismatches in {len(bad)}; score changed in {len(changed)}")
    for r in changed:
        print(f"  {r['trace']}: success {r['raw']['success']} -> {r['rescored']['success']}; "
              f"failed {r['raw']['failed_expectations']} -> {r['rescored']['failed_expectations']}")
    for t in bad[:20]:
        print("  MISMATCH", t)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
