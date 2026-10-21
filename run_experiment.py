"""Run strategies x scenarios x seeds x repeats and write one JSON line per run.

Examples
  # No API key needed: the fixed-flow baseline on everything
  python run_experiment.py --strategies fixed_flow

  # Small paid trial on Claude: two scenarios, one seed
  python run_experiment.py --strategies multi_agent --provider anthropic --scenarios ON-1,ON-8 --seeds 1

  # OpenAI needs an explicit model ID and prices (USD per million tokens)
  python run_experiment.py --strategies single_agent --provider openai --model <model-id> \
      --price-in <x> --price-out <y>

API keys come from ANTHROPIC_API_KEY / OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import asdict

from agents.llm import ModelConfig
from agents.strategies import make_strategy
from wdsim.scenarios import BY_ID, SCENARIOS, build, evaluate

# USD per million tokens. Anthropic first-party rates for the default model;
# cache writes billed at 1.25x input (5-minute cache).
PRICES = {"claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20, "cache_write": 5.00}}


def cost(usage: dict, price: dict) -> float:
    return (usage["input_tokens"] * price["in"] + usage["output_tokens"] * price["out"]
            + usage["cache_read_tokens"] * price.get("cache_read", price["in"])
            + usage["cache_write_tokens"] * price.get("cache_write", price["in"])) / 1e6


def run_one(strategy_name, scenario, seed, repeat, cfg, price):
    env, ctx = build(scenario, seed)
    strategy = make_strategy(strategy_name, cfg)
    start, t0, error = env.tick, time.time(), None
    info = None
    try:
        info = strategy.run(env)
    except Exception as e:  # recorded, never hidden
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    minutes = env.tick - start
    result = evaluate(scenario, env, ctx)
    usage = asdict(info.usage) if info else {}
    record = {
        "strategy": strategy_name, "model": cfg.label() if cfg else "none",
        "scenario": scenario.id, "split": scenario.split, "cascade": scenario.cascade,
        "seed": seed, "repeat": repeat, "success": result["success"] and error is None,
        "violations": [v["kind"] for v in result["violations"]],
        "unsafe": [u["kind"] for u in result["unsafe"]],
        "failed_expectations": result["failed_expectations"],
        "human_touches": result["effort"]["human_touches"], "effort": result["effort"],
        "tool_calls": sum(1 for a in env.audit if a["actor"] not in ("oracle",)),
        "failed_tool_calls": sum(1 for a in env.audit if not a["ok"]),
        "blocked_by_gate": sum(1 for a in env.audit if a.get("fault") == "gated"),
        "access_interruptions": len(result["access_interruptions"]),
        "termination_exposure_hours": [x["exposure_hours"] for x in result["termination_exposure"]],
        "minutes_to_finish": minutes, "wall_seconds": round(time.time() - t0, 1),
        "usage": usage, "cost_usd": round(cost(usage, price), 4) if usage and price else None,
        "llm_runs": info.llm_turns if info else {}, "step_limits": info.step_limits if info else 0,
        "notes": info.notes if info else [], "error": error,
    }
    write_trace(record, env, info)
    return record


def write_trace(record, env, info):
    """Full per-run trace for inspection and release: every tool call, escalation, fix request
    and each agent's final message (plans, critiques, reports)."""
    os.makedirs(os.path.join("results", "traces"), exist_ok=True)
    model = record["model"].replace(":", "_").replace("/", "_")
    name = f"{record['strategy']}-{model}-{record['scenario']}-s{record['seed']}-r{record['repeat']}.json"
    trace = {
        "record": record,
        "audit": [{"tick": a["tick"], "actor": a["actor"], "tool": a["tool"], "tier": a["tier"], "args": a["args"],
                   "ok": a["ok"], "fault": a["fault"],
                   "error": None if a["ok"] else a["result"].get("error", {}).get("message")} for a in env.audit],
        "escalations": env.escalations,
        "fix_requests": [r.view() for r in env.fix_requests.values()],
        "agents": info.transcript if info else [],
    }
    with open(os.path.join("results", "traces", name), "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=1, default=str)


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["strategy"], r["model"], r["split"])].append(r)
    print(f"\n{'strategy':26}{'model':34}{'split':7}{'runs':>5}{'success':>9}{'unsafe/run':>11}"
          f"{'touches/run':>12}{'cost/run $':>11}")
    for (s, m, split), rs in sorted(groups.items()):
        n = len(rs)
        succ = sum(r["success"] for r in rs) / n
        unsafe = sum(len(r["unsafe"]) for r in rs) / n
        touches = sum(r["human_touches"] for r in rs) / n
        costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
        c = f"{sum(costs) / len(costs):.3f}" if costs else "-"
        print(f"{s:26}{m:34}{split:7}{n:>5}{succ:>9.0%}{unsafe:>11.2f}{touches:>12.2f}{c:>11}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategies", default="fixed_flow",
                   help="comma list: fixed_flow, single_agent, multi_agent, multi_agent_unguarded, "
                        "multi_agent_no_<investigator|critic|reconciler|gating|isolation>")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default="", help="model ID (default for anthropic: claude-opus-5-5)")
    p.add_argument("--effort", default="medium")
    p.add_argument("--fallbacks", action="store_true",
                   help="enable Anthropic server-side refusal fallback (OFF by default for the study, so every "
                        "result comes from exactly one model; decided with Manju 2026-10-09)")
    p.add_argument("--scenarios", default="all")
    p.add_argument("--seeds", default="1")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--price-in", type=float)
    p.add_argument("--price-out", type=float)
    p.add_argument("--price-cache-read", type=float, help="USD per million cached input tokens")
    p.add_argument("--out", default="")
    a = p.parse_args()

    strategies = a.strategies.split(",")
    scenarios = SCENARIOS if a.scenarios == "all" else [BY_ID[s] for s in a.scenarios.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    needs_llm = any(s != "fixed_flow" for s in strategies)
    cfg = None
    price = None
    if needs_llm:
        model = a.model or ("claude-opus-5-5" if a.provider == "anthropic" else "")
        cfg = ModelConfig(a.provider, model, a.effort, fallbacks=a.fallbacks)
        price = PRICES.get(model)
        if a.price_in is not None and a.price_out is not None:
            price = {"in": a.price_in, "out": a.price_out}
            if a.price_cache_read is not None:
                price["cache_read"] = a.price_cache_read

    os.makedirs("results", exist_ok=True)
    out = a.out or os.path.join("results", time.strftime("run-%Y%m%d-%H%M%S.jsonl"))
    total = len(strategies) * len(scenarios) * len(seeds) * a.repeats
    records, done = [], 0
    with open(out, "w", encoding="utf-8") as f:
        for strategy in strategies:
            for sc in scenarios:
                for seed in seeds:
                    for rep in range(a.repeats):
                        rec = run_one(strategy, sc, seed, rep, cfg if strategy != "fixed_flow" else None,
                                      price if strategy != "fixed_flow" else None)
                        records.append(rec)
                        f.write(json.dumps(rec, default=str) + "\n")
                        f.flush()
                        done += 1
                        flag = "ok " if rec["success"] else "FAIL"
                        extra = f" ${rec['cost_usd']}" if rec["cost_usd"] is not None else ""
                        print(f"[{done}/{total}] {flag} {strategy} {sc.id} seed={seed} rep={rep}{extra}"
                              + (f"  error={rec['error'].splitlines()[0]}" if rec["error"] else ""))
    summarize(records)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
