"""Print how much work the reference orchestrator needs per scenario.

A sanity check on scenario difficulty only. The reference is written with hindsight
of every scenario, so these are not results and are never reported as a baseline.
"""
from wdsim.oracle import Oracle
from wdsim.scenarios import SCENARIOS, build, evaluate

print(f"{'ID':6}{'split':7}{'calls':>6}{'errors':>7}{'human':>6}{'minutes':>8}  title")
for sc in SCENARIOS:
    env, ctx = build(sc, 1)
    start = env.tick
    Oracle(env).run()
    minutes = env.tick - start
    r = evaluate(sc, env, ctx, actor="oracle")
    assert r["success"], sc.id
    print(f"{sc.id:6}{sc.split:7}{r['tool_calls']:>6}{r['failed_tool_calls']:>7}"
          f"{r['effort']['human_touches']:>6}{minutes:>8}  {sc.title}")
