"""Every scenario must be (a) solvable with the public tools and (b) not already solved.

(a) The reference orchestrator reaches a fully correct end state.
(b) Doing nothing fails, so each scenario actually tests something.
Run over several seeds so results don't depend on one synthetic company.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wdsim.oracle import Oracle  # noqa: E402
from wdsim.scenarios import SCENARIOS, build, evaluate  # noqa: E402

SEEDS = (1, 2, 3)


class ScenarioTests(unittest.TestCase):
    def test_reference_solves_every_scenario(self):
        for sc in SCENARIOS:
            for seed in SEEDS:
                with self.subTest(scenario=sc.id, seed=seed):
                    env, ctx = build(sc, seed)
                    Oracle(env).run()
                    result = evaluate(sc, env, ctx, actor="oracle")
                    self.assertTrue(result["success"], {k: result[k] for k in
                                                        ("violations", "unsafe", "failed_expectations")})

    def test_doing_nothing_fails_every_scenario(self):
        for sc in SCENARIOS:
            with self.subTest(scenario=sc.id):
                env, ctx = build(sc, 1)
                self.assertFalse(evaluate(sc, env, ctx)["success"])

    def test_conversion_keeps_prior_equipment(self):
        # Correction 2026-10-15: a converted worker keeps the equipment on the contractor record, so no new
        # order is needed, unless that equipment was recovered.
        from wdsim.scenarios import BY_ID, expect_onboarded
        for sid in ("ON-8", "TE-4"):
            with self.subTest(scenario=sid):
                env, ctx = build(BY_ID[sid], 1)
                Oracle(env).run()
                w, prior = ctx["worker"], ctx["contractor"].wid
                for tid in [t for t, v in env.itsm.tickets.items()
                            if v["worker_wid"] == w.wid and v["type"] == "equipment_order"]:
                    del env.itsm.tickets[tid]
                self.assertNotIn("no equipment order", expect_onboarded(env, w, prior_wid=prior))
                self.assertIn("no equipment order", expect_onboarded(env, w))
                env.itsm.tickets["tkt-test-rec"] = {"ticket_id": "tkt-test-rec", "worker_wid": prior,
                                                    "type": "asset_recovery", "status": "Open"}
                self.assertIn("no equipment order", expect_onboarded(env, w, prior_wid=prior))

    def test_catalog_shape(self):
        ids = [s.id for s in SCENARIOS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(SCENARIOS), 31)
        self.assertEqual(sum(s.split == "known" for s in SCENARIOS), 8)
        self.assertEqual({s.split for s in SCENARIOS}, {"known", "novel"})


if __name__ == "__main__":
    unittest.main()
