# Agentic orchestration of HCM-driven cross-system change: environment, agents and results

Code, data and run traces for the paper

> M. Mathew, "Agentic Orchestration of Cross-System Enterprise Change: LLM Agents for Workday HCM-Driven Cascades," submitted to *IEEE Access*.

The repository contains a reproducible mock environment of an HCM system of record and connected systems, the 31 evaluation scenarios, the agent strategies that were compared, every result reported in the paper, and the full trace of every run.

**All data are synthetic.** Workers, organizations and identifiers are generated from a seed; government identifiers are synthetic tokens. No employer, client or tenant data and no real personal data are included. Workday and Workday HCM are trademarks of Workday, Inc.; this is independent research, not affiliated with or endorsed by Workday, Inc., and no Workday software or tenant was used.

## Contents

| Path | Contents |
|---|---|
| `wdsim/` | The mock environment: HCM system of record (workers, organizations, cost centers, onboarding tasks, asynchronous business processes, event feed), identity, finance, IT service management and legal holds; 45 typed tools with risk tiers; fault injection; simulated owner teams; the 31 scenarios (`scenarios.py`) and their scoring (`checks.py`); a hindsight reference orchestrator (`oracle.py`) used only to show that every scenario is solvable, never as a baseline |
| `agents/` | Model adapters (Anthropic Messages API, OpenAI Responses API), the agent loop, the bounded-autonomy gate, prompts and the strategies (`fixed_flow`, `single_agent`, `multi_agent_unguarded`, `multi_agent`) |
| `run_experiment.py` | Runs strategies x scenarios x seeds and writes one JSON line per run plus a full trace per run |
| `rescore.py` | Re-scores saved runs by replaying their traces (no model calls) |
| `freeze_manifest.py`, `FREEZE-*.json` | SHA-256 fingerprints of the frozen code used for all reported runs |
| `stage2_batch.ps1` | The launcher used for the main experiment (Windows PowerShell) |
| `tests/` | 43 tests: environment behaviour, scenario solvability, agent harness (no API keys needed) |
| `results/` | Main-experiment results (`stage2-*.jsonl`), re-scored results (`stage2-rescored.jsonl`), the fixed-flow baseline (`fixed_flow_final-2026-10-15.jsonl`) and `traces/` (one JSON file per run: every tool call, escalation, fix request and agent message) |
| `analysis/` | `extract_results.py` computes every number in the paper into `analysis/data.json`; `make_figures.py` draws the figures |
| `SCENARIOS.md` | What happens in each scenario and what counts as a correct end state |

## Reproducing the paper's results (no API keys, no cost)

Requires Python 3.12 and the packages in `requirements.txt`.

```
pip install -r requirements.txt
python freeze_manifest.py verify scenarios        # the code matches the frozen versions
python freeze_manifest.py verify harness
python -m unittest discover -s tests             # 43 tests
python run_experiment.py --strategies fixed_flow --seeds 1,2,3 --out results/fixed_flow_check.jsonl
python rescore.py --out results/rescore_check.jsonl
python analysis/extract_results.py
python analysis/make_figures.py
```

`rescore.py` rebuilds each scenario and replays the recorded tool calls of every saved run. The environment is deterministic given the seed and the sequence of calls, so the replay reproduces each run's end state exactly; the tool reports any run whose replay differs.

## Re-running the agents (paid)

The agent runs call the Anthropic and OpenAI APIs and need `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the environment. The main experiment used Claude Opus 5.5 (`claude-opus-5-5`, effort medium) and GPT-5.5 (`gpt-5.5-2026-04-23`, reasoning effort medium), 372 runs in total, at a model cost of about USD 187 at list prices. For example:

```
python run_experiment.py --strategies multi_agent --provider anthropic --scenarios ON-1 --seeds 1 --out results/my_run.jsonl
python run_experiment.py --strategies single_agent --provider openai --model gpt-5.5-2026-04-23 --price-in 5 --price-out 30 --price-cache-read 0.5 --scenarios ON-1 --seeds 1 --out results/my_gpt_run.jsonl
```

Runs write traces to `results/traces/` using the name strategy-model-scenario-seed-repeat, so a re-run with the same settings replaces the saved trace of that run; use a copy of the repository to keep the original traces.

Model outputs are not deterministic, so re-runs can differ from the recorded runs; the recorded traces are the reference for the reported results.

## Scoring note

After the main experiment, one check was corrected: in the two contractor-conversion scenarios (ON-8, TE-4) a person may keep the equipment already issued under the contractor record instead of receiving a new order. The correction is in `wdsim/scenarios.py` (`expect_onboarded`, `prior_wid`) and applies to every strategy; `results/stage2-rescored.jsonl` holds both the original and the corrected score of every run, and the paper reports both.

## Citation

See `CITATION.cff`.

## Terms of use

Copyright (c) 2026 Manju Mathew. All rights reserved.

The code, data and traces are shared as a research sample so that readers can inspect them and reproduce the results reported in the paper. No license is granted for any other use, redistribution or modification. The material is provided as is, without warranty of any kind.
