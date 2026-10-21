"""Record or verify SHA-256 fingerprints of frozen files.

  python freeze_manifest.py record scenarios   # write FREEZE-scenarios.json
  python freeze_manifest.py verify scenarios   # check nothing changed since the freeze

Scope "scenarios" covers everything that defines the mock systems and the 31 scenarios.
Scope "harness" (frozen later, after the shakedown) adds the agents and the runner.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SCOPES = {
    "scenarios": ["wdsim/systems.py", "wdsim/env.py", "wdsim/tools.py", "wdsim/owners.py", "wdsim/builder.py",
                  "wdsim/checks.py", "wdsim/scenarios.py"],
    "harness": ["agents/llm.py", "agents/harness.py", "agents/prompts.py", "agents/strategies.py",
                "run_experiment.py"],
}


def digest(rel: str) -> str:
    return hashlib.sha256((HERE / rel).read_bytes()).hexdigest()


def main(action: str, scope: str) -> int:
    files = SCOPES[scope] if scope == "scenarios" else SCOPES["scenarios"] + SCOPES["harness"]
    manifest = HERE / f"FREEZE-{scope}.json"
    if action == "record":
        data = {"scope": scope, "frozen_at": time.strftime("%Y-%m-%d %H:%M"), "files": {f: digest(f) for f in files}}
        manifest.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"recorded {len(files)} files -> {manifest.name}")
        return 0
    data = json.loads(manifest.read_text(encoding="utf-8"))
    changed = [f for f, h in data["files"].items() if digest(f) != h]
    print(f"frozen {data['frozen_at']}: " + ("UNCHANGED" if not changed else "CHANGED: " + ", ".join(changed)))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
