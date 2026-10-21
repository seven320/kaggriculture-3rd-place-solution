"""Generic JSON-lines worker hosting one Kaggle-style Kaggriculture agent.

Usage: worker.py <agent_file.py>

Protocol: each stdin line is one observation JSON object; the reply line is
the agent's action JSON. The process exits when stdin closes. The agent file
follows the Kaggle convention: a callable named ``agent`` if present,
otherwise the last top-level function definition.
"""

from __future__ import annotations

import ast
import json
import sys
import types


def load_agent(path: str):
    source = open(path).read()
    module = types.ModuleType("submission")
    module.__dict__["__name__"] = "submission"
    exec(compile(source, path, "exec"), module.__dict__)
    entry = module.__dict__.get("agent")
    if not callable(entry):
        names = [
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        ]
        if not names:
            raise SystemExit(f"no callable entry point in {path}")
        entry = module.__dict__[names[-1]]
    return entry


def main() -> None:
    entry = load_agent(sys.argv[1])
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        observation = json.loads(line)
        try:
            action = entry(observation, None)
        except TypeError:
            action = entry(observation)
        if not isinstance(action, dict):
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        sys.stdout.write(json.dumps(action, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
