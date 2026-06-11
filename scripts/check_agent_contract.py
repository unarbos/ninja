#!/usr/bin/env python3
"""Check that the bundle's agent.py satisfies the validator entry-point contract.

Loads agent.py exactly the way the validator's harness runner does (by file
path, with the bundle root on sys.path) and verifies the solve() signature.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path


def main() -> int:
    bundle_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    entrypoint = bundle_root / "agent.py"
    if not entrypoint.is_file():
        print(f"error: {entrypoint} not found", file=sys.stderr)
        return 1
    sys.path.insert(0, str(bundle_root))
    spec = importlib.util.spec_from_file_location("submitted_agent", str(entrypoint))
    if spec is None or spec.loader is None:
        print(f"error: unable to load {entrypoint}", file=sys.stderr)
        return 1
    module = importlib.util.module_from_spec(spec)
    sys.modules["submitted_agent"] = module
    spec.loader.exec_module(module)

    solve = getattr(module, "solve", None)
    if not callable(solve):
        print("error: agent.py must define solve(repo_path, issue, model, api_base, api_key)", file=sys.stderr)
        return 1
    params = list(inspect.signature(solve).parameters)
    expected = ["repo_path", "issue", "model", "api_base", "api_key"]
    if params[: len(expected)] != expected:
        print(f"error: solve() parameters are {params}; must start with {expected}", file=sys.stderr)
        return 1
    print(f"Import OK: {entrypoint}")
    print(f"Signature OK: {params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
