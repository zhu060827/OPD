"""Evaluate Student checkpoints with Pass@1 and cumulative repair success.

Generation and verification are supplied as ``module:function`` factories so the
same evaluator can drive local Transformers, vLLM, or a hardened code sandbox.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_rewrite_feedback_expander.multi_expert.mbpp_workflow import evaluate_with_repairs


def _load_factory(spec: str):
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="Fixed MBPP test JSONL")
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path; repeatable")
    parser.add_argument("--generator-factory", required=True, help="module:function(checkpoint_path)")
    parser.add_argument("--verifier-factory", required=True, help="module:function()")
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.test).read_text(encoding="utf-8").splitlines() if line.strip()]
    generator_factory = _load_factory(args.generator_factory)
    verifier = _load_factory(args.verifier_factory)()
    results = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        generator = generator_factory(path)
        results[name] = evaluate_with_repairs(rows, generator, verifier, args.max_repairs)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
