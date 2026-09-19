"""审计 train/validation/test 的原始问题族泄漏和样本重复。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .io_utils import read_jsonl


def audit(paths: dict[str, str]) -> dict[str, object]:
    problem_sets: dict[str, set[str]] = {}
    code_sets: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split, path in paths.items():
        rows = list(read_jsonl(path))
        counts[split] = len(rows)
        missing = [index for index, row in enumerate(rows, 1) if not row.get("problem_id")]
        if missing:
            raise ValueError(f"{split} 缺少 problem_id；行号示例：{missing[:5]}")
        problem_sets[split] = {str(row["problem_id"]) for row in rows}
        code_sets[split] = {
            hashlib.sha256(str(row.get("target_code", "")).strip().encode()).hexdigest() for row in rows
        }
    overlaps = {}
    names = tuple(paths)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            problem_overlap = problem_sets[left] & problem_sets[right]
            code_overlap = code_sets[left] & code_sets[right]
            overlaps[f"{left}__{right}"] = {
                "problem_id_overlap": len(problem_overlap),
                "target_code_hash_overlap": len(code_overlap),
                "problem_examples": sorted(problem_overlap)[:5],
            }
    valid = all(
        item["problem_id_overlap"] == 0 and item["target_code_hash_overlap"] == 0
        for item in overlaps.values()
    )
    return {"valid": valid, "counts": counts, "overlaps": overlaps}


def main() -> None:
    parser = argparse.ArgumentParser(description="审计三个数据划分是否存在问题族或代码重复泄漏。")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit({"train": args.train, "validation": args.validation, "test": args.test})
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
