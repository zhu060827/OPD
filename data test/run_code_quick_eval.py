from __future__ import annotations

"""Small Code rewrite smoke test.

Math is the main experiment in this folder. This script only checks whether the
integrated Code rewrite pipeline can run on a few HumanEval rows.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
DEFAULT_INPUT = BASE_DIR / "datasets" / "code_humaneval_3.jsonl"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import code_expander  # noqa: E402


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}. Run download_real_datasets.py first.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # 测试产物全部放在 data test，避免污染主工程 outputs。
    code_expander.OUTPUT_DIR = RESULT_DIR
    rows = read_jsonl(args.input, args.limit)
    results = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] running {row.get('task_id')}")
        result = code_expander.run_code_pipeline(row)
        data = result.to_dict()
        results.append(data)

    out_jsonl = RESULT_DIR / "code_quick_records.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_csv = RESULT_DIR / "code_quick_summary.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "accepted", "quality_score", "retained_count", "attempt_count"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "task_id": row.get("input_summary", {}).get("task_id"),
                    "accepted": row.get("accepted"),
                    "quality_score": row.get("metrics", {}).get("quality_score"),
                    "retained_count": row.get("metrics", {}).get("retained_count"),
                    "attempt_count": row.get("metrics", {}).get("attempt_count"),
                }
            )

    print(f"Records: {out_jsonl}")
    print(f"Summary: {out_csv}")


if __name__ == "__main__":
    main()
