"""Split the complete Stage-1 handoff into fixed MBPP workflow partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_rewrite_feedback_expander.multi_expert.mbpp_workflow import write_workflow_splits
from code_rewrite_feedback_expander.multi_expert.stage2_open_mopd import convert_stage1_handoff_to_parquet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Complete 974-row Stage-1 handoff JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    counts = write_workflow_splits(args.input, args.output_dir, not args.allow_incomplete)
    output = Path(args.output_dir)
    if args.parquet:
        for split in ("train", "validation"):
            convert_stage1_handoff_to_parquet(
                output / f"mbpp_{split}.jsonl", output / f"mbpp_{split}.parquet"
            )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
