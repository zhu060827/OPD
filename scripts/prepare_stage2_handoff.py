"""Convert the validated Stage-1 routing handoff to Open-MOPD Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code_rewrite_feedback_expander.multi_expert.stage2_open_mopd import (
    convert_stage1_handoff_to_parquet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Stage-1 mt_opd_handoff JSONL")
    parser.add_argument("--output", required=True, help="Destination training Parquet")
    args = parser.parse_args()
    report = convert_stage1_handoff_to_parquet(args.input, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
