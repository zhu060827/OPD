#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m code_rewrite_feedback_expander.multi_expert validate-config \
  --config configs/stage1_multi_expert.json

python -m unittest discover -s tests -p "test_*.py" -v

python -m code_rewrite_feedback_expander.multi_expert run \
  --config configs/stage1_multi_expert.json \
  --input code_rewrite_feedback_expander/data/code_real_16.jsonl \
  --output-dir outputs/stage1_multi_expert_smoke \
  --limit "${STAGE1_SMOKE_LIMIT:-3}"
