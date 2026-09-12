#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage2_open_mopd_five_teacher.json}"
MODE="${2:---preflight-only}"

case "${MODE}" in
    --preflight-only|--run)
        ;;
    --dry-run)
        MODE=""
        ;;
    *)
        echo "Usage: $0 [config.json] [--preflight-only|--dry-run|--run]" >&2
        exit 2
        ;;
esac

python3 -m code_rewrite_feedback_expander.multi_expert.stage2_open_mopd \
    --config "${CONFIG}" ${MODE:+"${MODE}"}
