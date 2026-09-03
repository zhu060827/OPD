from __future__ import annotations

import json
from typing import Any, Dict, List

REQUIRED_FIELDS = ("preserve", "fix", "avoid", "evidence")


def parse_feedback(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or any(field not in value for field in REQUIRED_FIELDS):
            raise ValueError("feedback schema is incomplete")
        return {field: _as_list(value[field]) for field in REQUIRED_FIELDS}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {field: _as_list(fallback.get(field, [])) for field in REQUIRED_FIELDS}


def render_feedback(structured: Dict[str, Any], selected_strategy: str) -> str:
    return (
        f"下一轮重点采用 {selected_strategy} 改写。\n"
        f"保留：{'；'.join(structured['preserve']) or '保持已验证的语义和接口。'}\n"
        f"修复：{'；'.join(structured['fix']) or '仅进行有证据的改动。'}\n"
        f"避免：{'；'.join(structured['avoid']) or '避免无依据的结构变化。'}\n"
        f"证据：{'；'.join(structured['evidence']) or '依据本轮结构化评估记录。'}"
    )


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
