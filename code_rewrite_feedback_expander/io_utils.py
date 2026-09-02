from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .models import CodeRecord


def read_jsonl(path: str) -> List[CodeRecord]:
    records: List[CodeRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            records.append(
                CodeRecord(
                    task_id=str(data.get("task_id") or data.get("id") or f"row_{line_no}"),
                    prompt=str(data.get("prompt") or data.get("question") or ""),
                    # An expansion record is a valid next-round input. Prefer
                    # the accepted expanded fields, then fall back to raw data.
                    code=str(
                        data.get("expanded_code")
                        or data.get("code")
                        or data.get("reference_code")
                        or data.get("solution")
                        or data.get("answer")
                        or data.get("starter_code")
                        or ""
                    ),
                    reasoning=_parse_reasoning(data),
                    tests=_parse_tests(data),
                    language=str(data.get("language", "python")),
                    metadata={
                        **dict(data.get("metadata", {})),
                        **{
                            key: data[key]
                            for key in ("method", "domain", "rewrite_method")
                            if key in data and data[key] is not None
                        },
                    },
                )
            )
    return records


def _parse_reasoning(data: Dict) -> List[str]:
    value = (
        data.get("expanded_reasoning")
        or data.get("reasoning")
        or data.get("cot")
        or data.get("chain_of_thought")
        or data.get("explanation")
        or []
    )
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip() for line in value.replace("\r\n", "\n").split("\n") if line.strip()]
        return lines if lines else ([value.strip()] if value.strip() else [])
    return []


def _parse_tests(data: Dict) -> List[str]:
    """Normalize list-style assertions and HumanEval-style test scripts.

    Some project datasets store the complete HumanEval harness as one string.
    Iterating that value would silently turn it into one test per character.
    """
    value = data.get("tests") or data.get("test") or data.get("test_code") or []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)] if value else []


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
