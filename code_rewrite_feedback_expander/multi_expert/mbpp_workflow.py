"""Dataset preparation and evaluation primitives for the 974-item MBPP workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping


EXPECTED_TOTAL = 974


def mbpp_numeric_id(record: Mapping) -> int:
    value = str(record.get("task_id", ""))
    match = re.search(r"(\d+)$", value)
    if not match:
        raise ValueError(f"Invalid MBPP task_id: {value!r}")
    return int(match.group(1))


def mbpp_split(record: Mapping) -> str:
    task_id = mbpp_numeric_id(record)
    if 1 <= task_id <= 10:
        return "prompt"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "validation"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError(f"MBPP task_id outside 1..974: {task_id}")


def split_handoff_records(records: Iterable[dict], require_complete: bool = True) -> dict[str, list[dict]]:
    groups = {name: [] for name in ("prompt", "test", "validation", "train")}
    seen: set[int] = set()
    for record in records:
        task_id = mbpp_numeric_id(record)
        if task_id in seen:
            raise ValueError(f"Duplicate MBPP task_id: {task_id}")
        seen.add(task_id)
        groups[mbpp_split(record)].append(dict(record))
    if require_complete and seen != set(range(1, EXPECTED_TOTAL + 1)):
        missing = sorted(set(range(1, EXPECTED_TOTAL + 1)) - seen)
        raise ValueError(f"Expected all 974 MBPP records; missing {missing[:10]}")
    return groups


def write_workflow_splits(input_path: str | Path, output_dir: str | Path, require_complete: bool = True) -> dict:
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = split_handoff_records(rows, require_complete=require_complete)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in groups.items():
        path = destination / f"mbpp_{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return {name: len(records) for name, records in groups.items()}


@dataclass
class AttemptResult:
    passed: bool
    code: str
    feedback: str = ""


def evaluate_with_repairs(
    records: Iterable[dict],
    generate: Callable[[dict, str], str],
    verify: Callable[[dict, str], tuple[bool, str]],
    max_repairs: int = 3,
) -> dict:
    """Evaluate Pass@1 and cumulative success after up to ``max_repairs`` repairs."""
    rows = list(records)
    cumulative = [0] * (max_repairs + 1)
    details = []
    for record in rows:
        feedback = ""
        attempts = []
        first_success = None
        for attempt in range(max_repairs + 1):
            code = generate(record, feedback)
            passed, feedback = verify(record, code)
            attempts.append({"attempt": attempt + 1, "passed": passed, "code": code, "feedback": feedback})
            if passed:
                first_success = attempt
                break
        if first_success is not None:
            for index in range(first_success, max_repairs + 1):
                cumulative[index] += 1
        details.append({"task_id": record.get("task_id"), "attempts": attempts})
    total = len(rows)
    return {
        "total": total,
        "pass_at_1": cumulative[0] / total if total else 0.0,
        "cumulative_pass_rates": [value / total if total else 0.0 for value in cumulative],
        "cumulative_pass_counts": cumulative,
        "records": details,
    }
