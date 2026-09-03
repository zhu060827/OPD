# jsonl_to_opd_parquet.py
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def record_is_valid(record: dict) -> bool:
    """Use expansion gates when present, otherwise use source-data verification."""
    if "accepted" in record or "semantic_result" in record:
        semantic = record.get("semantic_result", {})
        return bool(record.get("accepted", False)) and bool(semantic.get("passed", False))
    return bool(record.get("baseline_verification", {}).get("passed", False))


def response_code(record: dict) -> str:
    for key in ("expanded_code", "code", "solution", "answer", "reference_code"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def tests_text(record: dict) -> str:
    tests = record.get("tests", "")
    if isinstance(tests, list):
        return "\n".join(str(item) for item in tests if str(item).strip()).strip()
    return str(tests).strip()


def user_prompt(record: dict) -> str:
    prompt = str(record.get("prompt") or record.get("question") or "").strip()
    mode = str(record.get("mode", "")).strip()
    starter_code = str(record.get("starter_code", "")).strip()
    tests = tests_text(record)

    sections = [prompt]
    if starter_code:
        sections.append(f"Starter code:\n```python\n{starter_code}\n```")
    if tests:
        sections.append(f"The implementation must pass these tests:\n```python\n{tests}\n```")
    if mode == "rewrite":
        sections.append("Return the complete rewritten Python code and preserve its behavior.")
    elif mode == "generate_repair":
        sections.append("Return a complete corrected Python implementation.")
    return "\n\n".join(section for section in sections if section).strip()


def convert_record(record: dict, index: int) -> dict:
    prompt = user_prompt(record)
    code = response_code(record)

    task_id = str(record.get("task_id") or f"row_{index}")
    valid = record_is_valid(record)
    accepted = bool(record.get("accepted", valid))

    return {
        "data_source": str(record.get("dataset") or "code_rewrite_feedback_expander"),
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "code",
        # 保留 response 供分析或 SFT 使用；OPD rollout 默认重新生成。
        "response": code,
        "reward_model": {
            "style": "rule",
            "ground_truth": str(record.get("reference_code") or code).strip(),
        },
        "extra_info": {
            "index": index,
            "task_id": task_id,
            "accepted": accepted,
            "valid": valid,
            "language": str(record.get("language", "python")),
            "dataset": str(record.get("dataset", "")),
            "mode": str(record.get("mode", "")),
            "tests": tests_text(record),
            "source_url": str(record.get("source_url", "")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="只输出 accepted=true 且 semantic_result.passed=true 的记录",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    rows = []
    filtered = 0

    with Path(args.input).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue

            record = json.loads(line)

            if args.valid_only and not record_is_valid(record):
                filtered += 1
                continue

            converted = convert_record(record, index)
            if converted["prompt"][0]["content"] and converted["response"]:
                rows.append(converted)

    if not rows:
        mode = "有效记录" if args.valid_only else "可转换记录"
        raise ValueError(f"没有找到{mode}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output), compression="snappy")

    print(f"Written: {len(rows)}")
    print(f"Filtered: {filtered}")
    print(f"Valid-only filter: {args.valid_only}")
    print(f"Output: {output}")
    print(f"Columns: {table.column_names}")


if __name__ == "__main__":
    main()
