from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .domains import DOMAIN_SPECS, require_domain
from .io_utils import read_jsonl, write_jsonl
from .validation import validate_domain


def _first(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if row.get(key) not in (None, "", []):
            return row[key]
    return default


def _text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def normalize_row(row: dict[str, Any], domain_override: str | None = None, language_override: str | None = None, validate: bool = True, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    domain = require_domain(domain_override or _first(row, "domain", "method", "rewrite_method"))
    source_code = _text(_first(row, "source_code", "before_code", "original_code", default=row.get("extra_info", {}).get("original_code", "")))
    target_code = _text(_first(row, "target_code", "after_code", "rewritten_code", "expanded_code", "response", "code"))
    source_reasoning = _text(_first(row, "source_reasoning", "original_reasoning"))
    target_reasoning = _text(_first(row, "target_reasoning", "expanded_reasoning", "reasoning"))
    task = _text(_first(row, "task", "instruction", "question", "text", "prompt"))
    if isinstance(row.get("prompt"), list):
        task = "\n".join(str(item.get("content", "")) for item in row["prompt"] if isinstance(item, dict))
    if not source_code or not target_code:
        raise ValueError("每行都必须包含源代码和目标改写代码")
    language = (language_override or _first(row, "language", default="python")).strip().lower()
    if language == "python":
        try:
            ast.parse(source_code)
            ast.parse(target_code)
        except SyntaxError as exc:
            raise ValueError(f"Python 语法检查失败：{exc}") from exc
    source_id = str(_first(row, "source_id", "task_id", "id", default=""))
    if not source_id:
        source_id = hashlib.sha256((task + "\n" + source_code).encode()).hexdigest()[:16]
    tests = _first(row, "tests", default=row.get("extra_info", {}).get("tests", []))
    if isinstance(tests, str):
        tests = [tests]
    spec = DOMAIN_SPECS[domain]
    validation_row = dict(row)
    validation_row["source_id"] = source_id
    validation_row["tests"] = list(tests or [])
    accepted, validation_reason, evidence = validate_domain(domain, source_code, target_code, validation_row, language, thresholds) if validate else (True, "已跳过验证", {})
    if not accepted:
        raise ValueError(validation_reason)
    user = f"Task:\n{task}\n\nOriginal reasoning:\n{source_reasoning or '(not provided)'}\n\nOriginal code:\n{source_code}"
    assistant_parts = []
    if target_reasoning:
        assistant_parts.append(f"Rewritten reasoning:\n{target_reasoning}")
    assistant_parts.append(f"Rewritten code:\n{target_code}")
    return {
        "source_id": source_id,
        "domain": domain,
        "language": language,
        "messages": [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "\n\n".join(assistant_parts)},
        ],
        "source_code": source_code,
        "target_code": target_code,
        "tests": list(tests or []),
        "semantic_pass": bool(row.get("semantic_pass", row.get("verification_status") == "semantic_pass")),
        "validation_status": validation_reason,
        "experiment_status": "verified_positive",
        "evidence": evidence,
        "source_dataset": _text(_first(row, "source_dataset", "data_source", default="unknown")),
        "source_paper": _text(row.get("source_paper", "")),
        "metadata": dict(row.get("metadata") or {}),
    }


def prepare(input_path: str, output_dir: str, domain: str | None, seed: int, train_ratio: float, validation_ratio: float, require_semantic_pass: bool, language: str | None = None, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    normalized, rejected = [], []
    for index, row in enumerate(read_jsonl(input_path), 1):
        try:
            item = normalize_row(row, domain, language, validate=True, thresholds=thresholds)
            if require_semantic_pass and not item["semantic_pass"]:
                raise ValueError("semantic_pass is required")
            normalized.append(item)
        except (TypeError, ValueError) as exc:
            rejected.append({"line": index, "reason": str(exc), "row": row})

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        groups.setdefault(item["source_id"], []).append(item)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    train_end = int(len(ids) * train_ratio)
    validation_end = train_end + int(len(ids) * validation_ratio)
    split_ids = {"train": ids[:train_end], "validation": ids[train_end:validation_end], "test": ids[validation_end:]}
    root = Path(output_dir)
    counts = {}
    for split, selected in split_ids.items():
        rows = [item for source_id in selected for item in groups[source_id]]
        counts[split] = write_jsonl(root / f"{split}.jsonl", rows)
    write_jsonl(root / "rejected.jsonl", rejected)
    write_jsonl(root / "accepted.jsonl", normalized)
    manifest = {
        "input": str(Path(input_path).resolve()),
        "domain_override": domain,
        "language_override": language,
        "seed": seed,
        "split_by": "source_id",
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": 1 - train_ratio - validation_ratio},
        "counts": counts,
        "rejected": len(rejected),
        "domains": dict(Counter(item["domain"] for item in normalized)),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="标准化并划分一个 Teacher 领域的 JSONL 数据集。")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", choices=tuple(DOMAIN_SPECS))
    parser.add_argument("--language", help="覆盖源代码语言，例如 python 或 java")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--thresholds", help="人工标注验证集校准得到的 JSON 阈值文件")
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.validation_ratio < 0 or args.train_ratio + args.validation_ratio >= 1:
        parser.error("划分比例必须为测试集留出非空比例")
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8")) if args.thresholds else None
    if thresholds and "domains" in thresholds and args.domain:
        thresholds = thresholds["domains"].get(args.domain, {}).get("thresholds", {})
    result = prepare(args.input, args.output_dir, args.domain, args.seed, args.train_ratio, args.validation_ratio, not args.allow_unverified, args.language, thresholds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
