"""将 Hugging Face save_to_disk 快照适配为 Teacher 构造器的统一 JSONL。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from teacher_training.io_utils import write_jsonl
from teacher_training.test_execution import run_stdin_stdout_tests


SOURCES = {
    "commitpackft": ("bigcode/commitpackft", "CommitPackFT; Palma et al., STYLER"),
    "codesearchnet": ("code-search-net/code_search_net", "Husain et al. (2019), CodeSearchNet"),
    "code_contests": ("deepmind/code_contests", "Li et al. (2022), AlphaCode / CodeContests"),
}


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) not in (None, "", [], {}):
            return row[key]
    return ""


def _stable_id(*parts: Any) -> str:
    text = ":".join(str(part) for part in parts if part not in (None, ""))
    return text or hashlib.sha256(repr(parts).encode()).hexdigest()[:20]


def _valid_python(code: str, min_chars: int, max_chars: int) -> bool:
    if not min_chars <= len(code) <= max_chars:
        return False
    try:
        ast.parse(code)
        compile(code, "<adapted_source>", "exec")
    except SyntaxError:
        return False
    return True


def adapt_commitpackft(row: dict[str, Any], min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    language = str(_first(row, "language", "lang")).lower()
    if language and language not in {"python", "py"}:
        return []
    code = str(_first(row, "new_contents", "new_content", "after", "code"))
    if not _valid_python(code, min_chars, max_chars):
        return []
    commit, path = _first(row, "commit", "sha"), _first(row, "file_path", "path")
    problem_id = _stable_id(commit, path)
    return [{
        "source_id": problem_id, "problem_id": problem_id, "code": code,
        "task": str(_first(row, "message", "commit_message")), "tests": [],
        "test_format": "none", "language": "python", "source_dataset": SOURCES["commitpackft"][0],
        "source_paper": SOURCES["commitpackft"][1], "commit": commit, "file_path": path,
        "license": row.get("license"), "repos": row.get("repos"),
    }]


def adapt_codesearchnet(row: dict[str, Any], min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    code = str(_first(row, "code", "func_code_string", "whole_func_string"))
    if not _valid_python(code, min_chars, max_chars):
        return []
    repo, path = _first(row, "repository_name", "repo"), _first(row, "path", "file_path")
    function = _first(row, "func_name", "function_name", "name")
    sha = _first(row, "sha", "commit")
    problem_id = _stable_id(repo, path, function, sha)
    return [{
        "source_id": problem_id, "problem_id": problem_id, "code": code,
        "task": str(_first(row, "func_documentation_string", "docstring", "documentation")),
        "tests": [], "test_format": "none", "language": "python",
        "source_dataset": SOURCES["codesearchnet"][0], "source_paper": SOURCES["codesearchnet"][1],
        "metadata": {"repository": repo, "path": path, "function_name": function, "sha": sha},
    }]


def _parallel_values(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        candidate = value.get(key, [])
        return candidate if isinstance(candidate, list) else [candidate]
    return []


def _is_python_language(value: Any) -> bool:
    text = str(value).lower().replace("_", " ")
    # CodeContests TFDS enum: PYTHON3=3. 同时兼容已解码的字符串标签。
    return value == 3 or "python" in text


def _contest_tests(row: dict[str, Any]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for field in ("public_tests", "private_tests", "generated_tests"):
        block = row.get(field) or {}
        inputs, outputs = _parallel_values(block, "input"), _parallel_values(block, "output")
        cases.extend({"input": str(given), "output": str(expected)} for given, expected in zip(inputs, outputs))
    return cases


def adapt_code_contests(
    row: dict[str, Any], min_chars: int, max_chars: int, timeout: int, verify_tests: bool
) -> list[dict[str, Any]]:
    solutions = row.get("solutions") or {}
    languages = _parallel_values(solutions, "language")
    codes = _parallel_values(solutions, "solution")
    tests = _contest_tests(row)
    if not tests:
        return []
    problem_id = _stable_id(_first(row, "name", "problem_id", "id"))
    output = []
    for index, (language, code_value) in enumerate(zip(languages, codes)):
        code = str(code_value)
        if not _is_python_language(language) or not _valid_python(code, min_chars, max_chars):
            continue
        if verify_tests and not run_stdin_stdout_tests(code, tests, timeout)[0]:
            continue
        output.append({
            "source_id": f"{problem_id}:solution:{index}", "problem_id": problem_id,
            "code": code, "task": str(_first(row, "description", "problem")),
            "tests": tests, "test_format": "stdin_stdout", "test_timeout": timeout,
            "language": "python", "source_dataset": SOURCES["code_contests"][0],
            "source_paper": SOURCES["code_contests"][1],
            "metadata": {"solution_index": index, "original_language": language},
        })
    return output


def adapt_rows(
    rows: Iterable[dict[str, Any]], dataset: str, min_chars: int = 40, max_chars: int = 20000,
    limit: int | None = None, timeout: int = 5, verify_tests: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted, seen, scanned = [], set(), 0
    for row in rows:
        scanned += 1
        if dataset == "commitpackft":
            candidates = adapt_commitpackft(row, min_chars, max_chars)
        elif dataset == "codesearchnet":
            candidates = adapt_codesearchnet(row, min_chars, max_chars)
        else:
            candidates = adapt_code_contests(row, min_chars, max_chars, timeout, verify_tests)
        for candidate in candidates:
            digest = hashlib.sha256(candidate["code"].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            accepted.append(candidate)
            if limit and len(accepted) >= limit:
                return accepted, {"dataset": dataset, "scanned": scanned, "accepted": len(accepted)}
    return accepted, {"dataset": dataset, "scanned": scanned, "accepted": len(accepted)}


def main() -> None:
    parser = argparse.ArgumentParser(description="将原始 HF 快照适配为领域构造器 JSONL。")
    parser.add_argument("--dataset", required=True, choices=tuple(SOURCES))
    parser.add_argument("--input", required=True, help="datasets.save_to_disk 产生的 dataset 目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--skip-test-verification", action="store_true", help="仅用于调试；正式 CodeContests 适配不得使用")
    args = parser.parse_args()
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("请先安装 teacher_training/requirements.txt") from exc
    rows, manifest = adapt_rows(
        load_from_disk(args.input), args.dataset, args.min_chars, args.max_chars,
        args.limit, args.timeout, not args.skip_test_verification,
    )
    write_jsonl(args.output, rows)
    manifest.update({"input": str(Path(args.input).resolve()), "output": str(Path(args.output).resolve()),
                     "test_verification": not args.skip_test_verification})
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
