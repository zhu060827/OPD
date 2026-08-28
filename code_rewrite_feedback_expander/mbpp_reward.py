from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


FORBIDDEN_CALLS = {"compile", "eval", "exec", "input", "open", "__import__"}
FORBIDDEN_MODULES = {"ctypes", "multiprocessing", "os", "pathlib", "requests", "shutil", "socket", "subprocess"}
PYTHON_BLOCK = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_code(text: str) -> str:
    blocks = PYTHON_BLOCK.findall(text or "")
    if blocks:
        return max((block.strip() for block in blocks), key=len, default="")
    without_thinking = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    return without_thinking.strip()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _safety_error(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    issues: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            issues.update(alias.name for alias in node.names if alias.name.split(".")[0] in FORBIDDEN_MODULES)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                issues.add(node.module)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS or name.split(".")[0] in FORBIDDEN_MODULES:
                issues.add(name)
    return f"Forbidden operations: {sorted(issues)}" if issues else None


def _decode_payload(ground_truth: Any, extra_info: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(ground_truth, dict):
        payload.update(ground_truth)
    elif isinstance(ground_truth, str):
        try:
            decoded = json.loads(ground_truth)
            if isinstance(decoded, dict):
                payload.update(decoded)
        except json.JSONDecodeError:
            pass
    if extra_info:
        payload.setdefault("setup_code", extra_info.get("setup_code", ""))
        payload.setdefault("tests", extra_info.get("tests", []))
        payload.setdefault("task_id", extra_info.get("task_id", ""))
    tests = payload.get("tests", [])
    if isinstance(tests, str):
        tests = [line for line in tests.splitlines() if line.strip()]
    payload["tests"] = [str(test).strip() for test in tests if str(test).strip()]
    return payload


def _limit_child() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (ImportError, OSError, ValueError):
        pass


def _run_tests(code: str, setup_code: str, tests: list[str], timeout: float) -> tuple[bool, str]:
    # Some MBPP setup snippets instantiate classes defined by the candidate, while imports
    # in setup snippets are still available when the candidate functions are called later.
    script = "\n\n".join(part for part in (code.strip(), setup_code.strip(), "\n".join(tests)) if part).strip()
    with tempfile.TemporaryDirectory(prefix="mbpp_reward_") as temp_dir:
        script_path = Path(temp_dir) / "candidate.py"
        script_path.write_text(script + "\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(script_path)],
                cwd=temp_dir,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                preexec_fn=_limit_child,
            )
        except subprocess.TimeoutExpired:
            return False, "Timeout"
    output = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
    return result.returncode == 0, output[-1000:]


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    timeout: float = 5.0,
    **_: Any,
) -> dict[str, Any]:
    """Evaluate an MBPP rollout for metrics; OPD still trains on teacher token rewards."""
    code = extract_python_code(solution_str)
    payload = _decode_payload(ground_truth, extra_info)
    if data_source.lower() != "mbpp":
        return {"score": 0.0, "passed": False, "error_type": "unsupported_data_source"}
    if not code:
        return {"score": 0.0, "passed": False, "error_type": "missing_code"}
    safety_error = _safety_error(code)
    if safety_error:
        error_type = "syntax_error" if safety_error.startswith("SyntaxError") else "unsafe_code"
        return {"score": 0.0, "passed": False, "error_type": error_type, "error": safety_error}
    tests = payload.get("tests", [])
    if not tests:
        return {"score": 0.0, "passed": False, "error_type": "missing_tests"}

    passed, output = _run_tests(code, str(payload.get("setup_code", "")), tests, timeout)
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "error_type": "none" if passed else "test_failure",
        "test_output": output,
        "task_id": str(payload.get("task_id", "")),
    }
