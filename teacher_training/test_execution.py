"""统一执行 Teacher 数据中的可复现测试契约。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


def normalize_stdout(value: str) -> str:
    """按竞赛评测的常见文本口径：忽略行末空白和文件末空行。"""
    return "\n".join(line.rstrip() for line in str(value).replace("\r\n", "\n").splitlines()).rstrip()


def run_python_assert_tests(code: str, tests: list[Any], timeout: int = 5) -> tuple[bool, str]:
    script = code.rstrip() + "\n\n\ndef __run_tests__():"
    for test in tests:
        if not isinstance(test, str):
            return False, "python_assert 测试必须是字符串"
        script += "\n" + textwrap.indent(test.rstrip(), "    ")
    script += "\n    return True\n\n__run_tests__()\n"
    return _run_script(script, "", timeout)


def run_stdin_stdout_tests(code: str, tests: list[Any], timeout: int = 5) -> tuple[bool, str]:
    if not tests:
        return False, "stdin_stdout 测试为空"
    for index, case in enumerate(tests):
        if not isinstance(case, dict) or "input" not in case or "output" not in case:
            return False, f"第 {index + 1} 个 stdin_stdout 测试缺少 input/output"
        passed, output = _run_script(code, str(case["input"]), timeout)
        if not passed:
            return False, f"第 {index + 1} 个测试运行失败：{output[-500:]}"
        if normalize_stdout(output) != normalize_stdout(str(case["output"])):
            return False, (
                f"第 {index + 1} 个测试输出不匹配："
                f"expected={normalize_stdout(str(case['output']))!r}, actual={normalize_stdout(output)!r}"
            )
    return True, "all stdin/stdout tests passed"


def run_registered_tests(
    code: str, tests: list[Any], test_format: str = "python_assert", timeout: int = 5
) -> tuple[bool, str]:
    if not tests:
        return False, "未提供测试"
    if test_format == "python_assert":
        return run_python_assert_tests(code, tests, timeout)
    if test_format == "stdin_stdout":
        return run_stdin_stdout_tests(code, tests, timeout)
    return False, f"未注册的 test_format={test_format!r}"


def _run_script(code: str, stdin: str, timeout: int) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="teacher_test_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)], input=stdin, text=True, capture_output=True,
                timeout=timeout, cwd=directory, check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "Timed out."
    output = (completed.stdout or "")
    if completed.returncode != 0:
        output += completed.stderr or ""
    return completed.returncode == 0, output
