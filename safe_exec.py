from __future__ import annotations

"""代码沙盒执行工具。

这里不直接裸 exec 候选代码，而是把候选代码写成临时 .py 文件，再用 subprocess 跑。
这样即使代码里有死循环、断言失败或运行时报错，也能通过 timeout 和错误分类把结果收回来。
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

from config import CODE_TIMEOUT


def classify_error(stderr: str, stdout: str = "", timed_out: bool = False) -> str:
    """把 Python 执行错误归类，方便前端展示和 Code 修复反馈。"""
    text = f"{stderr}\n{stdout}"
    if timed_out:
        return "Timeout"
    if "SyntaxError" in text:
        return "SyntaxError"
    if "AssertionError" in text:
        return "AssertionError"
    if "ImportError" in text or "ModuleNotFoundError" in text:
        return "ImportError"
    if "Traceback" in text:
        return "RuntimeError"
    if text.strip():
        return "UnknownError"
    return "None"


def run_python_tests(code: str, tests: str, timeout: int = CODE_TIMEOUT) -> Dict[str, object]:
    """在独立子进程中运行 candidate + tests，返回 passed/error/stdout/stderr/runtime。"""
    start = time.perf_counter()
    script = f"{code.rstrip()}\n\n{tests.strip()}\n"
    with tempfile.TemporaryDirectory(prefix="octree_code_") as temp_dir:
        path = Path(temp_dir) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=temp_dir,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            runtime = time.perf_counter() - start
            passed = completed.returncode == 0
            error_type = "None" if passed else classify_error(completed.stderr, completed.stdout)
            return {
                "passed": passed,
                "error_type": error_type,
                "error_message": completed.stderr.strip() or completed.stdout.strip(),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "runtime": runtime,
            }
        except subprocess.TimeoutExpired as exc:
            runtime = time.perf_counter() - start
            return {
                "passed": False,
                "error_type": "Timeout",
                "error_message": f"Execution exceeded timeout of {timeout} seconds.",
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "runtime": runtime,
            }
