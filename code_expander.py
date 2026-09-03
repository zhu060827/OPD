from __future__ import annotations

"""Code 数据扩充主入口。

这里保留了两条路径：
1. 如果输入只有 prompt / starter_code / tests，就走原来的“生成代码 -> 跑测试 -> 错误反馈 -> 修复”流程；
2. 如果输入里已经有原始 code 和 reasoning，就走队友的 rewrite 扩充流程：
   同步改写 reasoning 和 code，先做语义等价检查，再做质量评估，只有有提升才保留。

这样既不破坏原来的演示，也把结题阶段的 Code 扩充做得更像真正的数据增强。
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from code_rewrite_feedback_expander import CodeRecord, CodeRewriteExpansionPipeline
from code_rewrite_feedback_expander.llm import MockRewriteClient, parse_rewrite_response
from code_rewrite_feedback_expander.models import RewriteCandidate
from code_rewrite_feedback_expander.visualization import write_quality_svg
from config import MAX_ITERATIONS, OUTPUT_DIR, SAMPLE_DATA_DIR
from llm_client import LLMClient, strip_code_fence
from result_schema import ResultRecord, append_jsonl
from safe_exec import run_python_tests


def sample_code_tasks() -> List[Dict[str, Any]]:
    """本地最小样例。没有联网或不拉 HumanEval 时，用它保证网页 demo 能跑。"""
    sample_path = SAMPLE_DATA_DIR / "sample_code.json"
    if sample_path.exists():
        try:
            data = json.loads(sample_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass
    return [
        {
            "task_id": "sample_add_rewrite",
            "prompt": "Write a function add(a, b) that returns the sum of a and b.",
            "reasoning": ["The function receives two numbers.", "Returning a + b gives their sum."],
            "code": "def add(a, b):\n    return a + b\n",
            "tests": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
                "assert add(10, 5) == 15",
            ],
        },
        {
            "task_id": "sample_close_generate",
            "prompt": "Return True if any two numbers are closer than threshold.",
            "starter_code": "def has_close_elements(numbers, threshold):\n    pass\n",
            "tests": (
                "assert has_close_elements([1.0, 2.0, 3.0], 0.5) is False\n"
                "assert has_close_elements([1.0, 2.0, 3.0], 1.1) is True\n"
                "print('ok')"
            ),
        },
    ]


def load_humaneval_tasks(limit: int = 3) -> List[Dict[str, Any]]:
    """HumanEval 入口。默认不联网，只有设置 USE_HUMANEVAL_DEMO=1 时才尝试加载。"""
    if os.getenv("USE_HUMANEVAL_DEMO", "0").lower() not in {"1", "true", "yes"}:
        return sample_code_tasks()[:limit]
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/openai_humaneval", split=f"test[:{limit}]")
        tasks: List[Dict[str, Any]] = []
        for row in dataset:
            entry = row.get("entry_point", "")
            tests = row.get("test", "")
            if entry:
                tests = f"{tests}\ncheck({entry})\nprint('ok')"
            tasks.append(
                {
                    "task_id": row.get("task_id", entry or "humaneval"),
                    "prompt": row.get("prompt", ""),
                    "starter_code": row.get("prompt", ""),
                    "tests": tests,
                }
            )
        return tasks or sample_code_tasks()[:limit]
    except Exception:
        return sample_code_tasks()[:limit]


def normalize_task(task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """兼容不同来源字段名，统一成 Code pipeline 内部格式。"""
    if task:
        tests = task.get("tests") or task.get("test") or task.get("test_cases") or ""
        return {
            "task_id": str(task.get("task_id") or task.get("id") or "custom_task"),
            "prompt": str(task.get("prompt") or task.get("question") or task.get("instruction") or ""),
            "starter_code": str(task.get("starter_code") or task.get("starter") or ""),
            "code": str(task.get("code") or task.get("solution") or task.get("canonical_solution") or ""),
            "reasoning": normalize_reasoning(task.get("reasoning") or task.get("cot") or task.get("chain_of_thought") or task.get("explanation") or []),
            "tests": tests,
            "language": str(task.get("language") or "python"),
        }
    return load_humaneval_tasks(limit=1)[0]


def normalize_reasoning(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.replace("\r\n", "\n").split("\n") if line.strip()]
    return []


def normalize_tests(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        # HumanEval 的 tests 通常包含 def check(candidate):，必须保留函数体缩进。
        return [value.replace("\r\n", "\n").strip()]
    return []


def should_use_rewrite(task: Dict[str, Any]) -> bool:
    """有原始 code 和 reasoning 时才走 rewrite；否则保持原来的生成修复流程。"""
    return bool(str(task.get("code", "")).strip() and task.get("reasoning"))


class ProjectRewriteClient:
    """把全项目统一的 LLMClient 适配成队友 rewrite pipeline 所需接口。"""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.mock = MockRewriteClient()

    def rewrite(self, record: CodeRecord, strategy: str, feedback: str = "") -> RewriteCandidate:
        provider = os.getenv("CODE_REWRITE_PROVIDER", "mock").lower()
        if provider not in {"openai", "real", "project_llm"}:
            return self.mock.rewrite(record, strategy=strategy, feedback=feedback)
        prompt = build_rewrite_prompt(record, strategy, feedback)
        try:
            raw = self.llm_client.chat(prompt, system_prompt=REWRITE_SYSTEM_PROMPT, temperature=0.2)
            data = parse_rewrite_response(raw)
            code = data.get("code") or record.code
            reasoning = data.get("reasoning") or record.reasoning
            return RewriteCandidate(
                code=str(code).strip(),
                reasoning=reasoning,
                rationale=str(data.get("rationale") or f"LLM rewrite with {strategy}."),
                strategy=strategy,
                raw_response=raw,
                metadata={"provider": "project_llm"},
            )
        except Exception:
            return self.mock.rewrite(record, strategy=strategy, feedback=feedback)


REWRITE_SYSTEM_PROMPT = (
    "你是代码数据扩充助手。请在保持函数签名和行为不变的前提下，"
    "同步改写 reasoning 和 code。输出必须是 JSON，不要输出 Markdown。"
)


def build_rewrite_prompt(record: CodeRecord, strategy: str, feedback: str = "") -> str:
    return f"""请按照策略 {strategy} 改写下面的代码样本。

题目:
{record.prompt}

原始 reasoning:
{json.dumps(record.reasoning, ensure_ascii=False, indent=2)}

原始代码:
```python
{record.code}
```

单元测试:
{json.dumps(record.tests, ensure_ascii=False, indent=2)}

上一轮反馈:
{feedback or "无"}

策略含义:
- cot: 增加更清楚的推理说明或中间变量。
- style: 改善代码风格。
- ast: 做 AST 等价重构。
- variable: 替换为更清晰的变量名。
- control_flow: 调整控制流，但不改变行为。

请只返回 JSON:
{{"reasoning": ["..."], "code": "...", "rationale": "..."}}
"""


def run_code_rewrite_pipeline(
    task: Dict[str, Any],
    llm_client: Optional[LLMClient] = None,
    max_iterations: int = 5,
) -> ResultRecord:
    """队友 rewrite 包接入后的 Code 扩充路径。"""
    tests = normalize_tests(task.get("tests"))
    record = CodeRecord(
        task_id=task["task_id"],
        prompt=task["prompt"],
        code=task["code"],
        reasoning=task["reasoning"],
        tests=tests,
        language=task.get("language", "python"),
        metadata={"source": "integrated_rewrite_pipeline"},
    )
    pipeline = CodeRewriteExpansionPipeline(
        llm=ProjectRewriteClient(llm_client),
        max_iterations=max_iterations,
        max_refine_iterations=3,
        patience=2,
        accept_threshold=0.78,
    )
    expansion = pipeline.expand_record(record)
    expansion_dict = expansion.to_dict()
    svg_path = OUTPUT_DIR / "code_quality.svg"
    write_quality_svg(svg_path, expansion_dict)

    semantic = expansion_dict.get("semantic_result", {})
    quality = expansion_dict.get("final_quality", {})
    stats = expansion_dict.get("expansion_stats", {})
    accepted = bool(expansion_dict.get("accepted"))
    feedback = "Code rewrite 扩充通过：语义保持，且至少有一次质量提升。" if accepted else "Code rewrite 未保留：没有同时满足语义通过和质量提升。"

    result = ResultRecord.create(
        modality="code",
        input_summary={
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "mode": "rewrite",
            "has_reasoning": bool(task["reasoning"]),
            "tests": tests,
        },
        structured_representation={
            "strategies": ["cot", "style", "ast", "variable", "control_flow"],
            "semantic_gate": ["ast_parse", "signature_consistency", "safety", "unit_tests", "ast_similarity", "codebleu_like"],
            "quality_dimensions": ["readability", "complexity", "length_balance", "diversity", "style"],
        },
        generated={
            "prompt": task["prompt"],
            "original_reasoning": expansion.original_reasoning,
            "expanded_reasoning": expansion.expanded_reasoning,
            "original_code": expansion.original_code,
            "generated_code": expansion.expanded_code,
            "final_code": expansion.expanded_code,
            "repair_trace": expansion.iteration_trace,
            "retained_rewrites": expansion.retained_rewrites,
            "quality_svg": str(svg_path),
        },
        verification={
            "passed": accepted,
            "error_type": "None" if accepted else "RewriteNotRetained",
            "error_message": "" if accepted else semantic.get("feedback", "质量没有提升或语义检查未通过。"),
            "attempts": stats.get("attempt_count", 0),
            "semantic_result": semantic,
            "quality_result": quality,
        },
        feedback=feedback,
        accepted=accepted,
        round=int(stats.get("attempt_count", 0)),
        errors=[] if accepted else [semantic.get("feedback", "rewrite 未通过保留条件")],
        metrics={
            "quality_score": quality.get("aggregate_score"),
            "retained_count": stats.get("retained_count", 0),
            "attempt_count": stats.get("attempt_count", 0),
            "reasoning_steps_before": stats.get("original_reasoning_steps", 0),
            "reasoning_steps_after": stats.get("expanded_reasoning_steps", 0),
            "code_lines_before": stats.get("original_code_lines", 0),
            "code_lines_after": stats.get("expanded_code_lines", 0),
        },
    )
    path = OUTPUT_DIR / "code_repair_traces.jsonl"
    result.saved_path = str(path)
    append_jsonl(path, result.to_dict())
    return result


def clean_generated_code(code: str) -> str:
    """清理 LLM 常见 Markdown 代码块，留下纯 Python。"""
    return strip_code_fence(code).replace("\r\n", "\n").strip()


def function_signature(starter_code: str) -> Optional[str]:
    match = re.search(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:", starter_code or "")
    return match.group(0) if match else None


def assemble_code(starter_code: str, generated_code: str) -> str:
    """LLM 可能返回完整函数，也可能只返回函数体，这里统一拼成可执行代码。"""
    generated_code = clean_generated_code(generated_code)
    if re.search(r"^\s*def\s+", generated_code, flags=re.M):
        return generated_code
    signature = function_signature(starter_code)
    if signature:
        body_lines = generated_code.splitlines() or ["pass"]
        body = "\n".join(("    " + line if line.strip() else line) for line in body_lines)
        return f"{signature}\n{body}\n"
    return generated_code or starter_code


def ast_check(code: str) -> Dict[str, Any]:
    try:
        ast.parse(code)
        return {"passed": True, "error_type": "None", "error_message": ""}
    except SyntaxError as exc:
        return {"passed": False, "error_type": "SyntaxError", "error_message": str(exc)}


def feedback_from_error(error_type: str, error_message: str) -> str:
    """把机器错误转成自然语言修复建议。"""
    if error_type == "SyntaxError":
        return f"代码存在语法错误，请检查缩进、括号、冒号或非法字符。错误信息：{error_message}"
    if error_type == "AssertionError":
        return f"单元测试断言失败，说明返回值不符合题意。请检查边界条件和核心逻辑。错误信息：{error_message}"
    if error_type == "Timeout":
        return "代码执行超时，请避免死循环，降低复杂度，并确保函数能在限制时间内返回。"
    if error_type == "ImportError":
        return f"代码导入了不可用模块，请尽量只使用 Python 标准能力。错误信息：{error_message}"
    if error_type == "RuntimeError":
        return f"代码运行时报错，请根据 traceback 修复异常。错误信息：{error_message}"
    return f"代码未通过验证，请重新生成更稳健的实现。错误信息：{error_message}"


def run_code_generate_repair_pipeline(
    task: Dict[str, Any],
    llm_client: Optional[LLMClient] = None,
    max_attempts: int = MAX_ITERATIONS,
) -> ResultRecord:
    """原来的代码生成/修复路径，作为没有原始 code/reasoning 时的兜底。"""
    llm_client = llm_client or LLMClient()
    prompt = task["prompt"]
    starter_code = task.get("starter_code") or task.get("code") or ""
    tests_value = task.get("tests") or ""
    tests = "\n".join(normalize_tests(tests_value))

    repair_trace: List[Dict[str, Any]] = []
    feedback = ""
    wrong_code = ""
    final_code = ""
    final_result: Dict[str, Any] = {
        "passed": False,
        "error_type": "UnknownError",
        "error_message": "No attempt was executed.",
        "stdout": "",
        "stderr": "",
        "runtime": 0.0,
    }

    for attempt in range(1, max_attempts + 1):
        raw_code = llm_client.generate_code(prompt, starter_code, tests, feedback=feedback)
        candidate_code = assemble_code(starter_code, raw_code)
        final_code = candidate_code
        syntax = ast_check(candidate_code)
        if not syntax["passed"]:
            final_result = {
                "passed": False,
                "error_type": syntax["error_type"],
                "error_message": syntax["error_message"],
                "stdout": "",
                "stderr": syntax["error_message"],
                "runtime": 0.0,
            }
        else:
            final_result = run_python_tests(candidate_code, tests)

        attempt_feedback = "" if final_result["passed"] else feedback_from_error(
            str(final_result["error_type"]),
            str(final_result["error_message"]),
        )
        if not final_result["passed"] and not wrong_code:
            wrong_code = candidate_code
        repair_trace.append(
            {
                "attempt": attempt,
                "generated_code": candidate_code,
                "test_result": final_result,
                "feedback": attempt_feedback,
            }
        )
        if final_result["passed"]:
            feedback = "代码通过 AST 检查和单元测试，可以作为扩充后的正确实现监督信号。"
            break
        feedback = attempt_feedback

    accepted = bool(final_result.get("passed"))
    result = ResultRecord.create(
        modality="code",
        input_summary={
            "task_id": task["task_id"],
            "prompt": prompt,
            "starter_code": starter_code,
            "tests": tests,
            "mode": "generate_repair",
        },
        structured_representation={
            "ast_checked": True,
            "sandbox": "subprocess",
            "max_attempts": max_attempts,
        },
        generated={
            "prompt": prompt,
            "starter_code": starter_code,
            "generated_code": repair_trace[0]["generated_code"] if repair_trace else "",
            "wrong_code": wrong_code,
            "final_code": final_code,
            "fixed_code": final_code if accepted else "",
            "repair_trace": repair_trace,
        },
        verification={
            "passed": accepted,
            "error_type": final_result.get("error_type"),
            "error_message": final_result.get("error_message"),
            "attempts": len(repair_trace),
            "test_result": final_result,
        },
        feedback=feedback,
        accepted=accepted,
        round=len(repair_trace),
        errors=[] if accepted else [str(final_result.get("error_message"))],
        metrics={"passed_count": 1 if accepted else 0, "failed_count": 0 if accepted else 1},
    )
    path = OUTPUT_DIR / "code_repair_traces.jsonl"
    result.saved_path = str(path)
    append_jsonl(path, result.to_dict())
    return result


def run_code_pipeline(
    task: Optional[Dict[str, Any]] = None,
    llm_client: Optional[LLMClient] = None,
    max_attempts: int = MAX_ITERATIONS,
) -> ResultRecord:
    """Code 总入口：优先 rewrite，有缺字段时自动退回生成/修复。"""
    normalized = normalize_task(task)
    if should_use_rewrite(normalized):
        return run_code_rewrite_pipeline(normalized, llm_client=llm_client, max_iterations=max(3, max_attempts + 2))
    return run_code_generate_repair_pipeline(normalized, llm_client=llm_client, max_attempts=max_attempts)


def run_code_demo(limit: int = 1) -> ResultRecord:
    tasks = load_humaneval_tasks(limit=limit)
    return run_code_pipeline(tasks[0] if tasks else None)
