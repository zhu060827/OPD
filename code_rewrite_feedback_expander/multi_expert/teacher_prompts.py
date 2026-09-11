"""Versioned prompt library for prompt-conditioned Teacher ablations."""
from __future__ import annotations
import hashlib

PROMPT_VERSION = "teacher-lens-v2"
_LENSES = {
    "cot": "你是代码推理结构分析专家。\n仅关注：推理步骤清晰度；中间变量对算法表达的帮助；代码与解释结构一致性；关键计算逻辑顺序。\n忽略：变量名称美观程度；排版；与推理表达无关的 AST 差异。",
    "style": "你是代码风格与可读性分析专家。\n仅关注：可读性；命名和格式；函数结构；注释/docstring；代码组织一致性。\n忽略：算法推理；AST 结构本身；变量数据流；控制流优化。",
    "ast": "你是 AST 结构分析专家。\n仅关注：表达式树结构；语句节点组织；语义等价语法结构；辅助变量；提前返回。\n忽略：变量名称美观程度；注释风格；格式排版；与 AST 无关的文字表达。",
    "variable": "你是变量语义与数据流分析专家。\n仅关注：变量命名语义；定义与使用关系；数据流；临时变量；作用域和依赖。\n忽略：注释/docstring；纯格式；控制流结构；与变量语义无关的 AST 排版。",
    "control_flow": "你是控制流结构分析专家。\n仅关注：条件分支；循环；提前返回；嵌套层级；控制流路径；等价控制流重构。\n忽略：变量名称；注释风格；纯格式；与控制流无关的表达式细节。",
}

def teacher_prompt(strategy: str, task: str, mode: str = "directional") -> str:
    if mode == "none":
        return f"Task:\n{task.rstrip()}\nCandidate code:\n"
    if strategy not in _LENSES:
        raise ValueError(f"Unknown Teacher strategy: {strategy}")
    return f"{_LENSES[strategy]}\n\n不要重新生成答案，只评估给定 Student completion。必须保持函数签名和输入输出行为。\n\nTask:\n{task.rstrip()}\nCandidate code:\n"

def prompt_metadata(strategy: str, task: str, mode: str) -> dict[str, str]:
    prompt = teacher_prompt(strategy, task, mode)
    return {"teacher_prompt_mode": mode, "teacher_prompt_version": PROMPT_VERSION, "teacher_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
