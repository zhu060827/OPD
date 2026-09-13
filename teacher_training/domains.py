from __future__ import annotations

from dataclasses import dataclass


DOMAINS = ("cot", "style", "ast", "variable", "control_flow")


@dataclass(frozen=True)
class DomainSpec:
    name: str
    title: str
    system_prompt: str
    dataset_source: str
    literature_basis: tuple[str, ...]


DOMAIN_SPECS = {
    "cot": DomainSpec(
        "cot",
        "代码推理专家",
        "你是代码推理专家。只改进算法解释、中间推理、边界情况和复杂度说明；保持函数签名和可观察行为不变。",
        "CodeContests（优先）或转换为题目/代码/推理对的 MBPP",
        (
            "Li et al. (2022), Competition-Level Code Generation with AlphaCode",
            "Wei et al. (2022), Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        ),
    ),
    "style": DomainSpec(
        "style",
        "代码风格专家",
        "你是代码风格专家。改进可读性、格式、文档和函数组织；不得改变算法、函数签名或可观察行为。",
        "筛选非功能性风格/组织改动后的 CodeXGLUE Code Refinement",
        ("Lu et al. (2021), CodeXGLUE", "Tufano et al. (2019), Learning to Represent Programs with Program Graphs"),
    ),
    "ast": DomainSpec(
        "ast",
        "AST 结构重构专家",
        "你是 AST 结构重构专家。执行有意义的语法树级重构并保持语义；纯格式或纯标识符变化不算有效改写。",
        "RefactoringMiner 检测的结构重构，或 Refactory 风格的前后代码对",
        ("Tsantalis et al. (2018), RefactoringMiner", "Allamanis et al. (2018), Learning to Represent Programs with Graphs"),
    ),
    "variable": DomainSpec(
        "variable",
        "变量与数据流专家",
        "你是变量与数据流专家。改善标识符语义和定义-使用关系；保持作用域、公开函数签名、控制流和可观察行为不变。",
        "CodeXGLUE Variable Misuse 加上真实 Rename Variable 提交转换的改写对",
        ("Lu et al. (2021), CodeXGLUE", "Allamanis et al. (2015), Suggesting Accurate Method and Class Names"),
    ),
    "control_flow": DomainSpec(
        "control_flow",
        "控制流专家",
        "你是控制流专家。改进条件、循环、提前返回、嵌套和执行路径；保持函数签名及可观察行为不变，避免纯命名或纯格式改写。",
        "用 RefactoringMiner 挖掘的控制流重构；ManySStuBs4J 仅作补充",
        ("Tsantalis et al. (2018), RefactoringMiner", "Karampatsis and Sutton (2020), How Often Do Single-Statement Bugs Occur?"),
    ),
}


def require_domain(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in DOMAIN_SPECS:
        raise ValueError(f"未知领域 {value!r}；应为以下之一：{', '.join(DOMAINS)}")
    return normalized
