from __future__ import annotations

from dataclasses import dataclass


DOMAINS = ("cot", "style", "ast", "variable", "control_flow")
OUTPUT_SCHEMA_VERSION = "teacher-plan-code-v1"
SYSTEM_PROMPT_VERSION = "teacher-specialization-v2"

# 内部 ID 保持稳定，以兼容现有 JSONL、配置和后续 MOPD 路由。
PUBLIC_DOMAIN_NAMES = {
    "cot": "Planning/CoT",
    "style": "Style/Documentation",
    "variable": "Identifier/Rename",
    "ast": "Extract/Inline",
    "control_flow": "Control-flow",
}

# 非 CoT 专家的 plan 是结构化改写意图，仅用于统一数据协议。
PLAN_ROLES = {
    "cot": "详细规划与代码实现步骤",
    "style": "简短的风格/文档改写意图（接口统一字段）",
    "variable": "简短的标识符重命名意图（接口统一字段）",
    "ast": "简短的 Extract/Inline 重构意图（接口统一字段）",
    "control_flow": "简短的控制流改写意图（接口统一字段）",
}

PLAN_TYPES = {
    "cot": "reasoning_plan",
    "style": "interface_intent",
    "variable": "interface_intent",
    "ast": "interface_intent",
    "control_flow": "interface_intent",
}


@dataclass(frozen=True)
class DomainSpec:
    name: str
    title: str
    system_prompt: str
    dataset_source: str
    literature_basis: tuple[str, ...]
    primary_metric: str
    primary_metric_reference: str
    fallback_plan: str


DOMAIN_SPECS = {
    "cot": DomainSpec(
        "cot",
        "规划引导的代码改写专家",
        "你是规划引导的代码改写专家。先给出保持行为不变的改写计划，再依据计划输出改写代码；保持函数签名和可观察行为不变。",
        "CodeContests（优先）或转换为题目/代码/推理对的 MBPP",
        (
            "Li et al. (2022), Competition-Level Code Generation with AlphaCode",
            "Wei et al. (2022), Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        ),
        "pass@1",
        "Chen et al. (2021), Evaluating Large Language Models Trained on Code",
        "分析任务约束和边界情况，制定保持函数签名与可观察行为不变的改写步骤，并据此生成通过原测试的代码。",
    ),
    "style": DomainSpec(
        "style",
        "风格与文档专家",
        "你是风格与文档专家。只改进格式、注释、docstring、可读性和局部组织；不得改变算法、数据流、主要 AST 结构、控制流、函数签名或可观察行为。",
        "筛选非功能性风格/组织改动后的 CodeXGLUE Code Refinement",
        ("Lu et al. (2021), CodeXGLUE", "Tufano et al. (2019), Learning to Represent Programs with Program Graphs"),
        "code readability score change",
        "Buse and Weimer (2010), Learning a Metric for Code Readability",
        "改善可读性、文档或代码组织；不改变算法、函数签名和可观察行为。",
    ),
    "ast": DomainSpec(
        "ast",
        "局部结构重构专家",
        "你是局部结构重构专家。只执行 Extract/Inline、表达式拆分或局部语句组织等结构重构；纯格式、纯命名和主要控制流变化不属于本领域。",
        "RefactoringMiner 检测的结构重构，或 Refactory 风格的前后代码对",
        ("Tsantalis et al. (2018), RefactoringMiner", "Allamanis et al. (2018), Learning to Represent Programs with Graphs"),
        "refactoring-type exact-match accuracy",
        "Tsantalis et al. (2018), RefactoringMiner",
        "执行 Extract/Inline、表达式拆分或局部语句组织，并保持程序行为不变。",
    ),
    "variable": DomainSpec(
        "variable",
        "标识符重命名专家",
        "你是标识符重命名专家。只改善变量或局部标识符名称，并同步更新定义和使用；保持作用域、公开 API、表达式结构、控制流和可观察行为不变。",
        "CodeXGLUE Variable Misuse 加上真实 Rename Variable 提交转换的改写对",
        ("Lu et al. (2021), CodeXGLUE", "Allamanis et al. (2015), Suggesting Accurate Method and Class Names"),
        "rename exact-match accuracy",
        "Allamanis et al. (2018), Learning to Represent Programs with Graphs；Lu et al. (2021), CodeXGLUE Variable Misuse",
        "依据变量的定义、使用、作用域和语义角色改进标识符，并同步更新全部引用。",
    ),
    "control_flow": DomainSpec(
        "control_flow",
        "控制流专家",
        "你是控制流专家。改进条件、循环、提前返回、嵌套和执行路径；保持函数签名及可观察行为不变，避免纯命名或纯格式改写。",
        "用 RefactoringMiner 挖掘的控制流重构；ManySStuBs4J 仅作补充",
        ("Tsantalis et al. (2018), RefactoringMiner", "Karampatsis and Sutton (2020), How Often Do Single-Statement Bugs Occur?"),
        "control-flow refactoring-type exact-match accuracy",
        "Tsantalis et al. (2018), RefactoringMiner",
        "重构条件、循环、提前返回或嵌套路径，并保持函数签名和可观察行为不变。",
    ),
}


def require_domain(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in DOMAIN_SPECS:
        raise ValueError(f"未知领域 {value!r}；应为以下之一：{', '.join(DOMAINS)}")
    return normalized


def public_domain_name(domain: str) -> str:
    return PUBLIC_DOMAIN_NAMES[require_domain(domain)]
