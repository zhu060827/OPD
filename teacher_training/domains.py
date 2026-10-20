from __future__ import annotations

from dataclasses import dataclass


DOMAINS = ("formatting", "identifier", "local_structure", "control_flow")
OUTPUT_SCHEMA_VERSION = "teacher-code-only-v1"
SYSTEM_PROMPT_VERSION = "teacher-specialization-v5"
SYSTEM_PROMPT_VERSIONS = {
    "formatting": SYSTEM_PROMPT_VERSION,
    "identifier": SYSTEM_PROMPT_VERSION,
    "local_structure": SYSTEM_PROMPT_VERSION,
    "control_flow": "teacher-control-flow-v1",
}

PUBLIC_DOMAIN_NAMES = {
    "formatting": "Style/Formatting",
    "identifier": "Identifier Deobfuscation/Rename",
    "local_structure": "Extract Variable Transformation",
    "control_flow": "Conditional Rewrite Transformation",
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


DOMAIN_SPECS = {
    "formatting": DomainSpec(
        "formatting",
        "风格与格式规范化专家",
        "你是代码风格与格式规范化专家。只修复缩进、空白、换行、括号布局和语言格式规范问题；不得改变标识符、表达式含义、控制流、函数签名、公开 API 或可观察行为。只输出完整改写代码。",
        "真实 formatting/checkstyle 修复对，或按 STYLER 范式构造的格式违规代码到规范代码对",
        ("Palma et al., STYLER: Learning Formatting Conventions to Repair Checkstyle Warnings", "Allamanis et al. (2014), Learning Natural Coding Conventions"),
        "exact formatting repair accuracy",
        "Palma et al., STYLER: Learning Formatting Conventions to Repair Checkstyle Warnings",
    ),
    "identifier": DomainSpec(
        "identifier",
        "标识符去混淆与重命名专家",
        "你是标识符去混淆与重命名专家。只恢复或改善局部变量、参数和局部标识符名称，并同步更新全部定义和使用；不得改变作用域、公开 API、表达式结构、控制流、函数签名或可观察行为。只输出完整改写代码。",
        "按 DOBF 目标从自然代码构造的标识符混淆代码到原始代码对；真实 Rename 提交仅作补充",
        ("Lachaux et al. (2021), DOBF: A Deobfuscation Pre-Training Objective for Programming Languages", "Allamanis et al. (2015), Suggesting Accurate Method and Class Names"),
        "identifier recovery exact-match accuracy",
        "Lachaux et al. (2021), DOBF",
    ),
    "local_structure": DomainSpec(
        "local_structure",
        "Extract Variable 局部重构专家",
        "你是 Extract Variable 局部重构专家。按照用户指定的唯一目标位置，将 return 表达式提取为一个局部变量并返回该变量。每次只执行一次指定变换；不得修改其他语句、标识符、格式、控制流、函数签名或可观察行为。只输出完整改写代码，并使用 <code>...</code> 包裹。",
        "按 NatGen/Fowler 范式对指定的单个 return 表达式应用语义保持的 Extract Variable 变换后形成的代码对",
        ("Chakraborty et al. (2022), NatGen: Generative Pre-training by Naturalizing Source Code", "Ren et al. (2020), CodeBLEU"),
        "CodeBLEU",
        "Ren et al. (2020), CodeBLEU: a Method for Automatic Evaluation of Code Synthesis",
    ),
    "control_flow": DomainSpec(
        "control_flow",
        "Conditional Rewrite 条件分支重写专家",
        "你是 Conditional Rewrite 条件分支重写专家。只执行一次用户指定的条件分支重写：将目标 if-else 条件取反，并交换其两个分支。不得修改其他语句、标识符、格式、循环、函数签名或可观察行为。不得执行 guard clause、early return、loop rewrite 或 break/continue 变换。只输出完整改写代码，并使用 <code>...</code> 包裹。",
        "按 NatGen/ContraCode 范式构造并经测试验证的语义保持控制流变换对",
        ("Chakraborty et al. (2022), NatGen: Generative Pre-training by Naturalizing Source Code", "Jain et al. (2021), ContraCode: Learning Contrastive Representations for Code"),
        "Pass@1",
        "Chen et al. (2021), Evaluating Large Language Models Trained on Code",
    ),
}


def require_domain(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in DOMAIN_SPECS:
        raise ValueError(f"未知领域 {value!r}；应为以下之一：{', '.join(DOMAINS)}")
    return normalized


def public_domain_name(domain: str) -> str:
    return PUBLIC_DOMAIN_NAMES[require_domain(domain)]
