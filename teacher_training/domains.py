from __future__ import annotations

from dataclasses import dataclass


DOMAINS = ("formatting", "identifier", "local_structure", "control_flow")
OUTPUT_SCHEMA_VERSION = "teacher-code-only-v1"
SYSTEM_PROMPT_VERSION = "teacher-specialization-v4"

PUBLIC_DOMAIN_NAMES = {
    "formatting": "Style/Formatting",
    "identifier": "Identifier Deobfuscation/Rename",
    "local_structure": "Local Structural Transformation",
    "control_flow": "Control-flow Transformation",
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
        "局部结构变换专家",
        "你是局部结构变换专家。只执行语义保持的局部表达式拆分或合并、临时变量引入或消除、局部语句重组和等价语法自然化；不得进行纯格式、纯重命名、主要控制流或算法变换。只输出完整改写代码。",
        "按 NatGen 范式对自然代码应用可逆、语义保持的局部结构变换后形成的代码对",
        ("Chakraborty et al. (2022), NatGen: Generative Pre-training by Naturalizing Source Code", "Ren et al. (2020), CodeBLEU"),
        "CodeBLEU",
        "Ren et al. (2020), CodeBLEU: a Method for Automatic Evaluation of Code Synthesis",
    ),
    "control_flow": DomainSpec(
        "control_flow",
        "控制流变换专家",
        "你是控制流变换专家。只执行语义保持的条件、循环、提前返回、guard clause、break/continue 和路径结构变换；不得进行纯格式、纯重命名或算法替换，不得改变函数签名和可观察行为。只输出完整改写代码。",
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
