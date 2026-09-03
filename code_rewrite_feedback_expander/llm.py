from __future__ import annotations

import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import CodeRecord, RewriteCandidate, TokenDistribution
from .attribution import structural_token_weights
from .feedback_schema import parse_feedback, render_feedback


class RewriteLLMClient(ABC):
    @abstractmethod
    def rewrite(self, record: CodeRecord, strategy: str, feedback: str = "") -> RewriteCandidate:
        raise NotImplementedError


class OPDDistributionScorer(ABC):
    @abstractmethod
    def score_student_trajectory(
        self,
        record: CodeRecord,
        candidate: RewriteCandidate,
        strategy: str,
    ) -> List[TokenDistribution]:
        raise NotImplementedError


class IterationFeedbackLLM(ABC):
    @abstractmethod
    def generate_feedback(
        self,
        record: CodeRecord,
        selected_strategy: str,
        iteration_records: List[Dict[str, Any]],
    ) -> str:
        raise NotImplementedError


class MockIterationFeedbackLLM(IterationFeedbackLLM):
    def generate_feedback(self, record, selected_strategy, iteration_records):
        _ = record
        selected = next(
            (item for item in iteration_records if item.get("selected_for_next_round")),
            iteration_records[-1] if iteration_records else {},
        )
        aspect_kl = selected.get("aspect_kl", {}).get(selected_strategy, 0.0)
        overlap = selected.get("aspect_distribution_overlap", {}).get(selected_strategy, 0.0)
        nll_reduction = selected.get("aspect_token_nll_reduction", {}).get(selected_strategy, 0.0)
        gain = selected.get("gain", 0.0)
        semantic = selected.get("semantic_passed", False)
        structured = {
            "preserve": ["函数签名、输入输出行为和通过的测试"],
            "fix": [f"关注 {selected_strategy} 方向；KL={aspect_kl:.3f}，教师 token NLL 降低={nll_reduction:+.3f}"],
            "avoid": ["语义未验证的结构变化", "重复且没有质量指标收益的改动"],
            "evidence": [f"semantic_passed={semantic}", f"distribution_overlap={overlap:.3f}", f"metric_gain={gain:+.3f}"],
        }
        return render_feedback(structured, selected_strategy)


class MockRewriteClient(RewriteLLMClient):
    """Offline deterministic rewriter used for local tests."""

    def rewrite(self, record: CodeRecord, strategy: str, feedback: str = "") -> RewriteCandidate:
        code = record.code.strip()
        reasoning = list(record.reasoning)
        if strategy == "style":
            rewritten = self._style_rewrite(code)
            if reasoning and "Add a short docstring while preserving behavior." not in reasoning:
                reasoning = reasoning + ["Add a short docstring while preserving behavior."]
        elif strategy == "variable":
            rewritten = self._variable_rewrite(code)
            reasoning = [step.replace("nums", "values").replace("arr", "values") for step in reasoning]
        else:
            rewritten = code
        return RewriteCandidate(
            code=rewritten,
            reasoning=reasoning,
            rationale=f"Mock rewrite using {strategy}.",
            strategy=strategy,
            raw_response=rewritten,
            metadata={"provider": "mock"},
        )

    def _style_rewrite(self, code: str) -> str:
        if '"""' in code or "'''" in code:
            return code
        lines = code.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def "):
                indent = line[: len(line) - len(line.lstrip())] + "    "
                lines.insert(idx + 1, f'{indent}"""Expanded rewrite preserving the original behavior."""')
                return "\n".join(lines)
        return code

    def _variable_rewrite(self, code: str) -> str:
        return code.replace("nums", "values").replace("arr", "values")


class MockOPDDistributionScorer(OPDDistributionScorer):
    def score_student_trajectory(
        self,
        record: CodeRecord,
        candidate: RewriteCandidate,
        strategy: str,
    ) -> List[TokenDistribution]:
        _ = record
        profile: List[TokenDistribution] = []
        tokens = re.findall(r"#[^\n]*|\"\"\".*?\"\"\"|[A-Za-z_]\w*|==|!=|<=|>=|[-+*/%=<>]|\S", candidate.code, re.S)
        weights = structural_token_weights(candidate.code, candidate.reasoning)
        for index, token in enumerate(tokens):
            alternative = "value" if token.isidentifier() else "pass"
            aspect = weights[index] if index < len(weights) else {"style": 1.0}
            teacher_bonus = 0.35 if strategy in aspect else 0.12
            profile.append(
                TokenDistribution(
                    token=token,
                    student_logprobs={token: -0.65, alternative: -1.15},
                    teacher_logprobs={token: -0.65 + teacher_bonus, alternative: -1.15 - teacher_bonus},
                    student_token_logprob=-0.65,
                    teacher_token_logprob=-0.65 + teacher_bonus,
                    aspect_weights=aspect,
                    attribution_source="structural_heuristic",
                )
            )
        return profile


class OpenAICompatibleRewriteClient(RewriteLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 90,
    ):
        self.model = model or os.getenv("CODE_EXPANDER_MODEL", "gpt-3.5-turbo")
        self.base_url = (base_url or os.getenv("CODE_EXPANDER_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("CODE_EXPANDER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Missing API key. Set CODE_EXPANDER_API_KEY or OPENAI_API_KEY.")

    def rewrite(self, record: CodeRecord, strategy: str, feedback: str = "") -> RewriteCandidate:
        prompt = build_rewrite_prompt(record, strategy, feedback)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"]
        data = parse_rewrite_response(raw)
        reasoning = data["reasoning"] or record.reasoning
        return RewriteCandidate(
            code=data["code"],
            reasoning=reasoning,
            rationale=data["rationale"],
            strategy=strategy,
            raw_response=raw,
            metadata={"provider": "openai_compatible"},
        )


class OpenAICompatibleIterationFeedbackClient(IterationFeedbackLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 90,
    ):
        self.model = model or os.getenv("CODE_EXPANDER_FEEDBACK_MODEL") or os.getenv(
            "CODE_EXPANDER_MODEL", "gpt-3.5-turbo"
        )
        self.base_url = (
            base_url or os.getenv("CODE_EXPANDER_FEEDBACK_BASE_URL") or os.getenv(
                "CODE_EXPANDER_BASE_URL", "https://api.openai.com/v1"
            )
        ).rstrip("/")
        self.api_key = api_key or os.getenv("CODE_EXPANDER_FEEDBACK_API_KEY") or os.getenv(
            "CODE_EXPANDER_API_KEY"
        ) or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Missing feedback API key.")

    def generate_feedback(self, record, selected_strategy, iteration_records):
        compact_records = [
            {
                "strategy": item.get("strategy"),
                "semantic_passed": item.get("semantic_passed"),
                "quality_score": item.get("quality_score"),
                "gain": item.get("gain"),
                "metric_deltas": item.get("metric_deltas"),
                "token_kl": item.get("token_kl"),
                "aspect_kl": item.get("aspect_kl"),
                "aspect_total_variation": item.get("aspect_total_variation"),
                "aspect_distribution_overlap": item.get("aspect_distribution_overlap"),
                "aspect_token_nll_reduction": item.get("aspect_token_nll_reduction"),
                "strategy_ranking": item.get("strategy_ranking"),
                "selected_for_next_round": item.get("selected_for_next_round"),
            }
            for item in iteration_records
        ]
        prompt = (
            "你是代码改写训练的反馈教师。请把本轮结构化评估记录压缩成给学生模型下一轮使用的中文反馈。\n"
            f"任务：{record.prompt}\n下一轮策略：{selected_strategy}\n"
            f"本轮记录：{json.dumps(compact_records, ensure_ascii=False)}\n"
            "要求：明确保留什么、修复什么、下一轮重点是什么；强调语义等价、函数签名和测试必须保持；"
            "不要虚构记录中没有的问题；只输出一段简洁自然语言，不输出 JSON 或 Markdown。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你负责将代码评估记录转化为可执行的迭代反馈。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"].strip()
        fallback = {
            "preserve": ["函数签名、输入输出行为和通过的测试"],
            "fix": [f"重点改写 {selected_strategy}"],
            "avoid": ["未被评估记录支持的改动"],
            "evidence": ["反馈必须来自本轮结构化评估记录"],
        }
        structured = parse_feedback(raw, fallback)
        return render_feedback(structured, selected_strategy)


def _token_aspect_weights(token: str) -> dict[str, float]:
    stripped = token.strip()
    if not stripped:
        return {"style": 1.0}
    if stripped.startswith("#") or stripped.startswith(('"""', "'''")):
        return {"cot": 0.6, "style": 0.4}
    if stripped in {"if", "elif", "else", "for", "while", "break", "continue", "return", "try", "except"}:
        return {"control_flow": 1.0}
    if stripped.isidentifier() and stripped not in {
        "def", "class", "and", "or", "not", "in", "is", "True", "False", "None",
    }:
        return {"variable": 1.0}
    if stripped in {"+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">="}:
        return {"ast": 1.0}
    return {"style": 0.5, "ast": 0.5}


SYSTEM_PROMPT = (
    "你是一个专业的代码数据扩充助手。你的任务是在保持语义等价的前提下改写代码，"
    "用于构造更高质量、更多样的代码训练数据。必须保留函数签名和输入输出行为，"
    "不得改变题目要求，不得添加网络、文件系统或危险系统调用。输出必须是 JSON："
    "{\"reasoning\": [\"...\"], \"code\": \"...\", \"rationale\": \"...\"}。"
)


def build_rewrite_prompt(record: CodeRecord, strategy: str, feedback: str = "") -> str:
    return (
        "请根据指定策略改写代码，并保持语义等价。\n\n"
        f"任务 ID: {record.task_id}\n"
        f"语言: {record.language}\n"
        f"改写策略: {strategy}\n"
        f"题目/需求:\n{record.prompt}\n\n"
        f"原始思维链 reasoning:\n{json.dumps(record.reasoning, ensure_ascii=False, indent=2)}\n\n"
        f"原始代码:\n```{record.language}\n{record.code}\n```\n\n"
        f"可用单元测试:\n{json.dumps(record.tests, ensure_ascii=False, indent=2)}\n\n"
        f"上一轮反馈:\n{feedback or 'None'}\n\n"
        "策略说明:\n"
        "- cot: 增加清晰注释或中间变量，表达推理过程。\n"
        "- style: 调整代码风格、结构和可读性。\n"
        "- ast: 在 AST 层面做等价重构，例如拆分表达式、提前返回、辅助变量。\n"
        "- variable: 更换变量名，使语义更清楚。\n"
        "- control_flow: 在不改变行为的前提下调整控制流。\n\n"
        "请同步更新 reasoning，使其与改写后的代码保持一致。reasoning 应是步骤列表，"
        "不要包含无法从代码或题目验证的内容。\n\n"
        "输出 JSON，不要输出 Markdown。"
    )


def parse_rewrite_response(raw: str) -> dict:
    text = raw.strip()
    try:
        data = json.loads(text)
        return {
            "code": str(data.get("code", "")).strip(),
            "reasoning": _normalize_reasoning(data.get("reasoning", [])),
            "rationale": str(data.get("rationale", "")).strip(),
        }
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        return parse_rewrite_response(fenced.group(1))
    code_block = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    code = code_block.group(1).strip() if code_block else text
    return {"code": code, "reasoning": [], "rationale": "Parsed from non-JSON model response."}


def _normalize_reasoning(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.replace("\r\n", "\n").split("\n") if line.strip()]
    return []


def create_rewrite_client(
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RewriteLLMClient:
    if provider == "mock":
        return MockRewriteClient()
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleRewriteClient(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")
