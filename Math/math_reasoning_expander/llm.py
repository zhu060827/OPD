from __future__ import annotations

import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import FillCandidate, MaskedTask


class LLMClient(ABC):
    @abstractmethod
    def generate_fill(self, task: MaskedTask, feedback: str = "") -> FillCandidate:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic fallback used for tests and offline demos."""

    def generate_fill(self, task: MaskedTask, feedback: str = "") -> FillCandidate:
        target = task.target_nodes
        if not target:
            text = "No missing reasoning step was found."
        else:
            generated = []
            for node in target:
                content = node.content
                if feedback:
                    content = self._make_more_explicit(content)
                generated.append(content)
            text = "\n".join(generated)
        steps = [line.strip() for line in text.split("\n") if line.strip()]
        return FillCandidate(text=text, steps=steps, raw_response=text, metadata={"provider": "mock"})

    def _make_more_explicit(self, text: str) -> str:
        if any(word in text.lower() for word in ["therefore", "so", "thus"]):
            return text
        return f"Therefore, {text[0].lower() + text[1:] if text else text}"


class OpenAICompatibleClient(LLMClient):
    """Minimal OpenAI-compatible chat client using only the Python standard library."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 90,
    ):
        self.model = model or os.getenv("MATH_EXPANDER_MODEL", "gpt-3.5-turbo")
        self.base_url = (base_url or os.getenv("MATH_EXPANDER_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("MATH_EXPANDER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Missing API key. Set MATH_EXPANDER_API_KEY or OPENAI_API_KEY.")

    def generate_fill(self, task: MaskedTask, feedback: str = "") -> FillCandidate:
        prompt = build_fill_prompt(task, feedback)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"]
        steps = parse_steps_from_response(raw)
        return FillCandidate(text="\n".join(steps), steps=steps, raw_response=raw, metadata={"provider": "openai_compatible"})


SYSTEM_PROMPT = (
    "你是一个专业的数学推理数据扩充助手，擅长根据未遮盖的前后推理步骤，"
    "补全被遮盖的中间推理节点。要求保持原题和最终答案不变，不引入无依据的新条件，"
    "只生成必要、清晰、可验证的中间推理步骤。输出必须是 JSON，格式为："
    "{\"steps\": [\"步骤1\", \"步骤2\"]}。"
)


def build_fill_prompt(task: MaskedTask, feedback: str = "") -> str:
    payload = task.to_prompt_payload()
    return (
        "Fill the masked mathematical reasoning node(s).\n"
        "Requirements:\n"
        "1. Use the visible prefix and suffix as constraints.\n"
        "2. Preserve notation and final answer.\n"
        "3. Add only useful intermediate reasoning.\n"
        "4. Avoid duplicating the suffix.\n"
        "5. Output JSON: {\"steps\": [\"...\"]}.\n\n"
        f"Task payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Previous evaluator feedback:\n{feedback or 'None'}"
    )


def parse_steps_from_response(raw: str) -> List[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        steps = data.get("steps", [])
        if isinstance(steps, list):
            return [str(step).strip() for step in steps if str(step).strip()]
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
    if fenced:
        return parse_steps_from_response(fenced.group(1))

    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines or [raw]


def create_llm_client(
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleClient(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")
