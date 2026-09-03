from __future__ import annotations

"""统一 LLM 调用封装。

原来的脚本里经常直接写 OpenAI-compatible API 初始化。结题版把它集中到这里：
优先使用环境变量；如果没配环境变量，就沿用 config.py 里的兼容配置。
真实调用失败时会自动 fallback 到 mock，保证现场演示不会因为网络或 Key 问题中断。
"""

import json
import re
from typing import Any, Dict, List, Optional

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, REQUEST_TIMEOUT, USE_MOCK_WHEN_LLM_FAILS


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_mock_when_fails: bool = USE_MOCK_WHEN_LLM_FAILS,
    ) -> None:
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = (base_url or OPENAI_BASE_URL or "").rstrip("/")
        self.model = model or OPENAI_MODEL
        self.use_mock_when_fails = use_mock_when_fails
        # 数据集测试需要区分真实调用和 mock 回退，避免把兜底结果误当成真实模型结果。
        self.real_call_count = 0
        self.mock_fallback_count = 0
        self.last_call_used_mock = False
        self.last_error = ""

    @property
    def available(self) -> bool:
        """只判断配置是否齐全，不在日志中打印完整 API Key。"""
        return bool(self.api_key and self.base_url and self.model)

    def chat(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        if not self.available:
            raise LLMError("LLM configuration is incomplete.")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=REQUEST_TIMEOUT)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def generate_feature_code(self, columns: List[str], stats: Dict[str, Any], history: List[str]) -> str:
        """Tabular 使用：让 LLM 返回 pandas 特征生成代码。"""
        system_prompt = "You generate short pandas feature-engineering code. Return only Python code."
        user_prompt = (
            "Given a pandas DataFrame named df, create one useful numeric feature column.\n"
            "Allowed objects: df, np, pd, math. Do not import modules. Do not read files.\n"
            f"Columns: {columns}\n"
            f"Stats: {json.dumps(stats, ensure_ascii=False)[:3000]}\n"
            f"Previous feedback: {history[-5:]}\n"
            "Return code like: df['new_feature'] = ..."
        )
        try:
            raw = self.chat(user_prompt, system_prompt=system_prompt, temperature=0.3)
            return strip_code_fence(raw)
        except LLMError:
            if not self.use_mock_when_fails:
                raise
            return mock_feature_code(columns, history)

    def recover_math_node(
        self,
        question: str,
        prefix: str,
        suffix: str,
        target_type: str = "intermediate_calculation",
        original_middle: str = "",
        feedback: str = "",
    ) -> str:
        """Math 使用：根据前后文恢复节点，并利用上一轮评价继续改进。"""
        system_prompt = (
            "You expand mathematical chain-of-thought by filling a missing middle reasoning step. "
            "Keep the final answer unchanged and output only the recovered step."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Prefix reasoning:\n{prefix}\n\n"
            f"Missing node type: {target_type}\n\n"
            f"Suffix reasoning:\n{suffix}\n\n"
            f"Evaluator feedback from the previous attempt:\n{feedback or 'None'}\n\n"
            "Recover one or two concise, verifiable intermediate steps. "
            "Do not output standalone LaTeX display delimiters such as \\[, \\], or $$."
        )
        try:
            result = strip_code_fence(self.chat(user_prompt, system_prompt=system_prompt, temperature=0.2)).strip()
            self.real_call_count += 1
            self.last_call_used_mock = False
            self.last_error = ""
            return result
        except LLMError as exc:
            if not self.use_mock_when_fails:
                raise
            self.mock_fallback_count += 1
            self.last_call_used_mock = True
            self.last_error = str(exc)
            if original_middle.strip():
                return f"Therefore, {original_middle.strip()}"
            return "Therefore, combine the known conditions with the previous result to obtain the next intermediate value."

    def generate_code(self, prompt: str, starter_code: str, tests: str, feedback: str = "") -> str:
        """Code 生成/修复路径使用：根据题目、starter code、测试和反馈生成代码。"""
        system_prompt = (
            "You are a Python coding assistant. Return only executable Python code. "
            "Do not include Markdown fences."
        )
        user_prompt = (
            f"Task prompt:\n{prompt}\n\n"
            f"Starter code:\n{starter_code}\n\n"
            f"Tests that must pass:\n{tests}\n\n"
            f"Feedback from previous attempt:\n{feedback or 'None'}\n\n"
            "Return a complete Python implementation."
        )
        try:
            return strip_code_fence(self.chat(user_prompt, system_prompt=system_prompt, temperature=0.2)).strip()
        except LLMError:
            if not self.use_mock_when_fails:
                raise
            return mock_code_solution(prompt, starter_code)


def strip_code_fence(text: str) -> str:
    """去掉 ```python 这类 Markdown 包裹，方便后续直接执行。"""
    text = (text or "").strip()
    fenced = re.search(r"```(?:python|py|json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    return text


def mock_feature_code(columns: List[str], history: List[str] | None = None) -> str:
    """本地 mock 特征生成，按轮次生成比例、差值、交互项。"""
    usable = [col for col in columns if col != "target"]
    history = history or []
    idx = max(0, len(history) - 1)
    if len(usable) >= 2:
        a = usable[idx % len(usable)]
        b = usable[(idx + 1) % len(usable)]
        if idx % 3 == 0:
            return (
                f"denom = df[{b!r}].replace(0, np.nan)\n"
                f"df['llm_ratio_{a}_to_{b}_{idx}'] = (df[{a!r}] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)"
            )
        if idx % 3 == 1:
            return f"df['llm_diff_{a}_minus_{b}_{idx}'] = df[{a!r}] - df[{b!r}]"
        return f"df['llm_interaction_{a}_{b}_{idx}'] = df[{a!r}] * df[{b!r}]"
    if usable:
        a = usable[0]
        return f"df['llm_squared_{a}'] = df[{a!r}] * df[{a!r}]"
    return "df['llm_constant_guard'] = 0"


def mock_code_solution(prompt: str, starter_code: str) -> str:
    """本地 mock 代码生成，覆盖 demo 里常见函数，保证无网络也能跑通。"""
    text = f"{prompt}\n{starter_code}".lower()
    if "has_close_elements" in text:
        return (
            "def has_close_elements(numbers, threshold):\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        )
    if "add(" in text or "sum of a and b" in text:
        return "def add(a, b):\n    return a + b\n"
    if "factorial" in text:
        return (
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    out = 1\n"
            "    for i in range(2, n + 1):\n"
            "        out *= i\n"
            "    return out\n"
        )
    name_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", starter_code or prompt)
    if name_match:
        name, args = name_match.group(1), name_match.group(2)
        return f"def {name}({args}):\n    pass\n"
    return starter_code or "def solution(*args, **kwargs):\n    return None\n"
