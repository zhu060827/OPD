from __future__ import annotations

"""项目配置文件。

所有路径、模型名、API 地址和开关都集中在这里，避免散落在各个 pipeline。
优先读取 .env / 环境变量；如果没有配置，就沿用原 app.py 中的兼容配置。
注意：不要在日志或文档里打印完整 API Key。
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # 依赖尚未安装时仍允许使用系统环境变量和 mock。
    def load_dotenv(*args, **kwargs):
        return False


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 旧代码仍可通过 LEGACY_API_KEY 环境变量兼容，但密钥不能写进压缩包或源码。
LEGACY_API_KEY = os.getenv("LEGACY_API_KEY", "")
LEGACY_BASE_URL = "https://api.gpt.ge/v1"
LEGACY_MODEL = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or LEGACY_API_KEY
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or LEGACY_BASE_URL
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or LEGACY_MODEL

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
MATH_MAX_MASK_ROUNDS = int(os.getenv("MATH_MAX_MASK_ROUNDS", "10"))
MATH_MAX_REFINE_ROUNDS = int(os.getenv("MATH_MAX_REFINE_ROUNDS", "3"))
MATH_PATIENCE = int(os.getenv("MATH_PATIENCE", "2"))
MATH_ACCEPT_THRESHOLD = float(os.getenv("MATH_ACCEPT_THRESHOLD", "0.80"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs"))).resolve()
SAMPLE_DATA_DIR = Path(os.getenv("SAMPLE_DATA_DIR", str(BASE_DIR / "sample_data"))).resolve()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))).resolve()

USE_MOCK_WHEN_LLM_FAILS = os.getenv("USE_MOCK_WHEN_LLM_FAILS", "1").lower() in {"1", "true", "yes", "y"}
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
CODE_TIMEOUT = int(os.getenv("CODE_TIMEOUT", "5"))


def ensure_directories() -> None:
    """启动时自动创建必要目录，避免第一次运行时报 outputs/uploads 不存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def masked_api_key() -> str:
    """调试时只显示脱敏 Key。"""
    if not OPENAI_API_KEY:
        return ""
    if len(OPENAI_API_KEY) <= 8:
        return "****"
    return f"{OPENAI_API_KEY[:4]}****{OPENAI_API_KEY[-4:]}"


ensure_directories()
