from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .domains import DOMAINS, require_domain


REGISTRY_PATH = Path(__file__).with_name("dataset_sources.json")


@lru_cache(maxsize=1)
def load_dataset_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def required_fields(domain: str) -> tuple[str, ...]:
    return tuple(load_dataset_registry()[require_domain(domain)]["正式必需字段"])


def missing_required_fields(row: dict[str, Any], domain: str) -> list[str]:
    return [key for key in required_fields(domain) if row.get(key) in (None, "", [])]


def validate_source_registry() -> dict[str, Any]:
    registry = load_dataset_registry()
    errors: list[str] = []
    for domain in DOMAINS:
        spec = registry.get(domain, {})
        for key in ("论文名称", "推荐主数据", "下载地址", "论文", "匹配等级", "转换", "正式必需字段", "正式适配状态"):
            if not spec.get(key):
                errors.append(f"{domain} 缺少 {key}")
        if "semantic_pass" not in spec.get("正式必需字段", []):
            errors.append(f"{domain} 未要求 semantic_pass")
    if errors:
        raise ValueError("；".join(errors))
    return {"valid": True, "registry_version": registry["版本"], "domains": list(DOMAINS)}
