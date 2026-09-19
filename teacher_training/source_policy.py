"""四个 Teacher 的唯一训练数据来源策略。"""

from __future__ import annotations


EXPECTED_SOURCE_DATASETS = {
    "formatting": {"commitpackft", "bigcode/commitpackft"},
    "identifier": {"codesearchnet", "code_search_net"},
    "local_structure": {"codesearchnet", "code_search_net"},
    "control_flow": {"code_contests", "deepmind/code_contests"},
}


def canonical_source_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def validate_source_for_domain(domain: str, source_dataset: str) -> None:
    actual = canonical_source_name(source_dataset)
    expected = {canonical_source_name(item) for item in EXPECTED_SOURCE_DATASETS[domain]}
    if actual not in expected:
        raise ValueError(
            f"{domain} Teacher 只允许单一训练来源 {sorted(EXPECTED_SOURCE_DATASETS[domain])}；"
            f"当前样本 source_dataset={source_dataset!r}"
        )
