"""生成可复现的四 Teacher 跨领域矩阵，不构造统一综合分。

每个测试领域列使用该领域已经注册的主指标；因此不同列之间不能比较数值大小。
解析、签名、编译、测试四个门禁分别输出矩阵，不合并为 all-gates 分数。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .domains import DOMAINS
from .io_utils import read_jsonl


PRIMARY_KEYS = {
    "formatting": "primary_exact_formatting_repair",
    "identifier": "primary_identifier_exact_match",
    "local_structure": "primary_codebleu",
    "control_flow": "primary_pass_at_1",
}
GATE_KEYS = ("parse_pass", "signature_pass", "compile_pass", "test_pass")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_matrices(rows: list[dict[str, object]]) -> dict[str, object]:
    primary: dict[tuple[str, str], list[float]] = defaultdict(list)
    gates: dict[str, dict[tuple[str, str], list[float]]] = {
        gate: defaultdict(list) for gate in GATE_KEYS
    }
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        teacher, sample = str(row["teacher_domain"]), str(row["sample_domain"])
        if teacher not in DOMAINS or sample not in DOMAINS:
            raise ValueError("teacher_domain 和 sample_domain 必须属于配置的四个领域")
        primary_key = PRIMARY_KEYS[sample]
        if primary_key not in row:
            raise ValueError(f"{teacher}×{sample} 缺少该测试领域的正式主指标 {primary_key}")
        missing_gates = [gate for gate in GATE_KEYS if gate not in row]
        if missing_gates:
            raise ValueError(f"{teacher}×{sample} 缺少门禁字段：{', '.join(missing_gates)}")
        primary[(teacher, sample)].append(float(row[primary_key]))
        for gate in GATE_KEYS:
            gates[gate][(teacher, sample)].append(float(bool(row[gate])))
        counts[(teacher, sample)] += 1

    def matrix(values: dict[tuple[str, str], list[float]]) -> dict[str, dict[str, float | None]]:
        return {
            teacher: {sample: _mean(values[(teacher, sample)]) for sample in DOMAINS}
            for teacher in DOMAINS
        }

    return {
        "protocol": "teacher-cross-domain-v2",
        "warning": "每列使用不同领域主指标，只允许同列比较 Teacher；禁止跨列比较或加权。",
        "column_primary_metrics": PRIMARY_KEYS,
        "sample_counts": {
            teacher: {sample: counts[(teacher, sample)] for sample in DOMAINS}
            for teacher in DOMAINS
        },
        "primary_metric_matrix": matrix(primary),
        "gate_matrices": {gate: matrix(values) for gate, values in gates.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成按列使用注册主指标的 4×4 跨领域矩阵。")
    parser.add_argument(
        "--input", required=True,
        help="JSONL：teacher_domain、sample_domain、对应领域主指标及四个门禁字段",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_matrices(list(read_jsonl(args.input)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
