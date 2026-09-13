from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .domains import DOMAINS
from .io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 JSONL 预测结果生成 5×5 Teacher/领域评分矩阵。")
    parser.add_argument("--input", required=True, help="每行需要包含 teacher_domain、sample_domain 和 score 字段")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    pass_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_jsonl(args.input):
        teacher = row["teacher_domain"]
        sample = row["sample_domain"]
        if teacher not in DOMAINS or sample not in DOMAINS:
            raise ValueError("teacher_domain 和 sample_domain 必须属于配置的五个领域")
        values[(teacher, sample)].append(float(row["score"]))
        pass_values[(teacher, sample)].append(float(bool(row.get("semantic_pass", False))))
    matrix = {
        teacher: {
            sample: {
                "count": len(values[(teacher, sample)]),
                "mean_score": sum(values[(teacher, sample)]) / len(values[(teacher, sample)]) if values[(teacher, sample)] else None,
                "semantic_pass_rate": sum(pass_values[(teacher, sample)]) / len(pass_values[(teacher, sample)]) if pass_values[(teacher, sample)] else None,
            }
            for sample in DOMAINS
        }
        for teacher in DOMAINS
    }
    result = {"matrix": matrix, "diagonal_expected_to_be_highest": True}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
