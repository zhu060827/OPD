from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .io_utils import read_jsonl


def _configure_plot_font(plt, font_manager) -> bool:
    paths = []
    for pattern in ("/usr/share/fonts/opentype/noto/*CJK*.ttc", "/usr/share/fonts/truetype/wqy/*.ttc", "/usr/share/fonts/truetype/wqy/*.ttf"):
        paths.extend(Path("/").glob(pattern.lstrip("/")))
    if paths:
        prop = font_manager.FontProperties(fname=str(paths[0]))
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        return True
    plt.rcParams["font.family"] = "DejaVu Sans"
    return False


COMPLEXITY = re.compile(r"(?:时间复杂度|time complexity)\s*(?:为|is|:)?\s*([Oo]\s*\([^\n,;。]+\))", re.I)
BOUNDARY_PATTERNS = {
    "empty": re.compile(r"空输入|空列表|empty input|empty list", re.I),
    "single": re.compile(r"单个|单元素|single element|one element", re.I),
    "duplicate": re.compile(r"重复|duplicate", re.I),
    "negative": re.compile(r"负数|negative", re.I),
    "large": re.compile(r"大输入|极大|large input|maximum", re.I),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", _normalize(text))


def _ngram_repetition(text: str, n: int = 4) -> float:
    tokens = _tokens(text)
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def _token_f1(left: str, right: str) -> float:
    a, b = Counter(_tokens(left)), Counter(_tokens(right))
    overlap = sum((a & b).values())
    if not a or not b:
        return 0.0
    precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _structure_score(text: str) -> float:
    patterns = [r"算法|algorithm|approach", r"步骤|step|first|then", r"复杂度|complexity|\bO\s*\(", r"边界|edge case|boundary"]
    return sum(bool(re.search(pattern, text, re.I)) for pattern in patterns) / len(patterns)


def _complexity(text: str) -> str | None:
    match = COMPLEXITY.search(text)
    return re.sub(r"\s+", "", match.group(1)).upper() if match else None


def _boundary_coverage(text: str, expected: list[str]) -> float:
    if not expected:
        return float("nan")
    return sum(bool(BOUNDARY_PATTERNS.get(name, re.compile(re.escape(name), re.I)).search(text)) for name in expected) / len(expected)


def _bootstrap(values: list[float], seed: int, samples: int = 2000) -> dict[str, float | None]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choice(finite) for _ in finite) for _ in range(samples))
    return {"mean": statistics.fmean(finite), "ci95_low": estimates[int(0.025 * samples)], "ci95_high": estimates[int(0.975 * samples)], "n": len(finite)}


def _load_optional(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        print(f"提示：可选文件不存在，已跳过：{target}")
        return {}
    return {str(row["source_id"]): row for row in read_jsonl(target)}


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="正式评估基座模型与 LoRA Teacher 的批量输出。")
    parser.add_argument("--predictions", required=True, help="compare_base_lora 生成的 JSONL")
    parser.add_argument("--metadata", help="可选：source_id、expected_complexity、expected_boundaries、base_test_pass、lora_test_pass")
    parser.add_argument("--human-scores", help="可选：source_id、base_score、lora_score")
    parser.add_argument("--judge-scores", help="可选：source_id、base_score、lora_score")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise SystemExit("生成图表需要可用的 matplotlib；服务器请安装 matplotlib，数据统计本身不依赖该库") from exc
    has_chinese_font = _configure_plot_font(plt, font_manager)
    predictions = list(read_jsonl(args.predictions))
    if len(predictions) < 2:
        raise ValueError("正式评估至少需要 2 条样本；论文结果建议不少于 100 条")
    metadata = _load_optional(args.metadata)
    human = _load_optional(args.human_scores)
    judge = _load_optional(args.judge_scores)
    rows = []
    for item in predictions:
        source_id = str(item["source_id"])
        meta = metadata.get(source_id, {})
        base, lora = item.get("base_output", ""), item.get("lora_output", "")
        reference = item.get("reference_output", "")
        expected_complexity = str(meta.get("expected_complexity", "")).replace(" ", "").upper() or None
        expected_boundaries = list(meta.get("expected_boundaries") or [])
        base_complexity, lora_complexity = _complexity(base), _complexity(lora)
        row = {
            "source_id": source_id,
            "base_chars": len(base), "lora_chars": len(lora),
            "base_repetition": _ngram_repetition(base), "lora_repetition": _ngram_repetition(lora),
            "base_reference_f1": _token_f1(base, reference), "lora_reference_f1": _token_f1(lora, reference),
            "base_structure_score": _structure_score(base), "lora_structure_score": _structure_score(lora),
            "base_complexity_correct": float(base_complexity == expected_complexity) if expected_complexity else None,
            "lora_complexity_correct": float(lora_complexity == expected_complexity) if expected_complexity else None,
            "base_boundary_coverage": _boundary_coverage(base, expected_boundaries),
            "lora_boundary_coverage": _boundary_coverage(lora, expected_boundaries),
            "base_test_pass": meta.get("base_test_pass"), "lora_test_pass": meta.get("lora_test_pass"),
            "base_consistency": meta.get("base_reasoning_code_consistent"), "lora_consistency": meta.get("lora_reasoning_code_consistent"),
            "base_human_score": human.get(source_id, {}).get("base_score"), "lora_human_score": human.get(source_id, {}).get("lora_score"),
            "base_judge_score": judge.get(source_id, {}).get("base_score"), "lora_judge_score": judge.get(source_id, {}).get("lora_score"),
        }
        rows.append(row)
    metrics = {}
    pairs = [
        ("输出长度", "base_chars", "lora_chars"), ("4-gram 重复率", "base_repetition", "lora_repetition"),
        ("参考推理 Token-F1", "base_reference_f1", "lora_reference_f1"), ("推理结构完整度", "base_structure_score", "lora_structure_score"),
        ("复杂度正确率", "base_complexity_correct", "lora_complexity_correct"), ("边界覆盖率", "base_boundary_coverage", "lora_boundary_coverage"),
        ("测试通过率", "base_test_pass", "lora_test_pass"), ("推理代码一致率", "base_consistency", "lora_consistency"),
        ("人工评分", "base_human_score", "lora_human_score"), ("Judge 评分", "base_judge_score", "lora_judge_score"),
    ]
    for label, base_key, lora_key in pairs:
        base_mean, lora_mean = _mean(rows, base_key), _mean(rows, lora_key)
        deltas = [float(row[lora_key]) - float(row[base_key]) for row in rows if row.get(base_key) is not None and row.get(lora_key) is not None and math.isfinite(float(row[base_key])) and math.isfinite(float(row[lora_key]))]
        metrics[label] = {"base_mean": base_mean, "lora_mean": lora_mean, "improvement": _bootstrap(deltas, args.seed)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    missing = [label for label, value in metrics.items() if value["base_mean"] is None or value["lora_mean"] is None]
    summary = {"samples": len(rows), "metrics": metrics, "missing_inputs": missing, "references": {"bootstrap_ci": "Efron (1979), Bootstrap Methods", "repetition": "4-gram 重复率，项目定义的生成多样性代理指标", "complexity": "McCabe (1976) 与人工/静态分析复杂度标签", "judge": "Zheng et al. (2023), Judging LLM-as-a-Judge"}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    available = [(label, value) for label, value in metrics.items() if value["base_mean"] is not None and value["lora_mean"] is not None]
    if available:
        figure, axis = plt.subplots(figsize=(max(8, len(available) * 1.5), 5), constrained_layout=True)
        x = list(range(len(available))); width = 0.38
        axis.bar([i - width / 2 for i in x], [v["base_mean"] for _, v in available], width, label="Base")
        axis.bar([i + width / 2 for i in x], [v["lora_mean"] for _, v in available], width, label="LoRA")
        display_labels = [label for label, _ in available] if has_chinese_font else [f"Metric {i + 1}" for i in x]
        axis.set_xticks(x, display_labels, rotation=30, ha="right")
        axis.set_title("基座模型与 LoRA Teacher 正式评估指标" if has_chinese_font else "Base model vs LoRA Teacher metrics")
        axis.legend(); axis.grid(axis="y", alpha=0.25)
        figure.savefig(output / "metric_comparison.png", dpi=180); plt.close(figure)
    lines = ["# Teacher 正式输出评估", "", f"样本数：{len(rows)}", ""]
    if missing:
        lines += ["## 缺失输入", "", "以下指标未生成，因为没有提供对应的 metadata、人工评分或 Judge 评分文件：", "", *[f"- {label}" for label in missing], ""]
    lines += ["| 指标 | Base | LoRA | 平均改进 | 95% CI |", "|---|---:|---:|---:|---:|"]
    for label, value in metrics.items():
        imp = value["improvement"]
        fmt = lambda x: "N/A" if x is None else f"{x:.4f}"
        lines.append(f"| {label} | {fmt(value['base_mean'])} | {fmt(value['lora_mean'])} | {fmt(imp['mean'])} | [{fmt(imp['ci95_low'])}, {fmt(imp['ci95_high'])}] |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"完成": True, "样本数": len(rows), "输出目录": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
