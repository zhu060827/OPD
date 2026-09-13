from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score


METRIC_RULES = {
    "style": ("style_gain", ">="),
    "variable": ("naming_gain", ">="),
    "ast_min": ("ast_similarity", ">="),
    "ast_max": ("ast_similarity", "<="),
    "control_flow": ("control_node_delta", ">="),
}


def _select(rows: list[dict[str, Any]], metric: str, operator: str, min_precision: float) -> dict[str, float]:
    values = sorted({float(row["metrics"][metric]) for row in rows if metric in row.get("metrics", {})})
    best: dict[str, float] | None = None
    positives = sum(bool(row.get("accepted_by_human")) for row in rows)
    for threshold in values:
        predicted = [row for row in rows if metric in row.get("metrics", {}) and ((float(row["metrics"][metric]) >= threshold) if operator == ">=" else (float(row["metrics"][metric]) <= threshold))]
        true_positive = sum(bool(row.get("accepted_by_human")) for row in predicted)
        precision = true_positive / len(predicted) if predicted else 1.0
        recall = true_positive / positives if positives else 0.0
        candidate = {"value": threshold, "precision": precision, "recall": recall, "predicted_positive": len(predicted)}
        if precision >= min_precision and (best is None or (recall, precision) > (best["recall"], best["precision"])):
            best = candidate
    if best is None:
        raise ValueError(f"指标 {metric} 找不到满足 precision>={min_precision:.2f} 的阈值")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="用人工标注验证集校准领域过滤阈值。")
    parser.add_argument("--input", required=True, help="JSONL：每行包含 domain、accepted_by_human 和 metrics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--plot", help="输出 ROC/PR 曲线 PNG 路径")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[str, Any] = {"min_precision": args.min_precision, "domains": {}, "说明": "阈值由人工标注验证集校准，并非论文统一规定值。", "auc": {}}
    figure = plt.figure(figsize=(12, 10)) if args.plot else None
    axes = figure.subplots(2, 1) if figure else []
    for rule_name, (metric, operator) in METRIC_RULES.items():
        domain = "ast" if rule_name.startswith("ast_") else rule_name
        subset = [row for row in rows if row.get("domain") == domain]
        if not subset:
            continue
        selected = _select(subset, metric, operator, args.min_precision)
        domain_result = result["domains"].setdefault(domain, {"samples": len(subset), "thresholds": {}, "calibration": {}})
        config_key = {"style": "style_gain", "variable": "naming_gain", "ast_min": "ast_similarity_min", "ast_max": "ast_similarity_max", "control_flow": "control_node_delta_min"}[rule_name]
        domain_result["thresholds"][config_key] = selected["value"]
        domain_result["calibration"][config_key] = selected
        labels = [int(bool(row.get("accepted_by_human"))) for row in subset]
        scores = [float(row.get("score", row.get("metrics", {}).get(metric, 0.0))) for row in subset]
        if len(set(labels)) == 2:
            roc_x, roc_y, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            roc_value = float(roc_auc_score(labels, scores))
            pr_value = float(auc(recall, precision))
            result["auc"][domain] = {"roc_auc": roc_value, "pr_auc": pr_value, "samples": len(subset)}
            if figure:
                axes[0].plot(roc_x, roc_y, label=f"{domain} AUC={roc_value:.3f}")
                axes[1].plot(recall, precision, label=f"{domain} AP={pr_value:.3f}")
        else:
            result["auc"][domain] = {"roc_auc": None, "pr_auc": None, "samples": len(subset), "warning": "人工标签必须同时包含正例和负例"}
    if figure:
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].set(title="五领域人工标注 ROC 曲线", xlabel="假阳性率", ylabel="真阳性率")
        axes[1].set(title="五领域人工标注 Precision-Recall 曲线", xlabel="召回率", ylabel="精确率")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.25)
        figure.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=160)
        plt.close(figure)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
