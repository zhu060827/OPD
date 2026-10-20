"""Identifier Teacher 的 Base–LoRA 主指标、辅助指标和硬门禁图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from teacher_training.scripts.plot_formatting_evaluation import (
    BASE_COLOR, GATE_COLOR, GRID_COLOR, LORA_COLOR, PALETTE,
    bootstrap_ci, configure_font, numeric, read_rows, wilson_ci,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Identifier Base–LoRA 学术对比图。")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--lora-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font-file")
    parser.add_argument("--strict-arial", action="store_true")
    args = parser.parse_args()
    family = configure_font(args.font_file, args.strict_arial)
    base, lora = read_rows(args.base_csv), read_rows(args.lora_csv)
    if not base or {r["source_id"] for r in base} != {r["source_id"] for r in lora}:
        raise ValueError("Base 与 LoRA 必须是同一组非空测试样本。")
    mapping = {row["source_id"]: row for row in lora}
    lora = [mapping[row["source_id"]] for row in base]
    output = Path(args.output_dir)
    for subdir in ("primary", "auxiliary", "gates", "data", "metadata"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    methods = (("Base", base, BASE_COLOR, "//"), ("LoRA Teacher", lora, LORA_COLOR, ""))
    exported: list[dict[str, object]] = []

    def style(axis) -> None:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    def save(figure, directory: str, name: str) -> list[str]:
        paths = []
        for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
            path = output / directory / f"{name}{suffix}"
            figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
            paths.append(str(path.resolve()))
        plt.close(figure)
        return paths

    def add_percent_label(axis, bar, value: float) -> None:
        """在柱上直接标出数值；零值放在基线上方以保持可见。"""
        if value >= 10:
            offset, vertical, color = -7, "top", "white"
        else:
            offset, vertical, color = 5, "bottom", "#333333"
        axis.annotate(
            f"{value:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset), textcoords="offset points",
            ha="center", va=vertical, color=color, fontweight="bold", fontsize=9,
            zorder=5,
        )

    # 主指标单独成图：DOBF identifier recovery exact match。
    figure, axis = plt.subplots(figsize=(3.8, 3.25))
    width = 0.34
    for x, (name, rows, color, hatch) in zip((-width / 2, width / 2), methods):
        values = numeric(rows, "primary_identifier_exact_match")
        low, high = wilson_ci(values); mean = float(values.mean())
        yerr = np.array([[max(0.0, (mean-low)*100)], [max(0.0, (high-mean)*100)]])
        bar = axis.bar(x, mean * 100, width, color=color, hatch=hatch, edgecolor="white", linewidth=1.2,
                       yerr=yerr, capsize=3, zorder=3)[0]
        add_percent_label(axis, bar, mean * 100)
        exported.append({"panel": "primary", "metric": "Identifier exact match", "method": name,
                         "mean": mean*100, "ci95_low": low*100, "ci95_high": high*100,
                         "ci_method": "Wilson score 95% CI", "n": len(values)})
    axis.set(xlim=(-0.55, .55), ylim=(0, 108), ylabel="Accuracy (%)")
    axis.set_xticks([0], ["Identifier recovery"])
    legend_handles = [
        Patch(facecolor=BASE_COLOR, edgecolor="white", hatch="//", label="Base"),
        Patch(facecolor=LORA_COLOR, edgecolor="white", label="LoRA Teacher"),
    ]
    axis.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.16))
    style(axis); figure.tight_layout()
    primary = save(figure, "primary", "identifier_exact_match")

    # 辅助指标分面板，不与主指标加权。
    auxiliary_specs = [
        ("Identifier subtoken F1", "identifier_subtoken_f1", False),
        ("Full-code exact match", "full_code_exact_match", True),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    for panel, (axis, (label, field, binary)) in enumerate(zip(axes, auxiliary_specs)):
        for x, (name, rows, color, hatch) in zip((-width/2, width/2), methods):
            values = numeric(rows, field)
            low, high = wilson_ci(values) if binary else bootstrap_ci(values)
            mean = float(values.mean())
            yerr = np.array([[max(0.0, (mean-low)*100)], [max(0.0, (high-mean)*100)]])
            bar = axis.bar(x, mean*100, width, color=color, hatch=hatch, edgecolor="white", linewidth=1.2,
                           yerr=yerr, capsize=3, zorder=3)[0]
            add_percent_label(axis, bar, mean * 100)
            exported.append({"panel": f"auxiliary-{panel+1}", "metric": label, "method": name,
                             "mean": mean*100, "ci95_low": low*100, "ci95_high": high*100,
                             "ci_method": "Wilson score 95% CI" if binary else "bootstrap 95% CI (10,000; seed=42)",
                             "n": len(values)})
        axis.set(xlim=(-.55, .55), ylim=(0, 108), ylabel="Score (%)")
        axis.set_xticks([0], [label]); axis.text(.02, .98, f"({chr(97+panel)})", transform=axis.transAxes,
                                               va="top", fontweight="bold")
        style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    # bar 未传 label，用显式代理确保图例与纹理编码完整。
    figure.legend([Patch(facecolor=BASE_COLOR, hatch="//"), Patch(facecolor=LORA_COLOR)],
                  ["Base", "LoRA Teacher"], frameon=False, ncol=2, loc="upper center")
    figure.tight_layout(rect=(0, 0, 1, .89))
    auxiliary = save(figure, "auxiliary", "identifier_auxiliary_metrics")

    gate_specs = [("Parse", "parse_pass"), ("Signature", "signature_pass"), ("Compile", "compile_pass")]
    table_rows, gate_data = [], []
    for name, rows, _, _ in methods:
        cells = [name]
        for label, field in gate_specs:
            values = numeric(rows, field); low, high = wilson_ci(values); mean = float(values.mean())
            cells.append(f"{mean*100:.1f}%")
            gate_data.append({"model": name, "gate": label, "rate_percent": mean*100,
                              "ci95_low": low*100, "ci95_high": high*100, "n": len(values)})
        cells.append("N/A")
        table_rows.append(cells)
    figure, axis = plt.subplots(figsize=(6.7, 1.65)); axis.axis("off")
    table = axis.table(cellText=table_rows, colLabels=["Model", "Parse", "Signature", "Compile", "Tests"],
                       cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(.8)
        cell.set_facecolor(LORA_COLOR if row == 0 else PALETTE["base_light"] if row == 1 else GATE_COLOR)
        if row == 0: cell.set_text_props(color="white", fontweight="bold")
    figure.text(.5, .04, "Tests are N/A because CodeSearchNet provides no executable unit tests.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .1, 1, 1))
    gates = save(figure, "gates", "identifier_hard_gates")

    plot_csv = output / "data" / "plot_data.csv"
    with plot_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "metric", "method", "mean", "ci95_low", "ci95_high", "ci_method", "n"])
        writer.writeheader(); writer.writerows(exported)
    gate_csv = output / "data" / "hard_gates.csv"
    with gate_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gate", "rate_percent", "ci95_low", "ci95_high", "n"])
        writer.writeheader(); writer.writerows(gate_data)
    manifest = {"artifact_type": "identifier_teacher_evaluation", "sample_count": len(base),
                "font_family": family, "primary": primary, "auxiliary": auxiliary, "gates": gates,
                "data": [str(plot_csv.resolve()), str(gate_csv.resolve())],
                "palette": PALETTE, "png_dpi": 600, "design_skill": "figure-designer"}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
