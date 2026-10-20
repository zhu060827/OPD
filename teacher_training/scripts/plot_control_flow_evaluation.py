"""绘制 Conditional Rewrite Teacher 的 Base–LoRA 多指标对比图。"""

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
    configure_font, numeric, read_rows, wilson_ci,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Conditional Rewrite Base–LoRA 多指标图。")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--lora-csv", required=True)
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--lora-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font-file")
    parser.add_argument("--strict-arial", action="store_true")
    args = parser.parse_args()
    family = configure_font(args.font_file, args.strict_arial)
    base, lora = read_rows(args.base_csv), read_rows(args.lora_csv)
    if not base or {r["source_id"] for r in base} != {r["source_id"] for r in lora}:
        raise ValueError("Base 与 LoRA 必须是同一组非空测试样本。")
    lora_map = {r["source_id"]: r for r in lora}
    lora = [lora_map[r["source_id"]] for r in base]
    bs = json.loads(Path(args.base_summary).read_text(encoding="utf-8"))["domains"]["control_flow"]
    ls = json.loads(Path(args.lora_summary).read_text(encoding="utf-8"))["domains"]["control_flow"]
    output = Path(args.output_dir)
    for folder in ("primary", "auxiliary", "gates", "data"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    methods = (("Base", base, BASE_COLOR, "//", bs), ("LoRA Teacher", lora, LORA_COLOR, "", ls))
    legend = [Patch(facecolor=BASE_COLOR, hatch="//", label="Base"), Patch(facecolor=LORA_COLOR, label="LoRA Teacher")]
    exported = []

    def style(ax):
        ax.grid(axis="y", color=GRID_COLOR, linewidth=.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, folder, name):
        paths = []
        for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
            path = output / folder / f"{name}{suffix}"
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
            paths.append(str(path.resolve()))
        plt.close(fig)
        return paths

    def label(ax, bar, value):
        ax.annotate(f"{value:.1f}%", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, -7 if value >= 10 else 5), textcoords="offset points",
                    ha="center", va="top" if value >= 10 else "bottom",
                    color="white" if value >= 10 else "#333333", fontweight="bold", fontsize=9)

    def metric(ax, title, field, scale, binary, panel):
        width = .34
        for x, (name, rows, color, hatch, summary) in zip((-width/2, width/2), methods):
            values = numeric(rows, field)
            if values.size == 0 and field == "codebleu":
                value = float(summary["metrics"][field]["mean"]) * scale
                low = high = value / scale
            else:
                value = float(values.mean()) * scale
                low, high = wilson_ci(values) if binary else (float(values.mean()), float(values.mean()))
            yerr = np.array([[max(0, (value - low*scale))], [max(0, (high*scale - value))]])
            bar = ax.bar(x, value, width, color=color, hatch=hatch, edgecolor="white", yerr=yerr, capsize=3, zorder=3)[0]
            label(ax, bar, value)
            exported.append({"panel": panel, "metric": title, "method": name, "mean": value, "n": len(rows)})
        ax.set(xlim=(-.55, .55), ylim=(0, 108), ylabel="Score (%)")
        ax.set_xticks([0], [title]); style(ax)

    # 主指标单独成图：Control Flow 的主指标为可执行测试通过率/Pass@1。
    fig, ax = plt.subplots(figsize=(3.8, 3.25))
    metric(ax, "Pass@1", "primary_pass_at_1", 100, True, "primary")
    fig.legend(handles=legend, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .89))
    primary = save(fig, "primary", "conditional_rewrite_primary_metrics")

    # 其他指标合并为辅助图，不与主指标加权。
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.25))
    auxiliary_specs = [
        ("CodeBLEU", "codebleu", False),
        ("Test pass", "test_pass", True),
        ("Parse pass", "parse_pass", True),
        ("Compile pass", "compile_pass", True),
    ]
    for index, (ax, (title, field, binary)) in enumerate(zip(axes, auxiliary_specs)):
        metric(ax, title, field, 100, binary, f"auxiliary-{index+1}")
        ax.text(.02, .98, f"({chr(97+index)})", transform=ax.transAxes, va="top", fontweight="bold")
    fig.legend(handles=legend, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .89))
    auxiliary = save(fig, "auxiliary", "conditional_rewrite_auxiliary_metrics")

    rows = []
    for name, data, *_ in methods:
        cells = [name]
        for field in ("parse_pass", "compile_pass", "signature_pass"):
            cells.append(f"{numeric(data, field).mean()*100:.1f}%")
        rows.append(cells)
    fig, ax = plt.subplots(figsize=(6.7, 1.65)); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Model", "Parse", "Compile", "Signature"], cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white"); cell.set_facecolor(LORA_COLOR if row == 0 else PALETTE["base_light"] if row == 1 else GATE_COLOR)
        if row == 0: cell.set_text_props(color="white", fontweight="bold")
    gates = save(fig, "gates", "conditional_rewrite_hard_gates")
    (output / "manifest.json").write_text(json.dumps({"artifact_type": "conditional_rewrite_teacher_evaluation", "sample_count": len(base), "font_family": family, "primary": primary, "auxiliary": auxiliary, "gates": gates}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"primary": primary, "auxiliary": auxiliary, "gates": gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
