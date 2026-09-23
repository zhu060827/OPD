"""Local Structure Teacher 的 Base–LoRA 可复现评估图。"""

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
    parser = argparse.ArgumentParser(description="绘制 Local Structure Base–LoRA 对比图。")
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
    if not base or {row["source_id"] for row in base} != {row["source_id"] for row in lora}:
        raise ValueError("Base 与 LoRA 必须是同一组非空测试样本。")
    lora_by_id = {row["source_id"]: row for row in lora}
    lora = [lora_by_id[row["source_id"]] for row in base]
    base_summary = json.loads(Path(args.base_summary).read_text(encoding="utf-8"))
    lora_summary = json.loads(Path(args.lora_summary).read_text(encoding="utf-8"))
    corpus = {
        "Base": base_summary["domains"]["local_structure"]["corpus_metrics"],
        "LoRA Teacher": lora_summary["domains"]["local_structure"]["corpus_metrics"],
    }
    output = Path(args.output_dir)
    for folder in ("primary", "auxiliary", "gates", "data"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    methods = (("Base", base, BASE_COLOR, "//"), ("LoRA Teacher", lora, LORA_COLOR, ""))
    legend = [Patch(facecolor=BASE_COLOR, hatch="//", label="Base"), Patch(facecolor=LORA_COLOR, label="LoRA Teacher")]
    exported = []

    def style(axis) -> None:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    def save(figure, folder: str, name: str) -> list[str]:
        paths = []
        for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
            path = output / folder / f"{name}{suffix}"
            figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs); paths.append(str(path.resolve()))
        plt.close(figure); return paths

    def plot_metric(axis, label: str, field: str, scale: float, binary: bool, panel: str) -> None:
        width = .34
        for x, (method, rows, color, hatch) in zip((-width/2, width/2), methods):
            values = numeric(rows, field)
            low, high = wilson_ci(values) if binary else bootstrap_ci(values)
            mean = float(values.mean())
            axis.bar(x, mean*scale, width, color=color, hatch=hatch, edgecolor="white", linewidth=1.2,
                     yerr=np.array([[(mean-low)*scale], [(high-mean)*scale]]), capsize=3, zorder=3)
            exported.append({"panel": panel, "metric": label, "method": method, "mean": mean*scale,
                             "ci95_low": low*scale, "ci95_high": high*scale,
                             "ci_method": "Wilson score 95% CI" if binary else "bootstrap 95% CI (10,000; seed=42)",
                             "n": len(values)})
        axis.set(xlim=(-.55, .55), ylim=(0, 108), ylabel="Score (%)")
        axis.set_xticks([0], [label]); style(axis)

    def plot_corpus_metric(axis, label: str, field: str, scale: float, panel: str) -> None:
        width = .34
        for x, (method, _, color, hatch) in zip((-width/2, width/2), methods):
            entry = corpus[method][field]
            value = float(entry["value"]) * scale
            axis.bar(x, value, width, color=color, hatch=hatch, edgecolor="white", linewidth=1.2, zorder=3)
            exported.append({"panel": panel, "metric": label, "method": method, "mean": value,
                             "ci95_low": "", "ci95_high": "", "ci_method": entry["aggregation"],
                             "n": entry["n"]})
        axis.set(xlim=(-.55, .55), ylim=(0, 108), ylabel="Score (%)")
        axis.set_xticks([0], [label]); style(axis)

    figure, axis = plt.subplots(figsize=(3.8, 3.25))
    plot_corpus_metric(axis, "Corpus CodeBLEU", "corpus_codebleu", 100, "primary")
    axis.legend(handles=legend, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.16))
    figure.tight_layout(); primary = save(figure, "primary", "local_structure_codebleu")

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    plot_corpus_metric(axes[0], "Corpus BLEU", "corpus_bleu", 1, "auxiliary-1")
    plot_metric(axes[1], "Exact match", "exact_match", 100, True, "auxiliary-2")
    for index, axis in enumerate(axes):
        axis.text(.02, .98, f"({chr(97+index)})", transform=axis.transAxes, va="top", fontweight="bold")
    figure.legend(handles=legend, frameon=False, ncol=2, loc="upper center")
    figure.tight_layout(rect=(0, 0, 1, .89)); auxiliary = save(figure, "auxiliary", "local_structure_auxiliary")

    gates, gate_rows = [("Parse", "parse_pass"), ("Signature", "signature_pass"), ("Compile", "compile_pass")], []
    table_rows = []
    for method, rows, _, _ in methods:
        cells = [method]
        for label, field in gates:
            values = numeric(rows, field); low, high = wilson_ci(values); mean = float(values.mean())
            cells.append(f"{mean*100:.1f}%")
            gate_rows.append({"model": method, "gate": label, "rate_percent": mean*100,
                              "ci95_low": low*100, "ci95_high": high*100, "n": len(values)})
        cells.append("N/A"); table_rows.append(cells)
    figure, axis = plt.subplots(figsize=(6.7, 1.65)); axis.axis("off")
    table = axis.table(cellText=table_rows, colLabels=["Model", "Parse", "Signature", "Compile", "Tests"],
                       cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(.8)
        cell.set_facecolor(LORA_COLOR if row == 0 else PALETTE["base_light"] if row == 1 else GATE_COLOR)
        if row == 0: cell.set_text_props(color="white", fontweight="bold")
    figure.text(.5, .04, "Tests are N/A because CodeSearchNet provides no executable unit tests.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .1, 1, 1)); gate_outputs = save(figure, "gates", "local_structure_hard_gates")

    plot_csv = output / "data" / "plot_data.csv"
    with plot_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "metric", "method", "mean", "ci95_low", "ci95_high", "ci_method", "n"])
        writer.writeheader(); writer.writerows(exported)
    gate_csv = output / "data" / "hard_gates.csv"
    with gate_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gate", "rate_percent", "ci95_low", "ci95_high", "n"])
        writer.writeheader(); writer.writerows(gate_rows)
    manifest = {"artifact_type": "local_structure_teacher_evaluation", "sample_count": len(base),
                "font_family": family, "primary": primary, "auxiliary": auxiliary, "gates": gate_outputs,
                "data": [str(plot_csv.resolve()), str(gate_csv.resolve())], "palette": PALETTE, "png_dpi": 600}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
