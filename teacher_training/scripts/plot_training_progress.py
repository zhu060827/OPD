"""从 training_progress.jsonl 生成可复现的训练过程图与绘图数据。"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


COLORS = {"train": "#4E79B7", "validation": "#E76F51", "lr": "#59A14F", "gpu": "#B07AA1"}


def configure_font(font_file: str | None) -> str:
    if font_file:
        font_manager.fontManager.addfont(font_file)
        family = font_manager.FontProperties(fname=font_file).get_name()
    else:
        installed = {item.name for item in font_manager.fontManager.ttflist}
        family = "Arial" if "Arial" in installed else "Liberation Sans"
        if family != "Arial":
            warnings.warn("Arial 不可用，预览图使用 Liberation Sans。", stacklevel=2)
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 1.2,
        "lines.linewidth": 1.4, "svg.fonttype": "none",
    })
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Teacher 训练过程。")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--progress", help="ProgressCallback 产生的 training_progress.jsonl")
    source.add_argument("--trainer-state", help="Transformers checkpoint 中的 trainer_state.json（旧训练兼容）")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--font-file")
    args = parser.parse_args()
    family = configure_font(args.font_file)
    if args.progress:
        source_path = Path(args.progress)
        rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        source_format = "training_progress_jsonl"
    else:
        source_path = Path(args.trainer_state)
        state = json.loads(source_path.read_text(encoding="utf-8"))
        rows = list(state.get("log_history") or [])
        source_format = "transformers_trainer_state"
    if not rows:
        raise ValueError("训练历史为空：找不到 log_history")
    train = [row for row in rows if "loss" in row]
    evaluation = [row for row in rows if "eval_loss" in row]
    panels = ["loss"]
    if any("learning_rate" in row for row in train):
        panels.append("learning_rate")
    if any("gpu_allocated_gb" in row for row in rows):
        panels.append("gpu")
    figure, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.35), squeeze=False)
    axes = axes[0]
    loss_axis = axes[0]
    if train:
        loss_axis.plot([row["step"] for row in train], [row["loss"] for row in train], marker="o", ms=4,
                       color=COLORS["train"], label="Training loss")
    if evaluation:
        loss_axis.plot([row["step"] for row in evaluation], [row["eval_loss"] for row in evaluation], marker="s", ms=4,
                       color=COLORS["validation"], label="Validation loss")
    loss_axis.set(xlabel="Optimizer step", ylabel="Loss", title="Loss trajectory")
    loss_axis.legend(frameon=False)
    panel = 1
    if "learning_rate" in panels:
        values = [row for row in train if "learning_rate" in row]
        axes[panel].plot([row["step"] for row in values], [row["learning_rate"] for row in values], marker="o", ms=4,
                         color=COLORS["lr"])
        axes[panel].set(xlabel="Optimizer step", ylabel="Learning rate", title="Learning-rate schedule")
        panel += 1
    if "gpu" in panels:
        values = [row for row in rows if "gpu_allocated_gb" in row]
        axes[panel].plot([row["elapsed_seconds"] / 60 for row in values], [row["gpu_allocated_gb"] for row in values],
                         marker="o", ms=4, color=COLORS["gpu"], label="Allocated")
        axes[panel].plot([row["elapsed_seconds"] / 60 for row in values], [row["gpu_reserved_gb"] for row in values],
                         linestyle="--", color="#79706E", label="Reserved")
        axes[panel].set(xlabel="Elapsed time (min)", ylabel="GPU memory (GiB)", title="GPU memory")
        axes[panel].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#E6F0FF", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.04)
    figure.tight_layout()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        path = prefix.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(str(path.resolve()))
    plt.close(figure)
    csv_path = prefix.with_name(prefix.name + "_data.csv")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    metadata = prefix.with_name(prefix.name + "_metadata.json")
    metadata.write_text(json.dumps({
        "source": str(source_path.resolve()), "source_format": source_format,
        "gpu_memory_available": any("gpu_allocated_gb" in row for row in rows),
        "gpu_memory_note": None if any("gpu_allocated_gb" in row for row in rows) else
            "旧 trainer_state.json 未记录 GPU 显存，无法事后恢复。",
        "font_family": family,
        "palette": COLORS, "png_dpi": 600, "outputs": outputs, "data": str(csv_path.resolve()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "data": str(csv_path.resolve()), "metadata": str(metadata.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
