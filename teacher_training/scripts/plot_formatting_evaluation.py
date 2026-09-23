"""绘制 Formatting Teacher 的 Base–LoRA 正式评估对比图。

数据与绘图分离：本程序只读取 evaluate_domains 生成的两个 per_sample.csv，
不在代码中硬编码实验数值。所有制品按 figures/data/metadata 子目录集中输出。
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# 采用用户提供的第 3 套蓝—橙数据可视化色板（原始色值）。
# 深色用于主系列，中间色用于表格，浅色用于背景；纹理继续提供非颜色冗余编码。
PALETTE = {
    "base_light": "#E6F0FF",
    "base_mid": "#A7C7E7",
    "base": "#4E79B7",
    "lora_light": "#FAD7B1",
    "lora_mid": "#F4A261",
    "lora": "#E76F51",
}
BASE_COLOR = PALETTE["base"]
LORA_COLOR = PALETTE["lora"]
TEXT_COLOR = "#333333"
GRID_COLOR = PALETTE["base_light"]
PANEL_COLOR = "#FFFFFF"
GATE_COLOR = PALETTE["lora_light"]


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, str]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        value = str(row.get(field, "")).strip().lower()
        if value in {"true", "false"}:
            values.append(float(value == "true"))
        elif value:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def bootstrap_ci(values: np.ndarray, seed: int = 42, repeats: int = 10_000) -> tuple[float, float]:
    """固定随机种子的非参数 bootstrap 95% CI；仅用于展示均值不确定性。"""
    if values.size < 2:
        mean = float(values.mean()) if values.size else float("nan")
        return mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def wilson_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """二元比例的 Wilson score interval；比小样本普通 bootstrap 更稳健。"""
    n = values.size
    if n == 0:
        return float("nan"), float("nan")
    proportion = float(values.mean())
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2)) / denominator
    return float(center - margin), float(center + margin)


def configure_font(font_file: str | None, strict_arial: bool) -> str:
    """优先使用 Arial；无管理员权限时允许显式字体文件或兼容字体回退。"""
    if font_file:
        path = Path(font_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"字体文件不存在：{path}")
        font_manager.fontManager.addfont(str(path))
        selected = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        if "Arial" in installed:
            selected = "Arial"
        elif strict_arial:
            raise RuntimeError(
                "系统未安装 Arial。请通过 --font-file 指定用户目录中的 Arial.ttf，"
                "或去掉 --strict-arial 使用 Liberation Sans 生成预览图。"
            )
        elif "Liberation Sans" in installed:
            selected = "Liberation Sans"
            warnings.warn(
                "系统未安装 Arial，当前使用度量兼容的 Liberation Sans。"
                "该图适合检查和实验记录；严格投稿版请在安装 Arial 后重新导出。",
                stacklevel=2,
            )
        else:
            selected = "DejaVu Sans"
            warnings.warn("Arial 与 Liberation Sans 均不可用，回退到 DejaVu Sans。", stacklevel=2)
    plt.rcParams.update({
        "font.family": selected,
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 12,
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.3,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "svg.fonttype": "none",  # SVG 保留可编辑文本。
    })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Formatting Base–LoRA 学术对比图。")
    parser.add_argument("--base-csv", required=True, help="Base evaluate_domains/per_sample.csv")
    parser.add_argument("--lora-csv", required=True, help="LoRA evaluate_domains/per_sample.csv")
    parser.add_argument("--output-prefix", required=True, help="输出前缀，不含扩展名")
    parser.add_argument("--font-file", help="可选：Arial .ttf/.otf 文件路径，无需 sudo 即可加载")
    parser.add_argument("--strict-arial", action="store_true", help="Arial 不可用时终止；投稿最终导出时建议启用")
    parser.add_argument(
        "--clean-legacy", action="store_true",
        help="成功生成新制品包后，删除同一 output-prefix 的旧散落图表文件；不会删除预测或评估目录",
    )
    args = parser.parse_args()

    selected_font = configure_font(args.font_file, args.strict_arial)
    base_rows, lora_rows = read_rows(args.base_csv), read_rows(args.lora_csv)
    if len(base_rows) != len(lora_rows) or not base_rows:
        raise ValueError("Base 与 LoRA CSV 必须包含相同数量且非空的样本。")
    base_ids = [row["source_id"] for row in base_rows]
    lora_by_id = {row["source_id"]: row for row in lora_rows}
    if set(base_ids) != set(lora_by_id):
        raise ValueError("Base 与 LoRA 的 source_id 不一致。")
    lora_rows = [lora_by_id[source_id] for source_id in base_ids]

    primary_metric = ("Exact formatting repair", "primary_exact_formatting_repair")
    auxiliary_metrics = [
        ("Style violations", "pycodestyle_violation_count", "Mean count"),
        ("Edit distance to target", "levenshtein_distance_to_target", "Mean distance"),
    ]
    gate_metrics = [
        ("AST identity", "ast_identity_pass"),
        ("Parse pass", "parse_pass"),
        ("Signature pass", "signature_pass"),
        ("Compile pass", "compile_pass"),
    ]
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_prefix.with_name(output_prefix.name + "_artifacts")
    primary_dir = bundle_dir / "figures" / "primary"
    auxiliary_dir = bundle_dir / "figures" / "auxiliary"
    gates_dir = bundle_dir / "figures" / "gates"
    data_dir = bundle_dir / "data"
    metadata_dir = bundle_dir / "metadata"
    for directory in (primary_dir, auxiliary_dir, gates_dir, data_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    exported = []

    methods = (
        ("Base", base_rows, BASE_COLOR, "//"),
        ("LoRA Teacher", lora_rows, LORA_COLOR, ""),
    )

    def style_axis(axis) -> None:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.65, zorder=0)
        axis.set_facecolor(PANEL_COLOR)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    def save_figure(figure, directory: Path, basename: str) -> list[str]:
        paths = []
        for suffix, options in ((".png", {"dpi": 600}), (".svg", {}), (".pdf", {})):
            path = directory / f"{basename}{suffix}"
            figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
            paths.append(str(path.resolve()))
        plt.close(figure)
        return paths

    def add_bar_label(axis, bar, mean: float, percent: bool) -> None:
        label = f"{mean:.1f}%" if percent else f"{mean:.2f}"
        if mean > 0:
            axis.annotate(
                label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, -7), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="white", fontweight="bold",
            )
        else:
            axis.annotate(
                label, (bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=TEXT_COLOR, fontweight="bold",
            )

    # Figure 1：只突出领域主指标，避免与门禁和解释性指标混淆。
    primary_figure, primary_axis = plt.subplots(figsize=(3.6, 3.25), facecolor="white")
    width = 0.34
    for offset, (method, rows, color, hatch) in zip((-width / 2, width / 2), methods):
        values = numeric(rows, primary_metric[1])
        low, high = wilson_ci(values)
        mean = float(values.mean() * 100)
        low, high = low * 100, high * 100
        bar = primary_axis.bar(
            offset, mean, width, label=method, color=color, hatch=hatch,
            edgecolor="white", linewidth=1.2,
            yerr=np.asarray([[mean - low], [high - mean]]),
            error_kw={"elinewidth": 1.05, "capsize": 3, "capthick": 1.05, "ecolor": "#4A4A4A"},
            zorder=3,
        )[0]
        add_bar_label(primary_axis, bar, mean, True)
        exported.append({
            "panel": "Primary metric", "metric": primary_metric[0], "method": method,
            "mean": mean, "ci95_low": low, "ci95_high": high,
            "ci_method": "Wilson score 95% CI", "n": len(values),
        })
    primary_axis.set_xlim(-0.55, 0.55)
    primary_axis.set_ylim(0, 108)
    primary_axis.set_xticks([0], ["Exact formatting repair"])
    primary_axis.set_ylabel("Accuracy (%)")
    primary_axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    style_axis(primary_axis)
    primary_figure.text(0.5, 0.01, f"n = {len(base_rows)}; error bars: Wilson 95% CI.", ha="center", fontsize=8, color="#555555")
    primary_figure.subplots_adjust(top=0.82, bottom=0.20, left=0.19, right=0.97)
    primary_outputs = save_figure(primary_figure, primary_dir, "primary_metric")

    # Figure 2：两个方向一致（越低越好）的辅助指标组成双面板。
    auxiliary_figure, auxiliary_axes = plt.subplots(1, 2, figsize=(6.7, 3.15), facecolor="white")
    for panel_index, (axis, (label, field, ylabel)) in enumerate(zip(auxiliary_axes, auxiliary_metrics)):
        ci_highs = []
        for offset, (method, rows, color, hatch) in zip((-width / 2, width / 2), methods):
            values = numeric(rows, field)
            low, high = bootstrap_ci(values)
            mean = float(values.mean())
            ci_highs.append(high)
            bar = axis.bar(
                offset, mean, width, label=method, color=color, hatch=hatch,
                edgecolor="white", linewidth=1.2,
                yerr=np.asarray([[mean - low], [high - mean]]),
                error_kw={"elinewidth": 1.05, "capsize": 3, "capthick": 1.05, "ecolor": "#4A4A4A"},
                zorder=3,
            )[0]
            add_bar_label(axis, bar, mean, False)
            exported.append({
                "panel": f"Auxiliary {panel_index + 1}", "metric": label, "method": method,
                "mean": mean, "ci95_low": low, "ci95_high": high,
                "ci_method": "percentile bootstrap 95% CI (10,000; seed=42)", "n": len(values),
            })
        axis.set_xlim(-0.55, 0.55)
        axis.set_ylim(0, max(1.0, max(ci_highs) * 1.18))
        axis.set_xticks([0], [label])
        axis.set_ylabel(ylabel)
        axis.text(0.02, 0.98, f"({chr(97 + panel_index)})", transform=axis.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
        style_axis(axis)
    handles, legend_labels = auxiliary_axes[0].get_legend_handles_labels()
    auxiliary_figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    auxiliary_figure.text(0.5, 0.01, f"n = {len(base_rows)}; error bars: percentile-bootstrap 95% CI.", ha="center", fontsize=8, color="#555555")
    auxiliary_figure.subplots_adjust(top=0.82, bottom=0.20, left=0.11, right=0.98, wspace=0.42)
    auxiliary_outputs = save_figure(auxiliary_figure, auxiliary_dir, "auxiliary_metrics")

    # Table：硬门禁是接纳条件，不与主指标混画。
    gate_headers = ["Model"] + [label for label, _ in gate_metrics]
    gate_rows = []
    gate_csv_rows = []
    for method, rows, _, _ in methods:
        cells = [method]
        for label, field in gate_metrics:
            values = numeric(rows, field)
            low, high = wilson_ci(values)
            mean = float(values.mean() * 100)
            cells.append(f"{mean:.1f}%")
            gate_csv_rows.append({"model": method, "gate": label, "rate_percent": mean, "ci95_low": low * 100, "ci95_high": high * 100, "n": len(values)})
            exported.append({
                "panel": "Hard gates", "metric": label, "method": method,
                "mean": mean, "ci95_low": low * 100, "ci95_high": high * 100,
                "ci_method": "Wilson score 95% CI", "n": len(values),
            })
        gate_rows.append(cells)
    gate_figure, gate_axis = plt.subplots(figsize=(7.2, 1.65), facecolor="white")
    gate_axis.axis("off")
    table = gate_axis.table(cellText=gate_rows, colLabels=gate_headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(LORA_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 1:
            cell.set_facecolor(PALETTE["base_light"])
        else:
            cell.set_facecolor(GATE_COLOR)
            if column == 0:
                cell.set_text_props(fontweight="bold")
    gate_figure.text(0.5, 0.04, f"All hard-gate rates are reported separately; n = {len(base_rows)}.", ha="center", fontsize=8, color="#555555")
    gate_figure.subplots_adjust(top=0.96, bottom=0.16, left=0.02, right=0.98)
    gate_outputs = save_figure(gate_figure, gates_dir, "hard_gates")

    gate_csv_path = data_dir / "hard_gates.csv"
    with gate_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gate", "rate_percent", "ci95_low", "ci95_high", "n"])
        writer.writeheader()
        writer.writerows(gate_csv_rows)

    plot_data_path = data_dir / "plot_data.csv"
    with plot_data_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "metric", "method", "mean", "ci95_low", "ci95_high", "ci_method", "n"])
        writer.writeheader()
        writer.writerows(exported)

    # 新制品全部成功落盘后，才允许清理旧版散落文件。
    deleted_legacy: list[str] = []
    if args.clean_legacy:
        legacy_paths = {
            output_prefix.with_suffix(extension)
            for extension in (".png", ".svg", ".pdf")
        }
        for stem in ("_primary", "_auxiliary", "_gates"):
            legacy_paths.update(
                output_prefix.with_name(output_prefix.name + stem).with_suffix(extension)
                for extension in (".png", ".svg", ".pdf")
            )
        legacy_paths.update({
            output_prefix.with_name(output_prefix.name + "_plot_data.csv"),
            output_prefix.with_name(output_prefix.name + "_gates.csv"),
            output_prefix.with_name(output_prefix.name + "_figure_metadata.json"),
        })
        for path in sorted(legacy_paths, key=str):
            if path.is_file():
                path.unlink()
                deleted_legacy.append(str(path.resolve()))

    metadata_path = metadata_dir / "figure_metadata.json"
    metadata_path.write_text(json.dumps({
        "font_family": selected_font,
        "strict_arial": bool(args.strict_arial),
        "font_file": str(Path(args.font_file).expanduser().resolve()) if args.font_file else None,
        "sample_count": len(base_rows),
        "bootstrap_repeats": 10_000,
        "bootstrap_seed": 42,
        "png_dpi": 600,
        "palette": {"name": "reference-blue-orange", **PALETTE},
        "design_skill": "HKUSTDial/Supervisor-Skills: figure-designer",
        "outputs": {"primary": primary_outputs, "auxiliary": auxiliary_outputs, "gates": gate_outputs, "gates_csv": str(gate_csv_path.resolve())},
        "bundle_dir": str(bundle_dir.resolve()),
        "deleted_legacy": deleted_legacy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "artifact_type": "formatting_teacher_evaluation",
        "primary_figures": primary_outputs,
        "auxiliary_figures": auxiliary_outputs,
        "gate_figures": gate_outputs,
        "data_files": [str(plot_data_path.resolve()), str(gate_csv_path.resolve())],
        "metadata_file": str(metadata_path.resolve()),
        "deleted_legacy": deleted_legacy,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Primary: {primary_outputs}")
    print(f"Auxiliary: {auxiliary_outputs}")
    print(f"Gates: {gate_outputs}")
    print(f"Gates CSV: {gate_csv_path.resolve()}")
    print(f"Data: {plot_data_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Bundle: {bundle_dir.resolve()}")
    if deleted_legacy:
        print(f"Deleted legacy files: {len(deleted_legacy)}")
    print(f"Font: {selected_font}")


if __name__ == "__main__":
    main()
