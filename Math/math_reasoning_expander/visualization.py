from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_ORDER = [
    "similarity",
    "logic_consistency",
    "formula_correctness",
    "completeness",
    "reasoning_gain",
]

METRIC_LABELS = {
    "similarity": "De-duplication",
    "logic_consistency": "Logic",
    "formula_correctness": "Formula",
    "completeness": "Completeness",
    "reasoning_gain": "Reasoning Gain",
}


def write_quality_report(path: str, records: Iterable[Dict]) -> None:
    records = list(records)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_quality_svg(records), encoding="utf-8")


def build_quality_svg(records: List[Dict]) -> str:
    width = 1180
    height = 790
    before_scores = [_aggregate_score(record, "original_evaluation") for record in records]
    after_scores = [_aggregate_score(record, "evaluation") for record in records]
    accepted_count = sum(1 for record in records if record.get("accepted"))
    before_avg = _average(before_scores)
    after_avg = _average(after_scores)
    avg_gain = after_avg - before_avg
    node_stats = _node_stats(records)
    accept_rate = accepted_count / len(records) if records else 0.0
    before_metrics = _metric_averages(records, "original_evaluation")
    after_metrics = _metric_averages(records, "evaluation")

    parts = [
        _svg_header(width, height),
        f'<rect width="{width}" height="{height}" fill="#f7f8fb"/>',
        _text(48, 54, "Before / After Data Quality Comparison", 28, "#18212f", weight="700"),
        _text(48, 82, "Quality change after reasoning-step expansion using the same evaluator.", 14, "#627083"),
        _summary_card(48, 112, "Records", str(len(records)), "#2563eb"),
        _summary_card(248, 112, "Before Avg.", f"{before_avg:.3f}", "#64748b"),
        _summary_card(448, 112, "After Avg.", f"{after_avg:.3f}", "#16a34a"),
        _summary_card(648, 112, "Avg. Gain", _format_gain(avg_gain), "#9333ea" if avg_gain >= 0 else "#dc2626"),
        _summary_card(848, 112, "Nodes", f"{node_stats['before']} -> {node_stats['after']}", "#0f766e"),
        _summary_card(48, 684, "Added Nodes", f"+{node_stats['added']}", "#0f766e"),
        _summary_card(248, 684, "Accepted", f"{accepted_count} ({accept_rate:.0%})", "#ea580c"),
        _aggregate_line_chart(48, 230, 520, 430, before_scores, after_scores),
        _metric_line_chart(640, 230, 460, 390, before_metrics, after_metrics),
        _legend(640, 664),
        "</svg>",
    ]
    return "\n".join(parts)


def _average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_gain(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def _aggregate_score(record: Dict, key: str) -> float:
    if key == "original_evaluation" and "original_evaluation" not in record:
        return float(record.get("evaluation", {}).get("aggregate_score", 0.0))
    return float(record.get(key, {}).get("aggregate_score", 0.0))


def _metric_averages(records: List[Dict], key: str) -> Dict[str, float]:
    totals = {metric: 0.0 for metric in METRIC_ORDER}
    counts = {metric: 0 for metric in METRIC_ORDER}
    for record in records:
        evaluation = record.get(key) or record.get("evaluation", {})
        for score in evaluation.get("scores", []):
            name = score.get("name")
            if name in totals:
                totals[name] += float(score.get("score", 0.0))
                counts[name] += 1
    return {metric: (totals[metric] / counts[metric] if counts[metric] else 0.0) for metric in METRIC_ORDER}


def _node_stats(records: List[Dict]) -> Dict[str, int]:
    before = 0
    after = 0
    for record in records:
        stats = record.get("expansion_stats", {})
        before += int(stats.get("original_node_count", 0))
        after += int(stats.get("expanded_node_count", 0))
    return {"before": before, "after": after, "added": max(0, after - before)}


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Quality report">'
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}"
        ".small{font-size:12px;fill:#627083}"
        "</style>"
    )


def _summary_card(x: int, y: int, title: str, value: str, color: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="168" height="82" rx="8" fill="#ffffff" stroke="#d9e0ea"/>',
            f'<rect x="{x}" y="{y}" width="6" height="82" rx="3" fill="{color}"/>',
            _text(x + 20, y + 30, title, 13, "#627083"),
            _text(x + 20, y + 60, value, 24, "#18212f", weight="700"),
        ]
    )


def _aggregate_line_chart(
    x: int,
    y: int,
    w: int,
    h: int,
    before_scores: List[float],
    after_scores: List[float],
) -> str:
    chart_h = h - 74
    chart_y = y + 42
    axis_x = x + 48
    axis_w = w - 72
    parts = [
        _panel(x, y, w, h, "Per-sample Aggregate Score Trend"),
        _grid(axis_x, chart_y, axis_w, chart_h),
    ]
    if not after_scores:
        parts.append(_text(x + 36, y + 90, "No data", 14, "#627083"))
        return "\n".join(parts)

    before_points = _series_points(before_scores, axis_x, chart_y, axis_w, chart_h)
    after_points = _series_points(after_scores, axis_x, chart_y, axis_w, chart_h)
    parts.append(_polyline(before_points, "#94a3b8", width=3))
    parts.append(_polyline(after_points, "#16a34a", width=3))
    parts.extend(_point_markers(before_points, "#94a3b8"))
    parts.extend(_point_markers(after_points, "#16a34a"))
    for idx, point in enumerate(after_points):
        if len(after_points) <= 24:
            parts.append(_text(point[0], chart_y + chart_h + 20, str(idx + 1), 11, "#627083", anchor="middle"))
    parts.append(_text(axis_x - 10, chart_y + 4, "1.0", 11, "#627083", anchor="end"))
    parts.append(_text(axis_x - 10, chart_y + chart_h + 4, "0", 11, "#627083", anchor="end"))
    parts.append(_text(axis_x + axis_w / 2, y + h - 18, "sample index", 12, "#627083", anchor="middle"))
    return "\n".join(parts)


def _metric_line_chart(
    x: int,
    y: int,
    w: int,
    h: int,
    before_averages: Dict[str, float],
    after_averages: Dict[str, float],
) -> str:
    chart_x = x + 68
    chart_y = y + 58
    chart_w = w - 112
    chart_h = h - 120
    before_values = [before_averages.get(metric, 0.0) for metric in METRIC_ORDER]
    after_values = [after_averages.get(metric, 0.0) for metric in METRIC_ORDER]
    before_points = _series_points(before_values, chart_x, chart_y, chart_w, chart_h)
    after_points = _series_points(after_values, chart_x, chart_y, chart_w, chart_h)
    parts = [
        _panel(x, y, w, h, "Average Metric Score Trend"),
        _grid(chart_x, chart_y, chart_w, chart_h),
        _polyline(before_points, "#94a3b8", width=3),
        _polyline(after_points, "#16a34a", width=3),
    ]
    parts.extend(_point_markers(before_points, "#94a3b8"))
    parts.extend(_point_markers(after_points, "#16a34a"))
    for idx, metric in enumerate(METRIC_ORDER):
        label = METRIC_LABELS[metric].replace(" ", "\n")
        x_pos = before_points[idx][0]
        parts.append(_wrapped_label(x_pos, chart_y + chart_h + 24, label, 10, "#627083"))
        parts.append(_text(x_pos, after_points[idx][1] - 10, f"{after_values[idx]:.2f}", 10, "#166534", anchor="middle"))
    return "\n".join(parts)


def _legend(x: int, y: int) -> str:
    return "\n".join(
        [
            _text(x, y, "Legend", 15, "#18212f", weight="700"),
            f'<line x1="{x}" y1="{y + 28}" x2="{x + 32}" y2="{y + 28}" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>',
            f'<circle cx="{x + 16}" cy="{y + 28}" r="4" fill="#94a3b8"/>',
            _text(x + 44, y + 34, "before expansion", 13, "#627083"),
            f'<line x1="{x + 190}" y1="{y + 28}" x2="{x + 222}" y2="{y + 28}" stroke="#16a34a" stroke-width="4" stroke-linecap="round"/>',
            f'<circle cx="{x + 206}" cy="{y + 28}" r="4" fill="#16a34a"/>',
            _text(x + 234, y + 34, "after expansion", 13, "#627083"),
        ]
    )


def _panel(x: int, y: int, w: int, h: int, title: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="#d9e0ea"/>',
            _text(x + 24, y + 30, title, 17, "#18212f", weight="700"),
        ]
    )


def _grid(x: int, y: int, w: int, h: int) -> str:
    lines = []
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y + h * ratio
        lines.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + w}" y2="{yy:.1f}" stroke="#e8edf5"/>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#b8c2d1"/>')
    lines.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#b8c2d1"/>')
    return "\n".join(lines)


def _score_color(score: float) -> str:
    if score >= 0.85:
        return "#16a34a"
    if score >= 0.75:
        return "#2563eb"
    if score >= 0.6:
        return "#f59e0b"
    return "#dc2626"


def _series_points(values: List[float], x: int, y: int, w: int, h: int) -> List[tuple[float, float]]:
    if not values:
        return []
    if len(values) == 1:
        return [(x + w / 2, y + h - _clamp01(values[0]) * h)]
    points = []
    for idx, value in enumerate(values):
        xx = x + (w * idx / (len(values) - 1))
        yy = y + h - _clamp01(value) * h
        points.append((xx, yy))
    return points


def _polyline(points: List[tuple[float, float]], color: str, width: int = 3) -> str:
    if not points:
        return ""
    if len(points) == 1:
        x, y = points[0]
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{data}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _point_markers(points: List[tuple[float, float]], color: str) -> List[str]:
    return [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#ffffff" stroke="{color}" stroke-width="3"/>' for x, y in points]


def _wrapped_label(x: float, y: float, content: str, size: int, color: str) -> str:
    lines = content.split("\n")
    parts = [
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="middle">'
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else size + 2
        parts.append(f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _text(
    x: float,
    y: float,
    content: str,
    size: int,
    color: str,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    safe = html.escape(str(content))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{safe}</text>'
    )
