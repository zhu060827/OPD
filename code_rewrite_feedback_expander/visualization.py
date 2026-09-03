from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_ORDER = [
    "style_violation_rate",
    "maintainability_index",
    "cyclomatic_complexity",
    "naming_convention_compliance",
    "codebleu_syntax_match",
]


def write_quality_report(path: str, records: Iterable[Dict]) -> None:
    records = list(records)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_quality_svg(records), encoding="utf-8")


def build_quality_svg(records: List[Dict]) -> str:
    width = 1180
    height = 790
    line_stats = _line_stats(records)
    reasoning_stats = _reasoning_stats(records)
    retained = sum(len(record.get("retained_rewrites", [])) for record in records)
    before_metrics = _metric_avgs(records, "original_quality")
    after_metrics = _metric_avgs(records, "final_quality")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}</style>",
        f'<rect width="{width}" height="{height}" fill="#f7f8fb"/>',
        _text(48, 54, "Code Rewrite Quality Comparison", 28, "#18212f", "700"),
        _text(48, 82, "Before/after quality trend for semantic-preserving code rewrites.", 14, "#627083"),
        _card(48, 112, "Records", str(len(records)), "#2563eb"),
        _card(248, 112, "Semantic Pass", str(sum(bool(r.get("semantic_result", {}).get("passed")) for r in records)), "#64748b"),
        _card(448, 112, "Accepted", str(sum(bool(r.get("accepted")) for r in records)), "#16a34a"),
        _card(648, 112, "Metrics", str(len(before_metrics)), "#9333ea"),
        _card(848, 112, "Lines", f"{line_stats['before']} -> {line_stats['after']}", "#0f766e"),
        _card(48, 684, "Added Lines", f"+{line_stats['added']}", "#0f766e"),
        _card(248, 684, "Reasoning", f"{reasoning_stats['before']} -> {reasoning_stats['after']}", "#2563eb"),
        _card(448, 684, "Retained", str(retained), "#ea580c"),
        _metric_chart(48, 230, 1052, 390, before_metrics, after_metrics),
        _legend(640, 664),
        "</svg>",
    ]
    return "\n".join(parts)


def _metric_avgs(records: List[Dict], key: str) -> Dict[str, float]:
    metric_names = _metric_names(records)
    totals = {metric: 0.0 for metric in metric_names}
    counts = {metric: 0 for metric in metric_names}
    for record in records:
        for score in record.get(key, {}).get("scores", []):
            name = score.get("name")
            if name in totals:
                totals[name] += float(score.get("score", 0.0))
                counts[name] += 1
    return {metric: totals[metric] / counts[metric] if counts[metric] else 0.0 for metric in metric_names}


def _metric_names(records: List[Dict]) -> List[str]:
    seen = set()
    names: List[str] = []
    for metric in METRIC_ORDER:
        seen.add(metric)
        names.append(metric)
    extras = set()
    for record in records:
        for key in ("original_quality", "final_quality"):
            for score in record.get(key, {}).get("scores", []):
                name = score.get("name")
                if isinstance(name, str) and name and name not in seen:
                    extras.add(name)
    for name in sorted(extras):
        names.append(name)
    return names


def _line_stats(records: List[Dict]) -> Dict[str, int]:
    before = 0
    after = 0
    for record in records:
        stats = record.get("expansion_stats", {})
        before += int(stats.get("original_line_count", 0))
        after += int(stats.get("expanded_line_count", 0))
    return {"before": before, "after": after, "added": max(0, after - before)}


def _reasoning_stats(records: List[Dict]) -> Dict[str, int]:
    before = 0
    after = 0
    for record in records:
        stats = record.get("expansion_stats", {})
        before += int(stats.get("original_reasoning_steps", 0))
        after += int(stats.get("expanded_reasoning_steps", 0))
    return {"before": before, "after": after, "added": max(0, after - before)}


def _line_chart(x: int, y: int, w: int, h: int, before: List[float], after: List[float], title: str, xlabel: str) -> str:
    chart_x = x + 48
    chart_y = y + 42
    chart_w = w - 72
    chart_h = h - 74
    parts = [_panel(x, y, w, h, title), _grid(chart_x, chart_y, chart_w, chart_h)]
    before_points = _points(before, chart_x, chart_y, chart_w, chart_h)
    after_points = _points(after, chart_x, chart_y, chart_w, chart_h)
    parts.append(_polyline(before_points, "#94a3b8"))
    parts.append(_polyline(after_points, "#16a34a"))
    parts.extend(_markers(before_points, "#94a3b8"))
    parts.extend(_markers(after_points, "#16a34a"))
    for idx, point in enumerate(after_points):
        if len(after_points) <= 24:
            parts.append(_text(point[0], chart_y + chart_h + 20, str(idx + 1), 11, "#627083", anchor="middle"))
    parts.append(_text(chart_x - 10, chart_y + 4, "1.0", 11, "#627083", anchor="end"))
    parts.append(_text(chart_x - 10, chart_y + chart_h + 4, "0", 11, "#627083", anchor="end"))
    parts.append(_text(chart_x + chart_w / 2, y + h - 18, xlabel, 12, "#627083", anchor="middle"))
    return "\n".join(parts)


def _metric_chart(x: int, y: int, w: int, h: int, before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> str:
    chart_x = x + 68
    chart_y = y + 58
    chart_w = w - 112
    chart_h = h - 120
    metrics = list(before_metrics.keys() or after_metrics.keys())
    before = [before_metrics.get(metric, 0.0) for metric in metrics]
    after = [after_metrics.get(metric, 0.0) for metric in metrics]
    before_points = _points(before, chart_x, chart_y, chart_w, chart_h)
    after_points = _points(after, chart_x, chart_y, chart_w, chart_h)
    parts = [_panel(x, y, w, h, "Average Metric Trend"), _grid(chart_x, chart_y, chart_w, chart_h)]
    parts.append(_polyline(before_points, "#94a3b8"))
    parts.append(_polyline(after_points, "#16a34a"))
    parts.extend(_markers(before_points, "#94a3b8"))
    parts.extend(_markers(after_points, "#16a34a"))
    for idx, metric in enumerate(metrics):
        parts.append(_wrapped_label(before_points[idx][0], chart_y + chart_h + 24, metric.replace("_", "\n"), 10, "#627083"))
        parts.append(_text(after_points[idx][0], after_points[idx][1] - 10, f"{after[idx]:.2f}", 10, "#166534", anchor="middle"))
    return "\n".join(parts)


def _card(x: int, y: int, title: str, value: str, color: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="168" height="82" rx="8" fill="#ffffff" stroke="#d9e0ea"/>',
            f'<rect x="{x}" y="{y}" width="6" height="82" rx="3" fill="{color}"/>',
            _text(x + 20, y + 30, title, 13, "#627083"),
            _text(x + 20, y + 60, value, 24, "#18212f", "700"),
        ]
    )


def _panel(x: int, y: int, w: int, h: int, title: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="#d9e0ea"/>\n' + _text(
        x + 24, y + 30, title, 17, "#18212f", "700"
    )


def _grid(x: int, y: int, w: int, h: int) -> str:
    lines = []
    for ratio in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y + h * ratio
        lines.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + w}" y2="{yy:.1f}" stroke="#e8edf5"/>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#b8c2d1"/>')
    lines.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#b8c2d1"/>')
    return "\n".join(lines)


def _points(values: List[float], x: int, y: int, w: int, h: int) -> List[tuple[float, float]]:
    if not values:
        return []
    if len(values) == 1:
        return [(x + w / 2, y + h - _clamp(values[0]) * h)]
    return [(x + w * idx / (len(values) - 1), y + h - _clamp(value) * h) for idx, value in enumerate(values)]


def _polyline(points: List[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'


def _markers(points: List[tuple[float, float]], color: str) -> List[str]:
    return [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#ffffff" stroke="{color}" stroke-width="3"/>' for x, y in points]


def _legend(x: int, y: int) -> str:
    return "\n".join(
        [
            _text(x, y, "Legend", 15, "#18212f", "700"),
            f'<line x1="{x}" y1="{y + 28}" x2="{x + 32}" y2="{y + 28}" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>',
            f'<circle cx="{x + 16}" cy="{y + 28}" r="4" fill="#94a3b8"/>',
            _text(x + 44, y + 34, "before rewrite", 13, "#627083"),
            f'<line x1="{x + 190}" y1="{y + 28}" x2="{x + 222}" y2="{y + 28}" stroke="#16a34a" stroke-width="4" stroke-linecap="round"/>',
            f'<circle cx="{x + 206}" cy="{y + 28}" r="4" fill="#16a34a"/>',
            _text(x + 234, y + 34, "after rewrite", 13, "#627083"),
        ]
    )


def _wrapped_label(x: float, y: float, content: str, size: int, color: str) -> str:
    parts = [f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="middle">']
    for idx, line in enumerate(content.split("\n")):
        dy = 0 if idx == 0 else size + 2
        parts.append(f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _text(
    x: float,
    y: float,
    content: str,
    size: int,
    color: str,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(content))}</text>'
    )


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
