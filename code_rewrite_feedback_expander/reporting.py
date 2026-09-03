from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .quality import METHOD_TO_QUALITY_METRIC


def build_paper_report(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    attempts = {name: 0 for name in ("cot", "style", "ast", "variable", "control_flow")}
    retained = dict.fromkeys(attempts, 0)
    metric_before: Dict[str, List[float]] = {}
    metric_after: Dict[str, List[float]] = {}
    raw_before: Dict[str, List[float]] = {}
    raw_after: Dict[str, List[float]] = {}
    semantic_passes = 0
    test_pass_rates: List[float] = []
    candidate_count = 0
    candidate_semantic_passes = 0
    candidate_test_pass_rates: List[float] = []

    for record in records:
        for trace in record.get("iteration_trace", []):
            candidate_count += 1
            candidate_semantic_passes += int(bool(trace.get("semantic_passed")))
            candidate_test_pass_rates.append(float(trace.get("semantic_metrics", {}).get("unit_test_pass_rate", 0.0)))
            strategy = trace.get("strategy")
            if strategy in attempts:
                attempts[strategy] += 1
                retained[strategy] += int(bool(trace.get("retained")))
        semantic = record.get("semantic_result", {})
        semantic_passes += int(bool(semantic.get("passed")))
        for score in semantic.get("scores", []):
            if score.get("name") == "unit_test_pass_rate":
                test_pass_rates.append(float(score.get("score", 0.0)))
        _collect_metrics(record.get("original_quality", {}), metric_before)
        _collect_metrics(record.get("final_quality", {}), metric_after)
        _collect_raw_metrics(record.get("original_quality", {}), raw_before)
        _collect_raw_metrics(record.get("final_quality", {}), raw_after)

    metric_names = sorted(set(metric_before) | set(metric_after))
    metric_improvements = {}
    for name in metric_names:
        before = _mean(metric_before.get(name, []))
        after = _mean(metric_after.get(name, []))
        metric_improvements[name] = {"before": before, "after": after, "delta": after - before}

    return {
        "record_count": len(records),
        "method_metric_mapping": dict(METHOD_TO_QUALITY_METRIC),
        "method_retention": {
            name: {
                "attempted": attempts[name],
                "retained": retained[name],
                "retention_rate": retained[name] / attempts[name] if attempts[name] else 0.0,
            }
            for name in attempts
        },
        "metric_improvements": metric_improvements,
        "raw_metric_changes": {
            name: {
                "before": _mean(raw_before.get(name, [])),
                "after": _mean(raw_after.get(name, [])),
                "delta": _mean(raw_after.get(name, [])) - _mean(raw_before.get(name, [])),
            }
            for name in sorted(set(raw_before) | set(raw_after))
        },
        "execution_correctness": {
            "final_output_semantic_pass_rate": semantic_passes / len(records) if records else 0.0,
            "final_output_unit_test_pass_rate": _mean(test_pass_rates),
            "candidate_semantic_pass_rate": candidate_semantic_passes / candidate_count if candidate_count else 0.0,
            "candidate_unit_test_pass_rate": _mean(candidate_test_pass_rates),
            "candidate_count": candidate_count,
        },
        "overall_retention_rate": sum(retained.values()) / sum(attempts.values()) if sum(attempts.values()) else 0.0,
    }


def _collect_metrics(quality: Dict[str, Any], destination: Dict[str, List[float]]) -> None:
    for score in quality.get("scores", []):
        destination.setdefault(score["name"], []).append(float(score.get("score", 0.0)))


def _collect_raw_metrics(quality: Dict[str, Any], destination: Dict[str, List[float]]) -> None:
    for score in quality.get("scores", []):
        raw_value = score.get("raw_value")
        if raw_value is not None:
            destination.setdefault(score["name"], []).append(float(raw_value))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
