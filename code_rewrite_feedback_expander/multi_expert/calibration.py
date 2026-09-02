from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable


def fit_robust_advantage_calibration(
    records: Iterable[Dict[str, Any]],
    expert_ids: Iterable[str],
    min_samples: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Fit per-Teacher median/MAD scales from held-out shared trajectories."""
    values = {expert_id: [] for expert_id in expert_ids}
    for record in records:
        for item in record.get("expert_assessments", []):
            expert_id = item.get("expert_id")
            trajectory = item.get("trajectory", {})
            if expert_id in values and trajectory.get("available"):
                values[expert_id].append(
                    float(trajectory["mean_teacher_student_nll_advantage"])
                )
    result: Dict[str, Dict[str, float]] = {}
    for expert_id, samples in values.items():
        if len(samples) < min_samples:
            raise ValueError(
                f"Need at least {min_samples} calibration samples for {expert_id}; "
                f"found {len(samples)}"
            )
        location = statistics.median(samples)
        mad = statistics.median(abs(value - location) for value in samples)
        scale = 1.4826 * mad
        if scale <= 1e-8:
            raise ValueError(
                f"Calibration scale collapsed for {expert_id}; use a larger, more varied split"
            )
        result[expert_id] = {
            "location": location,
            "scale": scale,
            "sample_count": len(samples),
            "estimator": "median_and_scaled_mad",
        }
    return result


def read_jsonl_dicts(path: str | Path) -> list[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_calibration(path: str | Path, calibration: Dict[str, Dict[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
