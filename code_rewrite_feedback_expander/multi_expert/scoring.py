from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable

from ..models import QualityResult, SemanticResult, TokenDistribution
from ..opd import HierarchicalTokenKLEvaluator
from ..quality import METHOD_TO_QUALITY_METRIC
from .config import RewardConfig
from .models import RewardEvidence, TrajectoryEvidence


class SharedCodeRewardScorer:
    """One deterministic reward definition shared by every strategy expert.

    Correctness is deliberately excluded from this scalar because it is enforced
    by the hard gate. This score only ranks feasible candidates. It is a routing
    utility, not a claim that code quality has one universally valid metric.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def score(
        self,
        strategy: str,
        baseline: QualityResult,
        candidate: QualityResult,
        changed: bool,
    ) -> RewardEvidence:
        before = {item.name: float(item.score) for item in baseline.scores}
        after = {item.name: float(item.score) for item in candidate.scores}
        names = sorted(set(before) | set(after))
        deltas = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in names}

        primary = METHOD_TO_QUALITY_METRIC[strategy]
        method_alignment = _centered_gain(
            deltas.get(primary, 0.0), self.config.quality_gain_scale
        )
        general_quality = (
            sum(_centered_gain(delta, self.config.quality_gain_scale) for delta in deltas.values())
            / max(len(deltas), 1)
        )
        non_regression = (
            sum(delta >= -self.config.regression_tolerance for delta in deltas.values())
            / max(len(deltas), 1)
        )
        change = 1.0 if changed else 0.0
        weights = {
            "method_alignment": self.config.method_alignment_weight,
            "general_quality": self.config.general_quality_weight,
            "non_regression": self.config.non_regression_weight,
            "change": self.config.change_weight,
        }
        denominator = sum(weights.values())
        total = (
            method_alignment * weights["method_alignment"]
            + general_quality * weights["general_quality"]
            + non_regression * weights["non_regression"]
            + change * weights["change"]
        ) / denominator
        return RewardEvidence(
            total=_clamp(total),
            method_alignment=method_alignment,
            general_quality=general_quality,
            non_regression=non_regression,
            change=change,
            metric_deltas=deltas,
            weights=weights,
        )


def build_gate_evidence(
    semantic: SemanticResult,
    tests_present: bool,
    changed: bool,
    require_tests: bool,
    require_candidate_change: bool,
    minimum_unit_test_pass_rate: float,
):
    from .models import GateEvidence

    metrics = {item.name: float(item.score) for item in semantic.scores}
    reasons: list[str] = []
    if require_tests and not tests_present:
        reasons.append("missing_tests")
    if not semantic.passed:
        reasons.append("semantic_check_failed")
    if require_candidate_change and not changed:
        reasons.append("candidate_unchanged")
    test_rate = metrics.get("unit_test_pass_rate", 0.0 if tests_present else 1.0)
    if tests_present and test_rate < minimum_unit_test_pass_rate:
        reasons.append("unit_test_threshold_failed")
    return GateEvidence(passed=not reasons, reasons=reasons, metrics=metrics)


def summarize_trajectory(tokens: Iterable[TokenDistribution]) -> TrajectoryEvidence:
    token_list = list(tokens)
    if not token_list:
        return TrajectoryEvidence(available=False, error="empty_token_trajectory")
    advantages = [
        float(item.teacher_token_logprob - item.student_token_logprob) for item in token_list
    ]
    profile = HierarchicalTokenKLEvaluator().evaluate(token_list, semantic_valid=True)
    total_variation_values = []
    for token in token_list:
        total_variation = _total_variation(token)
        total_variation_values.append(total_variation)
    return TrajectoryEvidence(
        available=True,
        token_count=len(token_list),
        mean_teacher_student_nll_advantage=statistics.fmean(advantages),
        median_teacher_student_nll_advantage=statistics.median(advantages),
        teacher_win_fraction=sum(value > 0.0 for value in advantages) / len(advantages),
        mean_forward_kl=profile.total_kl,
        mean_total_variation=statistics.fmean(total_variation_values),
    )


def unavailable_trajectory(error: Exception | str) -> TrajectoryEvidence:
    return TrajectoryEvidence(available=False, error=str(error))


def _centered_gain(delta: float, scale: float) -> float:
    # Smoothly maps a zero delta to 0.5 and avoids one extreme metric dominating.
    return _clamp(0.5 + 0.5 * math.tanh(delta / scale))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _total_variation(token: TokenDistribution) -> float:
    vocabulary = sorted(set(token.student_logprobs) | set(token.teacher_logprobs))
    if not vocabulary:
        return 0.0
    student = _probabilities(token.student_logprobs, vocabulary)
    teacher = _probabilities(token.teacher_logprobs, vocabulary)
    return 0.5 * sum(abs(teacher[name] - student[name]) for name in vocabulary)


def _probabilities(logprobs: Dict[str, float], vocabulary: list[str]) -> Dict[str, float]:
    finite = [value for value in logprobs.values() if math.isfinite(value)]
    floor = min(finite) - 20.0 if finite else -20.0
    values = {name: logprobs.get(name, floor) for name in vocabulary}
    maximum = max(values.values())
    exponentials = {name: math.exp(value - maximum) for name, value in values.items()}
    denominator = sum(exponentials.values()) or 1.0
    return {name: value / denominator for name, value in exponentials.items()}
