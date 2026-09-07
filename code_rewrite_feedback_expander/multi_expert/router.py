from __future__ import annotations

import math
from typing import Iterable, List

from .config import RoutingConfig
from .models import ExpertAssessment, RoutingDecision


class MultiExpertRouter:
    """Route known labels directly and unknown labels by calibrated OPD evidence."""

    def __init__(self, config: RoutingConfig, expert_order: Iterable[str]):
        self.config = config
        self.expert_order = {expert_id: index for index, expert_id in enumerate(expert_order)}

    def recorded_label(self, metadata: dict) -> str | None:
        for field_name in self.config.recorded_label_fields:
            value = metadata.get(field_name)
            if value is None:
                continue
            value = str(value).strip()
            if value in self.expert_order:
                return value
            expert_id = next(
                (key for key in self.expert_order if key.removeprefix("expert_") == value),
                None,
            )
            if expert_id:
                return expert_id
            raise ValueError(f"Unknown recorded routing label {value!r} in {field_name}")
        return None

    def route(
        self,
        assessments: List[ExpertAssessment],
        recorded_expert_id: str | None = None,
    ) -> RoutingDecision:
        if recorded_expert_id is not None:
            return self._route_recorded_label(assessments, recorded_expert_id)
        if self.config.policy == "heuristic_ablation":
            return self._route_heuristic_ablation(assessments)
        return self._route_calibrated_opd(assessments)

    def _route_recorded_label(
        self, assessments: List[ExpertAssessment], expert_id: str
    ) -> RoutingDecision:
        selected = next((item for item in assessments if item.expert_id == expert_id), None)
        if selected is None:
            raise ValueError(f"No assessment was produced for recorded expert {expert_id}")
        weights = {key: float(key == expert_id) for key in self.expert_order}
        # A recorded route is independent of whether the current Student
        # completion is correct.  Incorrect on-policy trajectories are valid OPD
        # training states; only missing/alignment-invalid trajectories block it.
        usable = selected.trajectory.available
        return RoutingDecision(
            status="recorded_label" if usable else "recorded_label_invalid",
            pseudo_method_label=selected.strategy if usable else None,
            selected_expert_id=expert_id,
            top_k=[self._diagnostic(selected, 1, 1.0)],
            expert_weights=weights,
            margin=1.0,
            usable_for_training=usable,
            reason=(
                "Used the augmentation provenance label; no Teacher selection was inferred."
                if usable
                else "The recorded route had no valid aligned token trajectory."
            ),
            routing_source="recorded_method_label",
        )

    def _route_calibrated_opd(self, assessments: List[ExpertAssessment]) -> RoutingDecision:
        # Semantic correctness is recorded separately from Teacher routing.
        # Every Teacher must score the same available Student trajectory.
        eligible = [item for item in assessments if item.trajectory.available]
        if not eligible:
            return self._no_valid(assessments)
        for item in eligible:
            stats = self.config.calibration[item.expert_id]
            location = float(stats.get("location", 0.0))
            scale = float(stats["scale"])
            item.normalized_nll_advantage = (
                item.trajectory.mean_teacher_student_nll_advantage - location
            ) / scale
            item.routing_score = item.normalized_nll_advantage
        eligible.sort(
            key=lambda item: (
                item.routing_score,
                -self.expert_order.get(item.expert_id, 10**6),
            ),
            reverse=True,
        )
        for rank, item in enumerate(eligible, start=1):
            item.rank = rank
        selected = eligible[: self.config.top_k]
        soft_weights = _softmax(
            [item.routing_score for item in selected], self.config.softmax_temperature
        )
        top_k = [
            self._diagnostic(item, item.rank or rank, weight)
            for rank, (item, weight) in enumerate(zip(selected, soft_weights), start=1)
        ]
        margin = eligible[0].routing_score - eligible[1].routing_score if len(eligible) > 1 else 1.0
        low_confidence = margin < self.config.minimum_margin
        if low_confidence and self.config.abstain_on_low_confidence:
            if self.config.fallback_expert_id:
                fallback = next(
                    (item for item in eligible if item.expert_id == self.config.fallback_expert_id),
                    None,
                )
                if fallback is not None:
                    weights = {key: float(key == fallback.expert_id) for key in self.expert_order}
                    return RoutingDecision(
                        status="fallback_low_confidence",
                        pseudo_method_label=fallback.strategy,
                        selected_expert_id=fallback.expert_id,
                        top_k=top_k,
                        expert_weights=weights,
                        margin=margin,
                        usable_for_training=True,
                        reason="Calibrated margin was low; used the configured fallback route.",
                        routing_source="calibrated_opd_fallback",
                    )
            return RoutingDecision(
                status="abstained_low_confidence",
                pseudo_method_label=None,
                selected_expert_id=eligible[0].expert_id,
                top_k=top_k,
                expert_weights={key: 0.0 for key in self.expert_order},
                margin=margin,
                usable_for_training=False,
                reason="Calibrated Top-1/Top-2 margin was too small; no pseudo-label was fabricated.",
                routing_source="calibrated_opd_abstention",
            )
        weights = {key: float(key == eligible[0].expert_id) for key in self.expert_order}
        return RoutingDecision(
            status="routed_calibrated_opd",
            pseudo_method_label=eligible[0].strategy,
            selected_expert_id=eligible[0].expert_id,
            top_k=top_k,
            expert_weights=weights,
            margin=margin,
            usable_for_training=True,
            reason="Selected Top-1 from calibrated Teacher advantage on one shared completion.",
            routing_source="calibrated_same_trajectory_opd",
        )

    def _route_heuristic_ablation(self, assessments: List[ExpertAssessment]) -> RoutingDecision:
        eligible = [
            item for item in assessments if item.gate.passed and item.trajectory.available
        ]
        if not eligible:
            return self._no_valid(assessments)

        reward_values = [item.reward.total for item in eligible]
        advantage_values = [
            item.trajectory.mean_teacher_student_nll_advantage for item in eligible
        ]
        normalized_reward = _minmax(reward_values)
        normalized_advantage = _minmax(advantage_values)
        weight_sum = self.config.reward_weight + self.config.nll_advantage_weight
        for item, reward, advantage in zip(eligible, normalized_reward, normalized_advantage):
            item.normalized_reward = reward
            item.normalized_nll_advantage = advantage
            item.routing_score = (
                self.config.reward_weight * reward
                + self.config.nll_advantage_weight * advantage
            ) / weight_sum

        eligible.sort(
            key=lambda item: (
                item.routing_score,
                item.trajectory.mean_teacher_student_nll_advantage,
                item.reward.total,
                -self.expert_order.get(item.expert_id, 10**6),
            ),
            reverse=True,
        )
        for rank, item in enumerate(eligible, start=1):
            item.rank = rank

        selected = eligible[: self.config.top_k]
        weights = _softmax(
            [item.routing_score for item in selected], self.config.softmax_temperature
        )
        top_k = [
            {
                "rank": item.rank,
                "expert_id": item.expert_id,
                "strategy": item.strategy,
                "weight": weight,
                "routing_score": item.routing_score,
                "reward_score": item.reward.total,
                "mean_nll_advantage": item.trajectory.mean_teacher_student_nll_advantage,
            }
            for item, weight in zip(selected, weights)
        ]
        # The first formal experiment follows Open-MOPD hard routing. Top-k
        # softmax values remain in ``top_k`` strictly as diagnostics/ablation data.
        expert_weights = {item.expert_id: 0.0 for item in assessments}
        expert_weights[eligible[0].expert_id] = 1.0

        margin = (
            eligible[0].routing_score - eligible[1].routing_score
            if len(eligible) > 1
            else 1.0
        )
        low_confidence = margin < self.config.minimum_margin
        usable = not (low_confidence and self.config.abstain_on_low_confidence)
        return RoutingDecision(
            status="low_confidence" if low_confidence else "routed",
            pseudo_method_label=eligible[0].strategy,
            selected_expert_id=eligible[0].expert_id,
            top_k=top_k,
            expert_weights=expert_weights,
            margin=margin,
            usable_for_training=usable,
            reason=(
                "Top-1/Top-2 margin is below the configured confidence threshold."
                if low_confidence
                else "Top-1 expert selected after hard gating and evidence fusion."
            ),
            router_design="heuristic_reward_and_nll_ablation",
            routing_source="legacy_heuristic_ablation",
        )

    def _no_valid(self, assessments: List[ExpertAssessment]) -> RoutingDecision:
        return RoutingDecision(
            status="no_valid_expert",
            pseudo_method_label=None,
            selected_expert_id=None,
            expert_weights={key: 0.0 for key in self.expert_order},
            usable_for_training=False,
            reason="No expert produced a valid aligned token trajectory.",
            routing_source="none",
        )

    @staticmethod
    def _diagnostic(item: ExpertAssessment, rank: int, weight: float) -> dict:
        return {
            "rank": rank,
            "expert_id": item.expert_id,
            "strategy": item.strategy,
            "weight": weight,
            "routing_score": item.routing_score,
            "reward_score": item.reward.total,
            "mean_nll_advantage": item.trajectory.mean_teacher_student_nll_advantage,
            "calibrated_nll_advantage": item.normalized_nll_advantage,
        }


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _softmax(values: List[float], temperature: float) -> List[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp((value - maximum) / temperature) for value in values]
    denominator = sum(exponentials) or 1.0
    return [value / denominator for value in exponentials]
