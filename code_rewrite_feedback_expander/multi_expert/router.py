from __future__ import annotations

import math
from typing import Iterable, List

from .config import RoutingConfig
from .models import ExpertAssessment, RoutingDecision


class MultiExpertRouter:
    """Hard feasibility routing followed by reward/NLL evidence fusion."""

    def __init__(self, config: RoutingConfig, expert_order: Iterable[str]):
        self.config = config
        self.expert_order = {expert_id: index for index, expert_id in enumerate(expert_order)}

    def route(self, assessments: List[ExpertAssessment]) -> RoutingDecision:
        eligible = [
            item for item in assessments if item.gate.passed and item.trajectory.available
        ]
        if not eligible:
            return RoutingDecision(
                status="no_valid_expert",
                pseudo_method_label=None,
                selected_expert_id=None,
                expert_weights={item.expert_id: 0.0 for item in assessments},
                usable_for_training=False,
                reason="No candidate passed both the code hard gate and trajectory scoring.",
            )

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
        )


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
