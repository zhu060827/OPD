from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .models import OPDTokenProfile, TokenDistribution


ASPECTS = ["cot", "style", "ast", "variable", "control_flow"]


class HierarchicalTokenKLEvaluator:
    def evaluate(self, tokens: Iterable[TokenDistribution], semantic_valid: bool) -> OPDTokenProfile:
        token_list = list(tokens)
        kl_totals = {aspect: 0.0 for aspect in ASPECTS}
        nll_reduction_totals = {aspect: 0.0 for aspect in ASPECTS}
        tv_totals = {aspect: 0.0 for aspect in ASPECTS}
        weights = {aspect: 0.0 for aspect in ASPECTS}
        total_kl = 0.0

        for token in token_list:
            token_kl, total_variation = self._token_distances(token)
            token_nll_reduction = token.teacher_token_logprob - token.student_token_logprob
            total_kl += token_kl
            aspect_weights = token.aspect_weights or {"ast": 1.0}
            for aspect, weight in aspect_weights.items():
                if aspect not in weights or weight <= 0.0:
                    continue
                kl_totals[aspect] += token_kl * weight
                nll_reduction_totals[aspect] += token_nll_reduction * weight
                tv_totals[aspect] += total_variation * weight
                weights[aspect] += weight

        aspect_kl = self._weighted_means(kl_totals, weights)
        aspect_total_variation = self._weighted_means(tv_totals, weights)
        aspect_distribution_overlap = {
            aspect: 1.0 - aspect_total_variation[aspect] for aspect in ASPECTS
        }
        aspect_token_nll_reduction = self._weighted_means(nll_reduction_totals, weights)
        strategy_ranking = sorted(
            ASPECTS,
            key=lambda aspect: (
                aspect_kl[aspect],
                aspect_token_nll_reduction[aspect],
                -aspect_total_variation[aspect],
            ),
            reverse=True,
        ) if semantic_valid and token_list else []
        dominant_aspect = strategy_ranking[0] if strategy_ranking and aspect_kl[strategy_ranking[0]] > 0.0 else ""
        return OPDTokenProfile(
            total_kl=total_kl / len(token_list) if token_list else 0.0,
            aspect_kl=aspect_kl,
            aspect_total_variation=aspect_total_variation,
            aspect_distribution_overlap=aspect_distribution_overlap,
            aspect_token_nll_reduction=aspect_token_nll_reduction,
            strategy_ranking=strategy_ranking,
            dominant_aspect=dominant_aspect,
            next_strategy=dominant_aspect,
            token_count=len(token_list),
        )

    def _token_distances(self, token: TokenDistribution) -> tuple[float, float]:
        vocabulary = sorted(set(token.student_logprobs) | set(token.teacher_logprobs))
        if not vocabulary:
            return 0.0, 0.0
        student = self._probabilities(token.student_logprobs, vocabulary)
        teacher = self._probabilities(token.teacher_logprobs, vocabulary)
        kl = sum(
            teacher[token_name] * math.log(teacher[token_name] / max(student[token_name], 1e-12))
            for token_name in vocabulary
        )
        total_variation = 0.5 * sum(
            abs(teacher[token_name] - student[token_name]) for token_name in vocabulary
        )
        return max(0.0, kl), min(1.0, max(0.0, total_variation))

    def _probabilities(self, logprobs: Dict[str, float], vocabulary: List[str]) -> Dict[str, float]:
        finite_values = [value for value in logprobs.values() if math.isfinite(value)]
        floor = min(finite_values) - 20.0 if finite_values else -20.0
        values = {token: logprobs.get(token, floor) for token in vocabulary}
        maximum = max(values.values())
        exponentials = {token: math.exp(value - maximum) for token, value in values.items()}
        total = sum(exponentials.values()) or 1.0
        return {token: value / total for token, value in exponentials.items()}

    def _weighted_means(self, totals: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
        return {
            aspect: totals[aspect] / weights[aspect] if weights[aspect] > 0.0 else 0.0
            for aspect in ASPECTS
        }
