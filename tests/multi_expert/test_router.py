from __future__ import annotations

import unittest

from code_rewrite_feedback_expander.models import RewriteCandidate
from code_rewrite_feedback_expander.multi_expert.config import RoutingConfig
from code_rewrite_feedback_expander.multi_expert.models import (
    ExpertAssessment,
    GateEvidence,
    RewardEvidence,
    TrajectoryEvidence,
)
from code_rewrite_feedback_expander.multi_expert.router import MultiExpertRouter


def assessment(expert_id: str, reward: float, advantage: float, passed: bool = True):
    strategy = expert_id.removeprefix("expert_")
    return ExpertAssessment(
        expert_id=expert_id,
        strategy=strategy,
        candidate=RewriteCandidate("pass", [], "", strategy, "pass"),
        semantic_result={},
        quality_result={},
        gate=GateEvidence(passed=passed, reasons=[], metrics={}),
        reward=RewardEvidence(reward, reward, reward, 1.0, 1.0, {}, {}),
        trajectory=TrajectoryEvidence(
            available=passed,
            token_count=1,
            mean_teacher_student_nll_advantage=advantage,
        ),
    )


class MultiExpertRouterTests(unittest.TestCase):
    def test_nll_advantage_can_select_top1(self):
        items = [
            assessment("expert_cot", 0.5, 0.1),
            assessment("expert_ast", 0.5, 0.9),
            assessment("expert_style", 0.5, 0.2),
        ]
        router = MultiExpertRouter(
            RoutingConfig(reward_weight=0.0, nll_advantage_weight=1.0, top_k=2),
            [item.expert_id for item in items],
        )
        decision = router.route(items)
        self.assertEqual("expert_ast", decision.selected_expert_id)
        self.assertEqual("ast", decision.pseudo_method_label)

    def test_top2_weights_sum_to_one_and_failed_expert_is_zero(self):
        items = [
            assessment("expert_cot", 0.8, 0.8),
            assessment("expert_ast", 0.7, 0.7),
            assessment("expert_style", 1.0, 1.0, passed=False),
        ]
        decision = MultiExpertRouter(
            RoutingConfig(top_k=2), [item.expert_id for item in items]
        ).route(items)
        self.assertAlmostEqual(1.0, sum(item["weight"] for item in decision.top_k))
        self.assertEqual(1.0, decision.expert_weights[decision.selected_expert_id])
        self.assertEqual(0.0, decision.expert_weights["expert_style"])

    def test_no_valid_expert_never_forces_a_label(self):
        items = [assessment("expert_cot", 1.0, 1.0, passed=False)]
        decision = MultiExpertRouter(RoutingConfig(top_k=1), ["expert_cot"]).route(items)
        self.assertEqual("no_valid_expert", decision.status)
        self.assertIsNone(decision.pseudo_method_label)
        self.assertFalse(decision.usable_for_training)


if __name__ == "__main__":
    unittest.main()
