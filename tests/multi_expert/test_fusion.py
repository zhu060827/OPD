from __future__ import annotations

import unittest

from code_rewrite_feedback_expander.multi_expert.fusion import (
    build_expert_weight_matrix,
    compute_expert_share_metrics,
    compute_teacher_conflict_metrics,
    compute_token_share_loss_weights,
    route_aligned_teacher_logprobs,
)


class OpenMOPDBridgeTests(unittest.TestCase):
    def test_hard_route_selects_exact_teacher_tensor(self):
        teachers = {
            "cot": [[[1.0, 2.0]], [[3.0, 4.0]]],
            "ast": [[[10.0, 20.0]], [[30.0, 40.0]]],
        }
        weights = build_expert_weight_matrix(["cot", "ast"], ["cot", "ast"])
        routed = route_aligned_teacher_logprobs(teachers, ["cot", "ast"], weights)
        self.assertEqual([[[1.0, 2.0]], [[30.0, 40.0]]], routed)

    def test_unknown_label_can_be_rejected_for_formal_runs(self):
        with self.assertRaisesRegex(ValueError, "Unknown expert label"):
            build_expert_weight_matrix([None], ["cot", "ast"], unknown_policy="error")

    def test_token_share_weights_hit_equal_target(self):
        labels = ["long", "short"]
        tokens = [100, 10]
        weights = compute_token_share_loss_weights(labels, tokens)
        weighted = [weight * count for weight, count in zip(weights, tokens)]
        self.assertAlmostEqual(weighted[0], weighted[1])

    def test_share_metrics_distinguish_prompt_token_and_budget_share(self):
        metrics = compute_expert_share_metrics(
            ["long", "short"], [100, 10], [0.1, 1.0]
        )
        self.assertEqual(0.5, metrics["short"]["prompt_share"])
        self.assertLess(metrics["short"]["token_share"], 0.1)
        self.assertAlmostEqual(0.5, metrics["short"]["effective_budget_share"])

    def test_teacher_conflict_uses_only_valid_aligned_tokens(self):
        metrics = compute_teacher_conflict_metrics(
            {
                "a": [[-0.1, -0.2, -9.0]],
                "b": [[-0.2, -1.5, 9.0]],
            },
            [[True, True, False]],
            threshold_nats=1.0,
        )
        self.assertEqual(2.0, metrics["valid_token_count"])
        self.assertEqual(0.5, metrics["conflict_fraction"])


if __name__ == "__main__":
    unittest.main()
