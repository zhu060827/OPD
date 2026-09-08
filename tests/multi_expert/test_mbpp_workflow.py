from __future__ import annotations

import unittest

from code_rewrite_feedback_expander.multi_expert.mbpp_workflow import (
    evaluate_with_repairs,
    mbpp_split,
    split_handoff_records,
)
from code_rewrite_feedback_expander.multi_expert.config import RoutingConfig
from code_rewrite_feedback_expander.multi_expert.router import MultiExpertRouter

from .test_router import assessment


class MBPPWorkflowTests(unittest.TestCase):
    def test_fixed_partitions(self):
        self.assertEqual("prompt", mbpp_split({"task_id": "MBPP/1"}))
        self.assertEqual("test", mbpp_split({"task_id": "MBPP/510"}))
        self.assertEqual("validation", mbpp_split({"task_id": "MBPP/511"}))
        self.assertEqual("train", mbpp_split({"task_id": "MBPP/974"}))

    def test_partial_split_for_development(self):
        groups = split_handoff_records(
            [{"task_id": "MBPP/1"}, {"task_id": "MBPP/11"}, {"task_id": "MBPP/601"}],
            require_complete=False,
        )
        self.assertEqual(1, len(groups["prompt"]))
        self.assertEqual(1, len(groups["test"]))
        self.assertEqual(1, len(groups["train"]))

    def test_missing_trajectory_fallback_assigns_domain(self):
        item = assessment("expert_cot", 0.5, 0.1)
        item.trajectory.available = False
        decision = MultiExpertRouter(
            RoutingConfig(
                policy="three_tier",
                calibration={"expert_cot": {"location": 0.0, "scale": 1.0}},
                top_k=1,
                fallback_expert_id="expert_cot",
                fallback_on_missing_trajectory=True,
            ),
            ["expert_cot"],
        ).route([item])
        self.assertEqual("cot", decision.pseudo_method_label)
        self.assertTrue(decision.usable_for_training)
        self.assertEqual(0.1, decision.opd_sample_weight)

    def test_balanced_hash_fallback_is_deterministic_and_uses_known_domains(self):
        items = []
        calibration = {}
        for name in ("cot", "style", "ast", "variable", "control_flow"):
            item = assessment(f"expert_{name}", 0.5, 0.1)
            item.trajectory.available = False
            items.append(item)
            calibration[f"expert_{name}"] = {"location": 0.0, "scale": 1.0}
        router = MultiExpertRouter(
            RoutingConfig(
                policy="three_tier",
                calibration=calibration,
                fallback_on_missing_trajectory=True,
                missing_trajectory_fallback_policy="balanced_hash",
            ),
            [item.expert_id for item in items],
        )
        first = router.route(items, routing_key="MBPP/601")
        second = router.route(items, routing_key="MBPP/601")
        self.assertEqual(first.selected_expert_id, second.selected_expert_id)
        self.assertIn(first.pseudo_method_label, calibration_domain_names(calibration))


def calibration_domain_names(calibration):
    return {name.removeprefix("expert_") for name in calibration}

    def test_repair_rates_are_cumulative(self):
        calls = {}
        def generate(record, feedback):
            key = record["task_id"]
            calls[key] = calls.get(key, 0) + 1
            return str(calls[key])
        def verify(record, code):
            passed = int(code) >= record["success_attempt"]
            return passed, "fix"
        result = evaluate_with_repairs(
            [{"task_id": "a", "success_attempt": 1}, {"task_id": "b", "success_attempt": 3}],
            generate,
            verify,
            max_repairs=3,
        )
        self.assertEqual([1, 1, 2, 2], result["cumulative_pass_counts"])


if __name__ == "__main__":
    unittest.main()
