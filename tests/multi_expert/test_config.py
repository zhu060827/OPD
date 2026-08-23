from __future__ import annotations

import unittest

from code_rewrite_feedback_expander.multi_expert.config import Stage1Config


def valid_raw_config():
    strategies = ["cot", "style", "ast", "variable", "control_flow"]
    return {
        "schema_version": "1.0",
        "experiment_name": "test",
        "seed": 7,
        "expected_expert_count": 5,
        "experts": [
            {"expert_id": f"expert_{strategy}", "strategy": strategy}
            for strategy in strategies
        ],
        "generation_backend": {"type": "mock"},
        "trajectory_backend": {"type": "mock"},
        "routing": {"top_k": 2},
    }


class Stage1ConfigTests(unittest.TestCase):
    def test_valid_five_expert_config(self):
        config = Stage1Config.from_dict(valid_raw_config())
        self.assertEqual(5, len(config.enabled_experts))

    def test_duplicate_expert_ids_are_rejected(self):
        raw = valid_raw_config()
        raw["experts"][1]["expert_id"] = raw["experts"][0]["expert_id"]
        with self.assertRaisesRegex(ValueError, "expert_id"):
            Stage1Config.from_dict(raw)

    def test_wrong_expert_count_is_rejected(self):
        raw = valid_raw_config()
        raw["experts"].pop()
        with self.assertRaisesRegex(ValueError, "Expected 5"):
            Stage1Config.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
