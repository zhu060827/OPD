from __future__ import annotations

import unittest

import torch

from code_rewrite_feedback_expander.multi_expert.opd_reward import (
    compute_topk_opd_reward,
    gap_following_scales,
)


class OpenMOPDRewardTests(unittest.TestCase):
    def test_reward_is_gap_times_normalized_student_probability(self):
        student = torch.log(torch.tensor([[[0.75, 0.25]]]))
        teacher = torch.log(torch.tensor([[[0.5, 0.5]]]))
        reward, metrics = compute_topk_opd_reward(student, teacher, torch.tensor([[1.0]]))
        expected = (teacher - student) * torch.softmax(student, dim=-1)
        self.assertTrue(torch.allclose(reward, expected))
        self.assertEqual(1.0, metrics["valid_token_count"])

    def test_padding_tokens_are_zero(self):
        student = torch.zeros((1, 2, 2))
        teacher = torch.ones((1, 2, 2))
        reward, _ = compute_topk_opd_reward(student, teacher, torch.tensor([[1.0, 0.0]]))
        self.assertTrue(torch.all(reward[:, 1] == 0))

    def test_gap_following_is_relative_and_clipped(self):
        scales = gap_following_scales({"a": 1.0, "b": 3.0}, alpha=1.0)
        self.assertAlmostEqual(0.5, scales["a"])
        self.assertAlmostEqual(1.5, scales["b"])


if __name__ == "__main__":
    unittest.main()
