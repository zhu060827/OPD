from __future__ import annotations

import unittest

from code_rewrite_feedback_expander.multi_expert.calibration import (
    fit_robust_advantage_calibration,
)


class CalibrationTests(unittest.TestCase):
    def test_fits_per_teacher_robust_location_and_scale(self):
        records = []
        for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
            records.append(
                {
                    "expert_assessments": [
                        {
                            "expert_id": "expert_a",
                            "trajectory": {
                                "available": True,
                                "mean_teacher_student_nll_advantage": 10.0 + offset,
                            },
                        },
                        {
                            "expert_id": "expert_b",
                            "trajectory": {
                                "available": True,
                                "mean_teacher_student_nll_advantage": 2.0 + 2.0 * offset,
                            },
                        },
                    ]
                }
            )
        fitted = fit_robust_advantage_calibration(
            records, ["expert_a", "expert_b"], min_samples=5
        )
        self.assertEqual(10.0, fitted["expert_a"]["location"])
        self.assertGreater(fitted["expert_b"]["scale"], fitted["expert_a"]["scale"])

    def test_rejects_too_few_samples(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            fit_robust_advantage_calibration([], ["expert_a"], min_samples=2)


if __name__ == "__main__":
    unittest.main()
