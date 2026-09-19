import unittest

from teacher_training.evaluate_teachers import build_matrices


class CrossDomainMatrixTests(unittest.TestCase):
    def test_uses_target_domain_primary_metric(self):
        row = {
            "teacher_domain": "identifier",
            "sample_domain": "formatting",
            "primary_exact_formatting_repair": 1.0,
            "parse_pass": True,
            "signature_pass": True,
            "compile_pass": True,
            "test_pass": True,
        }
        result = build_matrices([row])
        self.assertEqual(1.0, result["primary_metric_matrix"]["identifier"]["formatting"])
        self.assertNotIn("diagonal_expected_to_be_highest", result)

    def test_rejects_generic_score(self):
        row = {
            "teacher_domain": "identifier",
            "sample_domain": "formatting",
            "score": 1.0,
            "parse_pass": True,
            "signature_pass": True,
            "compile_pass": True,
            "test_pass": True,
        }
        with self.assertRaisesRegex(ValueError, "primary_exact_formatting_repair"):
            build_matrices([row])


if __name__ == "__main__":
    unittest.main()
