import unittest

from teacher_training.evaluate_domains import (
    extract_code,
    function_signatures,
    levenshtein_distance,
    recovered_identifier,
    subtoken_f1,
)


class DomainEvaluationTests(unittest.TestCase):
    def test_extracts_code_protocol(self):
        self.assertEqual("def f():\n    return 1", extract_code("<code>\ndef f():\n    return 1\n</code>"))

    def test_identifier_recovery_uses_corresponding_ast_positions(self):
        source = "def total(VAR_0):\n    return VAR_0 + 1"
        prediction = "def total(numbers):\n    return numbers + 1"
        self.assertEqual("numbers", recovered_identifier(source, prediction, "VAR_0"))

    def test_identifier_recovery_rejects_inconsistent_renames(self):
        source = "def total(VAR_0):\n    return VAR_0 + 1"
        prediction = "def total(numbers):\n    return values + 1"
        self.assertEqual("", recovered_identifier(source, prediction, "VAR_0"))

    def test_subtoken_f1(self):
        self.assertEqual(1.0, subtoken_f1("totalValue", "total_value"))

    def test_levenshtein_distance(self):
        self.assertEqual(3, levenshtein_distance("kitten", "sitting"))

    def test_function_signature_is_reproducible(self):
        before = "def f(x, *, limit=1):\n    return x"
        after = "def f(x, *, limit=1):\n    return x + limit"
        self.assertEqual(function_signatures(before), function_signatures(after))

    def test_function_signature_detects_default_change(self):
        before = "def f(x=1):\n    return x"
        after = "def f(x=2):\n    return x"
        self.assertNotEqual(function_signatures(before), function_signatures(after))


if __name__ == "__main__":
    unittest.main()
