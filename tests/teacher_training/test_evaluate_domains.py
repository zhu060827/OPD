import unittest
from types import SimpleNamespace
from unittest.mock import patch

from teacher_training.evaluate_domains import (
    evaluate_row,
    corpus_metrics,
    edit_scope_pass,
    extract_code,
    function_signatures,
    levenshtein_distance,
    recovered_identifier,
    subtoken_f1,
)


class DomainEvaluationTests(unittest.TestCase):
    def test_edit_scope_accepts_only_declared_return_rewrite(self):
        source = "def f(x):\n    keep = x + 1\n    return x * 2\n"
        prediction = "def f(x):\n    keep = x + 1\n    computed_value = x * 2\n    return computed_value\n"
        self.assertTrue(edit_scope_pass(source, prediction, {"start_line": 3, "end_line": 3}))

    def test_edit_scope_rejects_change_outside_declared_return(self):
        source = "def f(x):\n    keep = x + 1\n    return x * 2\n"
        prediction = "def f(x):\n    retained = x + 1\n    computed_value = x * 2\n    return computed_value\n"
        self.assertFalse(edit_scope_pass(source, prediction, {"start_line": 3, "end_line": 3}))

    @patch("teacher_training.evaluate_domains.bleu_score", return_value=100.0)
    @patch("teacher_training.evaluate_domains.codebleu_score", return_value=1.0)
    def test_local_structure_scope_failure_fails_structural_gate(self, _codebleu, _bleu):
        reference = {
            "source_id": "hard-one", "domain": "local_structure", "language": "python",
            "difficulty": "hard", "source_code": "def f(x):\n    keep = x + 1\n    return x * 2\n",
            "target_code": "def f(x):\n    keep = x + 1\n    value = x * 2\n    return value\n",
            "tests": [], "transformation_evidence": {"source_span": {"start_line": 3, "end_line": 3}},
        }
        prediction = {"prediction": "<code>\ndef f(x):\n    changed = x + 1\n    value = x * 2\n    return value\n</code>"}
        result = evaluate_row(reference, prediction, 1, "prediction")
        self.assertEqual("hard", result["difficulty"])
        self.assertEqual(0.0, result["edit_scope_pass"])
        self.assertEqual(0.0, result["structural_hard_gate_pass"])
        self.assertIsNone(result["semantic_gate_pass"])

    def test_extracts_code_protocol(self):
        self.assertEqual("def f():\n    return 1", extract_code("<code>\ndef f():\n    return 1\n</code>"))

    def test_extracts_markdown_fence(self):
        self.assertEqual("def f():\n    return 1", extract_code("```python\ndef f():\n    return 1\n```"))

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

    def test_explicit_prediction_field_is_used(self):
        reference = {
            "source_id": "format-one",
            "domain": "formatting",
            "language": "python",
            "source_code": "def f(x):\n  return x + 1",
            "target_code": "def f(x):\n    return x + 1",
            "tests": [],
        }
        prediction = {
            "base_output": "<code>\ndef f(x):\n  return x + 1\n</code>",
            "lora_output": "<code>\ndef f(x):\n    return x + 1\n</code>",
        }
        base = evaluate_row(reference, prediction, 1, "base_output")
        lora = evaluate_row(reference, prediction, 1, "lora_output")
        self.assertEqual(0.0, base["primary_exact_formatting_repair"])
        self.assertEqual(1.0, lora["primary_exact_formatting_repair"])
        self.assertEqual(1.0, base["ast_identity_pass"])
        self.assertEqual(1.0, lora["ast_identity_pass"])

    def test_formatting_ast_identity_rejects_semantic_structure_change(self):
        reference = {
            "source_id": "format-ast-change",
            "domain": "formatting",
            "language": "python",
            "source_code": "def f(x):\n    return x + 1",
            "target_code": "def f(x):\n    return x + 1",
            "tests": [],
        }
        prediction = {"prediction": "<code>\ndef f(x):\n    return x + 2\n</code>"}
        result = evaluate_row(reference, prediction, 1, "prediction")
        self.assertEqual(0.0, result["ast_identity_pass"])

    @patch.dict("sys.modules", {"sacrebleu": SimpleNamespace(corpus_bleu=lambda *args, **kwargs: SimpleNamespace(score=100.0))})
    @patch("teacher_training.evaluate_domains.codebleu_details")
    def test_local_structure_corpus_metrics_use_whole_test_set(self, mocked_codebleu):
        mocked_codebleu.return_value = {
            "codebleu": 0.8,
            "ngram_match_score": 0.7,
            "weighted_ngram_match_score": 0.75,
            "syntax_match_score": 0.85,
            "dataflow_match_score": 0.9,
        }
        references = [
            {"source_id": "one", "domain": "local_structure", "language": "python", "target_code": "x = 1"},
            {"source_id": "two", "domain": "local_structure", "language": "python", "target_code": "y = 2"},
        ]
        predictions = {
            "one": {"prediction": "<code>\nx = 1\n</code>"},
            "two": {"prediction": "<code>\ny = 2\n</code>"},
        }
        metrics = corpus_metrics(references, predictions, "prediction")
        mocked_codebleu.assert_called_once_with(["x = 1", "y = 2"], ["x = 1", "y = 2"], "python")
        self.assertEqual(0.8, metrics["corpus_codebleu"]["value"])
        self.assertEqual(100.0, metrics["corpus_bleu"]["value"])
        self.assertEqual(0.9, metrics["codebleu_dataflow_match_score"]["value"])
        self.assertEqual(2, metrics["corpus_codebleu"]["n"])


if __name__ == "__main__":
    unittest.main()
