import argparse
import json
import tempfile
import unittest
from pathlib import Path

from teacher_training.compare_base_lora import (
    _append_jsonl_durable,
    _code_contract,
    _extract_code,
    _generation_limit,
    _prepare_resume,
    _result_row,
    _run_spec,
)


class CompareBaseLoraTests(unittest.TestCase):
    def test_generation_limit_is_fixed_without_reference_length(self):
        self.assertEqual(2048, _generation_limit(2048))
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            _generation_limit(0)

    def test_plain_python_is_parsed_without_protocol_compliance(self):
        protocol, parse_pass, code, method = _code_contract("x = 1\n", "python")
        self.assertFalse(protocol)
        self.assertTrue(parse_pass)
        self.assertEqual(code, "x = 1")
        self.assertEqual(method, "plain_text")

    def test_markdown_python_is_extracted(self):
        protocol, parse_pass, code, method = _code_contract("```python\nx = 1\n```", "python")
        self.assertFalse(protocol)
        self.assertTrue(parse_pass)
        self.assertEqual(code, "x = 1")
        self.assertEqual(method, "markdown_fence")

    def test_code_tag_reports_protocol_compliance(self):
        protocol, parse_pass, code, method = _code_contract("<code>\nx = 1\n</code>", "python")
        self.assertTrue(protocol)
        self.assertTrue(parse_pass)
        self.assertEqual(code, "x = 1")
        self.assertEqual(method, "code_tag")

    def test_reference_and_prediction_can_be_compared_without_wrappers(self):
        reference, _, _ = _extract_code("<code>\nx = 1\n</code>")
        prediction, _, _ = _extract_code("```python\nx = 1\n```")
        self.assertEqual(reference.rstrip(), prediction.rstrip())

    def test_resume_skips_durably_completed_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_file = root / "test.jsonl"
            test_file.write_text('{"source_id":"one"}\n', encoding="utf-8")
            output = root / "predictions.jsonl"
            args = argparse.Namespace(
                base_model=str(root / "base"), adapter=str(root / "adapter"),
                test_file=str(test_file), limit=1, max_new_tokens=128, thinking=False,
            )
            spec = _run_spec(args)
            rows, completed = _prepare_resume(output, spec)
            self.assertEqual([], rows)
            self.assertEqual(set(), completed)
            _append_jsonl_durable(output, {
                "source_id": "one",
                "base_output": "base",
                "lora_output": "ok",
                "reference_output": "reference",
            })
            rows, completed = _prepare_resume(output, spec)
            self.assertEqual({"one"}, completed)
            self.assertEqual("ok", rows[0]["lora_output"])

    def test_resume_rejects_changed_generation_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_file = root / "test.jsonl"
            test_file.write_text('{"source_id":"one"}\n', encoding="utf-8")
            output = root / "predictions.jsonl"
            args = argparse.Namespace(
                base_model=str(root / "base"), adapter=str(root / "adapter"),
                test_file=str(test_file), limit=1, max_new_tokens=128, thinking=False,
            )
            _prepare_resume(output, _run_spec(args))
            args.max_new_tokens = 256
            with self.assertRaisesRegex(ValueError, "拒绝混合续跑"):
                _prepare_resume(output, _run_spec(args))

    def test_result_records_token_limits_and_problem_family(self):
        row = {
            "source_id": "problem:variant-0",
            "problem_id": "problem",
            "domain": "identifier",
            "language": "python",
            "messages": [{"role": "assistant", "content": "<code>\nvalue = 1\n</code>"}],
        }
        result = _result_row(
            row,
            ("<code>\nvalue = 1\n</code>", 12, 12, True),
            ("<code>\nvalue = 1\n</code>", 10, 12, False),
            False,
        )
        self.assertEqual("problem", result["problem_id"])
        self.assertEqual(12, result["base_generated_tokens"])
        self.assertTrue(result["base_max_tokens_reached"])
        self.assertFalse(result["lora_max_tokens_reached"])
        self.assertTrue(result["base_protocol_pass"])
        self.assertFalse(result["base_suspected_truncation"])


if __name__ == "__main__":
    unittest.main()
