from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from code_rewrite_feedback_expander.io_utils import read_jsonl
from code_rewrite_feedback_expander.models import RewriteCandidate
from code_rewrite_feedback_expander.semantic import SemanticEquivalenceChecker


class InputNormalizationTests(unittest.TestCase):
    def test_mbpp_reference_code_is_the_rewrite_source(self):
        row = {
            "task_id": "MBPP/example",
            "prompt": "Add two values.",
            "starter_code": "def add(a, b):\n    pass",
            "reference_code": "def add(a, b):\n    return a + b",
            "tests": "assert add(2, 3) == 5",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = read_jsonl(str(path))[0]
        self.assertEqual(row["reference_code"], record.code)

    def test_humaneval_test_script_is_not_split_into_characters(self):
        row = {
            "task_id": "example",
            "prompt": "Add two values.",
            "code": "def add(a, b):\n    return a + b",
            "tests": "assert add(2, 3) == 5\nassert add(0, 1) == 1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = read_jsonl(str(path))
        self.assertEqual(1, len(records[0].tests))
        self.assertIn("assert add(2, 3) == 5", records[0].tests[0])

    def test_multiline_humaneval_harness_executes_as_one_indented_block(self):
        row = {
            "task_id": "example",
            "prompt": "Add two values.",
            "code": "def add(a, b):\n    return a + b",
            "tests": (
                "def check(candidate):\n"
                "    assert candidate(2, 3) == 5\n"
                "    assert candidate(0, 1) == 1\n\n"
                "check(add)"
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = read_jsonl(str(path))[0]
        candidate = RewriteCandidate(
            code="# harmless rewrite\n" + record.code,
            reasoning=[],
            rationale="test",
            strategy="cot",
            raw_response=record.code,
        )
        result = SemanticEquivalenceChecker().check(record, candidate)
        self.assertTrue(result.passed, result.feedback)


if __name__ == "__main__":
    unittest.main()
