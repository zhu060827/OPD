from __future__ import annotations

import json
import unittest

from code_rewrite_feedback_expander.mbpp_reward import extract_python_code, reward_func
from code_rewrite_feedback_expander.mbpp_to_opd_parquet import (
    convert_row,
    required_interface_spec,
)


class MBPPOPDDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "task_id": 1,
            "text": "Write a function add(a, b) that returns the sum.",
            "code": "def add(a, b):\n    return a + b",
            "test_list": ["assert add(2, 3) == 5"],
            "test_setup_code": "",
            "challenge_test_list": ["assert add(-1, 1) == 0"],
        }

    def test_conversion_produces_verl_schema(self) -> None:
        row = convert_row(self.source, split="train", row_index=0, include_challenge_tests=True)
        self.assertEqual(row["data_source"], "mbpp")
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertIn("def add(a, b): ...", row["prompt"][0]["content"])
        payload = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(len(payload["tests"]), 2)
        self.assertEqual(payload["required_entrypoints"], ["add"])
        self.assertEqual(row["extra_info"]["required_interfaces"], ["def add(a, b): ..."])
        self.assertEqual(row["extra_info"]["prompt_schema"], "mbpp_required_interfaces_v1")

    def test_required_interface_matches_test_target_not_first_helper(self) -> None:
        code = """\
def helper(value):
    return list(value)

def target(value, blocked):
    return value
"""
        entrypoints, interfaces = required_interface_spec(
            code,
            ["assert target('abc', 'b') == 'ac'"],
        )
        self.assertEqual(entrypoints, ["target"])
        self.assertEqual(interfaces, ["def target(value, blocked): ..."])

    def test_required_interface_includes_class_constructor(self) -> None:
        code = """\
class Node:
    def __init__(self, value):
        self.value = value

def height(node):
    return 1
"""
        entrypoints, interfaces = required_interface_spec(
            code,
            ["assert height(root) == 1"],
            "root = Node(1)",
        )
        self.assertEqual(entrypoints, ["Node", "height"])
        self.assertEqual(interfaces, ["class Node:\n    def __init__(self, value): ...", "def height(node): ..."])

    def test_reward_accepts_canonical_and_rejects_bad_code(self) -> None:
        row = convert_row(self.source, split="train", row_index=0, include_challenge_tests=True)
        ground_truth = row["reward_model"]["ground_truth"]
        good = reward_func(
            "mbpp",
            "<think>Use addition.</think>\n```python\ndef add(a, b):\n    return a + b\n```",
            ground_truth,
            row["extra_info"],
        )
        bad = reward_func(
            "mbpp",
            "```python\ndef add(a, b):\n    return a - b\n```",
            ground_truth,
            row["extra_info"],
        )
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])

    def test_code_extraction_rejects_thinking_text(self) -> None:
        code = extract_python_code("<think>analysis</think>\n```python\ndef f():\n    return 1\n```")
        self.assertEqual(code, "def f():\n    return 1")


if __name__ == "__main__":
    unittest.main()
