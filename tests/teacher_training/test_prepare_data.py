import json
import tempfile
import unittest
from pathlib import Path

from teacher_training.prepare_data import normalize_row, prepare


class TeacherDataPreparationTests(unittest.TestCase):
    def test_normalizes_before_after_pair(self):
        row = {
            "source_id": "one",
            "before_code": "def f(x):\n    d = x + 1\n    return d",
            "after_code": "def f(x):\n    total_value = x + 1\n    return total_value",
            "domain": "variable",
            "semantic_pass": True,
            "tests": ["assert f(1) == 2"],
        }
        result = normalize_row(row, thresholds={"naming_gain": 0.0})
        self.assertEqual("variable", result["domain"])
        self.assertEqual("system", result["messages"][0]["role"])
        self.assertIn("<plan>", result["messages"][2]["content"])
        self.assertIn("<code>", result["messages"][2]["content"])
        self.assertFalse(result["plan_from_source"])
        self.assertEqual("interface_intent", result["plan_type"])
        self.assertEqual("teacher-plan-code-v1", result["output_schema_version"])

    def test_prefers_original_plan_over_fallback(self):
        row = {
            "source_id": "cot-one",
            "source_code": "def f(x):\n    return x + 1",
            "target_code": "def f(x):\n    result = x + 1\n    return result",
            "domain": "cot",
            "target_reasoning": "先保存计算结果，再返回；时间复杂度保持 O(1)。",
            "semantic_pass": True,
            "tests": ["assert f(1) == 2"],
        }
        result = normalize_row(row)
        self.assertTrue(result["plan_from_source"])
        self.assertEqual("原始 target_reasoning", result["plan_origin"])
        self.assertIn(row["target_reasoning"], result["messages"][2]["content"])
        self.assertEqual("reasoning_plan", result["plan_type"])

    def test_split_keeps_source_ids_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            rows = []
            for index in range(10):
                for variant in range(2):
                    rows.append({
                        "source_id": str(index),
                        "before_code": f"def f_{index}(x):\n    return x",
                        "after_code": f"def f_{index}(value):\n    return value",
                        "domain": "variable",
                        "semantic_pass": True,
                        "tests": [f"assert f_{index}(1) == 1"],
                        "variant": variant,
                    })
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            prepare(str(source), str(root / "out"), None, 42, 0.8, 0.1, True)
            split_ids = {}
            for split in ("train", "validation", "test"):
                split_ids[split] = {json.loads(line)["source_id"] for line in (root / "out" / f"{split}.jsonl").read_text().splitlines()}
            self.assertFalse(split_ids["train"] & split_ids["validation"])
            self.assertFalse(split_ids["train"] & split_ids["test"])
            self.assertFalse(split_ids["validation"] & split_ids["test"])

    def test_accepts_verified_java_pairs(self):
        result = normalize_row({
            "source_id": "java-one",
            "before_code": "int f(int x) { return x + 1; }",
            "after_code": "int f(int value) { return value + 1; }",
            "domain": "style",
            "language": "java",
            "semantic_pass": True,
        })
        self.assertEqual("java", result["language"])


if __name__ == "__main__":
    unittest.main()
