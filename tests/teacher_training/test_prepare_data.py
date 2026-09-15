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
        self.assertIn("<reasoning>", result["messages"][2]["content"])
        self.assertIn("<code>", result["messages"][2]["content"])
        self.assertFalse(result["reasoning_from_source"])
        self.assertEqual("transformation_rationale", result["reasoning_type"])
        self.assertEqual("teacher-reasoning-code-v2", result["output_schema_version"])

    def test_prefers_original_reasoning_over_fallback(self):
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
        self.assertTrue(result["reasoning_from_source"])
        self.assertEqual("原始 target_reasoning", result["reasoning_origin"])
        self.assertIn(row["target_reasoning"], result["messages"][2]["content"])
        self.assertEqual("full_reasoning", result["reasoning_type"])

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

    def test_formal_contract_rejects_missing_domain_gold_labels(self):
        row = {
            "source_id": "rename-without-label",
            "source_code": "def f(x):\n    d = x\n    return d",
            "target_code": "def f(x):\n    value = x\n    return value",
            "domain": "variable",
            "semantic_pass": True,
            "tests": ["assert f(1) == 1"],
            "source_dataset": "commitpack",
            "source_paper": "OctoPack",
        }
        with self.assertRaisesRegex(ValueError, "target_identifier"):
            normalize_row(row, thresholds={"naming_gain": 0.0}, formal=True)

    def test_formal_cot_rejects_fallback_reasoning(self):
        row = {
            "source_id": "cot-without-real-plan",
            "task": "返回输入加一",
            "source_code": "def f(x):\n    return x + 1",
            "target_code": "def f(x):\n    value = x + 1\n    return value",
            "domain": "cot",
            "semantic_pass": True,
            "tests": ["assert f(1) == 2"],
            "source_dataset": "CodeContests",
            "source_paper": "AlphaCode",
        }
        with self.assertRaisesRegex(ValueError, "禁止使用固定领域模板"):
            normalize_row(row, formal=True)


if __name__ == "__main__":
    unittest.main()
