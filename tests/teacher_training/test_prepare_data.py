import json
import tempfile
import unittest
from pathlib import Path

from teacher_training.prepare_data import normalize_row, prepare


class TeacherDataPreparationTests(unittest.TestCase):
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(sum(len(item["content"].split()) for item in messages)))

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    def test_normalizes_before_after_pair(self):
        row = {
            "source_id": "one",
            "before_code": "def f(x):\n    d = x + 1\n    return d",
            "after_code": "def f(x):\n    total_value = x + 1\n    return total_value",
            "domain": "identifier",
            "semantic_pass": True,
            "tests": ["assert f(1) == 2"],
        }
        result = normalize_row(row)
        self.assertEqual("identifier", result["domain"])
        self.assertEqual("system", result["messages"][0]["role"])
        self.assertIn("<code>", result["messages"][2]["content"])
        self.assertNotIn("<reasoning>", result["messages"][2]["content"])
        self.assertEqual("teacher-code-only-v1", result["output_schema_version"])

    def test_rejects_removed_cot_domain(self):
        row = {
            "source_id": "removed-domain-one",
            "source_code": "def f(x):\n    return x + 1",
            "target_code": "def f(x):\n    result = x + 1\n    return result",
            "domain": "cot",
            "target_reasoning": "先保存计算结果，再返回；时间复杂度保持 O(1)。",
            "semantic_pass": True,
            "tests": ["assert f(1) == 2"],
        }
        with self.assertRaisesRegex(ValueError, "未知领域"):
            normalize_row(row)

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
                        "domain": "identifier",
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

    def test_split_keeps_problem_families_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            rows = []
            for problem in range(10):
                for variant in range(2):
                    rows.append({
                        "source_id": f"{problem}:{variant}",
                        "problem_id": str(problem),
                        "source_code": f"def f_{problem}(x):\n    value_{variant} = x\n    return value_{variant}",
                        "target_code": f"def f_{problem}(x):\n    result_{variant} = x\n    return result_{variant}",
                        "domain": "identifier",
                        "semantic_pass": True,
                        "tests": [f"assert f_{problem}(1) == 1"],
                    })
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            prepare(str(source), str(root / "out"), None, 42, 0.8, 0.1, True)
            split_problems = {}
            for split in ("train", "validation", "test"):
                split_problems[split] = {
                    json.loads(line)["problem_id"]
                    for line in (root / "out" / f"{split}.jsonl").read_text().splitlines()
                }
            self.assertFalse(split_problems["train"] & split_problems["validation"])
            self.assertFalse(split_problems["train"] & split_problems["test"])
            self.assertFalse(split_problems["validation"] & split_problems["test"])

    def test_accepts_verified_java_pairs(self):
        result = normalize_row({
            "source_id": "java-one",
            "before_code": "int f(int x) { return x + 1; }",
            "after_code": "int f(int value) { return value + 1; }",
            "domain": "formatting",
            "language": "java",
            "semantic_pass": True,
            "evidence": {
                "parse_pass": True,
                "signature_pass": True,
                "compile_pass": True,
                "test_pass": True,
            },
        })
        self.assertEqual("java", result["language"])

    def test_formal_contract_rejects_missing_domain_gold_labels(self):
        row = {
            "source_id": "rename-without-label",
            "problem_id": "rename-problem",
            "source_code": "def f(x):\n    d = x\n    return d",
            "target_code": "def f(x):\n    value = x\n    return value",
            "domain": "identifier",
            "semantic_pass": True,
            "tests": ["assert f(1) == 1"],
            "source_dataset": "commitpack",
            "source_paper": "OctoPack",
        }
        with self.assertRaisesRegex(ValueError, "target_identifier"):
            normalize_row(row, formal=True)

    def test_formal_local_structure_requires_construction_fields(self):
        row = {
            "source_id": "local-structure-without-construction-fields",
            "problem_id": "local-structure-problem",
            "task": "返回输入加一",
            "source_code": "def f(x):\n    return x + 1",
            "target_code": "def f(x):\n    return 1 + x",
            "domain": "local_structure",
            "semantic_pass": True,
            "tests": ["assert f(1) == 2"],
            "source_dataset": "CodeSearchNet",
            "source_paper": "NatGen",
        }
        with self.assertRaisesRegex(ValueError, "transformation_type"):
            normalize_row(row, formal=True)

    def test_control_flow_requires_registered_rule_evidence(self):
        row = {
            "source_id": "bad-guard",
            "problem_id": "bad-guard-problem",
            "source_code": "def f(x):\n    if x > 0:\n        return x\n    return 0",
            "target_code": "def f(x):\n    while x > 0:\n        return x\n    return 0",
            "domain": "control_flow",
            "semantic_pass": True,
            "tests": ["assert f(0) == 0", "assert f(1) == 1"],
            "transformation_type": "guard_clause",
            "construction_method": "fixture",
            "source_dataset": "fixture",
            "source_paper": "fixture",
        }
        with self.assertRaisesRegex(ValueError, "预注册规则"):
            normalize_row(row, formal=True)

    def test_formatting_accepts_missing_tests_with_na_test_gate(self):
        row = {
            "source_id": "formatting-no-tests",
            "problem_id": "formatting-problem",
            "source_code": "def f(x):\n  return x + 1",
            "target_code": "def f(x):\n    return x + 1",
            "domain": "formatting",
            "semantic_pass": True,
            "transformation_type": "formatting_repair",
            "construction_method": "styler_style_deterministic",
            "source_dataset": "bigcode/commitpackft",
            "source_paper": "CommitPackFT/STYLER",
        }
        result = normalize_row(row, formal=True)
        self.assertIsNone(result["evidence"]["hard_gates"]["test_pass"])

    def test_control_flow_rejects_missing_tests(self):
        row = {
            "source_id": "control-no-tests",
            "problem_id": "control-problem",
            "source_code": "def f(x):\n    if x >= 0:\n        return x\n    else:\n        return 0",
            "target_code": "def f(x):\n    if x < 0:\n        return 0\n    return x",
            "domain": "control_flow",
            "semantic_pass": True,
            "transformation_type": "guard_clause",
            "construction_method": "deterministic_ast_transform",
            "source_dataset": "deepmind/code_contests",
            "source_paper": "AlphaCode",
        }
        with self.assertRaisesRegex(ValueError, "必须提供可执行测试"):
            normalize_row(row, formal=True)

    def test_prepare_rejects_training_context_that_exceeds_token_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            row = {
                "source_id": "long-one",
                "problem_id": "long-problem",
                "source_code": "def f(x):\n    value = x\n    return value",
                "target_code": "def f(x):\n    result = x\n    return result",
                "task": "rename the local variable",
                "domain": "identifier",
                "semantic_pass": True,
                "transformation_type": "identifier_deobfuscation",
            }
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = prepare(
                str(source), str(root / "out"), "identifier", 42, 0.8, 0.1,
                True, tokenizer=self.FakeTokenizer(), tokenizer_name="fake",
                max_length=3, max_generation_tokens=100,
            )
            self.assertEqual(1, manifest["rejected"])
            rejected = json.loads((root / "out" / "rejected.jsonl").read_text())
            self.assertIn("training_context_too_long", rejected["reason"])


if __name__ == "__main__":
    unittest.main()
