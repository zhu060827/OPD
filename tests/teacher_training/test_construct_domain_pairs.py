import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teacher_training.scripts.construct_domain_pairs import (
    classify_local_structure_rejection, construct_identifier, construct_identifier_variants, construct_local_structure,
)


class ConstructDomainPairsTests(unittest.TestCase):
    def test_local_structure_rejection_has_stable_reason_code(self):
        code, reason, details = classify_local_structure_rejection("def f(x):\n    return x\n")
        self.assertEqual("return_is_name_or_constant", code)
        self.assertTrue(reason)
        self.assertEqual(1, details["return_count"])

    def test_local_structure_rejection_distinguishes_unsupported_expression(self):
        code, _, details = classify_local_structure_rejection("async def f(x):\n    return await x\n")
        self.assertEqual("unsupported_expression_type", code)
        self.assertIn("Await", details["return_types"])

    def test_constructor_exception_is_counted_once(self):
        from teacher_training.scripts.construct_domain_pairs import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            rejected = root / "rejected.jsonl"
            source.write_text(json.dumps({"problem_id": "bad", "code": "def broken("}) + "\n", encoding="utf-8")
            argv = [
                "construct_domain_pairs",
                "--input", str(source),
                "--output", str(output),
                "--rejected", str(rejected),
                "--domain", "local_structure",
            ]
            with patch("sys.argv", argv):
                main()
            rows = [json.loads(line) for line in rejected.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(1, len(rows))
            self.assertEqual("construction_exception", rows[0]["rejection_code"])
    def test_identifier_preserves_formatting_and_comments(self):
        target = (
            "def total(values):\n"
            "    # keep this documentation\n"
            "    result  = sum(values)  # keep spacing\n"
            "    return result\n"
        )
        pair = construct_identifier({"problem_id": "identifier-format"}, target)
        self.assertIsNotNone(pair)
        self.assertIn("# keep this documentation", pair["source_code"])
        self.assertIn("VAR_0  = sum(values)  # keep spacing", pair["source_code"])
        self.assertEqual(target, pair["target_code"])
        self.assertEqual("result", pair["target_identifier"])

    def test_identifier_can_build_multiple_non_leaking_variants(self):
        target = (
            "def aggregate(values):\n"
            "    running_total = sum(values)\n"
            "    item_count = len(values)\n"
            "    return running_total / item_count\n"
        )
        variants = construct_identifier_variants({"problem_id": "aggregate"}, target, limit=2)
        self.assertEqual(2, len(variants))
        self.assertEqual({"aggregate"}, {row["problem_id"] for row in variants})
        self.assertEqual(2, len({row["source_id"] for row in variants}))
        self.assertNotIn("running_total", variants[0]["source_id"])
        self.assertNotIn("item_count", variants[1]["source_id"])

    def test_identifier_handles_inner_scope_without_selecting_self(self):
        target = (
            "def outer(values):\n"
            "    result = sum(values)\n"
            "    def inner(self):\n"
            "        result = self.value\n"
            "        return result\n"
            "    return result\n"
        )
        variants = construct_identifier_variants({"problem_id": "nested"}, target, limit=3)
        self.assertEqual(["result"], [row["target_identifier"] for row in variants])
        self.assertNotIn("self", [row["target_identifier"] for row in variants])

    def test_identifier_excludes_varargs_and_kwargs(self):
        target = "def collect(*items, **options):\n    result = len(items) + len(options)\n    return result\n"
        variants = construct_identifier_variants({"problem_id": "parameters"}, target, limit=3)
        self.assertEqual(["result"], [row["target_identifier"] for row in variants])

    def test_identifier_excludes_reserved_receiver_even_when_assigned(self):
        target = "def unusual(value):\n    self = value\n    return self\n"
        self.assertEqual([], construct_identifier_variants({"problem_id": "receiver"}, target, limit=3))

    def test_identifier_source_ids_disambiguate_same_problem_different_code(self):
        first = "def f(values):\n    result = sum(values)\n    return result\n"
        second = "def f(values):\n    total = max(values)\n    return total\n"
        left = construct_identifier_variants({"problem_id": "shared"}, first, limit=1)[0]
        right = construct_identifier_variants({"problem_id": "shared"}, second, limit=1)[0]
        self.assertEqual(left["problem_id"], right["problem_id"])
        self.assertNotEqual(left["source_id"], right["source_id"])

    def test_local_structure_only_replaces_registered_source_span(self):
        target = (
            "def total(values):\n"
            "    # documentation remains unchanged\n"
            "    result = sum(values)\n"
            "    return result\n"
        )
        pair = construct_local_structure({"problem_id": "local-span"}, target)
        self.assertIsNotNone(pair)
        self.assertEqual(
            "def total(values):\n"
            "    # documentation remains unchanged\n"
            "    return (sum(values))\n",
            pair["source_code"],
        )
        self.assertEqual(target, pair["target_code"])
        self.assertTrue(pair["transformation_evidence"]["formatting_outside_span_preserved"])

    def test_local_structure_rejects_comment_bearing_transform(self):
        target = "def total(values):\n    result = sum(values)  # explanation\n    return result\n"
        self.assertIsNone(construct_local_structure({"problem_id": "comment"}, target))

    def test_local_structure_supports_single_use_inside_return_expression(self):
        target = "def scaled(value):\n    doubled = value * 2\n    return doubled + 1\n"
        pair = construct_local_structure({"problem_id": "return-expression"}, target)
        self.assertIsNotNone(pair)
        self.assertEqual("def scaled(value):\n    return (value * 2) + 1\n", pair["source_code"])

    def test_local_structure_rejects_side_effecting_rhs(self):
        target = "def load():\n    value = fetch()\n    return normalize(value)\n"
        pair = construct_local_structure({"problem_id": "side-effect"}, target)
        # 旧临时变量不会被危险内联；fallback 只抽取原 return 表达式。
        self.assertIsNotNone(pair)
        self.assertEqual("fowler_extract_return_expression", pair["construction_method"])
        self.assertIn("normalize_result = normalize(value)", pair["target_code"])

    def test_local_structure_extracts_return_expression_without_global_reformat(self):
        source = (
            "def score(left, right):\n"
            "    # preserve this comment\n"
            "    return left + right\n"
        )
        pair = construct_local_structure({"problem_id": "extract-return"}, source)
        self.assertIsNotNone(pair)
        self.assertEqual(source, pair["source_code"])
        self.assertEqual(
            "def score(left, right):\n"
            "    # preserve this comment\n"
            "    computed_value = left + right\n"
            "    return computed_value\n",
            pair["target_code"],
        )
        self.assertEqual("teacher-local-structure-constructor-v7", pair["constructor_version"])

    def test_local_structure_rejects_weak_tuple_packaging(self):
        source = "def pair(left, right):\n    return left, right\n"
        self.assertIsNone(construct_local_structure({"problem_id": "weak-tuple"}, source))

    def test_local_structure_source_ids_disambiguate_different_pairs(self):
        first = "def f(value):\n    return value + 1\n"
        second = "def f(value):\n    return value + 2\n"
        left = construct_local_structure({"problem_id": "shared-local"}, first)
        right = construct_local_structure({"problem_id": "shared-local"}, second)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual("shared-local", left["problem_id"])
        self.assertEqual(left["problem_id"], right["problem_id"])
        self.assertNotEqual(left["source_id"], right["source_id"])
        self.assertRegex(left["source_id"], r":code-[0-9a-f]{12}$")

    def test_local_structure_source_id_is_deterministic(self):
        source = "def f(value):\n    return value + 1\n"
        first = construct_local_structure({"problem_id": "stable-local"}, source)
        second = construct_local_structure({"problem_id": "stable-local"}, source)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first["source_id"], second["source_id"])
        self.assertEqual(first["pair_digest"], second["pair_digest"])

    def test_local_structure_all_registered_branches_receive_pair_digest(self):
        natgen_target = "def total(values):\n    result = sum(values)\n    return result\n"
        fowler_source = "def total(values):\n    return sum(values)\n"
        natgen = construct_local_structure({"problem_id": "natgen"}, natgen_target)
        fowler = construct_local_structure({"problem_id": "fowler"}, fowler_source)
        self.assertIsNotNone(natgen)
        self.assertIsNotNone(fowler)
        for pair in (natgen, fowler):
            self.assertRegex(pair["source_id"], r":code-[0-9a-f]{12}$")
            self.assertEqual(12, len(pair["pair_digest"]))


if __name__ == "__main__":
    unittest.main()
