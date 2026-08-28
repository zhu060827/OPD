from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from code_rewrite_feedback_expander.evaluate_mbpp_local import (
    GenerationResult,
    _execution_feedback_message,
    _pass_at_k,
    evaluate,
    evaluate_with_execution_feedback,
    load_mbpp_rows,
    summarize_iterative_results,
    summarize_results,
)


def mbpp_row(task_id: str, function_name: str, expected: int) -> dict:
    tests = [f"assert {function_name}() == {expected}"]
    return {
        "data_source": "mbpp",
        "prompt": [{"role": "user", "content": f"Implement {function_name}()."}],
        "ability": "code",
        "response": f"def {function_name}():\n    return {expected}",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"task_id": task_id, "canonical_code": "", "setup_code": "", "tests": tests}
            ),
        },
        "extra_info": {
            "index": int(task_id),
            "task_id": task_id,
            "split": "test",
            "dataset": "mbpp",
            "language": "python",
            "setup_code": "",
            "tests": tests,
            "canonical_code": "",
            "required_entrypoints": [function_name],
            "required_interfaces": [f"def {function_name}(): ..."],
            "prompt_schema": "mbpp_required_interfaces_v1",
        },
    }


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, conversations):
        self.calls += 1
        outputs = []
        for conversation in conversations:
            function_name = conversation[0]["content"].split("Implement ", 1)[1].split("(", 1)[0]
            expected = 1 if function_name == "good" else -1
            text = f"```python\ndef {function_name}():\n    return {expected}\n```"
            outputs.append(
                [
                    GenerationResult(
                        text=text,
                        prompt_tokens=8,
                        generated_tokens=12,
                        latency_seconds=0.01,
                        hit_token_limit=False,
                    )
                ]
            )
        return outputs


class FakeIterativeGenerator:
    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []

    def generate(self, conversations):
        outputs = []
        for conversation in conversations:
            self.conversations.append(conversation)
            function_name = conversation[0]["content"].split("Implement ", 1)[1].split("(", 1)[0]
            is_repair = len(conversation) > 1
            expected = 1 if function_name == "good" or is_repair else -1
            text = f"```python\ndef {function_name}():\n    return {expected}\n```"
            outputs.append(
                [
                    GenerationResult(
                        text=text,
                        prompt_tokens=16 if is_repair else 8,
                        generated_tokens=12,
                        latency_seconds=0.01,
                        hit_token_limit=False,
                    )
                ]
            )
        return outputs


class MBPPLocalEvaluationTest(unittest.TestCase):
    def test_load_mbpp_rows_selects_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.parquet"
            pq.write_table(pa.Table.from_pylist([mbpp_row("0", "a", 1), mbpp_row("1", "b", 1)]), path)
            selected = load_mbpp_rows(path, start_index=1, max_samples=1)
            self.assertEqual(selected[0]["extra_info"]["task_id"], "1")

    def test_pass_at_k_estimator(self) -> None:
        self.assertEqual(_pass_at_k(5, 0, 1), 0.0)
        self.assertEqual(_pass_at_k(5, 5, 3), 1.0)
        self.assertAlmostEqual(_pass_at_k(2, 1, 1), 0.5)
        self.assertEqual(_pass_at_k(2, 1, 2), 1.0)

    def test_evaluate_writes_results_and_resumes(self) -> None:
        rows = [mbpp_row("0", "good", 1), mbpp_row("1", "bad", 1)]
        generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics = evaluate(
                rows=rows,
                generator=generator,
                output_dir=output_dir,
                existing_results=[],
                samples_per_task=1,
                batch_size=1,
                model_path="model",
                dataset_path="test.parquet",
                dataset_start_index=0,
                execution_timeout=1.0,
            )
            self.assertEqual(metrics["tasks_complete"], 2)
            self.assertEqual(metrics["passed_completions"], 1)
            self.assertEqual(metrics["pass@1"], 0.5)
            self.assertEqual(metrics["interface_match_rate"], 1.0)
            self.assertEqual(len((output_dir / "predictions.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((output_dir / "failures.jsonl").read_text().splitlines()), 1)

            existing = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            second_metrics = evaluate(
                rows=rows,
                generator=generator,
                output_dir=output_dir,
                existing_results=existing,
                samples_per_task=1,
                batch_size=1,
                model_path="model",
                dataset_path="test.parquet",
                dataset_start_index=0,
                execution_timeout=1.0,
            )
            self.assertEqual(second_metrics["pass@1"], 0.5)
            self.assertEqual(generator.calls, 2)
            self.assertEqual(len((output_dir / "predictions.jsonl").read_text().splitlines()), 2)

    def test_summary_uses_only_complete_task_groups(self) -> None:
        result = {
            "task_id": "0",
            "sample_id": 0,
            "passed": True,
            "error_type": "none",
            "code_extracted": True,
            "syntax_valid": True,
            "interface_match": True,
            "hit_token_limit": False,
            "prompt_tokens": 2,
            "generated_tokens": 3,
            "generation_seconds": 1.0,
            "grading_seconds": 0.1,
        }
        metrics = summarize_results(
            [result],
            target_task_ids=["0"],
            samples_per_task=2,
            model_path="model",
            dataset_path="dataset",
        )
        self.assertEqual(metrics["status"], "in_progress")
        self.assertEqual(metrics["tasks_complete"], 0)
        self.assertEqual(metrics["completions_evaluated"], 0)

    def test_summary_feedback_hides_held_out_assertion(self) -> None:
        previous = {
            "error_type": "test_failure",
            "test_output": 'AssertionError: assert hidden_case() == "secret"',
            "interface_match": True,
            "required_interfaces": ["def solve(): ..."],
            "defined_entrypoints": ["solve"],
        }
        summary = _execution_feedback_message(previous, mode="summary", max_chars=1000)
        full = _execution_feedback_message(previous, mode="full", max_chars=1000)
        self.assertNotIn("hidden_case", summary)
        self.assertNotIn("secret", summary)
        self.assertIn("held-out unit tests", summary)
        self.assertIn("hidden_case", full)
        self.assertIn("oracle-assisted", full)

    def test_iterative_feedback_repairs_only_failed_tasks(self) -> None:
        rows = [mbpp_row("0", "good", 1), mbpp_row("1", "bad", 1)]
        generator = FakeIterativeGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics = evaluate_with_execution_feedback(
                rows=rows,
                generator=generator,
                output_dir=output_dir,
                existing_results=[],
                max_attempts=3,
                batch_size=2,
                model_path="model",
                dataset_path="test.parquet",
                dataset_start_index=0,
                execution_feedback_mode="summary",
                feedback_max_chars=1000,
                execution_timeout=1.0,
            )
            self.assertEqual(metrics["status"], "complete")
            self.assertEqual(metrics["pass@1"], 0.5)
            self.assertEqual(metrics["iterative_solve_rate@1"], 0.5)
            self.assertEqual(metrics["iterative_solve_rate@2"], 1.0)
            self.assertEqual(metrics["iterative_pass@3"], 1.0)
            self.assertEqual(metrics["repair_gain@3"], 0.5)
            self.assertEqual(metrics["attempts_evaluated"], 3)
            self.assertEqual(metrics["repaired_after_initial_failure"], 1)
            self.assertFalse(metrics["standard_pass_at_k_applicable"])

            records = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            good_records = [record for record in records if record["task_id"] == "0"]
            bad_records = [record for record in records if record["task_id"] == "1"]
            self.assertEqual(len(good_records), 1)
            self.assertEqual(len(bad_records), 2)
            self.assertIsNone(bad_records[0]["feedback_used"])
            self.assertIn("previous implementation", bad_records[1]["feedback_used"])
            self.assertEqual(bad_records[1]["previous_attempt_id"], 0)
            self.assertEqual(generator.conversations[-1][-2]["role"], "assistant")
            self.assertIn("return -1", generator.conversations[-1][-2]["content"])
            self.assertEqual(generator.conversations[-1][-1]["role"], "user")

            calls_before_resume = len(generator.conversations)
            resumed = evaluate_with_execution_feedback(
                rows=rows,
                generator=generator,
                output_dir=output_dir,
                existing_results=records,
                max_attempts=3,
                batch_size=2,
                model_path="model",
                dataset_path="test.parquet",
                dataset_start_index=0,
                execution_feedback_mode="summary",
                feedback_max_chars=1000,
                execution_timeout=1.0,
            )
            self.assertEqual(resumed["iterative_solve_rate@3"], 1.0)
            self.assertEqual(len(generator.conversations), calls_before_resume)
            self.assertEqual(len((output_dir / "predictions.jsonl").read_text().splitlines()), 3)

    def test_iterative_summary_waits_for_terminal_attempt(self) -> None:
        failed = {
            "task_id": "0",
            "sample_id": 0,
            "passed": False,
            "error_type": "test_failure",
            "code_extracted": True,
            "syntax_valid": True,
            "interface_match": True,
            "hit_token_limit": False,
            "prompt_tokens": 2,
            "generated_tokens": 3,
            "generation_seconds": 1.0,
            "grading_seconds": 0.1,
        }
        metrics = summarize_iterative_results(
            [failed],
            target_task_ids=["0"],
            max_attempts=3,
            model_path="model",
            dataset_path="dataset",
            execution_feedback_mode="summary",
        )
        self.assertEqual(metrics["status"], "in_progress")
        self.assertEqual(metrics["tasks_complete"], 0)


if __name__ == "__main__":
    unittest.main()
