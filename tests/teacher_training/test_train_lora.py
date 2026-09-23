import json
import tempfile
import unittest
from pathlib import Path

from teacher_training.domains import (
    DOMAIN_SPECS,
    OUTPUT_SCHEMA_VERSION,
    PUBLIC_DOMAIN_NAMES,
    SYSTEM_PROMPT_VERSION,
)
from teacher_training.train_lora import build_dry_run_report


class TrainLoraDryRunTests(unittest.TestCase):
    def test_dry_run_validates_data_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for split, problem_id in (("train", "problem-train"), ("validation", "problem-validation")):
                path = root / f"{split}.jsonl"
                row = {
                    "source_id": f"source-{split}",
                    "problem_id": problem_id,
                    "domain": "identifier",
                    "domain_name": PUBLIC_DOMAIN_NAMES["identifier"],
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "system_prompt_version": SYSTEM_PROMPT_VERSION,
                    "experiment_status": "verified_positive",
                    "semantic_pass": True,
                    "messages": [
                        {"role": "system", "content": DOMAIN_SPECS["identifier"].system_prompt},
                        {"role": "user", "content": "Original code:\nx = 1"},
                        {"role": "assistant", "content": "<code>\nvalue = 1\n</code>"},
                    ],
                }
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                paths[split] = str(path)

            report = build_dry_run_report({
                "domain": "identifier",
                "train_file": paths["train"],
                "validation_file": paths["validation"],
                "formal_experiment": False,
            })

            self.assertEqual("ready_for_model_loading", report["status"])
            self.assertEqual(1, report["dataset_summary"]["train_file"]["samples"])
            self.assertEqual(1, report["dataset_summary"]["validation_file"]["samples"])


if __name__ == "__main__":
    unittest.main()
