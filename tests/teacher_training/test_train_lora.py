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
from teacher_training.train_lora import (
    _non_thinking_training_template,
    audit_assistant_mask,
    build_dry_run_report,
)


class TrainLoraDryRunTests(unittest.TestCase):
    def test_training_template_disables_thinking_and_keeps_mask_markers(self):
        template = "{% generation %}{{ content }}{% endgeneration %}"
        patched = _non_thinking_training_template(template)
        self.assertTrue(patched.startswith("{% set enable_thinking = false %}"))
        self.assertIn("{% generation %}", patched)

    def test_assistant_mask_audit_rejects_missing_target_mask(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 0]}

        with self.assertRaisesRegex(ValueError, "assistant token mask"):
            audit_assistant_mask(Tokenizer(), [{"role": "assistant", "content": "x"}])

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
