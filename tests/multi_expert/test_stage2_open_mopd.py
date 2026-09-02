from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from code_rewrite_feedback_expander.multi_expert.stage2_open_mopd import (
    EXPECTED_DOMAINS,
    Stage2OpenMOPDConfig,
    build_open_mopd_command,
    inspect_dataset_domains,
    validate_real_run,
)


def raw_config(root: Path) -> dict:
    teachers = []
    for domain in EXPECTED_DOMAINS:
        teachers.append({"domain": domain, "teacher_path": str(root / f"teacher-{domain}")})
    return {
        "open_mopd_backend": {"root": str(root / "Open-MOPD"), "expected_commit": ""},
        "models": {"student_path": str(root / "student"), "teachers": teachers},
        "data": {"train_file": str(root / "train.jsonl"), "val_file": str(root / "val.jsonl")},
        "output_dir": str(root / "output"),
        "runtime": {"gpus": 1, "nodes": 1},
        "method": {
            "label_policy": "recorded_method",
            "target_gradient_shares": {domain: 0.2 for domain in EXPECTED_DOMAINS},
            "gap_following_alpha": 1.0,
            "reward_refresh": True,
            "conflict_policy": "none",
        },
    }


class Stage2OpenMOPDTests(unittest.TestCase):
    def test_requires_exactly_five_ordered_real_teacher_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = raw_config(Path(temp))
            raw["models"]["teachers"] = raw["models"]["teachers"][:-1]
            with self.assertRaisesRegex(ValueError, "Exactly five Teachers"):
                Stage2OpenMOPDConfig.from_dict(raw)

    def test_builds_official_hard_routed_balanced_refreshed_command(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Stage2OpenMOPDConfig.from_dict(raw_config(Path(temp)))
            command = build_open_mopd_command(config, execute=True)
            rendered = " ".join(command)
            self.assertEqual(5, command.count("--teacher"))
            self.assertIn("--domains cot,style,ast,variable,control_flow", rendered)
            self.assertIn("domain_weighting=domain_routing", rendered)
            self.assertIn("target_share_values=[0.2,0.2,0.2,0.2,0.2]", rendered)
            self.assertIn("reward_scale_direction=multiply", rendered)
            self.assertIn("opd_refresh_advantage=true", rendered)
            self.assertEqual("--run", command[-1])

    def test_dataset_domain_validation_reads_real_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text(
                "\n".join(json.dumps({"domain": domain}) for domain in EXPECTED_DOMAINS),
                encoding="utf-8",
            )
            counts = inspect_dataset_domains(path)
            self.assertEqual(set(EXPECTED_DOMAINS), set(counts))

    def test_rejects_duplicate_teacher_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = raw_config(Path(temp))
            raw["models"]["teachers"][1]["teacher_path"] = raw["models"]["teachers"][0]["teacher_path"]
            with self.assertRaisesRegex(ValueError, "distinct"):
                Stage2OpenMOPDConfig.from_dict(raw)

    def test_strict_preflight_accepts_five_distinct_compatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = raw_config(root)
            launcher = root / "Open-MOPD" / "scripts" / "local" / "mt_opd.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            model_paths = [root / "student", *[root / f"teacher-{domain}" for domain in EXPECTED_DOMAINS]]
            for model_path in model_paths:
                model_path.mkdir()
                (model_path / "config.json").write_text(
                    json.dumps({"model_type": "qwen3", "vocab_size": 100}),
                    encoding="utf-8",
                )
                (model_path / "model.safetensors").write_bytes(b"real-path-placeholder")
                (model_path / "tokenizer.json").write_text("same-tokenizer", encoding="utf-8")
            rows = "\n".join(json.dumps({"domain": domain}) for domain in EXPECTED_DOMAINS)
            (root / "train.jsonl").write_text(rows, encoding="utf-8")
            (root / "val.jsonl").write_text(rows, encoding="utf-8")
            report = validate_real_run(Stage2OpenMOPDConfig.from_dict(raw))
            self.assertEqual(5, report["teacher_count"])
            self.assertEqual(set(EXPECTED_DOMAINS), set(report["training_domain_counts"]))


if __name__ == "__main__":
    unittest.main()
