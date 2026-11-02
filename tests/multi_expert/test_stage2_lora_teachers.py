from __future__ import annotations

import unittest
from pathlib import Path

from code_rewrite_feedback_expander.multi_expert.stage2_open_mopd import (
    EXPECTED_DOMAINS,
    Stage2OpenMOPDConfig,
    build_open_mopd_command,
)


class Stage2LoraTeacherTests(unittest.TestCase):
    def _config(self) -> Stage2OpenMOPDConfig:
        raw = {
            "open_mopd_backend": {"root": "/open-mopd", "expected_commit": ""},
            "models": {
                "student_path": "/models/student",
                "teacher_base_path": "/models/base",
                "teachers": [
                    {"domain": domain, "adapter_path": f"/adapters/{domain}"}
                    for domain in EXPECTED_DOMAINS
                ],
            },
            "data": {"train_file": "/data/train.parquet", "val_file": "/data/val.parquet"},
            "runtime": {"gpus": 1, "nodes": 1},
            "method": {
                "label_policy": "recorded_method",
                "target_gradient_shares": {domain: 0.25 for domain in EXPECTED_DOMAINS},
                "use_sft": False,
            },
        }
        return Stage2OpenMOPDConfig.from_dict(raw)

    def test_shared_base_is_passed_for_every_teacher(self):
        command = build_open_mopd_command(self._config())
        teacher_values = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--teacher"
        ]
        self.assertEqual(["/models/base"] * 4, teacher_values)

    def test_each_reward_worker_receives_its_adapter(self):
        command = build_open_mopd_command(self._config())
        expected = {
            "+reward_model.model.lora_adapter_path=/adapters/formatting",
            "+mt_reward_model_1.model.lora_adapter_path=/adapters/identifier",
            "+mt_reward_model_2.model.lora_adapter_path=/adapters/local_structure",
            "+mt_reward_model_3.model.lora_adapter_path=/adapters/control_flow",
        }
        self.assertTrue(expected.issubset(set(command)))

    def test_adapter_paths_must_be_distinct(self):
        config = self._config()
        self.assertEqual(Path("/models/base"), config.teacher_base_path)


if __name__ == "__main__":
    unittest.main()
