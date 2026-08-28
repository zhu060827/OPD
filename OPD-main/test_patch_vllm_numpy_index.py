from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from patch_vllm_numpy_index import ORIGINAL, PATCHED, patch_runner


RUNNER_SOURCE = f"""\
import numpy as np
import torch


def dummy_run(hidden_states, num_scheduled_tokens):
    {ORIGINAL}
    return hidden_states, hidden_states[logit_indices]
"""


class PatchVllmNumpyIndexTest(unittest.TestCase):
    def test_patch_is_idempotent_and_keeps_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            target = Path(temporary_dir) / "gpu_model_runner.py"
            target.write_text(RUNNER_SOURCE, encoding="utf-8")

            self.assertEqual(patch_runner(target), "patched")
            patched_source = target.read_text(encoding="utf-8")
            self.assertIn(PATCHED, patched_source)
            self.assertNotIn(ORIGINAL, patched_source)
            compile(patched_source, str(target), "exec")

            backup = target.with_name(f"{target.name}.opd-numpy-index.bak")
            self.assertEqual(backup.read_text(encoding="utf-8"), RUNNER_SOURCE)
            self.assertEqual(patch_runner(target), "already patched")

    def test_unknown_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            target = Path(temporary_dir) / "gpu_model_runner.py"
            target.write_text("def unrelated():\n    return None\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                patch_runner(target)


if __name__ == "__main__":
    unittest.main()
