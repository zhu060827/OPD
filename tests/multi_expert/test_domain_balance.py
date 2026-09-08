from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


balance_module = load_module(
    "project_domain_balance",
    ROOT / "OPD-main/verl/verl/trainer/ppo/domain_balance.py",
)

sampler_api = types.ModuleType("verl.experimental.dataset.sampler")
sampler_api.AbstractSampler = torch.utils.data.Sampler
sys.modules.setdefault("verl", types.ModuleType("verl"))
sys.modules.setdefault("verl.experimental", types.ModuleType("verl.experimental"))
sys.modules.setdefault("verl.experimental.dataset", types.ModuleType("verl.experimental.dataset"))
sys.modules["verl.experimental.dataset.sampler"] = sampler_api
sampler_module = load_module(
    "project_domain_balanced_sampler",
    ROOT / "OPD-main/verl/verl/experimental/dataset/domain_balanced_sampler.py",
)


class FakeBatch:
    def __init__(self):
        self.non_tensor_batch = {
            "domain": np.array(["cot", "style", "ast", "variable", "control_flow"], dtype=object)
        }
        self.batch = {
            "response_mask": torch.tensor(
                [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
                dtype=torch.float32,
            ),
            "advantages": torch.ones((5, 4, 2), dtype=torch.float32),
        }


class DomainBalanceTests(unittest.TestCase):
    def test_every_generation_batch_has_equal_domain_prompts(self):
        labels = ["cot"] * 8 + ["style"] * 2 + ["ast"] + ["variable"] + ["control_flow"]

        class Frame:
            column_names = ["domain"]

            def __getitem__(self, key):
                return labels

        class Dataset:
            dataframe = Frame()

            def __len__(self):
                return len(labels)

        sampler = sampler_module.DomainBalancedSampler(
            Dataset(),
            {
                "train_batch_size": 10,
                "seed": 7,
                "sampler": {
                    "domains": ["cot", "style", "ast", "variable", "control_flow"],
                    "target_shares": [0.2] * 5,
                },
            },
        )
        indices = list(sampler)
        expected = sorted([domain for domain in ("cot", "style", "ast", "variable", "control_flow") for _ in range(2)])
        for start in range(0, len(indices), 10):
            self.assertEqual(expected, sorted(labels[index] for index in indices[start : start + 10]))

    def test_effective_token_shares_are_balanced(self):
        batch = FakeBatch()
        metrics = balance_module.apply_domain_token_balance(
            batch,
            {
                "domain_balance": {
                    "enabled": True,
                    "domains": ["cot", "style", "ast", "variable", "control_flow"],
                    "target_shares": [0.2] * 5,
                    "min_weight": 0.1,
                    "max_weight": 10.0,
                }
            },
        )
        for domain in ("cot", "style", "ast", "variable", "control_flow"):
            self.assertAlmostEqual(0.2, metrics[f"domain_balance/{domain}/effective_share"], places=6)
        self.assertEqual((5,), tuple(batch.batch["domain_loss_weight"].shape))
        self.assertFalse(torch.allclose(batch.batch["advantages"], torch.ones((5, 4, 2))))

    def test_disabled_balance_is_noop(self):
        batch = FakeBatch()
        original = batch.batch["advantages"].clone()
        self.assertEqual({}, balance_module.apply_domain_token_balance(batch, {}))
        self.assertTrue(torch.equal(original, batch.batch["advantages"]))

    def test_multiply_reward_scale_favors_remaining_teacher_student_gap(self):
        batch = FakeBatch()
        batch.batch["advantages"][0] *= 4.0
        metrics = balance_module.apply_domain_token_balance(
            batch,
            {
                "domain_balance": {
                    "enabled": True,
                    "domains": ["cot", "style", "ast", "variable", "control_flow"],
                    "target_shares": [0.2] * 5,
                    "min_weight": 0.05,
                    "max_weight": 20.0,
                    "reward_scale_alpha": 1.0,
                    "reward_ema_decay": 0.0,
                }
            },
            {},
        )
        self.assertGreater(
            metrics["domain_balance/cot/reward_scale"],
            metrics["domain_balance/style/reward_scale"],
        )
        self.assertGreater(
            metrics["domain_balance/cot/effective_share"],
            metrics["domain_balance/style/effective_share"],
        )

    def test_natural_batch_may_omit_domains(self):
        batch = FakeBatch()
        keep = torch.tensor([True, True, False, False, False])
        batch.non_tensor_batch["domain"] = batch.non_tensor_batch["domain"][:2]
        batch.batch["response_mask"] = batch.batch["response_mask"][keep]
        batch.batch["advantages"] = batch.batch["advantages"][keep]
        metrics = balance_module.apply_domain_token_balance(
            batch,
            {
                "domain_balance": {
                    "enabled": True,
                    "domains": ["cot", "style", "ast", "variable", "control_flow"],
                    "target_shares": [0.2] * 5,
                    "token_share_enabled": True,
                    "reward_scale_enabled": True,
                    "reward_scale_alpha": 0.5,
                }
            },
            {},
        )
        self.assertEqual(0.0, metrics["domain_balance/ast/effective_share"])
        self.assertEqual(0.0, metrics["domain_balance/ast/loss_weight"])
        self.assertAlmostEqual(
            1.0,
            metrics["domain_balance/cot/effective_share"]
            + metrics["domain_balance/style/effective_share"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
