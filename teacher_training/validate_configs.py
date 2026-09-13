from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domains import DOMAINS
from .train_lora import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="检查五个 Teacher 是否使用同一基座模型和一致的超参数。")
    parser.add_argument("--config-dir", default="teacher_training/configs")
    args = parser.parse_args()
    configs = {domain: load_config(str(Path(args.config_dir) / f"{domain}.json")) for domain in DOMAINS}
    base_models = {cfg["model_name_or_path"] for cfg in configs.values()}
    if len(base_models) != 1:
        raise ValueError(f"五个 Teacher 必须共享同一基座模型；当前发现：{sorted(base_models)}")
    compared = ("num_train_epochs", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha", "lora_dropout", "seed")
    mismatches = {}
    for key in compared:
        values = {domain: cfg.get(key) for domain, cfg in configs.items()}
        if len(set(values.values())) != 1:
            mismatches[key] = values
    if mismatches:
        raise ValueError(f"Teacher 超参数不一致：{json.dumps(mismatches, ensure_ascii=False)}")
    print(json.dumps({"valid": True, "domains": list(DOMAINS), "base_model": next(iter(base_models)), "matched_hyperparameters": list(compared)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
