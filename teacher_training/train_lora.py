from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .domains import require_domain


def load_config(path: str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config["domain"] = require_domain(config["domain"])
    for key in ("model_name_or_path", "train_file", "validation_file", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing required config key: {key}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 LoRA/QLoRA 训练一个领域 Teacher；不会加载或更新 Student。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["qwen3_thinking"] = bool(cfg.get("qwen3_thinking", False))
    cfg["training_references"] = {
        "lora": "Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models",
        "qlora": "Dettmers et al. (2023), QLoRA",
        "sft": "Zhang et al. (2023), Instruction Tuning for Large Language Models",
        "assistant_only_loss": "TRL SFT assistant-only loss；只对目标回答计算监督损失",
        "base_model": "Qwen/Qwen3-4B model card",
    }
    if args.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("训练前请在 GPU 环境安装 teacher_training/requirements.txt") from exc

    quantization = None
    if cfg.get("load_in_4bit", True):
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=cfg.get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    if quantization is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    data = load_dataset("json", data_files={"train": cfg["train_file"], "validation": cfg["validation_file"]})

    lora = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
    )
    training = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        max_length=int(cfg.get("max_length", 2048)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_steps=int(cfg.get("save_steps", 100)),
        bf16=True,
        gradient_checkpointing=True,
        report_to=cfg.get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
        assistant_only_loss=bool(cfg.get("assistant_only_loss", True)),
    )
    trainer = SFTTrainer(model=model, args=training, train_dataset=data["train"], eval_dataset=data["validation"], peft_config=lora)
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    (Path(cfg["output_dir"]) / "teacher_manifest.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
