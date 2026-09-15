from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from .domains import (
    DOMAIN_SPECS,
    OUTPUT_SCHEMA_VERSION,
    PLAN_ROLES,
    PLAN_TYPES,
    PUBLIC_DOMAIN_NAMES,
    SYSTEM_PROMPT_VERSION,
    require_domain,
)


def load_config(path: str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config["domain"] = require_domain(config["domain"])
    for key in ("model_name_or_path", "train_file", "validation_file", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing required config key: {key}")
    if config.get("assistant_only_loss") is not True:
        raise ValueError("五个 Teacher 必须统一使用 assistant-only causal language modeling loss；assistant_only_loss 必须为 true")
    config["training_objective"] = "assistant-only causal language modeling loss"
    config["domain_name"] = PUBLIC_DOMAIN_NAMES[config["domain"]]
    config["plan_role"] = PLAN_ROLES[config["domain"]]
    config["plan_type"] = PLAN_TYPES[config["domain"]]
    config["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    config["system_prompt_version"] = SYSTEM_PROMPT_VERSION
    return config


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_contract(config: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for split_key in ("train_file", "validation_file"):
        path = Path(config[split_key])
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("domain") != config["domain"]:
                    raise ValueError(f"{path}:{line_number} 的 domain 与配置不一致")
                expected = {
                    "domain_name": PUBLIC_DOMAIN_NAMES[config["domain"]],
                    "plan_role": PLAN_ROLES[config["domain"]],
                    "plan_type": PLAN_TYPES[config["domain"]],
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "system_prompt_version": SYSTEM_PROMPT_VERSION,
                }
                for key, value in expected.items():
                    if row.get(key) != value:
                        raise ValueError(f"{path}:{line_number} 的 {key} 不是当前协议值 {value!r}；请重新运行 prepare_data")
                if row.get("experiment_status") != "verified_positive":
                    raise ValueError(f"{path}:{line_number} 不是 verified_positive 正样本")
                if row.get("semantic_pass") is not True:
                    raise ValueError(f"{path}:{line_number} semantic_pass 必须为 true")
                messages = row.get("messages") or []
                roles = [item.get("role") for item in messages]
                if roles != ["system", "user", "assistant"] or not messages[-1].get("content", "").strip():
                    raise ValueError(f"{path}:{line_number} 的 messages 必须为非空 system/user/assistant")
                if messages[0].get("content") != DOMAIN_SPECS[config["domain"]].system_prompt:
                    raise ValueError(f"{path}:{line_number} 的 system prompt 与当前 {config['domain_name']} 专家不一致")
                assistant = messages[-1]["content"]
                if "<plan>" not in assistant or "</plan>" not in assistant or "<code>" not in assistant or "</code>" not in assistant:
                    raise ValueError(f"{path}:{line_number} 的 assistant 输出必须包含完整的 <plan> 和 <code> 标签")
                rows.append(row)
        if not rows:
            raise ValueError(f"数据文件为空：{path}")
        source_ids = [row.get("source_id") for row in rows]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{path} 存在重复 source_id")
        summary[split_key] = {"path": str(path.resolve()), "samples": len(rows), "sha256": _file_sha256(str(path))}
    return summary


def _runtime_manifest(config: dict[str, Any], data_summary: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    manifest = {
        "domain": config["domain"],
        "base_model": config["model_name_or_path"],
        "qwen3_thinking": bool(config.get("qwen3_thinking", False)),
        "data": data_summary,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "tokenizer_vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for package in ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes"):
        try:
            module = __import__(package)
            manifest["packages"][package] = getattr(module, "__version__", "unknown")
        except Exception:
            manifest["packages"][package] = "unavailable"
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        manifest["git_commit"] = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(0)
            manifest["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return manifest


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
    data_summary = validate_dataset_contract(cfg)
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
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    if quantization is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
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
        assistant_only_loss=True,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
    )
    trainer = SFTTrainer(model=model, args=training, train_dataset=data["train"], eval_dataset=data["validation"], peft_config=lora)
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    manifest = {**cfg, "dataset_summary": data_summary, "runtime": _runtime_manifest(cfg, data_summary, tokenizer)}
    (Path(cfg["output_dir"]) / "teacher_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
