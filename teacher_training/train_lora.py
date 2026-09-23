from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .domains import (
    DOMAIN_SPECS,
    OUTPUT_SCHEMA_VERSION,
    PUBLIC_DOMAIN_NAMES,
    SYSTEM_PROMPT_VERSION,
    require_domain,
)
from .source_policy import validate_source_for_domain


def load_config(path: str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    # 支持实验室服务器通过环境变量切换项目、模型和输出目录。
    for key in ("model_name_or_path", "train_file", "validation_file", "output_dir"):
        if isinstance(config.get(key), str):
            config[key] = os.path.expandvars(config[key])
    config["domain"] = require_domain(config["domain"])
    for key in ("model_name_or_path", "train_file", "validation_file", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing required config key: {key}")
    if config.get("assistant_only_loss") is not True:
        raise ValueError("四个 Teacher 必须统一使用 assistant-only causal language modeling loss；assistant_only_loss 必须为 true")
    config["training_objective"] = "assistant-only causal language modeling loss"
    config["domain_name"] = PUBLIC_DOMAIN_NAMES[config["domain"]]
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
    split_problem_ids: dict[str, set[str]] = {}
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
                if config.get("formal_experiment", False) and row.get("formal_data_contract_pass") is not True:
                    raise ValueError(f"{path}:{line_number} 未通过正式数据契约；请使用 prepare_data --formal 重新处理")
                if config.get("formal_experiment", False):
                    validate_source_for_domain(config["domain"], str(row.get("source_dataset", "")))
                messages = row.get("messages") or []
                roles = [item.get("role") for item in messages]
                if roles != ["system", "user", "assistant"] or not messages[-1].get("content", "").strip():
                    raise ValueError(f"{path}:{line_number} 的 messages 必须为非空 system/user/assistant")
                if messages[0].get("content") != DOMAIN_SPECS[config["domain"]].system_prompt:
                    raise ValueError(f"{path}:{line_number} 的 system prompt 与当前 {config['domain_name']} 专家不一致")
                assistant = messages[-1]["content"]
                if "<code>" not in assistant or "</code>" not in assistant:
                    raise ValueError(f"{path}:{line_number} 的 assistant 输出必须包含完整的 <code> 标签")
                if "<reasoning>" in assistant or "</reasoning>" in assistant:
                    raise ValueError(f"{path}:{line_number} 仍含旧 reasoning 协议；请重新运行 prepare_data")
                rows.append(row)
        if not rows:
            raise ValueError(f"数据文件为空：{path}")
        source_ids = [row.get("source_id") for row in rows]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{path} 存在重复 source_id")
        problem_ids = {str(row.get("problem_id", row.get("source_id"))) for row in rows}
        split_problem_ids[split_key] = problem_ids
        summary[split_key] = {"path": str(path.resolve()), "samples": len(rows), "sha256": _file_sha256(str(path))}
    overlap = split_problem_ids["train_file"] & split_problem_ids["validation_file"]
    if overlap:
        raise ValueError(f"训练集与验证集存在 problem_id 泄漏；示例：{sorted(overlap)[:5]}")
    return summary


def build_dry_run_report(config: dict[str, Any]) -> dict[str, Any]:
    """在不加载模型的前提下完成真实的训练前数据审计。"""
    return {
        "status": "ready_for_model_loading",
        "config": config,
        "dataset_summary": validate_dataset_contract(config),
    }


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
    declared_version = os.environ.get("TEACHER_CODE_VERSION", "").strip()
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        manifest["git_commit"] = declared_version or "unknown"
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
        print(json.dumps(build_dry_run_report(cfg), ensure_ascii=False, indent=2))
        return
    data_summary = validate_dataset_contract(cfg)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
        from transformers.trainer_utils import get_last_checkpoint
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

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    class ProgressCallback(TrainerCallback):
        """把 Trainer 日志持久化为 JSONL，便于断线后检查正式训练进度。"""

        def __init__(self, path: Path) -> None:
            self.path = path
            self.started_at = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return
            record = {
                "step": int(state.global_step),
                "epoch": state.epoch,
                "elapsed_seconds": round(time.time() - self.started_at, 3),
                **{key: value for key, value in logs.items() if isinstance(value, (int, float, str, bool))},
            }
            if torch.cuda.is_available():
                record["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
                record["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Transformers 5 / TRL 1.x 移除了 warmup_ratio，仅保留 warmup_steps。
    # 在单卡或多卡环境中依据实际优化器步数确定性换算，避免静默丢弃预热策略。
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    epochs = float(cfg.get("num_train_epochs", 3))
    micro_batch = int(cfg.get("per_device_train_batch_size", 2))
    accumulation = int(cfg.get("gradient_accumulation_steps", 8))
    batches_per_epoch = math.ceil(len(data["train"]) / (micro_batch * world_size))
    optimizer_steps = max(1, math.ceil(batches_per_epoch * epochs / accumulation))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    warmup_steps = int(cfg.get("warmup_steps", math.ceil(optimizer_steps * warmup_ratio)))
    cfg["derived_training_steps"] = {
        "world_size": world_size,
        "train_samples": len(data["train"]),
        "optimizer_steps": optimizer_steps,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
    }

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
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
    )
    progress_path = output_dir / "training_progress.jsonl"
    resume_setting = cfg.get("resume_from_checkpoint", False)
    if resume_setting == "auto":
        resume_setting = get_last_checkpoint(str(output_dir))
    if not resume_setting:
        progress_path.write_text("", encoding="utf-8")
    trainer = SFTTrainer(model=model, args=training, train_dataset=data["train"], eval_dataset=data["validation"], peft_config=lora, callbacks=[ProgressCallback(progress_path)])
    trainer.train(resume_from_checkpoint=resume_setting or None)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    trainer.save_state()
    if trainer.is_world_process_zero():
        try:
            import matplotlib.pyplot as plt

            history = trainer.state.log_history
            train_points = [(row["step"], row["loss"]) for row in history if "step" in row and "loss" in row]
            eval_points = [(row["step"], row["eval_loss"]) for row in history if "step" in row and "eval_loss" in row]
            if train_points or eval_points:
                figure, axis = plt.subplots(figsize=(8, 5))
                if train_points:
                    axis.plot(*zip(*train_points), label="train loss")
                if eval_points:
                    axis.plot(*zip(*eval_points), marker="o", label="validation loss")
                axis.set_xlabel("optimizer step")
                axis.set_ylabel("loss")
                axis.set_title(f"{cfg['domain_name']} training curves")
                axis.grid(alpha=0.25)
                axis.legend()
                figure.tight_layout()
                figure.savefig(output_dir / "training_curves.png", dpi=180)
                plt.close(figure)
        except Exception as exc:
            print(f"警告：训练完成，但训练曲线生成失败：{exc}", flush=True)
    manifest = {**cfg, "dataset_summary": data_summary, "runtime": _runtime_manifest(cfg, data_summary, tokenizer)}
    (Path(cfg["output_dir"]) / "teacher_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
