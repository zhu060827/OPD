"""训练服务器预检：GPU、Python 依赖、模型目录和四份配置路径。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from .domains import DOMAINS
from .train_lora import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 QLoRA Teacher 训练服务器是否就绪。")
    parser.add_argument("--config-dir", default="teacher_training/configs")
    args = parser.parse_args()
    errors: list[str] = []
    required_env = ("TEACHER_PROJECT_ROOT", "TEACHER_MODEL_DIR", "TEACHER_OUTPUT_DIR")
    for name in required_env:
        if not os.environ.get(name):
            errors.append(f"未设置环境变量 {name}")
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            errors.append("未检测到 CUDA GPU")
        else:
            index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            if index >= gpu_count:
                errors.append(f"CUDA_VISIBLE_DEVICES={index} 超出可见 GPU 数量 {gpu_count}")
    except ImportError:
        errors.append("未安装 torch")
        gpu_count = 0
    except Exception as exc:
        errors.append(f"GPU 检查失败：{exc}")
        gpu_count = 0
    for package in ("transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes"):
        try:
            importlib.import_module(package)
        except ImportError:
            errors.append(f"未安装 Python 包 {package}")
    configs = {}
    for domain in DOMAINS:
        try:
            config = load_config(str(Path(args.config_dir) / f"{domain}.json"))
            configs[domain] = {
                "model_exists": Path(config["model_name_or_path"]).is_dir(),
                "train_exists": Path(config["train_file"]).is_file(),
                "validation_exists": Path(config["validation_file"]).is_file(),
                "output_dir": config["output_dir"],
            }
            for key in ("model_name_or_path", "train_file", "validation_file"):
                if not Path(config[key]).exists():
                    errors.append(f"{domain} 的 {key} 不存在：{config[key]}")
        except (OSError, ValueError) as exc:
            errors.append(f"{domain} 配置无效：{exc}")
    result = {"ready": not errors, "gpu_count": gpu_count, "configs": configs, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
