from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domains import DOMAINS
from .train_lora import load_config
from .dataset_contracts import validate_source_registry
from .evaluate_teachers import GATE_KEYS, PRIMARY_KEYS


def main() -> None:
    parser = argparse.ArgumentParser(description="检查四个 Teacher 是否使用同一基座模型和一致的超参数。")
    parser.add_argument("--config-dir", default="teacher_training/configs")
    args = parser.parse_args()
    configs = {domain: load_config(str(Path(args.config_dir) / f"{domain}.json")) for domain in DOMAINS}
    registry_path = Path(__file__).with_name("metric_registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    source_registry = validate_source_registry()
    missing_metrics = [domain for domain in DOMAINS if domain not in registry or not registry[domain].get("主指标") or not registry[domain].get("文献")]
    if missing_metrics:
        raise ValueError(f"以下领域缺少正式主指标或文献：{', '.join(missing_metrics)}")
    missing_auxiliary = [
        domain for domain in DOMAINS
        if not 2 <= len(registry.get(domain, {}).get("辅助指标", [])) <= 4
    ]
    if missing_auxiliary:
        raise ValueError(f"以下领域的辅助指标数量不在 2～4 个之间：{', '.join(missing_auxiliary)}")
    for domain in DOMAINS:
        spec = registry[domain]
        if spec.get("主指标字段") != PRIMARY_KEYS[domain]:
            raise ValueError(f"{domain} 的主指标字段与评估实现不一致")
        for metric in spec["辅助指标"]:
            if not all(metric.get(key) for key in ("指标", "输出字段", "实现", "文献")):
                raise ValueError(f"{domain} 存在缺少名称、输出字段、实现或文献的辅助指标")
    registered_gates = tuple(item.get("输出字段") for item in registry["通用硬门禁"]["核心项目"])
    if registered_gates != GATE_KEYS:
        raise ValueError("指标注册表中的硬门禁字段与评估实现不一致")
    base_models = {cfg["model_name_or_path"] for cfg in configs.values()}
    if len(base_models) != 1:
        raise ValueError(f"四个 Teacher 必须共享同一基座模型；当前发现：{sorted(base_models)}")
    output_dirs = [cfg["output_dir"] for cfg in configs.values()]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("四个 Teacher 必须使用彼此独立的 output_dir，防止适配器相互覆盖")
    compared = ("num_train_epochs", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha", "lora_dropout", "seed")
    mismatches = {}
    for key in compared:
        values = {domain: cfg.get(key) for domain, cfg in configs.items()}
        if len(set(values.values())) != 1:
            mismatches[key] = values
    if mismatches:
        raise ValueError(f"Teacher 超参数不一致：{json.dumps(mismatches, ensure_ascii=False)}")
    print(json.dumps({"valid": True, "training_objective": registry["训练目标"]["名称"], "domains": list(DOMAINS), "base_model": next(iter(base_models)), "matched_hyperparameters": list(compared), "dataset_registry_version": source_registry["registry_version"], "metric_protocol_version": registry["协议版本"], "primary_metrics": {domain: registry[domain]["主指标"] for domain in DOMAINS}, "auxiliary_metric_counts": {domain: len(registry[domain]["辅助指标"]) for domain in DOMAINS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
