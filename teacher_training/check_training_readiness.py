"""在正式训练前执行失败即停止的四领域数据就绪审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domains import DOMAINS
from .io_utils import read_jsonl
from .source_policy import validate_source_for_domain
from .train_lora import load_config, validate_dataset_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="检查四个 Teacher 是否已具备正式训练条件。")
    parser.add_argument("--config-dir", default="teacher_training/configs")
    parser.add_argument("--min-train", type=int, default=5000)
    parser.add_argument("--min-validation", type=int, default=500)
    parser.add_argument("--min-test", type=int, default=500)
    args = parser.parse_args()
    report, errors = {}, []
    for domain in DOMAINS:
        try:
            config = load_config(str(Path(args.config_dir) / f"{domain}.json"))
            if config.get("formal_experiment") is not True:
                raise ValueError("formal_experiment 尚未设为 true")
            summary = validate_dataset_contract(config)
            test_path = Path(config["validation_file"]).with_name("test.jsonl")
            if not test_path.exists():
                raise FileNotFoundError(f"测试集不存在：{test_path}")
            test_rows = list(read_jsonl(test_path))
            for row in test_rows:
                validate_source_for_domain(domain, str(row.get("source_dataset", "")))
                if not row.get("tests"):
                    raise ValueError(f"测试样本 {row.get('source_id')} 缺少可执行 tests")
            counts = {
                "train": summary["train_file"]["samples"],
                "validation": summary["validation_file"]["samples"],
                "test": len(test_rows),
            }
            minimums = {
                "train": args.min_train,
                "validation": args.min_validation,
                "test": args.min_test,
            }
            insufficient = [name for name in counts if counts[name] < minimums[name]]
            if insufficient:
                raise ValueError(
                    "样本量低于预注册最低值：" + ", ".join(
                        f"{name}={counts[name]}<{minimums[name]}" for name in insufficient
                    )
                )
            report[domain] = {"ready": True, "counts": counts}
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            report[domain] = {"ready": False, "reason": str(exc)}
            errors.append(domain)
    result = {"ready": not errors, "domains": report}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
