from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io_utils import read_jsonl, write_jsonl
from .backends import build_generator, build_trajectory_scorer
from .config import load_config
from .pipeline import MultiExpertStage1Pipeline
from .reporting import build_mt_opd_handoff, build_summary
from .transformers_backend import create_trajectory_scorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate auditable pseudo-routing labels for five code experts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config", help="Validate canonical JSON config.")
    validate.add_argument("--config", required=True)

    run = subparsers.add_parser("run", help="Run Stage-1 multi-expert routing.")
    run.add_argument("--config", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--local", action="store_true", help="Use local Transformers for generation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-config":
        print(
            f"Valid config: {config.experiment_name}; "
            f"experts={len(config.enabled_experts)}"
        )
        return
    run(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        use_local=args.local,
    )


def run(config_path: str, input_path: str, output_dir: str, limit: int = 0, use_local: bool = False) -> None:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(input_path)
    if limit > 0:
        records = records[:limit]

    # 如果使用本地生成，先创建评分器（加载模型），再复用模型给生成器
    scorer = None
    generator = None

    if use_local:
        print("🔧 使用本地模型模式（生成 + 评分共享权重）...")
        # 先加载评分器（内含模型）
        from .transformers_backend import create_trajectory_scorer
        scorer = create_trajectory_scorer(config)
        # 获取共享的模型和 tokenizer
        shared_model, shared_tokenizer = scorer.get_shared_model_and_tokenizer()
        # 用共享模型创建生成器
        generator = build_generator(config, shared_model=shared_model, shared_tokenizer=shared_tokenizer)
        print("✅ 生成器和评分器共享同一模型权重")
    else:
        # 原有逻辑：生成器和评分器独立
        generator = build_generator(config)
        scorer = build_trajectory_scorer(config)

    pipeline = MultiExpertStage1Pipeline(
        config=config,
        generator=generator,
        trajectory_scorer=scorer,
    )

    results = []
    for index, record in enumerate(records, start=1):
        result = pipeline.process(record)
        results.append(result)
        print(
            f"[{index}/{len(records)}] task={record.task_id} "
            f"status={result.routing.status} label={result.routing.pseudo_method_label} "
            f"margin={result.routing.margin:.4f}"
        )

    write_jsonl(str(output / "routing_labels.jsonl"), [item.to_dict() for item in results])
    handoff = [build_mt_opd_handoff(item) for item in results]
    write_jsonl(str(output / "mt_opd_handoff.jsonl"), [item for item in handoff if item])

    (output / "summary.json").write_text(
        json.dumps(build_summary(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "input": str(Path(input_path).resolve()),
                "record_count": len(results),
                "generation_backend": "local_transformers" if use_local else config.generation_backend.backend_type,
                "trajectory_backend": "local_transformers" if use_local else config.trajectory_backend.backend_type,
                "formal_training_result": True,
                "note": "Full local mode: generation + scoring using shared Qwen 7B model",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Stage-1 artifacts to {output.resolve()}")


if __name__ == "__main__":
    main()
