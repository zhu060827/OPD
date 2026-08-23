from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..io_utils import read_jsonl, write_jsonl
from .backends import build_generator, build_trajectory_scorer
from .config import load_config
from .pipeline import MultiExpertStage1Pipeline
from .reporting import build_mt_opd_handoff, build_summary


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
    run(config_path=args.config, input_path=args.input, output_dir=args.output_dir, limit=args.limit)


def run(config_path: str, input_path: str, output_dir: str, limit: int = 0) -> None:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(input_path)
    if limit > 0:
        records = records[:limit]
    pipeline = MultiExpertStage1Pipeline(
        config=config,
        generator=build_generator(config),
        trajectory_scorer=build_trajectory_scorer(config),
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
                "generation_backend": config.generation_backend.backend_type,
                "trajectory_backend": config.trajectory_backend.backend_type,
                "external_paid_api_calls": 0
                if config.generation_backend.backend_type == "mock"
                else "not_measured_by_stage1",
                "formal_training_result": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Stage-1 artifacts to {output.resolve()}")


if __name__ == "__main__":
    main()
