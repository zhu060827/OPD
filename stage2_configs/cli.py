from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set

from ..io_utils import read_jsonl, write_jsonl
from .backends import build_generator, build_trajectory_scorer
from .config import load_config
from .pipeline import MultiExpertStage1Pipeline


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
    run.add_argument("--resume", action="store_true", help="Resume from existing output")
    run.add_argument("--write-interval", type=int, default=10, help="Write results every N samples")
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
        resume=args.resume,
        write_interval=args.write_interval,
    )


def run(
    config_path: str,
    input_path: str,
    output_dir: str,
    limit: int = 0,
    resume: bool = False,
    write_interval: int = 10,
) -> None:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    routing_path = output / "routing_labels.jsonl"
    handoff_path = output / "mt_opd_handoff.jsonl"
    summary_path = output / "summary.json"

    processed_ids: Set[str] = set()
    existing_results = []
    if resume and routing_path.exists():
        print(f"📂 检测到已有结果文件，正在加载已处理的 task_id...")
        with routing_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    if task_id:
                        processed_ids.add(str(task_id))
                        existing_results.append(data)
                except json.JSONDecodeError:
                    continue
        print(f"✅ 已加载 {len(processed_ids)} 条已处理记录，将从断点继续")

    records = read_jsonl(input_path)
    if limit > 0:
        records = records[:limit]

    if resume and processed_ids:
        original_count = len(records)
        records = [r for r in records if r.task_id not in processed_ids]
        print(f"📊 跳过 {original_count - len(records)} 条已处理样本，剩余 {len(records)} 条待处理")

    if not records:
        print("✅ 所有样本已处理完成，无需继续")
        if existing_results:
            _write_final_outputs(existing_results, handoff_path, summary_path)
        return

    pipeline = MultiExpertStage1Pipeline(
        config=config,
        generator=build_generator(config),
        trajectory_scorer=build_trajectory_scorer(config),
    )

    results = existing_results.copy()
    total = len(records) + len(processed_ids)
    processed_in_session = 0

    for index, record in enumerate(records, start=1):
        result = pipeline.process(record)
        result_dict = result.to_dict()
        results.append(result_dict)
        processed_in_session += 1

        print(
            f"[{len(processed_ids) + processed_in_session}/{total}] "
            f"task={record.task_id} "
            f"status={result.routing.status} label={result.routing.pseudo_method_label} "
            f"margin={result.routing.margin:.4f}"
        )

        if processed_in_session % write_interval == 0 or index == len(records):
            _append_routing_results(results, routing_path)
            handoff = [build_mt_opd_handoff_for_dict(r) for r in results]
            handoff = [h for h in handoff if h]
            write_jsonl(str(handoff_path), handoff)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(build_summary_from_dicts(results), f, ensure_ascii=False, indent=2)

    _append_routing_results(results, routing_path)
    handoff = [build_mt_opd_handoff_for_dict(r) for r in results]
    handoff = [h for h in handoff if h]
    write_jsonl(str(handoff_path), handoff)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(build_summary_from_dicts(results), f, ensure_ascii=False, indent=2)

    print(f"✅ 全部完成！共 {len(results)} 条记录")
    print(f"   路由标签: {routing_path}")
    print(f"   训练数据: {handoff_path}")
    print(f"   汇总报告: {summary_path}")


def _append_routing_results(results, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_mt_opd_handoff_for_dict(result_dict: dict) -> dict | None:
    routing = result_dict.get("routing", {})
    if not routing.get("usable_for_training") or not routing.get("selected_expert_id"):
        return None
    selected = next(
        (item for item in result_dict.get("expert_assessments", [])
         if item.get("expert_id") == routing.get("selected_expert_id")),
        None,
    )
    if selected is None:
        return None
    return {
        "data_source": "code_multi_expert_stage1",
        "task_id": result_dict.get("task_id"),
        "prompt": [{"role": "user", "content": result_dict.get("prompt")}],
        "ability": "code",
        "response": selected.get("candidate", {}).get("code", ""),
        "domain": routing.get("pseudo_method_label"),
        "teacher_id": routing.get("selected_expert_id"),
        "teacher_weights": routing.get("expert_weights", {}),
        "routing_source": routing.get("routing_source", ""),
        "routing_confidence": routing.get("margin", 0.0),
        "routing_loss_weight": routing.get("opd_sample_weight", 1.0),
        "verification_status": result_dict.get("verification_status", "semantic_unverified"),
        "downstream_action": result_dict.get("downstream_action", "unverified_pool"),
        "opd_training_eligible": routing.get("usable_for_training", False),
        "requires_stage2_rescore": routing.get("status") == "fallback_missing_trajectory",
        "positive_augmentation_eligible": result_dict.get("downstream_action") == "positive_augmentation",
        "reward_model": {"style": "rule", "ground_truth": {"tests": result_dict.get("tests", [])}},
        "extra_info": {
            "tests": result_dict.get("tests", []),
            "original_code": result_dict.get("original_code", ""),
            "routing_status": routing.get("status", "unknown"),
            "routing_margin": routing.get("margin", 0.0),
            "top_k": routing.get("top_k", []),
            "routing_source": routing.get("routing_source", ""),
            "formal_training_result": False,
            "verification_status": result_dict.get("verification_status", "semantic_unverified"),
            "downstream_action": result_dict.get("downstream_action", "unverified_pool"),
        },
    }


def build_summary_from_dicts(records: list[dict]) -> dict:
    from collections import Counter, defaultdict
    import statistics

    labels = Counter(r.get("pseudo_method_label") or "unrouted" for r in records)
    statuses = Counter(r.get("routing", {}).get("status", "unknown") for r in records)
    verification_statuses = Counter(r.get("verification_status", "semantic_unverified") for r in records)
    downstream_actions = Counter(r.get("downstream_action", "unverified_pool") for r in records)

    per_expert = defaultdict(lambda: {
        "attempted": 0,
        "hard_gate_passed": 0,
        "trajectory_available": 0,
        "selected_top1": 0,
        "selected_top2": 0,
        "reward_scores": [],
        "nll_advantages": [],
    })

    for record in records:
        routing = record.get("routing", {})
        top2_ids = {item.get("expert_id") for item in routing.get("top_k", [])[:2]}
        for assessment in record.get("expert_assessments", []):
            expert_id = assessment.get("expert_id")
            if expert_id not in per_expert:
                continue
            metrics = per_expert[expert_id]
            metrics["attempted"] += 1
            metrics["hard_gate_passed"] += int(assessment.get("gate", {}).get("passed", False))
            metrics["trajectory_available"] += int(assessment.get("trajectory", {}).get("available", False))
            metrics["selected_top1"] += int(expert_id == routing.get("selected_expert_id"))
            metrics["selected_top2"] += int(expert_id in top2_ids)
            reward = assessment.get("reward", {})
            metrics["reward_scores"].append(reward.get("total", 0.0))
            trajectory = assessment.get("trajectory", {})
            if trajectory.get("available"):
                metrics["nll_advantages"].append(trajectory.get("mean_teacher_student_nll_advantage", 0.0))

    expert_summary = {}
    for expert_id, metrics in per_expert.items():
        attempted = metrics.pop("attempted")
        rewards = metrics.pop("reward_scores")
        advantages = metrics.pop("nll_advantages")
        expert_summary[expert_id] = {
            "attempted": attempted,
            **metrics,
            "hard_gate_pass_rate": metrics["hard_gate_passed"] / max(attempted, 1),
            "mean_reward_score": statistics.fmean(rewards) if rewards else 0.0,
            "mean_nll_advantage": statistics.fmean(advantages) if advantages else 0.0,
        }

    margins = [r.get("routing", {}).get("margin", 0.0) for r in records if r.get("routing", {}).get("selected_expert_id")]

    return {
        "schema_version": "stage1_multi_expert.summary.v1",
        "formal_training_result": False,
        "total_records": len(records),
        "usable_for_training": sum(1 for r in records if r.get("routing", {}).get("usable_for_training", False)),
        "routing_status_distribution": dict(sorted(statuses.items())),
        "verification_status_distribution": dict(sorted(verification_statuses.items())),
        "downstream_action_distribution": dict(sorted(downstream_actions.items())),
        "pseudo_label_distribution": dict(sorted(labels.items())),
        "mean_top1_margin": statistics.fmean(margins) if margins else 0.0,
        "experts": expert_summary,
    }


def _write_final_outputs(results: list[dict], handoff_path: Path, summary_path: Path) -> None:
    handoff = [build_mt_opd_handoff_for_dict(r) for r in results]
    handoff = [h for h in handoff if h]
    write_jsonl(str(handoff_path), handoff)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(build_summary_from_dicts(results), f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总已生成: {summary_path}")
    print(f"   可训练样本: {len(handoff)}/{len(results)}")


if __name__ == "__main__":
    main()
