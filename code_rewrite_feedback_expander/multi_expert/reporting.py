from __future__ import annotations

from collections import Counter, defaultdict
import statistics
from typing import Any, Dict, Iterable, List

from .models import Stage1RecordResult


def build_summary(results: Iterable[Stage1RecordResult]) -> Dict[str, Any]:
    rows = list(results)
    labels = Counter(item.routing.pseudo_method_label or "unrouted" for item in rows)
    statuses = Counter(item.routing.status for item in rows)
    verification_statuses = Counter(item.verification_status for item in rows)
    downstream_actions = Counter(item.downstream_action for item in rows)
    per_expert: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "attempted": 0,
            "hard_gate_passed": 0,
            "trajectory_available": 0,
            "selected_top1": 0,
            "selected_top2": 0,
            "reward_scores": [],
            "nll_advantages": [],
        }
    )
    for row in rows:
        top2_ids = {item["expert_id"] for item in row.routing.top_k[:2]}
        for assessment in row.assessments:
            metrics = per_expert[assessment.expert_id]
            metrics["attempted"] += 1
            metrics["hard_gate_passed"] += int(assessment.gate.passed)
            metrics["trajectory_available"] += int(assessment.trajectory.available)
            metrics["selected_top1"] += int(
                assessment.expert_id == row.routing.selected_expert_id
            )
            metrics["selected_top2"] += int(assessment.expert_id in top2_ids)
            metrics["reward_scores"].append(assessment.reward.total)
            if assessment.trajectory.available:
                metrics["nll_advantages"].append(
                    assessment.trajectory.mean_teacher_student_nll_advantage
                )
    expert_summary = {}
    for expert_id, metrics in sorted(per_expert.items()):
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
    margins = [item.routing.margin for item in rows if item.routing.selected_expert_id]
    return {
        "schema_version": "stage1_multi_expert.summary.v2",
        "formal_training_result": False,
        "total_records": len(rows),
        "usable_for_training": sum(item.routing.usable_for_training for item in rows),
        "routing_status_distribution": dict(sorted(statuses.items())),
        "verification_status_distribution": dict(sorted(verification_statuses.items())),
        "downstream_action_distribution": dict(sorted(downstream_actions.items())),
        "pseudo_label_distribution": dict(sorted(labels.items())),
        "mean_top1_margin": statistics.fmean(margins) if margins else 0.0,
        "experts": expert_summary,
    }


def build_mt_opd_handoff(result: Stage1RecordResult) -> Dict[str, Any] | None:
    if not result.routing.usable_for_training or not result.routing.selected_expert_id:
        return None
    selected = next(
        item for item in result.assessments if item.expert_id == result.routing.selected_expert_id
    )
    return {
        "data_source": "code_multi_expert_stage1",
        "task_id": result.task_id,
        "prompt": [{"role": "user", "content": result.prompt}],
        "ability": "code",
        "response": selected.candidate.code,
        "domain": result.routing.pseudo_method_label,
        "teacher_id": result.routing.selected_expert_id,
        "teacher_weights": result.routing.expert_weights,
        "routing_source": result.routing.routing_source,
        "routing_confidence": result.routing.margin,
        "verification_status": result.verification_status,
        "downstream_action": result.downstream_action,
        "opd_training_eligible": result.routing.usable_for_training,
        "positive_augmentation_eligible": result.downstream_action
        == "positive_augmentation",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"tests": result.tests},
        },
        "extra_info": {
            "tests": result.tests,
            "original_code": result.original_code,
            "routing_status": result.routing.status,
            "routing_margin": result.routing.margin,
            "top_k": result.routing.top_k,
            "routing_source": result.routing.routing_source,
            "formal_training_result": False,
            "verification_status": result.verification_status,
            "downstream_action": result.downstream_action,
        },
    }
