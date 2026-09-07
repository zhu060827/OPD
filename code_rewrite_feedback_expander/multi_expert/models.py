from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from ..models import RewriteCandidate


@dataclass
class GateEvidence:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]


@dataclass
class RewardEvidence:
    total: float
    method_alignment: float
    general_quality: float
    non_regression: float
    change: float
    metric_deltas: Dict[str, float]
    weights: Dict[str, float]
    design_status: str = "project_level_routing_utility_requires_ablation"


@dataclass
class TrajectoryEvidence:
    available: bool
    token_count: int = 0
    mean_teacher_student_nll_advantage: float = 0.0
    median_teacher_student_nll_advantage: float = 0.0
    teacher_win_fraction: float = 0.0
    mean_forward_kl: float = 0.0
    mean_total_variation: float = 0.0
    error: str = ""


@dataclass
class ExpertAssessment:
    expert_id: str
    strategy: str
    candidate: RewriteCandidate
    semantic_result: Dict[str, Any]
    quality_result: Dict[str, Any]
    gate: GateEvidence
    reward: RewardEvidence
    trajectory: TrajectoryEvidence
    normalized_reward: float = 0.0
    normalized_nll_advantage: float = 0.0
    routing_score: float = 0.0
    rank: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["candidate"] = {
            "code": self.candidate.code,
            "reasoning": self.candidate.reasoning,
            "rationale": self.candidate.rationale,
            "strategy": self.candidate.strategy,
            "metadata": self.candidate.metadata,
        }
        return data


@dataclass
class RoutingDecision:
    status: str
    pseudo_method_label: str | None
    selected_expert_id: str | None
    top_k: List[Dict[str, Any]] = field(default_factory=list)
    expert_weights: Dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    usable_for_training: bool = False
    reason: str = ""
    router_design: str = "three_tier_recorded_label_then_calibrated_opd"
    routing_source: str = ""


@dataclass
class Stage1RecordResult:
    task_id: str
    prompt: str
    original_code: str
    tests: List[str]
    language: str
    assessments: List[ExpertAssessment]
    routing: RoutingDecision
    verification_status: str
    downstream_action: str

    def to_dict(self) -> Dict[str, Any]:
        selected = next(
            (item for item in self.assessments if item.expert_id == self.routing.selected_expert_id),
            None,
        )
        return {
            "schema_version": "stage1_multi_expert.v2",
            "task_id": self.task_id,
            "prompt": self.prompt,
            "original_code": self.original_code,
            "tests": self.tests,
            "language": self.language,
            "pseudo_method_label": self.routing.pseudo_method_label,
            "domain": self.routing.pseudo_method_label,
            "selected_expert_id": self.routing.selected_expert_id,
            "routing_confidence": self.routing.margin,
            "verification_status": self.verification_status,
            "downstream_action": self.downstream_action,
            "routing": asdict(self.routing),
            "expert_assessments": [item.to_dict() for item in self.assessments],
            "selected_candidate": (
                {
                    "code": selected.candidate.code,
                    "reasoning": selected.candidate.reasoning,
                    "rationale": selected.candidate.rationale,
                }
                if selected is not None
                else None
            ),
            "mt_opd_handoff": {
                "domain": self.routing.pseudo_method_label,
                "teacher_id": self.routing.selected_expert_id,
                "teacher_weights": self.routing.expert_weights,
                "routing_source": self.routing.routing_source,
                "routing_confidence": self.routing.margin,
                "usable_for_training": self.routing.usable_for_training,
                "verification_status": self.verification_status,
                "downstream_action": self.downstream_action,
                "positive_augmentation_eligible": self.downstream_action
                == "positive_augmentation",
            },
        }
