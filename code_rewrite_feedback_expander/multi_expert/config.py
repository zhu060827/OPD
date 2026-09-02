from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExpertConfig:
    expert_id: str
    strategy: str
    enabled: bool = True
    generation: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateConfig:
    require_tests: bool = True
    require_candidate_change: bool = True
    minimum_unit_test_pass_rate: float = 1.0
    score_failed_candidates: bool = False


@dataclass(frozen=True)
class RewardConfig:
    quality_gain_scale: float = 0.05
    regression_tolerance: float = 0.05
    method_alignment_weight: float = 0.40
    general_quality_weight: float = 0.30
    non_regression_weight: float = 0.20
    change_weight: float = 0.10


@dataclass(frozen=True)
class RoutingConfig:
    policy: str = "heuristic_ablation"
    recorded_label_fields: List[str] = field(
        default_factory=lambda: ["method", "domain", "rewrite_method"]
    )
    calibration: Dict[str, Dict[str, float]] = field(default_factory=dict)
    shared_completion_source: str = "record_code"
    student_max_new_tokens: int = 512
    reward_weight: float = 0.55
    nll_advantage_weight: float = 0.45
    top_k: int = 2
    softmax_temperature: float = 0.20
    minimum_margin: float = 0.05
    abstain_on_low_confidence: bool = False
    fallback_expert_id: str | None = None


@dataclass(frozen=True)
class BackendConfig:
    backend_type: str = "mock"
    external_factory: str | None = None


@dataclass(frozen=True)
class Stage1Config:
    schema_version: str
    experiment_name: str
    seed: int
    expected_expert_count: int
    experts: List[ExpertConfig]
    generation_backend: BackendConfig = field(default_factory=BackendConfig)
    trajectory_backend: BackendConfig = field(default_factory=BackendConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Stage1Config":
        config = cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            experiment_name=str(raw.get("experiment_name", "stage1_multi_expert")),
            seed=int(raw.get("seed", 0)),
            expected_expert_count=int(raw.get("expected_expert_count", 5)),
            experts=[ExpertConfig(**item) for item in raw.get("experts", [])],
            generation_backend=BackendConfig(**_backend_dict(raw.get("generation_backend", {}))),
            trajectory_backend=BackendConfig(**_backend_dict(raw.get("trajectory_backend", {}))),
            gate=GateConfig(**raw.get("gate", {})),
            reward=RewardConfig(**raw.get("reward", {})),
            routing=RoutingConfig(**raw.get("routing", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        enabled = [expert for expert in self.experts if expert.enabled]
        if len(enabled) != self.expected_expert_count:
            raise ValueError(
                f"Expected {self.expected_expert_count} enabled experts, found {len(enabled)}"
            )
        expert_ids = [expert.expert_id for expert in enabled]
        strategies = [expert.strategy for expert in enabled]
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError("expert_id values must be unique")
        if len(strategies) != len(set(strategies)):
            raise ValueError("strategy values must be unique")
        if not 0.0 <= self.gate.minimum_unit_test_pass_rate <= 1.0:
            raise ValueError("minimum_unit_test_pass_rate must be in [0, 1]")
        reward_weights = (
            self.reward.method_alignment_weight,
            self.reward.general_quality_weight,
            self.reward.non_regression_weight,
            self.reward.change_weight,
        )
        if any(weight < 0.0 for weight in reward_weights) or sum(reward_weights) <= 0.0:
            raise ValueError("Reward weights must be non-negative with a positive sum")
        if self.reward.quality_gain_scale <= 0.0:
            raise ValueError("quality_gain_scale must be positive")
        if self.routing.reward_weight < 0.0 or self.routing.nll_advantage_weight < 0.0:
            raise ValueError("Routing weights must be non-negative")
        if self.routing.reward_weight + self.routing.nll_advantage_weight <= 0.0:
            raise ValueError("At least one routing weight must be positive")
        if not 1 <= self.routing.top_k <= len(enabled):
            raise ValueError("routing.top_k must be between 1 and the enabled expert count")
        if self.routing.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        if self.routing.policy not in {"three_tier", "heuristic_ablation"}:
            raise ValueError("routing.policy must be three_tier or heuristic_ablation")
        if self.routing.shared_completion_source not in {"record_code", "student_generate"}:
            raise ValueError(
                "routing.shared_completion_source must be record_code or student_generate"
            )
        if self.routing.student_max_new_tokens <= 0:
            raise ValueError("routing.student_max_new_tokens must be positive")
        if not self.routing.recorded_label_fields:
            raise ValueError("routing.recorded_label_fields must not be empty")
        if self.routing.fallback_expert_id not in {None, *expert_ids}:
            raise ValueError("routing.fallback_expert_id must name an enabled expert")
        if self.routing.policy == "three_tier":
            missing = set(expert_ids) - set(self.routing.calibration)
            if missing:
                raise ValueError(
                    "three_tier routing requires calibration for every expert: "
                    + ", ".join(sorted(missing))
                )
            for expert_id in expert_ids:
                entry = self.routing.calibration[expert_id]
                if float(entry.get("scale", 0.0)) <= 0.0:
                    raise ValueError(f"Calibration scale must be positive for {expert_id}")
        for backend in (self.generation_backend, self.trajectory_backend):
            if backend.backend_type not in {"mock", "openai_compatible", "external"}:
                raise ValueError(f"Unsupported backend_type: {backend.backend_type}")
            if backend.backend_type == "external" and not backend.external_factory:
                raise ValueError("external backends require external_factory=module:function")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def enabled_experts(self) -> List[ExpertConfig]:
        return [expert for expert in self.experts if expert.enabled]


def _backend_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(raw)
    if "type" in result and "backend_type" not in result:
        result["backend_type"] = result.pop("type")
    return result


def load_config(path: str | Path) -> Stage1Config:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Stage-1 config must be a JSON object: {source}")
    return Stage1Config.from_dict(raw)
