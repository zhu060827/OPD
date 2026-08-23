from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Any, Dict, Mapping

from ..llm import (
    MockOPDDistributionScorer,
    MockRewriteClient,
    OPDDistributionScorer,
    OpenAICompatibleRewriteClient,
    RewriteLLMClient,
)
from ..models import CodeRecord, RewriteCandidate, TokenDistribution
from .config import ExpertConfig, Stage1Config


class ExpertCandidateGenerator(ABC):
    @abstractmethod
    def generate(self, expert: ExpertConfig, record: CodeRecord) -> RewriteCandidate:
        raise NotImplementedError


class ExpertTrajectoryScorer(ABC):
    @abstractmethod
    def score(
        self,
        expert: ExpertConfig,
        record: CodeRecord,
        candidate: RewriteCandidate,
    ) -> list[TokenDistribution]:
        raise NotImplementedError


class RewriteClientExpertGenerator(ExpertCandidateGenerator):
    def __init__(
        self,
        default_client: RewriteLLMClient | None = None,
        clients: Mapping[str, RewriteLLMClient] | None = None,
    ):
        self.default_client = default_client
        self.clients = dict(clients or {})

    def generate(self, expert: ExpertConfig, record: CodeRecord) -> RewriteCandidate:
        client = self.clients.get(expert.expert_id, self.default_client)
        if client is None:
            raise KeyError(f"No rewrite client configured for expert {expert.expert_id}")
        candidate = client.rewrite(record, strategy=expert.strategy, feedback="")
        candidate.metadata = {
            **candidate.metadata,
            "expert_id": expert.expert_id,
            "expert_strategy": expert.strategy,
        }
        return candidate


class AuditMockRewriteClient(MockRewriteClient):
    """Exercise every expert path without pretending to be a trained model."""

    def rewrite(self, record: CodeRecord, strategy: str, feedback: str = "") -> RewriteCandidate:
        candidate = super().rewrite(record, strategy, feedback)
        if candidate.code.strip() == record.code.strip():
            candidate.code = f"# Mock-only {strategy} expert candidate.\n{record.code.strip()}"
            candidate.raw_response = candidate.code
        candidate.metadata = {
            **candidate.metadata,
            "formal_result": False,
            "purpose": "deterministic_cpu_pipeline_verification",
        }
        return candidate


class DistributionExpertTrajectoryScorer(ExpertTrajectoryScorer):
    def __init__(
        self,
        default_scorer: OPDDistributionScorer | None = None,
        scorers: Mapping[str, OPDDistributionScorer] | None = None,
    ):
        self.default_scorer = default_scorer
        self.scorers = dict(scorers or {})

    def score(self, expert, record, candidate):
        scorer = self.scorers.get(expert.expert_id, self.default_scorer)
        if scorer is None:
            raise KeyError(f"No trajectory scorer configured for expert {expert.expert_id}")
        return scorer.score_student_trajectory(record, candidate, expert.strategy)


def build_generator(config: Stage1Config) -> ExpertCandidateGenerator:
    backend = config.generation_backend
    if backend.backend_type == "mock":
        return RewriteClientExpertGenerator(default_client=AuditMockRewriteClient())
    if backend.backend_type == "openai_compatible":
        clients: Dict[str, RewriteLLMClient] = {}
        for expert in config.enabled_experts:
            generation = expert.generation
            clients[expert.expert_id] = OpenAICompatibleRewriteClient(
                model=generation.get("model"),
                base_url=generation.get("base_url"),
                api_key=_environment_value(generation.get("api_key_env")),
                temperature=float(generation.get("temperature", 0.2)),
                timeout=int(generation.get("timeout", 90)),
            )
        return RewriteClientExpertGenerator(clients=clients)
    return _load_external(backend.external_factory, config, ExpertCandidateGenerator)


def build_trajectory_scorer(config: Stage1Config) -> ExpertTrajectoryScorer:
    backend = config.trajectory_backend
    if backend.backend_type == "mock":
        return DistributionExpertTrajectoryScorer(default_scorer=MockOPDDistributionScorer())
    if backend.backend_type == "openai_compatible":
        raise ValueError(
            "Trajectory scoring requires aligned token log-probabilities. "
            "Use mock for CPU tests or external_factory for local model scoring."
        )
    return _load_external(backend.external_factory, config, ExpertTrajectoryScorer)


def _load_external(spec: str | None, config: Stage1Config, expected_type: type) -> Any:
    if not spec or ":" not in spec:
        raise ValueError("External backend factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    instance = factory(config)
    if not isinstance(instance, expected_type):
        raise TypeError(f"Factory {spec} must return {expected_type.__name__}")
    return instance


def _environment_value(name: str | None) -> str | None:
    if not name:
        return None
    import os

    return os.getenv(name)
