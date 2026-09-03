from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Mapping

from ..llm import (
    MockOPDDistributionScorer,
    MockRewriteClient,
    OPDDistributionScorer,
    OpenAICompatibleRewriteClient,
    RewriteLLMClient,
)
from ..models import CodeRecord, RewriteCandidate, TokenDistribution
from .base import ExpertCandidateGenerator, ExpertTrajectoryScorer
from .config import ExpertConfig, Stage1Config


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


def build_generator(config: Stage1Config, shared_model=None, shared_tokenizer=None) -> ExpertCandidateGenerator:
    backend = config.generation_backend

    if backend.backend_type == "mock":
        return RewriteClientExpertGenerator(default_client=MockRewriteClient())

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

    if backend.backend_type == "external":
        if shared_model is not None and shared_tokenizer is not None:
            model_path = os.getenv("STAGE1_STUDENT_MODEL")
            from .local_generator import LocalTransformersGenerator
            return LocalTransformersGenerator(
                model_path=model_path,
                tokenizer=shared_tokenizer,
                model=shared_model,
                load_in_4bit=os.getenv("STAGE1_LOAD_IN_4BIT", "true").lower() == "true",
            )
        return _load_external(backend.external_factory, config, ExpertCandidateGenerator)

    raise ValueError(f"Unsupported generation backend: {backend.backend_type}")


def build_trajectory_scorer(config: Stage1Config) -> ExpertTrajectoryScorer:
    backend = config.trajectory_backend
    if backend.backend_type == "mock":
        return DistributionExpertTrajectoryScorer(default_scorer=MockOPDDistributionScorer())
    if backend.backend_type == "openai_compatible":
        raise ValueError(
            "Trajectory scoring requires aligned token log-probabilities. "
            "Use mock for CPU tests or external_factory for local model scoring."
        )
    if backend.backend_type == "external":
        return _load_external(backend.external_factory, config, ExpertTrajectoryScorer)
    raise ValueError(f"Unsupported trajectory backend: {backend.backend_type}")


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
    return os.getenv(name)
