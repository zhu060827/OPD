from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CodeRecord:
    task_id: str
    prompt: str
    code: str
    reasoning: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewriteCandidate:
    code: str
    reasoning: List[str]
    rationale: str
    strategy: str
    raw_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenDistribution:
    token: str
    student_logprobs: Dict[str, float]
    teacher_logprobs: Dict[str, float]
    student_token_logprob: float
    teacher_token_logprob: float
    aspect_weights: Dict[str, float] = field(default_factory=dict)
    attribution_source: str = "structural_heuristic"


@dataclass
class OPDTokenProfile:
    total_kl: float
    aspect_kl: Dict[str, float]
    aspect_total_variation: Dict[str, float]
    aspect_distribution_overlap: Dict[str, float]
    aspect_token_nll_reduction: Dict[str, float]
    strategy_ranking: List[str]
    dominant_aspect: str
    next_strategy: str
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_kl": self.total_kl,
            "aspect_kl": self.aspect_kl,
            "aspect_total_variation": self.aspect_total_variation,
            "aspect_distribution_overlap": self.aspect_distribution_overlap,
            "aspect_token_nll_reduction": self.aspect_token_nll_reduction,
            "strategy_ranking": self.strategy_ranking,
            "dominant_aspect": self.dominant_aspect,
            "next_strategy": self.next_strategy,
            "token_count": self.token_count,
        }


@dataclass
class MetricScore:
    name: str
    score: float
    explanation: str
    raw_value: float | None = None
    implementation: str = "internal"
    reference: str = ""


@dataclass
class SemanticResult:
    passed: bool
    scores: List[MetricScore]
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "feedback": self.feedback,
            "scores": [score.__dict__ for score in self.scores],
        }


@dataclass
class QualityResult:
    scores: List[MetricScore]
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback": self.feedback,
            "scores": [score.__dict__ for score in self.scores],
            "metric_scores": {score.name: score.score for score in self.scores},
        }


@dataclass
class RetainedRewrite:
    iteration: int
    strategy: str
    original_quality: Dict[str, Any]
    rewritten_quality: Dict[str, Any]
    semantic_result: Dict[str, Any]
    gain: float
    code: str
    reasoning: List[str]
    rationale: str
    opd_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the retained rewrite without relying on dataclass inference."""
        return {
            "iteration": self.iteration,
            "strategy": self.strategy,
            "original_quality": self.original_quality,
            "rewritten_quality": self.rewritten_quality,
            "semantic_result": self.semantic_result,
            "gain": self.gain,
            "code": self.code,
            "reasoning": self.reasoning,
            "rationale": self.rationale,
            "opd_profile": self.opd_profile,
        }


@dataclass
class ExpansionRecord:
    task_id: str
    prompt: str
    original_reasoning: List[str]
    expanded_reasoning: List[str]
    original_code: str
    expanded_code: str
    retained_rewrites: List[Dict[str, Any]]
    iteration_trace: List[Dict[str, Any]]
    original_quality: Dict[str, Any]
    final_quality: Dict[str, Any]
    semantic_result: Dict[str, Any]
    expansion_stats: Dict[str, Any]
    accepted: bool
