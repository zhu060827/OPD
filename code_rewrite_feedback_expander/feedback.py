from __future__ import annotations

from .models import CodeRecord, QualityResult, RewriteCandidate, SemanticResult


class RewriteFeedbackBuilder:
    def build(
        self,
        record: CodeRecord,
        candidate: RewriteCandidate,
        semantic: SemanticResult,
        quality: QualityResult,
        baseline_quality: QualityResult,
    ) -> str:
        baseline = {score.name: score.score for score in baseline_quality.scores}
        deltas = {score.name: score.score - baseline.get(score.name, 0.0) for score in quality.scores}
        improved = [name for name, delta in deltas.items() if delta > 0.0]
        regressed = [name for name, delta in deltas.items() if delta < 0.0]
        return (
            f"Task {record.task_id}. Strategy={candidate.strategy}. "
            f"Original reasoning steps={len(record.reasoning)}, generated reasoning steps={len(candidate.reasoning)}. "
            f"Semantic feedback: {semantic.feedback} "
            f"Quality feedback: {quality.feedback} "
            f"Per-metric deltas={deltas}; improved={improved}; regressed={regressed}. "
            "Rewrite again with the same strategy, preserve function signatures and behavior, "
            "improve the unified code-quality metrics, and avoid unnecessary changes."
        )
