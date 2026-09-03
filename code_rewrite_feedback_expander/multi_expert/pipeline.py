from __future__ import annotations

from typing import List

from ..models import CodeRecord, RewriteCandidate
from ..quality import CodeQualityEvaluator
from ..semantic import SemanticEquivalenceChecker
from .backends import ExpertCandidateGenerator, ExpertTrajectoryScorer
from .config import Stage1Config
from .models import ExpertAssessment, Stage1RecordResult
from .router import MultiExpertRouter
from .scoring import (
    SharedCodeRewardScorer,
    build_gate_evidence,
    summarize_trajectory,
    unavailable_trajectory,
)


class MultiExpertStage1Pipeline:
    def __init__(
        self,
        config: Stage1Config,
        generator: ExpertCandidateGenerator,
        trajectory_scorer: ExpertTrajectoryScorer,
    ):
        self.config = config
        self.generator = generator
        self.trajectory_scorer = trajectory_scorer
        self.semantic_checker = SemanticEquivalenceChecker()
        self.quality_evaluator = CodeQualityEvaluator()
        self.reward_scorer = SharedCodeRewardScorer(config.reward)
        self.router = MultiExpertRouter(
            config.routing, [expert.expert_id for expert in config.enabled_experts]
        )

    def process(self, record: CodeRecord) -> Stage1RecordResult:
        assessments: List[ExpertAssessment] = []
        for expert in self.config.enabled_experts:
            baseline_candidate = RewriteCandidate(
                code=record.code,
                reasoning=list(record.reasoning),
                rationale="Original code baseline.",
                strategy=expert.strategy,
                raw_response=record.code,
                metadata={"provider": "original", "expert_id": expert.expert_id},
            )
            baseline_quality = self.quality_evaluator.evaluate(
                record, baseline_candidate, record.code
            )
            candidate = self.generator.generate(expert, record)
            semantic = self.semantic_checker.check(record, candidate)
            quality = self.quality_evaluator.evaluate(record, candidate, record.code)
            changed = candidate.code.strip() != record.code.strip()
            gate = build_gate_evidence(
                semantic=semantic,
                tests_present=bool(record.tests),
                changed=changed,
                require_tests=self.config.gate.require_tests,
                require_candidate_change=self.config.gate.require_candidate_change,
                minimum_unit_test_pass_rate=self.config.gate.minimum_unit_test_pass_rate,
            )
            reward = self.reward_scorer.score(
                strategy=expert.strategy,
                baseline=baseline_quality,
                candidate=quality,
                changed=changed,
            )
            if gate.passed or self.config.gate.score_failed_candidates:
                try:
                    trajectory = summarize_trajectory(
                        self.trajectory_scorer.score(expert, record, candidate)
                    )
                except Exception as exc:  # Preserve the sample and make failure auditable.
                    trajectory = unavailable_trajectory(exc)
            else:
                trajectory = unavailable_trajectory("skipped_by_hard_gate")
            assessments.append(
                ExpertAssessment(
                    expert_id=expert.expert_id,
                    strategy=expert.strategy,
                    candidate=candidate,
                    semantic_result=semantic.to_dict(),
                    quality_result=quality.to_dict(),
                    gate=gate,
                    reward=reward,
                    trajectory=trajectory,
                )
            )

        decision = self.router.route(assessments)
        return Stage1RecordResult(
            task_id=record.task_id,
            prompt=record.prompt,
            original_code=record.code,
            tests=list(record.tests),
            language=record.language,
            assessments=assessments,
            routing=decision,
        )
