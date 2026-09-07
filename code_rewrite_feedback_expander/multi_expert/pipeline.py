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
        if self.config.routing.policy == "three_tier":
            return self._process_three_tier(record)
        return self._process_heuristic_ablation(record)

    def _process_three_tier(self, record: CodeRecord) -> Stage1RecordResult:
        recorded_expert_id = self.router.recorded_label(record.metadata)
        experts = self.config.enabled_experts
        if recorded_expert_id:
            experts = [item for item in experts if item.expert_id == recorded_expert_id]
        if self.config.routing.shared_completion_source == "student_generate":
            shared_candidate = self.trajectory_scorer.generate_student_completion(
                record, self.config.routing.student_max_new_tokens
            )
        else:
            shared_candidate = RewriteCandidate(
                code=record.code,
                reasoning=list(record.reasoning),
                rationale="Shared recorded completion used for aligned five-Teacher scoring.",
                strategy="shared_trajectory",
                raw_response=record.code,
                metadata={"provider": "record", "shared_across_teachers": True},
            )
        assessments = [
            self._assess(expert, record, shared_candidate, require_change=False)
            for expert in experts
        ]
        decision = self.router.route(assessments, recorded_expert_id=recorded_expert_id)
        return self._result(record, assessments, decision)

    def _process_heuristic_ablation(self, record: CodeRecord) -> Stage1RecordResult:
        assessments: List[ExpertAssessment] = []
        for expert in self.config.enabled_experts:
            candidate = self.generator.generate(expert, record)
            assessments.append(self._assess(expert, record, candidate, require_change=True))

        decision = self.router.route(assessments)
        return self._result(record, assessments, decision)

    def _assess(self, expert, record, candidate, require_change: bool) -> ExpertAssessment:
        baseline_candidate = RewriteCandidate(
            code=record.code,
            reasoning=list(record.reasoning),
            rationale="Original code baseline.",
            strategy=expert.strategy,
            raw_response=record.code,
            metadata={"provider": "original", "expert_id": expert.expert_id},
        )
        baseline_quality = self.quality_evaluator.evaluate(record, baseline_candidate, record.code)
        semantic = self.semantic_checker.check(record, candidate)
        quality = self.quality_evaluator.evaluate(record, candidate, record.code)
        changed = candidate.code.strip() != record.code.strip()
        gate = build_gate_evidence(
            semantic=semantic,
            tests_present=bool(record.tests),
            changed=changed,
            require_tests=self.config.gate.require_tests,
            require_candidate_change=require_change and self.config.gate.require_candidate_change,
            minimum_unit_test_pass_rate=self.config.gate.minimum_unit_test_pass_rate,
        )
        reward = self.reward_scorer.score(
            strategy=expert.strategy,
            baseline=baseline_quality,
            candidate=quality,
            changed=changed,
        )
        # Canonical OPD routing must observe the Student's current trajectory,
        # including incorrect trajectories.  Semantic verification is therefore
        # audit metadata in three-tier mode; it remains a hard feasibility gate
        # for the legacy candidate-generation ablation.
        should_score_trajectory = (
            self.config.routing.policy == "three_tier"
            or gate.passed
            or self.config.gate.score_failed_candidates
        )
        if should_score_trajectory:
            try:
                trajectory = summarize_trajectory(
                    self.trajectory_scorer.score(expert, record, candidate)
                )
            except Exception as exc:
                trajectory = unavailable_trajectory(exc)
        else:
            trajectory = unavailable_trajectory("skipped_by_hard_gate")
        return ExpertAssessment(
            expert_id=expert.expert_id,
            strategy=expert.strategy,
            candidate=candidate,
            semantic_result=semantic.to_dict(),
            quality_result=quality.to_dict(),
            gate=gate,
            reward=reward,
            trajectory=trajectory,
        )

    @staticmethod
    def _result(record, assessments, decision) -> Stage1RecordResult:
        selected = next(
            (item for item in assessments if item.expert_id == decision.selected_expert_id),
            assessments[0] if assessments else None,
        )
        verification_status = _verification_status(selected, tests_present=bool(record.tests))
        downstream_action = _downstream_action(decision, verification_status)
        return Stage1RecordResult(
            task_id=record.task_id,
            prompt=record.prompt,
            original_code=record.code,
            tests=list(record.tests),
            language=record.language,
            assessments=assessments,
            routing=decision,
            verification_status=verification_status,
            downstream_action=downstream_action,
        )


def _verification_status(
    assessment: ExpertAssessment | None, tests_present: bool
) -> str:
    if assessment is None:
        return "semantic_unverified"
    semantic = assessment.semantic_result
    metrics = {
        str(item.get("name")): float(item.get("score", 0.0))
        for item in semantic.get("scores", [])
        if isinstance(item, dict)
    }
    structural_checks = ("ast_parse", "signature_consistency", "safety", "compile_pass_rate")
    if any(metrics.get(name, 0.0) < 0.99 for name in structural_checks):
        return "semantic_fail"
    if not tests_present:
        return "semantic_unverified"
    if metrics.get("unit_test_pass_rate", 0.0) < 0.99:
        return "semantic_fail"
    return "semantic_pass"


def _downstream_action(decision, verification_status: str) -> str:
    if not decision.usable_for_training:
        return "abstained_pool"
    if verification_status == "semantic_pass":
        return "positive_augmentation"
    if verification_status == "semantic_fail":
        return "repair_or_negative"
    return "unverified_pool"
