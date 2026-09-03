from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any, Dict, List

from .llm import (
    IterationFeedbackLLM,
    MockIterationFeedbackLLM,
    MockOPDDistributionScorer,
    OPDDistributionScorer,
    RewriteLLMClient,
)
from .models import CodeRecord, ExpansionRecord, RetainedRewrite, RewriteCandidate
from .opd import HierarchicalTokenKLEvaluator
from .quality import CodeQualityEvaluator, METHOD_TO_QUALITY_METRIC
from .feedback_schema import parse_feedback
from .semantic import SemanticEquivalenceChecker


DEFAULT_STRATEGIES = ["cot", "style", "ast", "variable", "control_flow"]

# Literature-backed risk boundaries exist for MI and McCabe CC, but not for
# universal before/after deltas. Delta thresholds below are conservative
# measurement tolerances and should be sensitivity-tested on a validation set.
DEFAULT_PARETO_THRESHOLDS = {
    "style_violation_rate": {"epsilon": 0.01, "delta": 0.02, "source": "PEP 8/pycodestyle scale; validation_parameter"},
    "maintainability_index": {"epsilon": 1.0 / 171.0, "delta": 5.0 / 171.0, "source": "MI scale; validation_parameter"},
    "cyclomatic_complexity": {"epsilon": 0.01, "delta": 0.08, "source": "McCabe risk bands; implementation scale"},
    "naming_convention_compliance": {"epsilon": 0.01, "delta": 0.02, "source": "PEP 8 scale; validation_parameter"},
    "codebleu_syntax_match": {"epsilon": 0.01, "delta": 0.05, "minimum": 0.50, "source": "CodeBLEU scale; validation_parameter"},
}

class CodeRewriteExpansionPipeline:
    def __init__(
        self,
        llm: RewriteLLMClient,
        opd_scorer: OPDDistributionScorer | None = None,
        feedback_llm: IterationFeedbackLLM | None = None,
        max_iterations: int = 10,
        accept_threshold: float = 0.78,
        strategies: List[str] | None = None,
        selection_policy: str = "adaptive",
        random_seed: int = 0,
        regression_tolerance: float = 0.05,
        pareto_thresholds: Dict[str, Dict[str, Any]] | None = None,
    ):
        self.llm = llm
        self.opd_scorer = opd_scorer or MockOPDDistributionScorer()
        self.feedback_llm = feedback_llm or MockIterationFeedbackLLM()
        self.max_iterations = max_iterations
        self.strategies = strategies or DEFAULT_STRATEGIES
        if selection_policy not in {"adaptive", "random", "fixed"}:
            raise ValueError("selection_policy must be adaptive, random, or fixed")
        self.selection_policy = selection_policy
        self.random = random.Random(random_seed)
        self.regression_tolerance = regression_tolerance
        self.pareto_thresholds = pareto_thresholds or DEFAULT_PARETO_THRESHOLDS
        self.semantic_checker = SemanticEquivalenceChecker()
        self.quality_evaluator = CodeQualityEvaluator(accept_threshold=accept_threshold)
        self.opd_evaluator = HierarchicalTokenKLEvaluator()

    def expand_record(self, record: CodeRecord) -> ExpansionRecord:
        working_code = record.code
        working_reasoning = list(record.reasoning)
        baseline_candidate = RewriteCandidate(
            code=record.code,
            reasoning=record.reasoning,
            rationale="Original code baseline.",
            strategy="original",
            raw_response=record.code,
            metadata={"provider": "original"},
        )
        original_quality = self.quality_evaluator.evaluate(record, baseline_candidate, record.code)
        final_quality = original_quality
        final_semantic = self.semantic_checker.check(record, baseline_candidate)
        retained_rewrites: List[RetainedRewrite] = []
        iteration_trace: List[Dict[str, Any]] = []
        attempted_iterations = 0
        next_strategy = ""
        next_feedback = ""
        fallback_strategies = list(self.strategies)
        active_record = CodeRecord(
            task_id=record.task_id,
            prompt=record.prompt,
            code=working_code,
            reasoning=working_reasoning,
            tests=record.tests,
            language=record.language,
            metadata=record.metadata,
        )

        for opd_step in range(1, self.max_iterations + 1):
            if opd_step == 1:
                strategies_to_try = list(self.strategies)
            else:
                # Use the previous round's aspect score to order semantic fallbacks.
                strategies_to_try = [next_strategy] + [
                    strategy for strategy in fallback_strategies if strategy != next_strategy
                ]
            strategies_to_try = list(dict.fromkeys(strategy for strategy in strategies_to_try if strategy))
            if not strategies_to_try:
                break

            round_attempts: List[Dict[str, Any]] = []
            for strategy in strategies_to_try:
                strategy_baseline_candidate = RewriteCandidate(
                    code=working_code,
                    reasoning=working_reasoning,
                    rationale="Current working-code baseline.",
                    strategy=strategy,
                    raw_response=working_code,
                )
                strategy_baseline_quality = self.quality_evaluator.evaluate(record, strategy_baseline_candidate, working_code)
                active_record = CodeRecord(
                    task_id=record.task_id,
                    prompt=record.prompt,
                    code=working_code,
                    reasoning=working_reasoning,
                    tests=record.tests,
                    language=record.language,
                    metadata=record.metadata,
                )
                attempted_iterations += 1
                feedback_used = next_feedback if opd_step > 1 else ""
                candidate = self.llm.rewrite(active_record, strategy=strategy, feedback=feedback_used)
                semantic = self.semantic_checker.check(record, candidate)
                quality = self.quality_evaluator.evaluate(record, candidate, working_code)
                token_distributions = self.opd_scorer.score_student_trajectory(active_record, candidate, strategy)
                opd_profile = self.opd_evaluator.evaluate(token_distributions, semantic_valid=semantic.passed)

                metric_deltas = self._metric_deltas(strategy_baseline_quality.scores, quality.scores)
                quality_metrics = {score.name: score.score for score in quality.scores}
                semantic_metrics = {score.name: score.score for score in semantic.scores}
                primary_metric = METHOD_TO_QUALITY_METRIC[strategy]
                primary_metric_delta = metric_deltas.get(primary_metric, 0.0)
                code_changed = candidate.code.strip() != working_code.strip()
                improved_metrics = [
                    name for name, delta in metric_deltas.items()
                    if delta >= self._metric_threshold(name, "epsilon")
                ]
                regressed_metrics = [
                    name for name, delta in metric_deltas.items()
                    if delta < -self._metric_threshold(name, "delta")
                    and not (
                        name == "codebleu_syntax_match"
                        and quality_metrics.get(name, 0.0) >= self._metric_threshold(name, "minimum")
                    )
                ]
                if strategy == "ast":
                    primary_condition = quality_metrics.get(primary_metric, 0.0) >= self._metric_threshold(
                        primary_metric, "minimum"
                    )
                else:
                    primary_condition = primary_metric_delta >= self._metric_threshold(primary_metric, "epsilon")
                retained = semantic.passed and code_changed and primary_condition and not regressed_metrics
                trace_entry = {
                    "iteration": attempted_iterations,
                    "opd_step": opd_step,
                    "strategy": strategy,
                    "student_semantic_passed": semantic.passed,
                    "semantic_passed": semantic.passed,
                    "metric_deltas": metric_deltas,
                    "quality_metrics": quality_metrics,
                    "semantic_metrics": semantic_metrics,
                    "primary_quality_metric": primary_metric,
                    "primary_metric_delta": primary_metric_delta,
                    "primary_metric_condition_passed": primary_condition,
                    "code_changed": code_changed,
                    "no_change": not code_changed,
                    "gain": max(metric_deltas.values(), default=0.0),
                    "improved_metrics": improved_metrics,
                    "regressed_metrics": regressed_metrics,
                    "pareto_thresholds": {
                        name: self.pareto_thresholds.get(name, {}) for name in metric_deltas
                    },
                    "token_kl": opd_profile.total_kl,
                    "aspect_kl": opd_profile.aspect_kl,
                    "aspect_total_variation": opd_profile.aspect_total_variation,
                    "aspect_distribution_overlap": opd_profile.aspect_distribution_overlap,
                    "aspect_token_nll_reduction": opd_profile.aspect_token_nll_reduction,
                    "strategy_ranking": opd_profile.strategy_ranking,
                    "dominant_aspect": opd_profile.dominant_aspect,
                    "next_strategy": opd_profile.next_strategy,
                    "opd_token_count": opd_profile.token_count,
                    "feedback_used": feedback_used,
                    "generated_feedback": "",
                    "retained": retained,
                    "selected_for_next_round": False,
                }
                iteration_trace.append(trace_entry)
                round_attempts.append(
                    {
                        "trace": trace_entry,
                        "candidate": candidate,
                        "semantic": semantic,
                        "quality": quality,
                        "opd_profile": opd_profile,
                        "baseline_quality": strategy_baseline_quality,
                        "gain": max(metric_deltas.values(), default=0.0),
                        "retained": retained,
                    }
                )
                # After the exploratory first round, try fallbacks sequentially.
                # Once one candidate is semantically valid, do not spend calls on
                # lower-priority methods in the same round.
                if opd_step > 1 and semantic.passed:
                    break
            direction_attempts = [attempt for attempt in round_attempts if attempt["semantic"].passed]
            if not direction_attempts:
                break

            selected = self._select_attempt(direction_attempts, opd_step)
            selected["trace"]["selected_for_next_round"] = True
            next_strategy = selected["trace"]["next_strategy"]
            fallback_strategies = selected["trace"]["strategy_ranking"] or list(self.strategies)
            if not next_strategy:
                break
            round_records = [attempt["trace"] for attempt in round_attempts]
            next_feedback = self.feedback_llm.generate_feedback(active_record, next_strategy, round_records)
            selected["trace"]["generated_feedback"] = next_feedback
            feedback_source_records = [
                {key: value for key, value in item.items() if key not in {"feedback_record", "generated_feedback"}}
                for item in round_records
            ]
            selected["trace"]["feedback_record"] = {
                "selected_strategy": next_strategy,
                "source_records": feedback_source_records,
                "rendered_feedback": next_feedback,
            }
            if not selected["retained"]:
                continue

            selected_candidate = selected["candidate"]
            working_code = selected_candidate.code
            if selected_candidate.reasoning:
                working_reasoning = selected_candidate.reasoning
            final_quality = selected["quality"]
            final_semantic = selected["semantic"]
            retained_rewrites.append(
                RetainedRewrite(
                    iteration=selected["trace"]["iteration"],
                    strategy=selected_candidate.strategy,
                    original_quality=selected["baseline_quality"].to_dict(),
                    rewritten_quality=selected["quality"].to_dict(),
                    semantic_result=selected["semantic"].to_dict(),
                    gain=max(selected["trace"]["metric_deltas"].values(), default=0.0),
                    code=selected_candidate.code,
                    reasoning=selected_candidate.reasoning,
                    rationale=selected_candidate.rationale,
                    opd_profile=selected["opd_profile"].to_dict(),
                )
            )

        return ExpansionRecord(
            task_id=record.task_id,
            prompt=record.prompt,
            original_reasoning=record.reasoning,
            expanded_reasoning=working_reasoning,
            original_code=record.code,
            expanded_code=working_code,
            retained_rewrites=[item.to_dict() for item in retained_rewrites],
            iteration_trace=iteration_trace,
            original_quality=original_quality.to_dict(),
            final_quality=final_quality.to_dict(),
            semantic_result=final_semantic.to_dict(),
            expansion_stats=self._expansion_stats(record.code, working_code, record.reasoning, working_reasoning, retained_rewrites),
            accepted=bool(retained_rewrites),
        )

    def _expansion_stats(
        self,
        original_code: str,
        expanded_code: str,
        original_reasoning: List[str],
        expanded_reasoning: List[str],
        retained: List[RetainedRewrite],
    ) -> Dict[str, Any]:
        original_lines = len([line for line in original_code.splitlines() if line.strip()])
        expanded_lines = len([line for line in expanded_code.splitlines() if line.strip()])
        original_reasoning_steps = len(original_reasoning)
        expanded_reasoning_steps = len(expanded_reasoning)
        return {
            "original_line_count": original_lines,
            "expanded_line_count": expanded_lines,
            "added_line_count": max(0, expanded_lines - original_lines),
            "original_reasoning_steps": original_reasoning_steps,
            "expanded_reasoning_steps": expanded_reasoning_steps,
            "added_reasoning_steps": max(0, expanded_reasoning_steps - original_reasoning_steps),
            "rewrite_count": len(retained),
            "expansion_ratio": expanded_lines / original_lines if original_lines else 0.0,
            "reasoning_expansion_ratio": (
                expanded_reasoning_steps / original_reasoning_steps if original_reasoning_steps else 0.0
            ),
        }

    def _metric_deltas(self, baseline_scores, rewritten_scores) -> Dict[str, float]:
        baseline = {score.name: score.score for score in baseline_scores}
        rewritten = {score.name: score.score for score in rewritten_scores}
        metric_names = sorted(set(baseline) | set(rewritten))
        return {name: rewritten.get(name, 0.0) - baseline.get(name, 0.0) for name in metric_names}

    def _metric_threshold(self, name: str, key: str) -> float:
        default = 1e-6 if key == "epsilon" else self.regression_tolerance
        return float(self.pareto_thresholds.get(name, {}).get(key, default))

    def _select_attempt(self, attempts: List[Dict[str, Any]], opd_step: int) -> Dict[str, Any]:
        if self.selection_policy == "random":
            return self.random.choice(attempts)
        if self.selection_policy == "fixed":
            expected = self.strategies[(opd_step - 1) % len(self.strategies)]
            return next((item for item in attempts if item["trace"]["strategy"] == expected), attempts[0])
        return max(
            attempts,
            key=lambda item: (
                item["trace"]["aspect_kl"].get(item["trace"]["strategy"], 0.0),
                item["trace"]["aspect_token_nll_reduction"].get(item["trace"]["strategy"], 0.0),
                -item["trace"]["aspect_total_variation"].get(item["trace"]["strategy"], 1.0),
                len(item["trace"]["improved_metrics"]),
                -len(item["trace"]["regressed_metrics"]),
            ),
        )

def expansion_record_to_dict(record: ExpansionRecord) -> Dict[str, Any]:
    return asdict(record)
