from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

from .models import EvaluationResult, FillCandidate, MaskedTask, MetricScore
from .parser import ReasoningGraphParser
from .safe_math import equation_is_consistent


class MultiFeedbackEvaluator:
    def __init__(self, accept_threshold: float = 0.80):
        self.accept_threshold = accept_threshold
        self.parser = ReasoningGraphParser()

    def evaluate(self, task: MaskedTask, candidate: FillCandidate) -> EvaluationResult:
        scores = [
            self._similarity(task, candidate),
            self._logic_consistency(task, candidate),
            self._formula_correctness(task, candidate),
            self._completeness(task, candidate),
            self._reasoning_gain(task, candidate),
        ]
        weights = {
            "similarity": 0.18,
            "logic_consistency": 0.24,
            "formula_correctness": 0.24,
            "completeness": 0.17,
            "reasoning_gain": 0.17,
        }
        aggregate = sum(score.score * weights[score.name] for score in scores)
        accepted = aggregate >= self.accept_threshold
        feedback = self._build_feedback(scores, aggregate, accepted)
        return EvaluationResult(scores=scores, accepted=accepted, aggregate_score=aggregate, feedback=feedback)

    def _similarity(self, task: MaskedTask, candidate: FillCandidate) -> MetricScore:
        generated = " ".join(candidate.steps)
        compare_targets = [node.content for node in task.prefix_nodes + task.suffix_nodes]
        if not generated or not compare_targets:
            return MetricScore("similarity", 0.0, "Generated text is empty or no comparison target exists.")
        max_ratio = max(SequenceMatcher(None, generated.lower(), target.lower()).ratio() for target in compare_targets)
        score = 1.0 - max(0.0, min(1.0, (max_ratio - 0.55) / 0.45))
        if max_ratio < 0.55:
            explanation = f"Low duplication risk; maximum similarity is {max_ratio:.2f}."
        elif max_ratio < 0.82:
            explanation = f"Moderate overlap; maximum similarity is {max_ratio:.2f}."
        else:
            explanation = f"High overlap with existing steps; maximum similarity is {max_ratio:.2f}."
        return MetricScore("similarity", score, explanation)

    def _logic_consistency(self, task: MaskedTask, candidate: FillCandidate) -> MetricScore:
        text = " ".join(candidate.steps)
        if not text.strip():
            return MetricScore("logic_consistency", 0.0, "No generated reasoning to validate.")
        bad_markers = ["contradiction", "cannot be determined", "not enough information", "undefined"]
        if any(marker in text.lower() for marker in bad_markers):
            return MetricScore("logic_consistency", 0.25, "Generated text contains uncertainty or contradiction markers.")
        prefix_symbols = self._symbols(" ".join(node.content for node in task.prefix_nodes + task.target_nodes))
        generated_symbols = self._symbols(text)
        suffix_symbols = self._symbols(" ".join(node.content for node in task.suffix_nodes[:2]))
        unknown = generated_symbols - prefix_symbols - suffix_symbols
        if len(unknown) >= 3:
            return MetricScore("logic_consistency", 0.45, f"Introduces many unsupported symbols: {sorted(unknown)}.")
        if candidate.steps and len(candidate.steps) <= max(3, len(task.target_nodes) + 2):
            return MetricScore("logic_consistency", 0.86, "Generated reasoning is concise and compatible with visible context.")
        return MetricScore("logic_consistency", 0.72, "Generated reasoning is mostly compatible but may be verbose.")

    def _formula_correctness(self, task: MaskedTask, candidate: FillCandidate) -> MetricScore:
        context = " ".join(node.content for node in task.prefix_nodes + task.suffix_nodes) + " " + candidate.text
        formulas = self.parser.extract_formulas(candidate.text)
        equations = [formula for formula in formulas if "=" in formula]
        if not equations:
            target_has_formula = any(node.formulas for node in task.target_nodes)
            if target_has_formula:
                return MetricScore("formula_correctness", 0.48, "Target contains formulas but generated fill has no checkable equation.")
            return MetricScore("formula_correctness", 0.75, "No explicit formula is required or generated.")
        checked = [equation_is_consistent(equation, context) for equation in equations]
        definite = [item for item in checked if item is not None]
        if not definite:
            return MetricScore("formula_correctness", 0.68, "Equations are present but not fully numeric, so only format was checked.")
        ratio = sum(1 for item in definite if item) / len(definite)
        explanation = f"{sum(1 for item in definite if item)}/{len(definite)} numeric equations are consistent."
        return MetricScore("formula_correctness", ratio, explanation)

    def _completeness(self, task: MaskedTask, candidate: FillCandidate) -> MetricScore:
        if not candidate.steps:
            return MetricScore("completeness", 0.0, "No generated steps.")
        target_formula_count = sum(len(node.formulas) for node in task.target_nodes)
        generated_formula_count = len(self.parser.extract_formulas(candidate.text))
        connective_count = len(re.findall(r"\b(so|therefore|thus|because|hence|then|since)\b", candidate.text, flags=re.I))
        score = 0.45
        if len(candidate.steps) >= len(task.target_nodes):
            score += 0.2
        if target_formula_count == 0 or generated_formula_count >= min(target_formula_count, 1):
            score += 0.2
        if connective_count > 0:
            score += 0.15
        return MetricScore("completeness", min(score, 1.0), "Generated fill covers the masked slot with adequate structure.")

    def _reasoning_gain(self, task: MaskedTask, candidate: FillCandidate) -> MetricScore:
        generated = " ".join(candidate.steps)
        suffix = " ".join(node.content for node in task.suffix_nodes[:1])
        if not generated:
            return MetricScore("reasoning_gain", 0.0, "No additional reasoning gain.")
        suffix_overlap = SequenceMatcher(None, generated.lower(), suffix.lower()).ratio() if suffix else 0.0
        formula_bonus = 0.15 if self.parser.extract_formulas(generated) else 0.0
        length_bonus = 0.2 if 8 <= len(generated.split()) <= 80 else 0.05
        novelty = max(0.0, 1.0 - suffix_overlap)
        score = min(1.0, 0.45 * novelty + length_bonus + formula_bonus + 0.2)
        return MetricScore("reasoning_gain", score, f"Novelty against suffix is {novelty:.2f}.")

    def _symbols(self, text: str) -> set:
        blacklist = {
            "Let",
            "The",
            "Therefore",
            "Thus",
            "So",
            "If",
            "Then",
            "This",
            "We",
            "Now",
        }
        return {token for token in re.findall(r"\b[A-Za-z]\w*\b", text) if token not in blacklist}

    def _build_feedback(self, scores: List[MetricScore], aggregate: float, accepted: bool) -> str:
        status = "ACCEPT" if accepted else "REVISE"
        weak = [score for score in scores if score.score < 0.75]
        details = "; ".join(f"{score.name}={score.score:.2f}: {score.explanation}" for score in scores)
        if not weak:
            action = "The generated fill is usable. Keep notation and avoid unnecessary expansion in later rounds."
        else:
            weak_names = ", ".join(score.name for score in weak)
            action = (
                f"Improve {weak_names}. Preserve the final answer, reduce duplication, "
                "add missing algebraic/logical bridge steps, and avoid unsupported variables."
            )
        return f"{status}; aggregate={aggregate:.2f}. {action} Details: {details}"
