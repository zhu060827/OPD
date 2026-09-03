from __future__ import annotations

from .models import EvaluationResult, FillCandidate, MaskedTask


class NaturalLanguageFeedbackGenerator:
    def build(self, task: MaskedTask, candidate: FillCandidate, evaluation: EvaluationResult) -> str:
        visible_prefix = " | ".join(node.content for node in task.prefix_nodes[-2:]) or "None"
        visible_suffix = " | ".join(node.content for node in task.suffix_nodes[:2]) or "None"
        generated = " | ".join(candidate.steps) or "None"
        return (
            f"Masked nodes: {', '.join(task.masked_node_ids)}. "
            f"Prefix context: {visible_prefix}. "
            f"Suffix context: {visible_suffix}. "
            f"Generated fill: {generated}. "
            f"Evaluator feedback: {evaluation.feedback}"
        )
