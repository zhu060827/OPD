from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .evaluators import MultiFeedbackEvaluator
from .feedback import NaturalLanguageFeedbackGenerator
from .llm import LLMClient
from .masking import GraphMasker
from .models import ExpansionRecord, FillCandidate, ReasoningGraph
from .parser import ReasoningGraphParser


class MathReasoningExpansionPipeline:
    def __init__(
        self,
        llm: LLMClient,
        max_iterations: int = 10,
        max_refine_iterations: int = 3,
        patience: int = 2,
        accept_threshold: float = 0.80,
        seed: int = 7,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_refine_iterations = max_refine_iterations
        self.patience = patience
        self.parser = ReasoningGraphParser()
        self.masker = GraphMasker(seed=seed)
        self.evaluator = MultiFeedbackEvaluator(accept_threshold=accept_threshold)
        self.feedback_builder = NaturalLanguageFeedbackGenerator()

    def expand_record(self, record: Dict[str, str], mask_strategy: str = "auto", mask_width: int = 1) -> ExpansionRecord:
        question = record.get("question", "")
        answer = record.get("answer") or record.get("cot") or record.get("solution", "")
        graph = self.parser.parse(question, answer)
        return self.expand_graph(graph, mask_strategy=mask_strategy, mask_width=mask_width)

    def expand_graph(self, graph: ReasoningGraph, mask_strategy: str = "auto", mask_width: int = 1) -> ExpansionRecord:
        working_graph = graph
        expanded_answer = graph.answer
        best_task = None
        best_original_evaluation = None
        best_candidate: Optional[FillCandidate] = None
        best_evaluation = None
        no_gain_nodes = 0
        best_score = -1.0
        iteration_trace = []
        retained_expansions: List[Dict[str, Any]] = []
        attempted_iterations = 0

        for expansion_round in range(1, self.max_iterations + 1):
            task = self.masker.mask(working_graph, strategy=mask_strategy, width=mask_width)
            original_candidate = FillCandidate(
                text="\n".join(node.content for node in task.target_nodes),
                steps=[node.content for node in task.target_nodes],
                raw_response="",
                metadata={"provider": "original_masked_nodes"},
            )
            original_evaluation = self.evaluator.evaluate(task, original_candidate)
            feedback = ""
            retained_this_node = False

            for refine_round in range(1, self.max_refine_iterations + 1):
                attempted_iterations += 1
                candidate = self.llm.generate_fill(task, feedback=feedback)
                evaluation = self.evaluator.evaluate(task, candidate)
                gain = evaluation.aggregate_score - original_evaluation.aggregate_score
                retained = gain > 1e-6
                iteration_trace.append(
                    {
                        "iteration": attempted_iterations,
                        "expansion_round": expansion_round,
                        "refine_round": refine_round,
                        "masked_node_ids": task.masked_node_ids,
                        "original_score": original_evaluation.aggregate_score,
                        "aggregate_score": evaluation.aggregate_score,
                        "gain": gain,
                        "retained": retained,
                        "accepted": evaluation.accepted,
                    }
                )
                if evaluation.aggregate_score > best_score + 1e-6:
                    best_task = task
                    best_original_evaluation = original_evaluation
                    best_candidate = candidate
                    best_evaluation = evaluation
                    best_score = evaluation.aggregate_score
                if retained:
                    expanded_answer = self._merge_steps(working_graph, task.masked_node_ids, candidate.steps)
                    retained_expansions.append(
                        {
                            "iteration": attempted_iterations,
                            "expansion_round": expansion_round,
                            "refine_round": refine_round,
                            "masked_node_ids": task.masked_node_ids,
                            "original_steps": original_candidate.steps,
                            "generated_steps": candidate.steps,
                            "gain": gain,
                            "original_evaluation": original_evaluation.to_dict(),
                            "evaluation": evaluation.to_dict(),
                        }
                    )
                    working_graph = self.parser.parse(graph.question, expanded_answer)
                    retained_this_node = True
                    no_gain_nodes = 0
                    break

                feedback = self._build_refine_feedback(task, candidate, evaluation, original_evaluation)

            if not retained_this_node:
                no_gain_nodes += 1
            if no_gain_nodes >= self.patience:
                break

        assert best_task is not None
        assert best_candidate is not None and best_evaluation is not None
        assert best_original_evaluation is not None
        if retained_expansions:
            masked_node_ids = [node_id for item in retained_expansions for node_id in item["masked_node_ids"]]
            generated_steps = [step for item in retained_expansions for step in item["generated_steps"]]
            original_evaluation_dict = self._combine_evaluations(retained_expansions, "original_evaluation")
            evaluation_dict = self._combine_evaluations(retained_expansions, "evaluation")
            accepted = any(item["evaluation"]["accepted"] for item in retained_expansions)
        else:
            expanded_answer = graph.answer
            masked_node_ids = best_task.masked_node_ids
            generated_steps = []
            original_evaluation_dict = best_original_evaluation.to_dict()
            evaluation_dict = best_original_evaluation.to_dict()
            accepted = False
        return ExpansionRecord(
            question=graph.question,
            original_answer=graph.answer,
            expanded_answer=expanded_answer,
            graph=graph.to_dict(),
            masked_node_ids=masked_node_ids,
            generated_steps=generated_steps,
            expansion_stats=self._build_expansion_stats(graph.answer, expanded_answer),
            original_evaluation=original_evaluation_dict,
            evaluation=evaluation_dict,
            retained_expansions=retained_expansions,
            iteration_trace=iteration_trace,
            iteration=attempted_iterations,
            accepted=accepted,
        )

    def _merge_steps(self, graph: ReasoningGraph, masked_node_ids: List[str], generated_steps: List[str]) -> str:
        masked = set(masked_node_ids)
        output_steps: List[str] = []
        inserted = False
        for node in graph.nodes:
            if node.node_id in masked:
                if not inserted:
                    output_steps.extend(generated_steps)
                    inserted = True
                continue
            output_steps.append(node.content)
        if not inserted:
            output_steps.extend(generated_steps)
        return "\n".join(f"{idx}. {step}" for idx, step in enumerate(output_steps, start=1))

    def _combine_evaluations(self, retained_expansions: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        evaluations = [item[key] for item in retained_expansions]
        aggregate = sum(item["aggregate_score"] for item in evaluations) / len(evaluations)
        scores = []
        score_names = [score["name"] for score in evaluations[0]["scores"]]
        for name in score_names:
            values = [
                score["score"]
                for evaluation in evaluations
                for score in evaluation["scores"]
                if score["name"] == name
            ]
            scores.append(
                {
                    "name": name,
                    "score": sum(values) / len(values) if values else 0.0,
                    "explanation": "Average over retained expansion rounds.",
                }
            )
        return {
            "accepted": any(item["accepted"] for item in evaluations),
            "aggregate_score": aggregate,
            "feedback": "Average over retained expansion rounds.",
            "scores": scores,
        }

    def _build_expansion_stats(self, original_answer: str, expanded_answer: str) -> Dict[str, Any]:
        original_graph = self.parser.parse("", original_answer)
        expanded_graph = self.parser.parse("", expanded_answer)
        original_count = len(original_graph.nodes)
        expanded_count = len(expanded_graph.nodes)
        added_count = max(0, expanded_count - original_count)
        expansion_ratio = expanded_count / original_count if original_count else 0.0
        return {
            "original_node_count": original_count,
            "expanded_node_count": expanded_count,
            "added_node_count": added_count,
            "expansion_ratio": expansion_ratio,
        }

    def _build_refine_feedback(
        self,
        task,
        candidate: FillCandidate,
        evaluation,
        original_evaluation,
    ) -> str:
        feedback = self.feedback_builder.build(task, candidate, evaluation)
        gap = original_evaluation.aggregate_score - evaluation.aggregate_score
        return (
            f"{feedback} The generated fill has not improved over the masked original node "
            f"(original={original_evaluation.aggregate_score:.3f}, generated={evaluation.aggregate_score:.3f}, "
            f"gap={gap:.3f}). In the next attempt, keep the same masked node, add genuinely useful "
            "intermediate reasoning, reduce copying from the visible context, and preserve the final answer."
        )


def expansion_record_to_dict(record: ExpansionRecord) -> Dict:
    return asdict(record)
