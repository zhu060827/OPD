import unittest

from code_rewrite_feedback_expander.llm import MockOPDDistributionScorer, MockRewriteClient, OPDDistributionScorer
from code_rewrite_feedback_expander.models import CodeRecord, RewriteCandidate, TokenDistribution
from code_rewrite_feedback_expander.opd import HierarchicalTokenKLEvaluator
from code_rewrite_feedback_expander.pipeline import CodeRewriteExpansionPipeline, expansion_record_to_dict
from code_rewrite_feedback_expander.quality import CodeQualityEvaluator
from code_rewrite_feedback_expander.reporting import build_paper_report
from code_rewrite_feedback_expander.semantic import SemanticEquivalenceChecker
from code_rewrite_feedback_expander.visualization import build_quality_svg


class RecordingFeedbackLLM:
    def __init__(self):
        self.calls = []

    def generate_feedback(self, record, selected_strategy, iteration_records):
        self.calls.append((record.task_id, selected_strategy, iteration_records))
        return f"反馈：下一轮重点改写 {selected_strategy}。"


class StrategyProfileScorer(OPDDistributionScorer):
    def score_student_trajectory(self, record, candidate, strategy):
        _ = record, candidate
        profiles = {
            "style": (-3.0, -0.2),
            "variable": (-1.2, -0.1),
        }
        alternative_student, generated_teacher = profiles.get(strategy, (-1.0, -0.4))
        return [
            TokenDistribution(
                token="value",
                student_logprobs={"value": -0.5, "other": alternative_student},
                teacher_logprobs={"value": generated_teacher, "other": -1.5},
                student_token_logprob=-0.5,
                teacher_token_logprob=generated_teacher,
                aspect_weights={strategy: 1.0},
            )
        ]


class CodeRewritePipelineTest(unittest.TestCase):
    def test_semantic_checker_passes_mock_style_rewrite(self):
        record = CodeRecord(
            task_id="factorial",
            prompt="Return factorial.",
            reasoning=[
                "Initialize result to 1.",
                "Multiply by each integer from 2 through n.",
                "Return the final product.",
            ],
            code="def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result",
            tests=["assert factorial(5) == 120"],
        )
        candidate = MockRewriteClient().rewrite(record, "style")
        result = SemanticEquivalenceChecker().check(record, candidate)
        self.assertTrue(result.passed)

    def test_pipeline_outputs_trace_and_stats(self):
        record = CodeRecord(
            task_id="two_sum",
            prompt="Return indices.",
            reasoning=[
                "Track values seen so far in a hash map.",
                "For each value, compute the complement needed to reach target.",
                "Return indices when the complement has already appeared.",
            ],
            code="def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        need = target - x\n        if need in seen:\n            return [seen[need], i]\n        seen[x] = i",
            tests=["assert two_sum([2, 7, 11, 15], 9) == [0, 1]"],
        )
        pipeline = CodeRewriteExpansionPipeline(llm=MockRewriteClient(), opd_scorer=MockOPDDistributionScorer(), max_iterations=3)
        result = pipeline.expand_record(record)
        self.assertTrue(result.iteration_trace)
        self.assertIn("original_line_count", result.expansion_stats)
        self.assertIn("expanded_line_count", result.expansion_stats)
        self.assertTrue(result.original_reasoning)
        self.assertTrue(result.expanded_reasoning)
        self.assertIn("original_reasoning_steps", result.expansion_stats)
        self.assertIn("token_kl", result.iteration_trace[0])
        self.assertIn("aspect_kl", result.iteration_trace[0])
        self.assertIn("opd_step", result.iteration_trace[0])
        self.assertIn("quality_metrics", result.iteration_trace[0])
        self.assertIn("aspect_total_variation", result.iteration_trace[0])
        self.assertIn("aspect_token_nll_reduction", result.iteration_trace[0])
        self.assertIn("selected_for_next_round", result.iteration_trace[0])
        self.assertNotIn("refine_round", result.iteration_trace[0])

    def test_iteration_records_are_summarized_into_next_round_feedback(self):
        record = CodeRecord(
            task_id="feedback",
            prompt="Return the sum.",
            reasoning=["Accumulate values."],
            code="def add(nums):\n    total = 0\n    for value in nums:\n        total += value\n    return total",
            tests=["assert add([1, 2]) == 3"],
        )
        feedback_llm = RecordingFeedbackLLM()
        result = CodeRewriteExpansionPipeline(
            llm=MockRewriteClient(),
            feedback_llm=feedback_llm,
            max_iterations=2,
        ).expand_record(record)
        self.assertTrue(feedback_llm.calls)
        self.assertTrue(any(trace["generated_feedback"] for trace in result.iteration_trace if trace["selected_for_next_round"]))
        self.assertTrue(any(trace["feedback_used"] for trace in result.iteration_trace if trace["opd_step"] == 2))

    def test_quality_evaluator_uses_unified_metrics(self):
        record = CodeRecord(
            task_id="variable_names",
            prompt="Return the sum.",
            reasoning=["Use descriptive variables.", "Return the accumulated result."],
            code="def add(nums):\n    total = 0\n    for value in nums:\n        total += value\n    return total",
            tests=["assert add([1, 2, 3]) == 6"],
        )
        evaluator = CodeQualityEvaluator()
        expected = {
            "style_violation_rate",
            "maintainability_index",
            "cyclomatic_complexity",
            "naming_convention_compliance",
            "codebleu_syntax_match",
        }
        for strategy in ("cot", "style", "ast", "variable", "control_flow"):
            candidate_code = record.code if strategy != "variable" else record.code.replace("total", "sum_total")
            result = evaluator.evaluate(
                record,
                RewriteCandidate(
                    code=candidate_code,
                    reasoning=record.reasoning,
                    rationale=strategy,
                    strategy=strategy,
                    raw_response=candidate_code,
                ),
                record.code,
            )
            self.assertEqual({score.name for score in result.scores}, expected)
            self.assertNotIn("aggregate_score", result.to_dict())
            semantic_result = SemanticEquivalenceChecker().check(
                record,
                RewriteCandidate(
                    code=candidate_code,
                    reasoning=record.reasoning,
                    rationale=strategy,
                    strategy=strategy,
                    raw_response=candidate_code,
                ),
            )
            self.assertIn("unit_test_pass_rate", {score.name for score in semantic_result.scores})

    def test_maintainability_index_is_not_saturated_by_normalization_bug(self):
        record = CodeRecord(
            task_id="mi",
            prompt="Return a value.",
            code="def value(x):\n    return x + 1",
        )
        result = CodeQualityEvaluator().evaluate(
            record,
            RewriteCandidate(record.code, [], "baseline", "cot", record.code),
            record.code,
        )
        mi = next(score for score in result.scores if score.name == "maintainability_index")
        self.assertLessEqual(mi.score, 1.0)
        self.assertAlmostEqual(mi.score, max(0.0, min(1.0, mi.raw_value / 171.0)))

    def test_unchanged_candidate_is_not_retained(self):
        record = CodeRecord(
            task_id="unchanged",
            prompt="Return a value.",
            code="def value(x):\n    return x + 1",
            tests=["assert value(1) == 2"],
        )
        result = CodeRewriteExpansionPipeline(MockRewriteClient(), max_iterations=1).expand_record(record)
        self.assertTrue(any(trace["no_change"] for trace in result.iteration_trace))
        self.assertFalse(any(trace["retained"] for trace in result.iteration_trace if trace["no_change"]))

    def test_pipeline_records_opd_direction(self):
        record = CodeRecord(
            task_id="direction",
            prompt="Return indices.",
            reasoning=[
                "Track seen values.",
                "Check complement.",
                "Return matching indices.",
            ],
            code="def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        need = target - x\n        if need in seen:\n            return [seen[need], i]\n        seen[x] = i",
            tests=["assert two_sum([2, 7, 11, 15], 9) == [0, 1]"],
        )
        result = CodeRewriteExpansionPipeline(llm=MockRewriteClient(), max_iterations=1).expand_record(record)
        self.assertTrue(result.iteration_trace)
        trace = result.iteration_trace[0]
        self.assertIn("dominant_aspect", trace)
        self.assertIn("next_strategy", trace)
        self.assertIn(trace["next_strategy"], {"cot", "style", "ast", "variable", "control_flow", ""})

    def test_hierarchical_kl_is_zeroed_by_semantic_gate(self):
        token = TokenDistribution(
            token="x",
            student_logprobs={"x": -1.5, "y": -0.2},
            teacher_logprobs={"x": -0.1, "y": -2.0},
            student_token_logprob=-1.5,
            teacher_token_logprob=-0.1,
            aspect_weights={"variable": 1.0},
        )
        evaluator = HierarchicalTokenKLEvaluator()
        valid = evaluator.evaluate([token], semantic_valid=True)
        invalid = evaluator.evaluate([token], semantic_valid=False)
        self.assertGreater(valid.total_kl, 0.0)
        self.assertGreater(valid.aspect_kl["variable"], 0.0)
        self.assertTrue(valid.strategy_ranking)
        self.assertEqual(invalid.strategy_ranking, [])
        self.assertEqual(invalid.next_strategy, "")

    def test_pipeline_uses_combined_opd_score_direction(self):
        record = CodeRecord(
            task_id="teacher_direction",
            prompt="Return indices.",
            reasoning=["Track seen values.", "Check complement.", "Return matching indices."],
            code="def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        need = target - x\n        if need in seen:\n            return [seen[need], i]\n        seen[x] = i",
            tests=["assert two_sum([2, 7, 11, 15], 9) == [0, 1]"],
        )
        pipeline = CodeRewriteExpansionPipeline(
            llm=MockRewriteClient(),
            opd_scorer=StrategyProfileScorer(),
            max_iterations=2,
        )
        result = pipeline.expand_record(record)
        self.assertTrue(result.iteration_trace)
        self.assertTrue(any("strategy_ranking" in trace for trace in result.iteration_trace))
        selected = [trace for trace in result.iteration_trace if trace["selected_for_next_round"]]
        self.assertTrue(selected)
        self.assertIn(selected[0]["next_strategy"], {"style", "variable"})

    def test_quality_svg_renders(self):
        record = CodeRecord(
            task_id="factorial",
            prompt="Return factorial.",
            reasoning=["Initialize result.", "Loop through factors.", "Return the product."],
            code="def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result",
            tests=["assert factorial(5) == 120"],
        )
        result = CodeRewriteExpansionPipeline(llm=MockRewriteClient(), max_iterations=2).expand_record(record)
        svg = build_quality_svg([expansion_record_to_dict(result)])
        self.assertIn("Code Rewrite Quality Comparison", svg)
        self.assertIn("polyline", svg)

    def test_paper_report_contains_independent_metrics_and_retention(self):
        record = CodeRecord(
            task_id="report",
            prompt="Return factorial.",
            reasoning=["Multiply factors."],
            code="def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result",
            tests=["assert factorial(5) == 120"],
        )
        output = expansion_record_to_dict(CodeRewriteExpansionPipeline(MockRewriteClient(), max_iterations=1).expand_record(record))
        report = build_paper_report([output])
        self.assertIn("method_retention", report)
        self.assertIn("metric_improvements", report)
        self.assertIn("execution_correctness", report)
        self.assertIn("candidate_semantic_pass_rate", report["execution_correctness"])
        self.assertNotIn("aggregate_score", report)


if __name__ == "__main__":
    unittest.main()
