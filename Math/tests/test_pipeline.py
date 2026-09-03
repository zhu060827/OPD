import unittest

from math_reasoning_expander.llm import MockLLMClient
from math_reasoning_expander.parser import ReasoningGraphParser
from math_reasoning_expander.pipeline import MathReasoningExpansionPipeline
from math_reasoning_expander.pipeline import expansion_record_to_dict
from math_reasoning_expander.visualization import build_quality_svg


class PipelineTest(unittest.TestCase):
    def test_parser_builds_sequential_graph(self):
        parser = ReasoningGraphParser()
        graph = parser.parse(
            "What is 125 / 5?",
            "Use division.\nCompute 125 / 5 = 25.\nTherefore the answer is 25.",
        )
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(graph.edges[0].source, "s1")
        self.assertEqual(graph.edges[0].target, "s2")
        self.assertTrue(graph.nodes[1].formulas)

    def test_pipeline_expands_record_with_mock_llm(self):
        pipeline = MathReasoningExpansionPipeline(llm=MockLLMClient(), max_iterations=2)
        result = pipeline.expand_record(
            {
                "question": "What is 125 / 5?",
                "answer": "Use division.\nCompute 125 / 5 = 25.\nTherefore the answer is 25.",
            },
            mask_strategy="formula_node",
        )
        self.assertIn("25", result.expanded_answer)
        self.assertGreaterEqual(result.evaluation["aggregate_score"], 0.0)
        self.assertIsInstance(result.generated_steps, list)
        self.assertIn("original_node_count", result.expansion_stats)
        self.assertIn("expanded_node_count", result.expansion_stats)
        self.assertIsInstance(result.retained_expansions, list)
        self.assertTrue(result.iteration_trace)

    def test_quality_svg_contains_metric_labels(self):
        pipeline = MathReasoningExpansionPipeline(llm=MockLLMClient(), max_iterations=2)
        result = pipeline.expand_record(
            {
                "question": "What is 125 / 5?",
                "answer": "Use division.\nCompute 125 / 5 = 25.\nTherefore the answer is 25.",
            },
            mask_strategy="formula_node",
        )
        svg = build_quality_svg([expansion_record_to_dict(result)])
        self.assertIn("Before / After Data Quality Comparison", svg)
        self.assertIn("Formula", svg)
        self.assertIn("before expansion", svg)


if __name__ == "__main__":
    unittest.main()
