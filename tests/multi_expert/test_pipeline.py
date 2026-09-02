from __future__ import annotations

import copy
import unittest

from code_rewrite_feedback_expander.models import CodeRecord, RewriteCandidate, TokenDistribution
from code_rewrite_feedback_expander.multi_expert.backends import (
    ExpertCandidateGenerator,
    ExpertTrajectoryScorer,
)
from code_rewrite_feedback_expander.multi_expert.config import Stage1Config
from code_rewrite_feedback_expander.multi_expert.pipeline import MultiExpertStage1Pipeline
from code_rewrite_feedback_expander.multi_expert.reporting import build_mt_opd_handoff

from .test_config import valid_raw_config


VALID_CODES = {
    "cot": "# Explain the direct addition.\ndef add(a, b):\n    return a + b",
    "style": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n    return a + b",
    "ast": "def add(a, b):\n    result = a + b\n    return result",
    "variable": "def add(a, b):\n    total = a + b\n    return total",
    "control_flow": "def add(a, b):\n    if a == 0:\n        return b\n    return a + b",
}


class ScriptedGenerator(ExpertCandidateGenerator):
    def __init__(self, broken: bool = False):
        self.broken = broken
        self.record_snapshots = []

    def generate(self, expert, record):
        self.record_snapshots.append(copy.deepcopy(record))
        code = "def add(a, b):\n    return a - b" if self.broken else VALID_CODES[expert.strategy]
        return RewriteCandidate(
            code=code,
            reasoning=[f"Apply {expert.strategy} while preserving behavior."],
            rationale="scripted test candidate",
            strategy=expert.strategy,
            raw_response=code,
            metadata={"expert_id": expert.expert_id},
        )


class ScriptedScorer(ExpertTrajectoryScorer):
    ADVANTAGE = {"cot": 0.1, "style": 0.2, "ast": 0.8, "variable": 0.3, "control_flow": 0.4}

    def score(self, expert, record, candidate):
        if not hasattr(self, "calls"):
            self.calls = []
        self.calls.append((expert.expert_id, candidate.code))
        advantage = self.ADVANTAGE[expert.strategy]
        return [
            TokenDistribution(
                token="return",
                student_logprobs={"return": -0.8, "pass": -1.2},
                teacher_logprobs={"return": -0.8 + advantage, "pass": -1.2 - advantage},
                student_token_logprob=-0.8,
                teacher_token_logprob=-0.8 + advantage,
                aspect_weights={expert.strategy: 1.0},
            )
        ]

    def generate_student_completion(self, record, max_new_tokens):
        self.generation_calls = getattr(self, "generation_calls", 0) + 1
        code = VALID_CODES["style"]
        return RewriteCandidate(
            code=code,
            reasoning=[],
            rationale="scripted Student rollout",
            strategy="shared_trajectory",
            raw_response=code,
            metadata={"shared_across_teachers": True},
        )


def sample_record():
    return CodeRecord(
        task_id="add",
        prompt="Implement add(a, b).",
        code="def add(a, b):\n    return a + b",
        reasoning=["Return the sum."],
        tests=["assert add(2, 3) == 5", "assert add(0, 7) == 7"],
    )


class MultiExpertPipelineTests(unittest.TestCase):
    def test_all_experts_receive_the_same_unmodified_record(self):
        raw = valid_raw_config()
        raw["routing"] = {"reward_weight": 0.0, "nll_advantage_weight": 1.0, "top_k": 2}
        config = Stage1Config.from_dict(raw)
        generator = ScriptedGenerator()
        record = sample_record()
        original = copy.deepcopy(record)
        result = MultiExpertStage1Pipeline(config, generator, ScriptedScorer()).process(record)
        self.assertEqual(original, record)
        self.assertEqual([original] * 5, generator.record_snapshots)
        self.assertEqual("expert_ast", result.routing.selected_expert_id)

    def test_code_tests_are_a_hard_gate(self):
        config = Stage1Config.from_dict(valid_raw_config())
        result = MultiExpertStage1Pipeline(
            config, ScriptedGenerator(broken=True), ScriptedScorer()
        ).process(sample_record())
        self.assertEqual("no_valid_expert", result.routing.status)
        self.assertTrue(
            all("semantic_check_failed" in item.gate.reasons for item in result.assessments)
        )

    def test_handoff_contains_open_mopd_compatible_domain_route(self):
        raw = valid_raw_config()
        raw["routing"] = {"reward_weight": 0.0, "nll_advantage_weight": 1.0, "top_k": 2}
        result = MultiExpertStage1Pipeline(
            Stage1Config.from_dict(raw), ScriptedGenerator(), ScriptedScorer()
        ).process(sample_record())
        handoff = build_mt_opd_handoff(result)
        self.assertIsNotNone(handoff)
        self.assertEqual("ast", handoff["domain"])
        self.assertEqual("expert_ast", handoff["teacher_id"])
        self.assertAlmostEqual(1.0, sum(handoff["teacher_weights"].values()))
        self.assertEqual(sample_record().tests, handoff["reward_model"]["ground_truth"]["tests"])

    def test_handoff_student_prompt_contains_no_teacher_or_reference_information(self):
        raw = valid_raw_config()
        raw["routing"] = {"reward_weight": 0.0, "nll_advantage_weight": 1.0, "top_k": 2}
        result = MultiExpertStage1Pipeline(
            Stage1Config.from_dict(raw), ScriptedGenerator(), ScriptedScorer()
        ).process(sample_record())
        handoff = build_mt_opd_handoff(result)
        student_prompt = handoff["prompt"][0]["content"]
        self.assertEqual(sample_record().prompt, student_prompt)
        self.assertNotIn("expert_", student_prompt)
        self.assertNotIn(sample_record().code, student_prompt)

    def test_recorded_label_routes_directly_without_generating_five_candidates(self):
        raw = valid_raw_config()
        raw["routing"] = three_tier_routing()
        generator = ScriptedGenerator()
        scorer = ScriptedScorer()
        record = sample_record()
        record.metadata["method"] = "variable"
        result = MultiExpertStage1Pipeline(
            Stage1Config.from_dict(raw), generator, scorer
        ).process(record)
        self.assertEqual([], generator.record_snapshots)
        self.assertEqual([("expert_variable", record.code)], scorer.calls)
        self.assertEqual("recorded_label", result.routing.status)
        self.assertEqual("expert_variable", result.routing.selected_expert_id)

    def test_unlabeled_record_scores_one_shared_completion_with_all_teachers(self):
        raw = valid_raw_config()
        raw["routing"] = three_tier_routing()
        generator = ScriptedGenerator()
        scorer = ScriptedScorer()
        record = sample_record()
        result = MultiExpertStage1Pipeline(
            Stage1Config.from_dict(raw), generator, scorer
        ).process(record)
        self.assertEqual([], generator.record_snapshots)
        self.assertEqual(5, len(scorer.calls))
        self.assertEqual({record.code}, {code for _, code in scorer.calls})
        self.assertEqual("expert_ast", result.routing.selected_expert_id)
        self.assertEqual("calibrated_same_trajectory_opd", result.routing.routing_source)

    def test_unlabeled_gpu_path_generates_exactly_one_student_rollout(self):
        raw = valid_raw_config()
        raw["routing"] = {
            **three_tier_routing(),
            "shared_completion_source": "student_generate",
        }
        scorer = ScriptedScorer()
        result = MultiExpertStage1Pipeline(
            Stage1Config.from_dict(raw), ScriptedGenerator(), scorer
        ).process(sample_record())
        self.assertEqual(1, scorer.generation_calls)
        self.assertEqual(5, len(scorer.calls))
        self.assertEqual({VALID_CODES["style"]}, {code for _, code in scorer.calls})
        self.assertTrue(result.routing.usable_for_training)


def three_tier_routing():
    return {
        "policy": "three_tier",
        "top_k": 2,
        "minimum_margin": 0.05,
        "abstain_on_low_confidence": True,
        "calibration": {
            f"expert_{name}": {"location": 0.0, "scale": 1.0}
            for name in ("cot", "style", "ast", "variable", "control_flow")
        },
    }


if __name__ == "__main__":
    unittest.main()
