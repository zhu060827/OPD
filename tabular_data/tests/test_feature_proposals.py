"""候选可解释性、逐行可复用变换及依赖消融检查。"""
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from llm_client import LLMClient
from tabular_data.ablation import dependency_group, permutation_scores, run_ablation
from tabular_data.feature_generation import OfflineFeatureClient
from tabular_data.feature_proposals import propose_feature_candidates, validate_expression
from tabular_data.tabular_octree import check_feature_robustness


class FeatureProposalTests(unittest.TestCase):
    def propose(self, *args, **kwargs):
        from tabular_data.model_guidance import build_guidance
        frame = args[0]
        probabilities = np.full((len(frame), 2), .5)
        context = build_guidance(frame, [c for c in frame if c != "target"],
                                 predictions=np.zeros(len(frame)), probabilities=probabilities)
        return propose_feature_candidates(*args, guidance=context, **kwargs)

    def setUp(self):
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=["a", "b", "c", "d"])
        self.frame["target"] = (self.frame.a * self.frame.b > 0).astype(int)

    def test_reasoned_families_are_explained_and_reusable(self):
        plans = self.propose(self.frame, OfflineFeatureClient(), 1, [], set())
        self.assertEqual({p.family for p in plans}, {"multicolumn", "nonlinear", "piecewise"})
        for plan in plans:
            self.assertGreaterEqual(len(plan.input_columns), 1)
            if plan.family == "multicolumn":
                self.assertGreaterEqual(len(plan.input_columns), 3)
            self.assertTrue(plan.hypothesis and plan.construction and plan.evidence)
            passed, whole, _, _ = check_feature_robustness(self.frame, plan.code)
            self.assertTrue(passed)
            # 同一行单独转换与放在整批数据转换必须一致，阈值不能在验证/测试重新计算。
            passed, single, _, _ = check_feature_robustness(self.frame.iloc[[5]], plan.code, allow_constant=True)
            self.assertTrue(passed)
            self.assertAlmostEqual(single[plan.name].iloc[0], whole[plan.name].iloc[5])
            self.assertEqual(plan.to_dict()["evidence_split"], "train")

    def test_no_duplicate_formulas_across_rounds(self):
        seen = set()
        for round_index in (1, 2, 3, 4, 5):
            plans = self.propose(self.frame, OfflineFeatureClient(), round_index, [], seen)
            self.assertEqual(len(plans), 5)
            for plan in plans:
                self.assertNotIn(plan.signature, seen)
                seen.add(plan.signature)

    def test_retired_generator_cannot_be_selected(self):
        with self.assertRaisesRegex(ValueError, "reasoned"):
            self.propose(self.frame, OfflineFeatureClient(), 1, [], set(), mode="retired")

    def test_expression_rejects_label_access_and_cross_row_statistics(self):
        for expression in ("df['target']", "np.mean(df['a'])", "df['a'].quantile(.5)",
                           "df['a'].iloc[0] + df['b']", "df.index + df['a']",
                           "__import__('os').getcwd()", "np.random.normal(size=120)"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                validate_expression(expression, {"a", "b"})

    def test_llm_returns_structured_explanation_and_is_audited(self):
        client = LLMClient(api_key="test-placeholder")
        payload = {"features": [{"name": "test_composite", "family": "multicolumn",
                    "expression": "df['a'] * df['b'] / (1 + np.abs(df['c']))",
                    "hypothesis": "三列的联合关系可能有效，等待验证。", "construction": "计算乘积后按第三列调制。"}]}
        with patch.object(client, "chat", return_value=json.dumps(payload)):
            plans = self.propose(self.frame, client, 1, [], set(), count=1)
        self.assertEqual(plans[0].source, "llm")
        self.assertEqual(client.real_call_count, 1)
        self.assertEqual(client.mock_fallback_count, 0)

    def test_invalid_llm_expression_falls_back_without_executing(self):
        client = LLMClient(api_key="test-placeholder")
        payload = {"features": [{"name": "leak", "family": "nonlinear", "expression": "df['target']",
                                 "hypothesis": "invalid", "construction": "invalid"}]}
        with patch.object(client, "chat", return_value=json.dumps(payload)):
            plans = self.propose(self.frame, client, 1, [], set())
        self.assertEqual(len(plans), 5)
        self.assertTrue(all(p.source in {"local_template", "cart_rule"} for p in plans))
        self.assertEqual(client.mock_fallback_count, 1)
        self.assertEqual(client.real_call_count, 6)

    def test_dependency_ablation_removes_descendants(self):
        proposals = [{"name": "f1", "input_columns": ["a", "b"]},
                     {"name": "f2", "input_columns": ["f1", "c"]},
                     {"name": "f3", "input_columns": ["f2", "d"]},
                     {"name": "other", "input_columns": ["a", "c"]}]
        self.assertEqual(dependency_group("f1", proposals), ["f1", "f2", "f3"])

    def test_permutation_is_reproducible_and_does_not_mutate_test_data(self):
        frame = self.frame.assign(signal=self.frame.target)
        original = frame.copy(deep=True)
        def predict(model, features, task):
            prediction = features.signal.to_numpy(dtype=int)
            return prediction, np.column_stack((1 - prediction, prediction)) * .98 + .01
        with patch("tabular_data.ablation.predict_model", side_effect=predict):
            first = permutation_scores(Mock(), frame, ["signal"], "classification", repeats=5)
            second = permutation_scores(Mock(), frame, ["signal"], "classification", repeats=5)
        self.assertEqual(first, second)
        self.assertGreater(first["mean_score_drop"]["f1_macro"], .2)
        pd.testing.assert_frame_equal(frame, original)

    def test_no_accepted_features_does_not_fabricate_importance(self):
        result = run_ablation(Mock(), Mock(), self.frame, self.frame, [])
        self.assertEqual(result["status"], "no_accepted_features")
        self.assertEqual(result["feature_rows"], [])


if __name__ == "__main__":
    unittest.main()
