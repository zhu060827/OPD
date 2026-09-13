from __future__ import annotations

import ast
import keyword
import re
from typing import Any

from code_rewrite_feedback_expander.models import CodeRecord, RewriteCandidate
from code_rewrite_feedback_expander.semantic import SemanticEquivalenceChecker
from code_rewrite_feedback_expander.quality import CodeQualityEvaluator


def _names(code: str) -> set[str]:
    tree = ast.parse(code)
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)
    }


DEFAULT_THRESHOLDS = {
    "require_tests": 1.0,
    "unit_test_pass_rate": 1.0,
    "style_gain": 0.02,
    "naming_gain": 0.02,
    "ast_similarity_min": 0.05,
    "ast_similarity_max": 0.98,
    "control_node_delta_min": 1,
}


def validate_domain(domain: str, source: str, target: str, row: dict[str, Any], language: str, thresholds: dict[str, float] | None = None) -> tuple[bool, str, dict[str, Any]]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if language != "python":
        evidence = dict(row.get("evidence") or {})
        if not row.get("semantic_pass"):
            return False, "非 Python 数据必须通过上游语义验证", evidence
        return True, "已验证的非 Python 代码对", evidence
    try:
        before, after = ast.parse(source), ast.parse(target)
    except SyntaxError as exc:
        return False, f"语法错误：{exc.msg}", {}
    dump_before, dump_after = ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False)
    source_names, target_names = _names(source), _names(target)
    changed_names = sorted(source_names ^ target_names)
    evidence: dict[str, Any] = {
        "ast_changed": dump_before != dump_after,
        "changed_identifiers": changed_names,
        "source_name_count": len(source_names),
        "target_name_count": len(target_names),
        "semantic_pass": bool(row.get("semantic_pass")),
        "metric_references": {
            "ast_parse": "multi_expert SemanticEquivalenceChecker；Python ast 模块",
            "signature_consistency": "multi_expert SemanticEquivalenceChecker；函数签名保持约束",
            "safety": "multi_expert SemanticEquivalenceChecker；危险调用/模块黑名单",
            "compile_pass_rate": "multi_expert SemanticEquivalenceChecker；Python compile",
            "unit_test_pass_rate": "multi_expert SemanticEquivalenceChecker；执行原始测试",
            "ast_edit_distance": "multi_expert SemanticEquivalenceChecker；AST 节点序列编辑相似度",
            "codebleu_syntax": "Ren et al. (2020), CodeBLEU；当前为 AST 兼容实现",
            "codebleu_dataflow": "Ren et al. (2020), CodeBLEU；当前为数据流边重合实现",
            "style_violation_rate": "PEP 8；pycodestyle 约定；CodeXGLUE Code Refinement 任务依据",
            "naming_convention_compliance": "PEP 8 命名约定；Allamanis et al. (2015)",
            "cyclomatic_complexity": "McCabe (1976), A Complexity Measure",
        },
    }
    record = CodeRecord(task_id=str(row.get("source_id", "unknown")), prompt=str(row.get("task", row.get("prompt", ""))), code=source, tests=list(row.get("tests") or []))
    candidate = RewriteCandidate(code=target, reasoning=[], rationale="", strategy=domain, raw_response="")
    semantic = SemanticEquivalenceChecker().check(record, candidate)
    semantic_metrics = {item.name: float(item.score) for item in semantic.scores}
    evidence["semantic_metrics"] = semantic_metrics
    evidence["semantic_feedback"] = semantic.feedback
    if thresholds["require_tests"] and not record.tests:
        return False, "正式实验要求提供可执行测试", evidence
    if not semantic.passed or semantic_metrics.get("unit_test_pass_rate", 0.0) < thresholds["unit_test_pass_rate"]:
        return False, "multi_expert 语义等价检查未通过", evidence
    quality = CodeQualityEvaluator().evaluate(record, candidate, source)
    quality_metrics = {item.name: float(item.score) for item in quality.scores}
    evidence["quality_metrics"] = quality_metrics
    before_quality = CodeQualityEvaluator().evaluate(record, RewriteCandidate(code=source, reasoning=[], rationale="", strategy=domain, raw_response=""), source)
    before_metrics = {item.name: float(item.score) for item in before_quality.scores}
    evidence["quality_deltas"] = {name: quality_metrics.get(name, 0.0) - before_metrics.get(name, 0.0) for name in quality_metrics}
    if not row.get("semantic_pass", True):
        return False, "必须通过语义验证", evidence
    if domain == "cot":
        reasoning = row.get("target_reasoning") or row.get("expanded_reasoning") or row.get("reasoning")
        if not reasoning:
            return False, "CoT 必须包含目标推理文本", evidence
        evidence["reasoning_present"] = True
    elif domain == "style":
        gain = quality_metrics.get("style_violation_rate", 0.0) - before_metrics.get("style_violation_rate", 0.0)
        evidence["style_gain"] = gain
        if source == target or gain < thresholds["style_gain"]:
            return False, "风格指标提升未达到阈值", evidence
    elif domain == "ast":
        similarity = semantic_metrics.get("ast_edit_distance", 0.0)
        evidence["ast_similarity"] = similarity
        if not evidence["ast_changed"] or changed_names or not (thresholds["ast_similarity_min"] <= similarity <= thresholds["ast_similarity_max"]):
            return False, "AST 必须存在结构变化，不能只有标识符变化", evidence
    elif domain == "variable":
        gain = quality_metrics.get("naming_convention_compliance", 0.0) - before_metrics.get("naming_convention_compliance", 0.0)
        evidence["naming_gain"] = gain
        if not changed_names or gain < thresholds["naming_gain"]:
            return False, "变量领域必须存在标识符变化", evidence
        if not evidence["ast_changed"]:
            return False, "缺少变量改写的 AST 证据", evidence
        evidence["identifier_change"] = True
    elif domain == "control_flow":
        control = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.Break, ast.Continue, ast.Return, ast.Try, ast.With)
        before_control = sum(isinstance(n, control) for n in ast.walk(before))
        after_control = sum(isinstance(n, control) for n in ast.walk(after))
        evidence.update({"source_control_nodes": before_control, "target_control_nodes": after_control, "control_count_changed": before_control != after_control})
        if not evidence["ast_changed"] or abs(after_control - before_control) < thresholds["control_node_delta_min"]:
            return False, "未发现可证明的控制流变化", evidence
    return True, "accepted", evidence
