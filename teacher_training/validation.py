from __future__ import annotations

import ast
from typing import Any

from code_rewrite_feedback_expander.models import CodeRecord, RewriteCandidate
from code_rewrite_feedback_expander.semantic import SemanticEquivalenceChecker


def _names(code: str) -> set[str]:
    tree = ast.parse(code)
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)
    }


ALLOWED_TRANSFORMATIONS = {
    "formatting": {
        "formatting_repair", "black_reformat", "pep8_repair", "checkstyle_repair",
    },
    "identifier": {
        "identifier_deobfuscation", "rename_local_variable", "rename_parameter",
    },
    "local_structure": {
        "extract_variable", "inline_variable", "expression_split",
        "expression_merge", "local_statement_rewrite", "natgen_local_transform",
    },
    "control_flow": {
        "guard_clause", "conditional_rewrite", "loop_rewrite", "early_return",
        "break_continue_rewrite", "natgen_control_flow_transform",
    },
}


def _normalized_ast_without_identifiers(tree: ast.AST) -> str:
    """比较程序结构时忽略局部标识符文本，不构造新的质量分数。"""
    tree = ast.parse(ast.unparse(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = "__identifier__"
        elif isinstance(node, ast.arg):
            node.arg = "__identifier__"
    return ast.dump(tree, include_attributes=False)


def _control_profile(tree: ast.AST) -> dict[str, int]:
    tracked = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Break, ast.Continue, ast.Return, ast.Try)
    return {kind.__name__: sum(isinstance(node, kind) for node in ast.walk(tree)) for kind in tracked}


def _maximum_if_depth(tree: ast.AST) -> int:
    def visit(node: ast.AST, depth: int = 0) -> int:
        current = depth + 1 if isinstance(node, ast.If) else depth
        return max([current, *(visit(child, current) for child in ast.iter_child_nodes(node))])
    return visit(tree)


def _validate_control_transformation(
    transformation_type: str, before: ast.AST, after: ast.AST, row: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """验证预注册控制流规则的可观察 AST 证据，不使用节点差阈值。"""
    before_profile, after_profile = _control_profile(before), _control_profile(after)
    before_depth, after_depth = _maximum_if_depth(before), _maximum_if_depth(after)
    evidence = {
        "rule_id": transformation_type,
        "source_control_profile": before_profile,
        "target_control_profile": after_profile,
        "source_max_if_depth": before_depth,
        "target_max_if_depth": after_depth,
        "constructor": row.get("construction_method"),
        "constructor_version": row.get("constructor_version"),
    }
    before_ifs = [node for node in ast.walk(before) if isinstance(node, ast.If)]
    after_ifs = [node for node in ast.walk(after) if isinstance(node, ast.If)]
    predicates = {
        "guard_clause": (
            any(node.orelse for node in before_ifs)
            and any(not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Return) for node in after_ifs)
        ),
        "early_return": after_profile["Return"] > before_profile["Return"],
        "loop_rewrite": (
            before_profile["For"] != after_profile["For"]
            or before_profile["While"] != after_profile["While"]
        ),
        "break_continue_rewrite": (
            before_profile["Break"] != after_profile["Break"]
            or before_profile["Continue"] != after_profile["Continue"]
        ),
        "conditional_rewrite": (
            before_profile["If"] != after_profile["If"]
            or [ast.dump(n, include_attributes=False) for n in ast.walk(before) if isinstance(n, (ast.If, ast.IfExp, ast.BoolOp))]
            != [ast.dump(n, include_attributes=False) for n in ast.walk(after) if isinstance(n, (ast.If, ast.IfExp, ast.BoolOp))]
        ),
        "natgen_control_flow_transform": bool(row.get("transformation_evidence", {}).get("natgen_rule_id")),
    }
    evidence["rule_predicate_pass"] = predicates.get(transformation_type, False)
    return evidence["rule_predicate_pass"], evidence


def validate_domain(domain: str, source: str, target: str, row: dict[str, Any], language: str) -> tuple[bool, str, dict[str, Any]]:
    if language != "python":
        evidence = dict(row.get("evidence") or {})
        required = ("parse_pass", "signature_pass", "compile_pass", "test_pass")
        missing = [name for name in required if evidence.get(name) is not True]
        if missing:
            return False, f"非 Python 数据缺少或未通过硬门禁证据：{', '.join(missing)}", evidence
        if row.get("require_safety_gate") and evidence.get("safety_pass") is not True:
            return False, "该任务启用了安全门禁，但 safety_pass 未通过", evidence
        return True, "已验证的非 Python 代码对", evidence
    try:
        before, after = ast.parse(source), ast.parse(target)
    except SyntaxError as exc:
        return False, f"语法错误：{exc.msg}", {}
    dump_before, dump_after = ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False)
    source_names, target_names = _names(source), _names(target)
    changed_names = sorted(source_names ^ target_names)
    transformation_type = str(row.get("transformation_type", "")).strip()
    evidence: dict[str, Any] = {
        "ast_changed": dump_before != dump_after,
        "changed_identifiers": changed_names,
        "source_name_count": len(source_names),
        "target_name_count": len(target_names),
        "semantic_pass": bool(row.get("semantic_pass")),
        "transformation_type": transformation_type,
        "hard_gate_references": {
            "parse": "语言标准解析器（Python ast）",
            "api_signature": "Fowler (2018), Refactoring：保持可观察行为与接口",
            "compile": "语言标准编译器（Python compile）",
            "tests": "Chen et al. (2021), HumanEval；Austin et al. (2021), MBPP",
        },
    }
    record = CodeRecord(task_id=str(row.get("source_id", "unknown")), prompt=str(row.get("task", row.get("prompt", ""))), code=source, tests=list(row.get("tests") or []))
    candidate = RewriteCandidate(code=target, reasoning=[], rationale="", strategy=domain, raw_response="")
    semantic = SemanticEquivalenceChecker().check(record, candidate)
    semantic_metrics = {item.name: float(item.score) for item in semantic.scores}
    evidence["semantic_metrics"] = semantic_metrics
    evidence["semantic_feedback"] = semantic.feedback
    if not record.tests:
        return False, "正式实验要求提供可执行测试", evidence
    core_gates = {
        "parse_pass": semantic_metrics.get("ast_parse") == 1.0,
        "signature_pass": semantic_metrics.get("signature_consistency") == 1.0,
        "compile_pass": semantic_metrics.get("compile_pass_rate") == 1.0,
        "test_pass": semantic_metrics.get("unit_test_pass_rate") == 1.0,
    }
    evidence["hard_gates"] = core_gates
    failed = [name for name, passed in core_gates.items() if not passed]
    if failed:
        return False, f"未通过核心硬门禁：{', '.join(failed)}", evidence
    if row.get("require_safety_gate") and semantic_metrics.get("safety") != 1.0:
        return False, "该任务启用了安全门禁，但安全检查未通过", evidence
    if not row.get("semantic_pass", True):
        return False, "必须通过语义验证", evidence
    if source.strip() == target.strip():
        return False, "源代码与目标代码完全相同，不构成改写样本", evidence
    if transformation_type and transformation_type not in ALLOWED_TRANSFORMATIONS[domain]:
        return False, f"transformation_type={transformation_type!r} 不属于 {domain} 的预注册变换", evidence
    if domain == "formatting":
        if dump_before != dump_after:
            return False, "Formatting 只能改变具体语法布局，AST 必须完全一致", evidence
        if changed_names:
            return False, "Formatting 领域不得修改标识符", evidence
    elif domain == "local_structure":
        if not evidence["ast_changed"]:
            return False, "局部结构必须发生 AST 变化", evidence
        if _normalized_ast_without_identifiers(before) == _normalized_ast_without_identifiers(after):
            return False, "局部结构样本不能只是标识符重命名", evidence
    elif domain == "identifier":
        if not changed_names:
            return False, "Identifier 领域必须存在标识符变化", evidence
        if _normalized_ast_without_identifiers(before) != _normalized_ast_without_identifiers(after):
            return False, "Identifier 领域除标识符外不得改变 AST 结构", evidence
        evidence["identifier_change"] = True
    elif domain == "control_flow":
        control = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.Break, ast.Continue, ast.Return, ast.Try, ast.With)
        before_control = [ast.dump(n, include_attributes=False) for n in ast.walk(before) if isinstance(n, control)]
        after_control = [ast.dump(n, include_attributes=False) for n in ast.walk(after) if isinstance(n, control)]
        evidence.update({"source_control_nodes": len(before_control), "target_control_nodes": len(after_control), "control_structure_changed": before_control != after_control})
        rule_pass, rule_evidence = _validate_control_transformation(transformation_type, before, after, row)
        evidence["control_transformation_evidence"] = rule_evidence
        if not evidence["ast_changed"] or before_control == after_control:
            return False, "未发现可证明的控制流变化", evidence
        if not rule_pass:
            return False, f"控制流改写不满足预注册规则 {transformation_type!r} 的 AST 证据", evidence
    return True, "accepted", evidence
