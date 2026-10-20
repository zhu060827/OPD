"""从带测试的 Python 自然代码 JSONL 构造四领域确定性训练对。

输入至少含 source_id/problem_id、code/target_code 和 tests。无法应用预注册变换的样本写入 rejected。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any

from teacher_training.domains import require_domain
from teacher_training.io_utils import read_jsonl, write_jsonl


class RenameLocal(ast.NodeTransformer):
    def __init__(self, old: str, new: str, root: ast.AST | None = None) -> None:
        self.old, self.new, self.root = old, new, root

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old:
            node.id = self.new
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self.generic_visit(node) if self.root is None or node is self.root else node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return self.generic_visit(node) if self.root is None or node is self.root else node


def _base(row: dict[str, Any], domain: str, source: str, target: str, kind: str) -> dict[str, Any]:
    # 一个 CodeContests problem_id 对应多个 solution；必须优先保留
    # 适配阶段生成的 solution 级 source_id，避免不同样本被压成同一 ID。
    original = str(row.get("source_id") or row.get("problem_id") or row.get("task_id") or row.get("id") or "")
    if not original:
        original = hashlib.sha256(target.encode()).hexdigest()[:16]
    result = {
        "source_id": f"{original}:{domain}:{kind}",
        "problem_id": str(row.get("problem_id") or original),
        "domain": domain,
        # Task 必须描述预注册变换，不直接用函数 docstring 充当指令。
        # 否则长 XML/示例 docstring 会压过改写目标并诱导模型输出文档片段。
        "task": "保持程序行为不变，执行指定的预注册代码变换。",
        "source_code": source,
        "target_code": target,
        "tests": list(row.get("tests") or []),
        "test_format": row.get("test_format", "python_assert" if row.get("tests") else "none"),
        "test_timeout": int(row.get("test_timeout", 5)),
        "language": "python",
        "semantic_pass": True,
        "transformation_type": kind,
        "construction_method": "deterministic_ast_transform",
        "constructor_version": "teacher-constructor-v1",
        "source_dataset": row.get("source_dataset", "unknown"),
        "source_paper": row.get("source_paper", "source dataset paper"),
        "source_task": row.get("task") or row.get("docstring") or "",
    }
    for key in ("commit", "file_path", "license", "repos"):
        if row.get(key) not in (None, "", [], {}):
            result[key] = row[key]
    return result


def classify_local_structure_rejection(target: str) -> tuple[str, str, dict[str, Any]]:
    """为 Local Structure 未命中样本返回稳定、可审计的拒绝原因。"""
    try:
        tree = ast.parse(target)
    except SyntaxError as exc:
        return "parse_failure", "源代码无法解析", {"error": str(exc)}
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return) and node.value is not None]
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    if not returns:
        return "no_return_expression", "没有可注册的 return 表达式", {}
    meaningful = (ast.Call, ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Dict, ast.List, ast.Set, ast.Subscript, ast.Attribute)
    if all(isinstance(node.value, (ast.Name, ast.Constant)) for node in returns):
        return "return_is_name_or_constant", "return 仅返回变量或常量", {"return_count": len(returns)}
    if not any(isinstance(node.value, meaningful) for node in returns):
        return "unsupported_expression_type", "return 表达式类型不在预注册变换范围内", {
            "return_types": sorted({type(node.value).__name__ for node in returns})
        }
    lines = target.splitlines()
    for node in returns:
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if "#" in line:
            return "comment_on_transformation_line", "变换行包含注释，无法保留可审计源码区间", {"line": node.lineno}
    if assignments:
        return "no_safe_extract_candidate", "存在赋值但没有满足副作用和单次使用约束的候选", {"assignment_count": len(assignments)}
    return "no_supported_return_expression", "未找到预注册 Extract Variable 适用点", {"return_count": len(returns)}


def _rejection(code: str, reason: str, row: dict[str, Any], **details: Any) -> dict[str, Any]:
    return {
        "rejection_code": code,
        "reason": reason,
        "source_id": row.get("source_id") or row.get("problem_id") or row.get("id"),
        "problem_id": row.get("problem_id") or row.get("source_id") or row.get("id"),
        "details": details,
        "row": row,
    }


def construct_formatting(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    try:
        import black
    except ImportError as exc:
        raise RuntimeError("Formatting 数据构造需要固定版本 black==25.1.0") from exc
    target = black.format_str(target, mode=black.Mode()).rstrip()
    target_ast = ast.dump(ast.parse(target), include_attributes=False)

    def doubled_indentation(code: str) -> str:
        return re.sub(r"(?m)^( +)", lambda match: " " * (len(match.group(1)) * 2), code)

    def compact_token_spacing(code: str) -> str:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        return tokenize.untokenize([(token.type, token.string) for token in tokens]).rstrip()

    def extra_blank_lines(code: str) -> str:
        lines = code.splitlines()
        output: list[str] = []
        inserted = False
        for index, line in enumerate(lines):
            output.append(line)
            if not inserted and line.strip() and index + 1 < len(lines) and lines[index + 1].strip():
                output.append("")
                inserted = True
        return "\n".join(output)

    candidates = [
        ("indentation", doubled_indentation(target)),
        ("token_spacing", compact_token_spacing(target)),
        ("blank_lines", extra_blank_lines(target)),
    ]
    try:
        narrow = black.format_str(target, mode=black.Mode(line_length=60)).rstrip()
        candidates.append(("line_wrapping", narrow))
    except black.InvalidInput:
        pass

    # 由 problem_id 的稳定哈希轮换首选扰动，使各类型分布可复现且不依赖随机状态。
    identity = str(row.get("problem_id") or row.get("source_id") or target)
    offset = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % len(candidates)
    candidates = candidates[offset:] + candidates[:offset]
    for perturbation_type, source in candidates:
        if source.strip() == target.strip():
            continue
        try:
            if ast.dump(ast.parse(source), include_attributes=False) != target_ast:
                continue
            compile(source, "<formatting_source>", "exec")
        except SyntaxError:
            continue
        result = _base(row, "formatting", source, target, "formatting_repair")
        result.update({
            "task": "保持程序行为不变，仅修复代码格式与布局，不改变标识符或 AST 结构。",
            "formatting_perturbation": perturbation_type,
            "construction_method": "styler_style_deterministic_formatting",
            "constructor_version": "teacher-constructor-v2",
            "transformation_evidence": {"ast_identity": True},
        })
        return result
    return None


def construct_identifier_variants(
    row: dict[str, Any], target: str, limit: int = 1
) -> list[dict[str, Any]]:
    tree = ast.parse(target)
    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    variants: list[dict[str, Any]] = []
    forbidden_identifiers = {
        "self", "cls", "args", "kwargs", "_",
        "i", "j", "k", "x", "y", "z",
    }
    target_digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:12]
    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        nested_scopes = [
            node for node in ast.walk(function)
            if node is not function and isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
            ))
        ]
        # Python 标识符恢复必须有明确作用域。当前构造器不跨嵌套作用域
        # 做名称解析，因此拒绝这类函数，避免把内层 self/推导式变量误当外层局部量。
        if nested_scopes:
            continue
        parameters = {argument.arg for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
        parameters.update({
            argument.arg for argument in (function.args.vararg, function.args.kwarg)
            if argument is not None
        })
        candidates = []
        for node in function.body:
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and child is not function:
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id not in parameters:
                    candidates.append(child.id)
        unique_candidates = []
        for natural in dict.fromkeys(candidates):
            occurrences = sum(
                isinstance(node, ast.Name) and node.id == natural
                for node in ast.walk(function)
            )
            # DOBF 恢复需要上下文：排除仅出现一次、单字符和占位名。
            if occurrences < 2 or len(natural) < 2 or natural in forbidden_identifiers:
                continue
            subtokens = [part for part in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", natural) if part]
            # 优先多次使用、多 subtoken 且语义更充分的局部标识符。
            score = (occurrences, len(subtokens), min(len(natural), 32))
            unique_candidates.append((score, natural))
        for variant_index, (_, natural) in enumerate(sorted(unique_candidates, reverse=True)[:limit]):
            obfuscated_index = 0
            while f"VAR_{obfuscated_index}" in all_names:
                obfuscated_index += 1
            obfuscated = f"VAR_{obfuscated_index}"
            locations = [
                {
                    "line": node.lineno,
                    "column": node.col_offset,
                    "end_column": node.end_col_offset,
                    "context": type(node.ctx).__name__,
                }
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and node.id == natural
            ]
            if not locations:
                continue
            # 仅替换 AST 确认的 Name 文本区间；保留原始缩进、空格、引号和注释。
            # 不使用 ast.unparse，否则 Identifier 数据会混入 Formatting/Documentation 变化。
            lines = target.splitlines(keepends=True)
            replacements: dict[int, list[tuple[int, int]]] = {}
            for location in locations:
                replacements.setdefault(location["line"] - 1, []).append(
                    (location["column"], location["end_column"])
                )
            for line_index, spans in replacements.items():
                line = lines[line_index]
                # AST 列偏移是 UTF-8 byte offset，转成 Python 字符下标以兼容非 ASCII 注释。
                char_spans = [
                    (
                        len(line.encode("utf-8")[:start].decode("utf-8", errors="ignore")),
                        len(line.encode("utf-8")[:end].decode("utf-8", errors="ignore")),
                    )
                    for start, end in spans
                ]
                for start, end in sorted(char_spans, reverse=True):
                    line = line[:start] + obfuscated + line[end:]
                lines[line_index] = line
            source = "".join(lines)
            # 可逆性是 DOBF 式构造的基本审计证据。
            restored = RenameLocal(obfuscated, natural).visit(ast.parse(source))
            ast.fix_missing_locations(restored)
            if ast.dump(restored, include_attributes=False) != ast.dump(ast.parse(target), include_attributes=False):
                continue
            result = _base(row, "identifier", source, target, "identifier_deobfuscation")
            result.update({
                "task": f"保持程序行为不变，将混淆局部标识符 {obfuscated} 恢复为与上下文一致的自然名称。",
                "old_identifier": obfuscated, "target_identifier": natural, "scope": "local",
                "definition_use_locations": locations,
                "transformation_evidence": {"reversible": True, "renamed_occurrences": len(locations)},
                "construction_method": "dobf_deterministic_local_obfuscation",
                "constructor_version": "teacher-identifier-constructor-v6",
            })
            # 不在 source_id 中泄露金标名称；所有变体仍共享原 problem_id。
            global_variant_index = len(variants)
            result["source_id"] = (
                f"{result['source_id']}:code-{target_digest}:variant-{global_variant_index}"
            )
            result["transformation_evidence"].update({
                "candidate_rank": variant_index + 1,
                "candidate_count": len(unique_candidates),
                "scope_resolution": "single_function_no_nested_scope",
                "parameters_excluded": sorted(parameters),
                "forbidden_identifiers_excluded": sorted(forbidden_identifiers),
                "target_code_digest": target_digest,
            })
            variants.append(result)
            if len(variants) >= limit:
                return variants
    return variants


def construct_identifier(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    variants = construct_identifier_variants(row, target, limit=1)
    return variants[0] if variants else None


def _finalize_local_structure_pair(result: dict[str, Any]) -> dict[str, Any]:
    """用整个改写对的稳定摘要使 source_id 唯一，但不改变防泄漏 problem_id。"""
    source = str(result["source_code"])
    target = str(result["target_code"])
    pair_digest = hashlib.sha256(
        (source + "\0" + target).encode("utf-8")
    ).hexdigest()[:12]
    result["source_id"] = f"{result['source_id']}:code-{pair_digest}"
    result["pair_digest"] = pair_digest
    return result


def _local_structure_difficulty(function: ast.AST, expression: str, code: str) -> tuple[str, list[str]]:
    """标注容易诱发错误定位或路径语义变化的 Extract Variable 样本。"""
    nodes = list(ast.walk(function))
    reasons = []
    if sum(isinstance(node, ast.Return) for node in nodes) > 1:
        reasons.append("multiple_returns")
    if any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in nodes):
        reasons.append("loop_state")
    if any(isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in nodes):
        reasons.append("multiple_paths")
    if code.count(expression) > 1:
        reasons.append("repeated_target_expression")
    return ("hard" if reasons else "standard"), reasons


def construct_local_structure(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    def is_pure_expression(node: ast.AST) -> bool:
        """只内联无调用、无赋值、无 await/yield 的表达式，避免改变副作用顺序。"""
        unsafe = (
            ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr,
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
        )
        return not any(isinstance(item, unsafe) for item in ast.walk(node))

    def byte_to_character(line: str, offset: int) -> int:
        return len(line.encode("utf-8")[:offset].decode("utf-8", errors="ignore"))

    def extracted_name(expression: ast.AST, existing: set[str]) -> str:
        """按表达式类型生成稳定且可读的名称，避免全部目标都叫 extracted_result。"""
        if isinstance(expression, ast.Call):
            function = expression.func
            raw = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else "call"
            raw = re.sub(r"\W+", "_", raw).strip("_").lower() or "call"
            base = f"{raw}_result"
        elif isinstance(expression, (ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            base = "computed_value"
        elif isinstance(expression, (ast.Dict, ast.List, ast.Set)):
            base = "constructed_value"
        elif isinstance(expression, (ast.Subscript, ast.Attribute)):
            base = "selected_value"
        else:
            base = "extracted_value"
        candidate, suffix = base, 0
        while candidate in existing:
            suffix += 1
            candidate = f"{base}_{suffix}"
        return candidate

    tree = ast.parse(target)
    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for index in range(len(function.body) - 1):
            assignment, returned = function.body[index:index + 2]
            if (
                isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                and isinstance(assignment.targets[0], ast.Name) and isinstance(returned, ast.Return)
                and returned.value is not None
                and assignment.lineno == assignment.end_lineno
                and returned.lineno == assignment.end_lineno + 1
            ):
                temporary = assignment.targets[0].id
                direct_return = isinstance(returned.value, ast.Name) and returned.value.id == temporary
                rhs_is_pure = is_pure_expression(assignment.value)
                # 只有 `temp = expr; return temp` 能在不做全程数据流和别名分析的
                # 情况下证明求值位置与次数不变。将 temp 内联到更大的 return
                # 表达式会改变求值顺序，例如 values.pop() + temp。
                if not direct_return:
                    continue
                uses = [
                    node for node in ast.walk(returned.value)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == temporary
                ]
                if len(uses) != 1:
                    continue
                expression = ast.get_source_segment(target, assignment.value)
                if not expression:
                    continue
                lines = target.splitlines(keepends=True)
                assignment_line = lines[assignment.lineno - 1]
                # 删除带注释的赋值会丢失文本证据，因此拒绝该样本。
                if "#" in assignment_line:
                    continue
                use = uses[0]
                use_line_index = use.lineno - 1
                use_line = lines[use_line_index]
                start = byte_to_character(use_line, use.col_offset)
                end = byte_to_character(use_line, use.end_col_offset)
                lines[use_line_index] = use_line[:start] + f"({expression})" + use_line[end:]
                del lines[assignment.lineno - 1]
                source = "".join(lines)
                try:
                    source_tree = ast.parse(source)
                except SyntaxError:
                    continue
                result = _base(row, "local_structure", source, target, "extract_variable")
                difficulty, hard_reasons = _local_structure_difficulty(function, expression, source)
                result.update({
                    "task": (
                        f"保持程序行为不变。仅对函数 {function.name} 中第 {assignment.lineno} 行的 "
                        f"return 表达式 `({expression})` 执行一次 Extract Variable：先将表达式赋给局部变量 "
                        f"{temporary}，再返回该变量。除这一处外，不得修改其他语句、标识符、格式或控制流。"
                    ),
                    "construction_method": "natgen_source_span_inline_variable",
                    "constructor_version": "teacher-local-structure-constructor-v9",
                    "difficulty": difficulty,
                    "transformation_evidence": {
                        "rule_id": "extract_variable",
                        "orientation": "inline_to_extract",
                        "temporary_identifier": temporary,
                        "temporary_use_count": 1,
                        "rhs_side_effect_free": rhs_is_pure,
                        "direct_return": direct_return,
                        "target_function": function.name,
                        "target_expression": expression,
                        "expected_edit_count": 1,
                        "hard_case_reasons": hard_reasons,
                        "source_span": {"start_line": assignment.lineno, "end_line": assignment.lineno},
                        "target_span": {"start_line": assignment.lineno, "end_line": returned.end_lineno},
                        "formatting_outside_span_preserved": True,
                    },
                })
                return _finalize_local_structure_pair(result)
    # Fallback：对原始自然代码中的单行 return expression 执行 Fowler Extract Variable。
    # source 保留原始内联表达式，target 是确定性的局部结构重构。
    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for returned in [node for node in ast.walk(tree) if isinstance(node, ast.Return) and node.value is not None]:
        if returned.lineno != returned.end_lineno or isinstance(returned.value, (ast.Name, ast.Constant)):
            continue
        # 单纯 tuple 打包等变换几乎不增加局部结构，且会产生大量弱信号，不纳入该 Teacher。
        meaningful = (ast.Call, ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Dict, ast.List, ast.Set, ast.Subscript, ast.Attribute)
        if not isinstance(returned.value, meaningful):
            continue
        lines = target.splitlines(keepends=True)
        return_line = lines[returned.lineno - 1]
        if "#" in return_line:
            continue
        expression = ast.get_source_segment(target, returned.value)
        if not expression:
            continue
        containing = [
            function for function in functions
            if function.lineno <= returned.lineno <= function.end_lineno
        ]
        if not containing:
            continue
        function = min(containing, key=lambda item: item.end_lineno - item.lineno)
        temporary = extracted_name(returned.value, all_names)
        indentation = return_line[: len(return_line) - len(return_line.lstrip(" \t"))]
        newline = "\n" if return_line.endswith("\n") else ""
        lines[returned.lineno - 1 : returned.end_lineno] = [
            f"{indentation}{temporary} = {expression}\n",
            f"{indentation}return {temporary}{newline}",
        ]
        extracted_target = "".join(lines)
        try:
            ast.parse(extracted_target)
            compile(extracted_target, "<local_structure_target>", "exec")
        except SyntaxError:
            continue
        result = _base(row, "local_structure", target, extracted_target, "extract_variable")
        difficulty, hard_reasons = _local_structure_difficulty(function, expression, target)
        result.update({
            "task": (
                f"保持程序行为不变。仅对函数 {function.name} 中第 {returned.lineno} 行的 "
                f"return 表达式 `{expression}` "
                f"执行一次 Extract Variable：先将表达式赋给局部变量 {temporary}，再返回该变量。"
                "除这一处外，不得修改其他语句、标识符、格式或控制流。"
            ),
            "construction_method": "fowler_extract_return_expression",
            "constructor_version": "teacher-local-structure-constructor-v9",
            "difficulty": difficulty,
            "transformation_evidence": {
                "rule_id": "extract_variable",
                "orientation": "inline_to_extract",
                "temporary_identifier": temporary,
                "extracted_expression": expression,
                "target_expression": expression,
                "target_function": function.name,
                "expected_edit_count": 1,
                "hard_case_reasons": hard_reasons,
                "source_span": {"start_line": returned.lineno, "end_line": returned.end_lineno},
                "target_span": {"start_line": returned.lineno, "end_line": returned.end_lineno + 1},
                "formatting_outside_span_preserved": True,
            },
        })
        return _finalize_local_structure_pair(result)
    return None


def construct_control_flow(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    """构造 Conditional Rewrite；其他控制流变换暂不进入当前专家。"""
    tree = ast.parse(target)
    # 竞赛解答多为顶层 stdin/stdout 程序；对有完整 else 的条件分支做可逆的分支交换。
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.orelse and not isinstance(node.orelse[0], ast.If):
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            node.body, node.orelse = node.orelse, node.body
            ast.fix_missing_locations(tree)
            result = _base(row, "control_flow", ast.unparse(tree), target, "conditional_rewrite")
            result["transformation_evidence"] = {
                "rule_id": "conditional_rewrite", "orientation": "negated_branch_swap"
            }
            return result
    return None


CONSTRUCTORS = {
    "formatting": construct_formatting,
    "identifier": construct_identifier,
    "local_structure": construct_local_structure,
    "control_flow": construct_control_flow,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="构造一个领域的确定性 Python 改写对。")
    parser.add_argument("--input", required=True, help="已适配为函数代码和可执行 assert tests 的 JSONL；不能直接传入 Hugging Face 原始快照")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--domain", required=True, choices=tuple(CONSTRUCTORS))
    parser.add_argument(
        "--variants-per-row", type=int, default=1,
        help="Identifier 每个原函数最多构造的独立 DOBF 变体数；其他领域必须为 1",
    )
    args = parser.parse_args()
    if args.variants_per_row < 1:
        parser.error("--variants-per-row 必须至少为 1")
    if args.domain != "identifier" and args.variants_per_row != 1:
        parser.error("--variants-per-row 目前仅支持 identifier")
    domain = require_domain(args.domain)
    accepted, rejected = [], []
    for row in read_jsonl(args.input):
        target = str(row.get("target_code") or row.get("code") or "")
        if not target:
            rejected.append(_rejection("missing_code", "缺少代码", row))
            continue
        if domain == "control_flow" and not row.get("tests"):
            rejected.append(_rejection("missing_tests", "Control-flow 缺少 tests", row))
            continue
        try:
            pairs = (
                construct_identifier_variants(row, target, args.variants_per_row)
                if domain == "identifier"
                else [CONSTRUCTORS[domain](row, target)]
            )
        except (SyntaxError, ValueError) as exc:
            rejected.append(_rejection("construction_exception", f"构造异常：{exc}", row, error=str(exc)))
            continue
        pairs = [pair for pair in pairs if pair is not None]
        if not pairs:
            if domain == "local_structure":
                code, reason, details = classify_local_structure_rejection(target)
                rejected.append(_rejection(code, reason, row, **details))
            else:
                rejected.append(_rejection("no_registered_transform", "样本不匹配预注册确定性变换", row))
        else:
            accepted.extend(pairs)
    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected, rejected)
    rejection_counts = {}
    for item in rejected:
        code = item.get("rejection_code", "unknown")
        rejection_counts[code] = rejection_counts.get(code, 0) + 1
    print(json.dumps({"domain": domain, "accepted": len(accepted), "rejected": len(rejected), "rejection_counts": rejection_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
