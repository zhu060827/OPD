"""从带测试的 Python 自然代码 JSONL 构造四领域确定性训练对。

输入至少含 source_id/problem_id、code/target_code 和 tests。无法应用预注册变换的样本写入 rejected。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from teacher_training.domains import require_domain
from teacher_training.io_utils import read_jsonl, write_jsonl


class RenameLocal(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old, self.new = old, new

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old:
            node.id = self.new
        return node


def _base(row: dict[str, Any], domain: str, source: str, target: str, kind: str) -> dict[str, Any]:
    original = str(row.get("problem_id") or row.get("source_id") or row.get("task_id") or row.get("id") or "")
    if not original:
        original = hashlib.sha256(target.encode()).hexdigest()[:16]
    return {
        "source_id": f"{original}:{domain}:{kind}",
        "problem_id": original,
        "domain": domain,
        "task": row.get("task") or row.get("docstring") or "保持程序行为不变，按指定领域改写代码。",
        "source_code": source,
        "target_code": target,
        "tests": list(row.get("tests") or []),
        "language": "python",
        "semantic_pass": True,
        "transformation_type": kind,
        "construction_method": "deterministic_ast_transform",
        "constructor_version": "teacher-constructor-v1",
        "source_dataset": row.get("source_dataset", "unknown"),
        "source_paper": row.get("source_paper", "source dataset paper"),
    }


def construct_formatting(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    try:
        import black
    except ImportError as exc:
        raise RuntimeError("Formatting 数据构造需要固定版本 black==25.1.0") from exc
    target = black.format_str(target, mode=black.Mode()).rstrip()
    source = re.sub(r"(?m)^( +)", lambda match: " " * (len(match.group(1)) * 2), target)
    if source == target:
        return None
    return _base(row, "formatting", source, target, "formatting_repair")


def construct_identifier(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    tree = ast.parse(target)
    candidates = [
        node.targets[0].id for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    ]
    if not candidates:
        return None
    natural = candidates[0]
    obfuscated = "VAR_0"
    transformed = RenameLocal(natural, obfuscated).visit(tree)
    ast.fix_missing_locations(transformed)
    result = _base(row, "identifier", ast.unparse(transformed), target, "identifier_deobfuscation")
    result.update({"old_identifier": obfuscated, "target_identifier": natural, "scope": "local"})
    return result


def construct_local_structure(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    tree = ast.parse(target)
    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for index in range(len(function.body) - 1):
            assignment, returned = function.body[index:index + 2]
            if (
                isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                and isinstance(assignment.targets[0], ast.Name) and isinstance(returned, ast.Return)
                and isinstance(returned.value, ast.Name) and returned.value.id == assignment.targets[0].id
            ):
                function.body[index:index + 2] = [ast.Return(value=assignment.value)]
                ast.fix_missing_locations(tree)
                return _base(row, "local_structure", ast.unparse(tree), target, "extract_variable")
    return None


def construct_control_flow(row: dict[str, Any], target: str) -> dict[str, Any] | None:
    """把自然 guard-clause target 确定性变为嵌套 source，训练方向为 source→target。"""
    tree = ast.parse(target)
    for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if len(function.body) == 2 and isinstance(function.body[0], ast.If) and not function.body[0].orelse:
            guard, tail = function.body
            if len(guard.body) == 1 and isinstance(guard.body[0], ast.Return) and isinstance(tail, ast.Return):
                nested = ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=guard.test),
                    body=[tail],
                    orelse=[guard.body[0]],
                )
                function.body = [nested]
                ast.fix_missing_locations(tree)
                result = _base(row, "control_flow", ast.unparse(tree), target, "guard_clause")
                result["transformation_evidence"] = {"rule_id": "guard_clause", "orientation": "nested_to_guard"}
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
    args = parser.parse_args()
    domain = require_domain(args.domain)
    accepted, rejected = [], []
    for row in read_jsonl(args.input):
        target = str(row.get("target_code") or row.get("code") or "")
        if not target or not row.get("tests"):
            rejected.append({"reason": "缺少代码或 tests", "row": row})
            continue
        try:
            pair = CONSTRUCTORS[domain](row, target)
        except (SyntaxError, ValueError) as exc:
            pair = None
            rejected.append({"reason": f"构造异常：{exc}", "row": row})
        if pair is None:
            rejected.append({"reason": "样本不匹配预注册确定性变换", "row": row})
        else:
            accepted.append(pair)
    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected, rejected)
    print(json.dumps({"domain": domain, "accepted": len(accepted), "rejected": len(rejected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
