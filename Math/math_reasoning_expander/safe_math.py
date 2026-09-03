from __future__ import annotations

import ast
import operator
import re
from typing import Dict, Optional


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def normalize_formula(expr: str) -> str:
    expr = expr.replace("^", "**").replace("$", "").strip()
    expr = re.sub(r"(?<=\d)(?=[A-Za-z])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z])(?=\d)", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z0-9)])(?=\()", "*", expr)
    expr = re.sub(r"(?<=\))(?=[A-Za-z0-9])", "*", expr)
    return expr


def extract_numeric_assignments(text: str) -> Dict[str, float]:
    assignments: Dict[str, float] = {}
    for name, value in re.findall(r"\b([A-Za-z]\w*)\s*=\s*(-?\d+(?:\.\d+)?)\b", text):
        assignments[name] = float(value)
    return assignments


def safe_eval(expr: str, variables: Optional[Dict[str, float]] = None) -> Optional[float]:
    variables = variables or {}
    expr = normalize_formula(expr)
    try:
        tree = ast.parse(expr, mode="eval")
        return float(_eval_node(tree.body, variables))
    except Exception:
        return None


def _eval_node(node: ast.AST, variables: Dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown variable: {node.id}")
        return float(variables[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left, variables), _eval_node(node.right, variables))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand, variables))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def equation_is_consistent(equation: str, context: str = "", tolerance: float = 1e-6) -> Optional[bool]:
    if "=" not in equation:
        return None
    parts = [part.strip() for part in equation.split("=") if part.strip()]
    if len(parts) < 2:
        return None
    variables = extract_numeric_assignments(context)
    values = [safe_eval(part, variables) for part in parts]
    if any(value is None for value in values):
        return None
    return all(abs(values[idx] - values[idx + 1]) <= tolerance for idx in range(len(values) - 1))
