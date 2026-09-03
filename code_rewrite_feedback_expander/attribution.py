from __future__ import annotations

import ast
import io
import tokenize
from typing import Dict, List

ASPECTS = ("cot", "style", "ast", "variable", "control_flow")


def structural_token_weights(code: str, reasoning: List[str] | None = None) -> List[Dict[str, float]]:
    """Return one multi-label weight map per lexical token.

    The labels are derived from syntax/definition-use evidence. This is an
    auditable heuristic attribution layer, not a claim of semantic ground truth.
    """
    reasoning = reasoning or []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [{"style": 1.0} for _ in _token_strings(code)]
    variable_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    control_keywords = {"if", "elif", "else", "for", "while", "try", "except", "finally", "with", "match", "case", "return", "break", "continue"}
    comments = {token.string for token in _tokens(code) if token.type == tokenize.COMMENT}
    weights: List[Dict[str, float]] = []
    for token in _tokens(code):
        text = token.string
        scores: Dict[str, float] = {}
        if text in variable_names or (text.isidentifier() and text not in dir(__builtins__)):
            scores["variable"] = 0.55
        if text in control_keywords:
            scores["control_flow"] = 0.80
        if text in {"=", "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">="}:
            scores["ast"] = 0.70
        if token.type == tokenize.COMMENT or text.startswith(('"""', "'''")) or text in comments:
            scores["cot"] = 0.55
            scores["style"] = 0.45
        if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL}:
            scores["style"] = max(scores.get("style", 0.0), 0.65)
        if not scores:
            scores["style"] = 0.35
            scores["ast"] = 0.65
        total = sum(scores.values()) or 1.0
        weights.append({aspect: value / total for aspect, value in scores.items()})
    return weights


def _tokens(code: str):
    try:
        return [token for token in tokenize.generate_tokens(io.StringIO(code).readline)
                if token.type not in {tokenize.ENCODING, tokenize.ENDMARKER}]
    except (tokenize.TokenError, IndentationError):
        return []


def _token_strings(code: str) -> List[str]:
    return [token.string for token in _tokens(code)]
