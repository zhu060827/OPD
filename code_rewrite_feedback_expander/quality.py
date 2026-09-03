from __future__ import annotations

import ast
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import CodeRecord, MetricScore, QualityResult, RewriteCandidate


UNIFIED_METRICS = [
    "style_violation_rate",
    "maintainability_index",
    "cyclomatic_complexity",
    "naming_convention_compliance",
    "codebleu_syntax_match",
]

METHOD_TO_QUALITY_METRIC = {
    "cot": "maintainability_index",
    "style": "style_violation_rate",
    "ast": "codebleu_syntax_match",
    "variable": "naming_convention_compliance",
    "control_flow": "cyclomatic_complexity",
}

class CodeQualityEvaluator:
    def __init__(self, accept_threshold: float = 0.78):
        self.accept_threshold = accept_threshold

    def evaluate(self, record: CodeRecord, candidate: RewriteCandidate, original_code: str) -> QualityResult:
        metric_names = self._strategy_profile(candidate.strategy)
        scores = [self._score_metric(name, record, candidate.code, original_code) for name in metric_names]
        feedback = self._feedback(scores)
        return QualityResult(scores=scores, feedback=feedback)

    def _strategy_profile(self, strategy: str) -> List[str]:
        _ = strategy
        return list(UNIFIED_METRICS)

    def _score_metric(self, name: str, record: CodeRecord, code: str, original_code: str) -> MetricScore:
        if name == "style_violation_rate":
            return self._style_violation_rate(code)
        if name in {"readability", "readability_score", "readability_proxy"}:
            return self._readability_score(code)
        if name == "maintainability_index":
            return self._maintainability_index(code)
        if name == "halstead_effort":
            return self._halstead_effort(code)
        if name == "local_complexity":
            return self._local_complexity(code)
        if name == "compile_pass_rate":
            return self._compile_pass_rate(code)
        if name == "linter_violation_score":
            return self._linter_violation_score(code)
        if name == "formatter_consistency":
            return self._formatter_consistency(code)
        if name == "style_conformance":
            return self._style_conformance(code)
        if name == "ast_edit_distance":
            return self._ast_edit_distance(original_code, code)
        if name == "tree_edit_distance":
            return self._tree_edit_distance(original_code, code)
        if name in {"codebleu_syntax_match", "ast"}:
            return self._ast_match(original_code, code)
        if name in {"naming_convention_compliance", "naming_conformance_rate", "naming_conformance"}:
            return self._naming_conformance_rate(code)
        if name == "abbreviation_score":
            return self._abbreviation_score(code)
        if name == "reference_resolution":
            return self._reference_resolution(code)
        if name == "identifier_context_consistency":
            return self._identifier_context_consistency(code)
        if name == "cyclomatic_complexity":
            return self._cyclomatic_complexity(code)
        if name == "cognitive_complexity":
            return self._cognitive_complexity(code)
        if name == "npath_complexity":
            return self._npath_complexity(code)
        if name == "max_nesting_depth":
            return self._max_nesting_depth_score(code)
        raise KeyError(f"Unknown metric: {name}")

    def _style_violation_rate(self, code: str) -> MetricScore:
        """PEP 8/pycodestyle-style conformance, normalized for code length."""
        lines = code.splitlines()
        logical_lines = [line for line in lines if line.strip()]
        violations = self._style_violations(code)
        density = len(violations) / max(1, len(logical_lines))
        score = 1.0 / (1.0 + density)
        return MetricScore(
            "style_violation_rate",
            score,
            f"PEP 8-style violations={len(violations)}, logical_lines={len(logical_lines)}, density={density:.3f}.",
            raw_value=float(len(violations)),
            implementation="pycodestyle_compatible_internal_checks",
            reference="PEP 8; pycodestyle style-checking convention.",
        )

    def _readability_score(self, code: str) -> MetricScore:
        lines = _logical_lines(code)
        if not lines:
            return MetricScore("readability_score", 0.0, "No code lines found.")
        avg_len = sum(len(line) for line in lines) / len(lines)
        long_lines = sum(1 for line in lines if len(line) > 88)
        comment_lines = sum(1 for line in lines if line.strip().startswith("#") or '"""' in line or "'''" in line)
        long_line_penalty = long_lines / len(lines) * 0.35
        avg_penalty = min(0.40, max(0.0, (avg_len - 72) / 100))
        comment_bonus = min(0.12, comment_lines / max(1, len(lines)) * 0.5)
        score = 1.0 - avg_penalty - long_line_penalty + comment_bonus
        return MetricScore(
            "readability",
            _clamp(score),
            f"Average line length={avg_len:.1f}, long lines={long_lines}, comments/docstrings={comment_lines}.",
            raw_value=avg_len,
            implementation="documented_proxy_not_buse_weimer",
            reference="Buse and Weimer (2010) motivates readability measurement; this fallback is not their learned model.",
        )

    def _maintainability_index(self, code: str) -> MetricScore:
        lines = max(1, len(_logical_lines(code)))
        cyclomatic = self._cyclomatic_number(code)
        halstead = self._halstead_metrics(code)
        volume = max(1.0, halstead["volume"])
        raw = 171 - 5.2 * math.log(volume) - 0.23 * cyclomatic - 16.2 * math.log(lines)
        normalized = _clamp(raw / 171)
        return MetricScore(
            "maintainability_index",
            normalized,
            f"SEI derivative MI={raw:.2f}/171; normalized MI={normalized * 100:.1f}/100, Halstead volume={volume:.1f}, cyclomatic={cyclomatic}, LOC={lines}.",
            raw_value=raw,
            implementation="sei_mi_without_comment_term",
            reference="Oman and Hagemeister (1992); SEI derivative formula.",
        )

    def _halstead_effort(self, code: str) -> MetricScore:
        metrics = self._halstead_metrics(code)
        effort = metrics["effort"]
        score = 1.0 / (1.0 + math.log10(max(1.0, effort)) / 6.0)
        return MetricScore(
            "halstead_effort",
            _clamp(score),
            f"Effort={effort:.1f}, vocabulary={metrics['vocabulary']}, length={metrics['length']}.",
        )

    def _local_complexity(self, code: str) -> MetricScore:
        values = self._function_cyclomatic_numbers(code)
        if not values:
            values = [self._cyclomatic_number(code)]
        worst = max(values)
        score = _complexity_score(worst, good=6, step=0.08)
        return MetricScore("local_complexity", score, f"Maximum function-level cyclomatic complexity={worst}.")

    def _compile_pass_rate(self, code: str) -> MetricScore:
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            return MetricScore("compile_pass_rate", 0.0, f"Compilation failed: {exc}.")
        return MetricScore("compile_pass_rate", 1.0, "Code compiles successfully.")

    def _linter_violation_score(self, code: str) -> MetricScore:
        violations = self._style_violations(code)
        score = max(0.0, 1.0 - len(violations) * 0.12)
        return MetricScore("linter_violation_score", score, f"Linter-style violations={len(violations)}: {violations[:5]}.")

    def _formatter_consistency(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("formatter_consistency", 0.0, "Code cannot be parsed for formatting round-trip.")
        try:
            formatted = ast.unparse(tree)
            round_trip_tree = ast.parse(formatted)
        except Exception as exc:
            return MetricScore("formatter_consistency", 0.0, f"AST formatter round-trip failed: {exc}.")
        same_ast = ast.dump(tree, include_attributes=False) == ast.dump(round_trip_tree, include_attributes=False)
        return MetricScore(
            "formatter_consistency",
            1.0 if same_ast else 0.0,
            "AST unparse/parse round-trip preserved structure." if same_ast else "AST round-trip changed structure.",
        )

    def _style_conformance(self, code: str) -> MetricScore:
        lines = _logical_lines(code)
        if not lines:
            return MetricScore("style_conformance", 0.0, "No code lines found.")
        violations = self._style_violations(code)
        score = max(0.0, 1.0 - len(violations) / max(1, len(lines)))
        return MetricScore("style_conformance", score, f"Style violations={len(violations)} over {len(lines)} logical lines.")

    def _ast_edit_distance(self, original: str, rewritten: str) -> MetricScore:
        original_tree = self._safe_parse(original)
        rewritten_tree = self._safe_parse(rewritten)
        if original_tree is None or rewritten_tree is None:
            return MetricScore("ast_edit_distance", 0.0, "Cannot compute AST edit distance without valid AST.")
        left = [type(node).__name__ for node in ast.walk(original_tree)]
        right = [type(node).__name__ for node in ast.walk(rewritten_tree)]
        distance = _levenshtein(left, right)
        denominator = max(1, max(len(left), len(right)))
        normalized_distance = distance / denominator
        return MetricScore(
            "ast_edit_distance",
            _clamp(1.0 - normalized_distance),
            f"Normalized AST edit distance={normalized_distance:.3f}.",
        )

    def _tree_edit_distance(self, original: str, rewritten: str) -> MetricScore:
        original_tree = self._safe_parse(original)
        rewritten_tree = self._safe_parse(rewritten)
        if original_tree is None or rewritten_tree is None:
            return MetricScore("tree_edit_distance", 0.0, "Cannot compute tree edit distance without valid AST.")
        left = self._tree_tokens(original_tree)
        right = self._tree_tokens(rewritten_tree)
        distance = _levenshtein(left, right)
        denominator = max(1, max(len(left), len(right)))
        normalized_distance = distance / denominator
        return MetricScore(
            "tree_edit_distance",
            _clamp(1.0 - normalized_distance),
            f"Normalized ordered-tree edit distance={normalized_distance:.3f}.",
        )

    def _ast_match(self, original: str, rewritten: str) -> MetricScore:
        original_tree = self._safe_parse(original)
        rewritten_tree = self._safe_parse(rewritten)
        if original_tree is None or rewritten_tree is None:
            return MetricScore("codebleu_syntax_match", 0.0, "Cannot compute CodeBLEU syntax match without valid AST.")
        reference_subtrees = _codebleu_ast_subtrees(original_tree)
        candidate_subtrees = _codebleu_ast_subtrees(rewritten_tree)
        if not reference_subtrees:
            score = 1.0 if not candidate_subtrees else 0.0
            return MetricScore("codebleu_syntax_match", score, "CodeBLEU syntax match is degenerate due to empty reference tree.")
        reference_counts = Counter(reference_subtrees)
        candidate_counts = Counter(candidate_subtrees)
        matched = sum(min(reference_counts[subtree], candidate_counts.get(subtree, 0)) for subtree in reference_counts)
        total = sum(reference_counts.values())
        score = matched / total if total else 0.0
        return MetricScore(
            "codebleu_syntax_match",
            _clamp(score),
            f"CodeBLEU syntax match={score:.3f}, matched_subtrees={matched}, reference_subtrees={total}.",
            raw_value=score,
            implementation="codebleu_compatible_fallback",
            reference="Ren et al. (2020). Install the official CodeBLEU/tree-sitter backend for paper results.",
        )

    def _naming_conformance_rate(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("naming_convention_compliance", 0.0, "Code cannot be parsed.")
        identifiers = self._collect_identifiers(tree)
        if not identifiers:
            return MetricScore("naming_convention_compliance", 1.0, "No explicit identifiers found.")
        conforming = sum(1 for name in identifiers if _is_pep8_identifier(name))
        return MetricScore(
            "naming_convention_compliance",
            conforming / len(identifiers),
            f"PEP 8-style conforming identifiers={conforming}/{len(identifiers)}.",
            raw_value=conforming / len(identifiers),
            implementation="pep8_identifier_conformance",
            reference="PEP 8 naming conventions; pep8-naming/Pylint invalid-name rules.",
        )

    def _abbreviation_score(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("abbreviation_score", 0.0, "Code cannot be parsed.")
        identifiers = self._collect_identifiers(tree)
        if not identifiers:
            return MetricScore("abbreviation_score", 1.0, "No explicit identifiers found.")
        abbreviated = 0
        for name in identifiers:
            tokens = _split_identifier(name)
            if any(len(token) <= 2 and not token.isdigit() for token in tokens):
                abbreviated += 1
        score = 1.0 - abbreviated / len(identifiers)
        return MetricScore("abbreviation_score", score, f"Abbreviated identifiers={abbreviated}/{len(identifiers)}.")

    def _reference_resolution(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("reference_resolution", 0.0, "Code cannot be parsed.")
        defined = set()
        unresolved = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
                for arg in node.args.args:
                    defined.add(arg.arg)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
                defined.add(node.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined and node.id not in dir(__builtins__):
                    unresolved.add(node.id)
        score = 1.0 if not unresolved else max(0.0, 1.0 - len(unresolved) * 0.2)
        return MetricScore("reference_resolution", score, f"Unresolved loaded names={sorted(unresolved)}.")

    def _identifier_context_consistency(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("identifier_context_consistency", 0.0, "Code cannot be parsed.")
        definitions = Counter()
        uses = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                definitions[node.arg] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
                definitions[node.id] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                uses[node.id] += 1
        user_names = set(definitions)
        if not user_names:
            score = 1.0
        else:
            score = sum(1 for name in user_names if uses[name] > 0 or definitions[name] > 1) / len(user_names)
        return MetricScore(
            "identifier_context_consistency",
            score,
            f"Defined identifiers with a consistent use context={score:.3f}; definitions={len(user_names)}.",
            raw_value=score,
            implementation="definition_use_consistency_proxy",
            reference="Definition-use consistency is standard static-analysis evidence; this ratio is reported as a proxy.",
        )

    def _cyclomatic_complexity(self, code: str) -> MetricScore:
        value = self._cyclomatic_number(code)
        score = _complexity_score(value, good=6, step=0.08)
        return MetricScore(
            "cyclomatic_complexity",
            score,
            f"Cyclomatic complexity={value}; McCabe low-risk boundary is commonly CC<=10.",
            raw_value=float(value),
            implementation="mccabe_decision_count",
            reference="McCabe (1976).",
        )

    def _cognitive_complexity(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("cognitive_complexity", 0.0, "Code cannot be parsed.")
        value = self._cognitive_complexity_number(tree)
        score = _complexity_score(value, good=8, step=0.06)
        return MetricScore("cognitive_complexity", score, f"Cognitive complexity={value}.")

    def _npath_complexity(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("npath_complexity", 0.0, "Code cannot be parsed.")
        value = max(1, self._npath(tree))
        score = 1.0 / (1.0 + math.log10(value) / 4.0)
        return MetricScore("npath_complexity", _clamp(score), f"NPath complexity={value}.")

    def _max_nesting_depth_score(self, code: str) -> MetricScore:
        tree = self._safe_parse(code)
        if tree is None:
            return MetricScore("max_nesting_depth", 0.0, "Code cannot be parsed.")
        depth = self._max_branch_depth(tree)
        score = _complexity_score(depth, good=3, step=0.14)
        return MetricScore("max_nesting_depth", score, f"Maximum nesting depth={depth}.")

    def _feedback(self, scores: List[MetricScore]) -> str:
        weak = [score for score in scores if score.score < 0.72]
        if weak:
            advice = "Improve " + ", ".join(score.name for score in weak) + "."
        else:
            advice = "Quality is usable; preserve semantic equivalence while improving the selected code-quality metrics."
        details = "; ".join(f"{score.name}={score.score:.2f}: {score.explanation}" for score in scores)
        return f"{advice} Metrics are reported independently. Details: {details}"

    def _safe_parse(self, code: str):
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def _style_violations(self, code: str) -> List[str]:
        violations = []
        for line_number, line in enumerate(code.splitlines(), start=1):
            if line.rstrip() != line:
                violations.append(f"L{line_number}: trailing whitespace")
            if "\t" in line:
                violations.append(f"L{line_number}: tab indentation")
            if len(line) > 88:
                violations.append(f"L{line_number}: line too long")
        tree = self._safe_parse(code)
        if tree is None:
            violations.append("syntax: cannot parse")
        return violations

    def _halstead_metrics(self, code: str) -> Dict[str, float]:
        operators: List[str] = []
        operands: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"vocabulary": 0, "length": 0, "volume": 0.0, "difficulty": 0.0, "effort": float("inf")}
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Assign, ast.Return)):
                operators.append(type(node).__name__)
            if isinstance(node, ast.operator):
                operators.append(type(node).__name__)
            if isinstance(node, ast.cmpop):
                operators.append(type(node).__name__)
            if isinstance(node, ast.Name):
                operands.append(node.id)
            elif isinstance(node, ast.Constant):
                operands.append(repr(node.value))
        distinct_operators = len(set(operators))
        distinct_operands = len(set(operands))
        total_operators = len(operators)
        total_operands = len(operands)
        vocabulary = distinct_operators + distinct_operands
        length = total_operators + total_operands
        volume = length * math.log2(vocabulary) if vocabulary > 1 and length else 0.0
        difficulty = (distinct_operators / 2) * (total_operands / distinct_operands) if distinct_operands else 0.0
        effort = difficulty * volume
        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
        }

    def _function_cyclomatic_numbers(self, code: str) -> List[int]:
        tree = self._safe_parse(code)
        if tree is None:
            return []
        values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                values.append(self._cyclomatic_number_for_node(node))
        return values

    def _cyclomatic_number(self, code: str) -> int:
        tree = self._safe_parse(code)
        if tree is None:
            return 999
        return self._cyclomatic_number_for_node(tree)

    def _cyclomatic_number_for_node(self, node: ast.AST) -> int:
        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.IfExp, ast.ExceptHandler)
        bool_ops = sum(len(item.values) - 1 for item in ast.walk(node) if isinstance(item, ast.BoolOp))
        branches = sum(1 for item in ast.walk(node) if isinstance(item, branch_nodes))
        return 1 + branches + bool_ops

    def _collect_identifiers(self, tree: ast.AST) -> List[str]:
        identifiers: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                identifiers.append(node.name)
            elif isinstance(node, ast.arg):
                identifiers.append(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                identifiers.append(node.id)
        return identifiers

    def _tree_tokens(self, node: ast.AST) -> List[str]:
        tokens = [f"({type(node).__name__}"]
        for child in ast.iter_child_nodes(node):
            tokens.extend(self._tree_tokens(child))
        tokens.append(")")
        return tokens

    def _cognitive_complexity_number(self, tree: ast.AST) -> int:
        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.IfExp)

        def walk(node: ast.AST, nesting: int) -> int:
            score = 0
            next_nesting = nesting
            if isinstance(node, branch_nodes):
                score += 1 + nesting
                next_nesting += 1
            elif isinstance(node, ast.BoolOp):
                score += max(0, len(node.values) - 1)
            for child in ast.iter_child_nodes(node):
                score += walk(child, next_nesting)
            return score

        return walk(tree, 0)

    def _npath(self, node: ast.AST) -> int:
        if isinstance(node, ast.Module):
            return _product(self._npath(child) for child in node.body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return _product(self._npath(child) for child in node.body)
        if isinstance(node, ast.If):
            body = _product(self._npath(child) for child in node.body)
            orelse = _product(self._npath(child) for child in node.orelse) if node.orelse else 1
            return max(1, body + orelse)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            body = _product(self._npath(child) for child in node.body)
            orelse = _product(self._npath(child) for child in node.orelse) if node.orelse else 1
            return max(1, body + orelse + 1)
        if isinstance(node, ast.Try):
            body = _product(self._npath(child) for child in node.body)
            handlers = sum(_product(self._npath(child) for child in handler.body) for handler in node.handlers) or 1
            orelse = _product(self._npath(child) for child in node.orelse) if node.orelse else 1
            finalbody = _product(self._npath(child) for child in node.finalbody) if node.finalbody else 1
            return max(1, (body + handlers + orelse) * finalbody)
        if isinstance(node, ast.BoolOp):
            return max(1, len(node.values))
        return 1

    def _max_branch_depth(self, tree: ast.AST) -> int:
        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.IfExp)

        def walk(node: ast.AST, depth: int) -> int:
            next_depth = depth + 1 if isinstance(node, branch_nodes) else depth
            child_depth = next_depth
            for child in ast.iter_child_nodes(node):
                child_depth = max(child_depth, walk(child, next_depth))
            return child_depth

        return max(1, walk(tree, 0))


def _logical_lines(code: str) -> List[str]:
    return [line for line in code.splitlines() if line.strip()]


def _tokens(code: str) -> List[str]:
    import tokenize
    from io import StringIO

    result = []
    try:
        for token in tokenize.generate_tokens(StringIO(code).readline):
            if token.type in {tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP}:
                result.append(token.string)
    except tokenize.TokenError:
        return code.split()
    return result


def _split_identifier(name: str) -> List[str]:
    parts = [part for part in re.split(r"[_\W]+", name) if part]
    tokens: List[str] = []
    for part in parts:
        tokens.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part))
    return tokens or [name]


def _is_pep8_identifier(name: str) -> bool:
    """Conservative PEP 8 check for function/variable-style identifiers."""
    if name in dir(__builtins__) or name.startswith("__") and name.endswith("__"):
        return True
    return bool(re.fullmatch(r"[a-z_][a-z0-9_]*", name)) and "__" not in name


def _is_snake_case(name: str) -> bool:
    return _is_pep8_identifier(name)


def _complexity_score(value: int, good: int, step: float) -> float:
    if value <= good:
        return 1.0
    return _clamp(1.0 - (value - good) * step)


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= max(1, value)
    return result


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace = previous[right_index - 1] + (0 if left_item == right_item else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _jaccard_multiset(left: List[str], right: List[str]) -> float:
    if not left and not right:
        return 1.0
    left_counts = {}
    right_counts = {}
    for item in left:
        left_counts[item] = left_counts.get(item, 0) + 1
    for item in right:
        right_counts[item] = right_counts.get(item, 0) + 1
    keys = set(left_counts) | set(right_counts)
    intersection = sum(min(left_counts.get(key, 0), right_counts.get(key, 0)) for key in keys)
    union = sum(max(left_counts.get(key, 0), right_counts.get(key, 0)) for key in keys)
    return intersection / union if union else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _codebleu_ast_subtrees(tree: ast.AST) -> List[Tuple]:
    signatures: List[Tuple] = []

    def visit(node: ast.AST) -> Tuple:
        structural_children = [child for child in ast.iter_child_nodes(node) if _has_structural_children(child)]
        signature = (type(node).__name__, tuple(visit(child) for child in structural_children))
        signatures.append(signature)
        return signature

    if _has_structural_children(tree):
        visit(tree)
    else:
        signatures.append((type(tree).__name__, ()))
    return signatures


def _has_structural_children(node: ast.AST) -> bool:
    return any(True for _ in ast.iter_child_nodes(node))
