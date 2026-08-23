from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from .models import CodeRecord, MetricScore, RewriteCandidate, SemanticResult


FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
}

FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
}


class SemanticEquivalenceChecker:
    def check(self, record: CodeRecord, candidate: RewriteCandidate) -> SemanticResult:
        scores: List[MetricScore] = []

        original_tree, original_error = self._parse(record.code)
        rewritten_tree, rewritten_error = self._parse(candidate.code)
        ast_parse_score = 1.0 if original_tree is not None and rewritten_tree is not None else 0.0
        parse_explanation = "Both original and rewritten code parse as Python AST."
        if original_error or rewritten_error:
            parse_explanation = f"AST parse failed. original={original_error}; rewritten={rewritten_error}"
        scores.append(MetricScore("ast_parse", ast_parse_score, parse_explanation))

        signature_score = self._signature_score(original_tree, rewritten_tree)
        scores.append(signature_score)

        safety_score = self._safety_score(rewritten_tree)
        scores.append(safety_score)

        compile_score = self._compile_pass_rate(candidate.code)
        scores.append(compile_score)

        ast_edit_distance = self._ast_edit_distance(original_tree, rewritten_tree)
        scores.append(MetricScore("ast_edit_distance", ast_edit_distance, f"Normalized AST edit similarity is {ast_edit_distance:.3f}."))

        tree_edit_distance = self._tree_edit_distance(original_tree, rewritten_tree)
        scores.append(MetricScore("tree_edit_distance", tree_edit_distance, f"Normalized tree edit similarity is {tree_edit_distance:.3f}."))

        codebleu_syntax = self._codebleu_syntax(original_tree, rewritten_tree)
        scores.append(MetricScore("codebleu_syntax", codebleu_syntax, f"AST syntax overlap is {codebleu_syntax:.3f}."))

        codebleu_dataflow = self._codebleu_dataflow(original_tree, rewritten_tree)
        scores.append(MetricScore("codebleu_dataflow", codebleu_dataflow, f"Data-flow overlap is {codebleu_dataflow:.3f}."))

        test_score = self._unit_test_score(record, candidate)
        scores.append(test_score)

        passed = (
            ast_parse_score == 1.0
            and signature_score.score >= 0.99
            and safety_score.score >= 0.99
            and compile_score.score >= 0.99
            and test_score.score >= 0.99
        )
        feedback = self._feedback(scores, passed)
        return SemanticResult(passed=passed, scores=scores, feedback=feedback)

    def _parse(self, code: str) -> Tuple[Optional[ast.AST], Optional[str]]:
        try:
            return ast.parse(code), None
        except SyntaxError as exc:
            return None, str(exc)

    def _signature_score(self, original_tree: Optional[ast.AST], rewritten_tree: Optional[ast.AST]) -> MetricScore:
        if original_tree is None or rewritten_tree is None:
            return MetricScore("signature_consistency", 0.0, "Cannot compare signatures without valid AST.")
        original = self._function_signatures(original_tree)
        rewritten = self._function_signatures(rewritten_tree)
        score = 1.0 if original == rewritten else 0.0
        return MetricScore("signature_consistency", score, f"Original signatures={original}; rewritten signatures={rewritten}.")

    def _function_signatures(self, tree: ast.AST) -> List[str]:
        signatures = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                signatures.append(f"{node.name}({','.join(args)})")
        return sorted(signatures)

    def _safety_score(self, tree: Optional[ast.AST]) -> MetricScore:
        if tree is None:
            return MetricScore("safety", 0.0, "Cannot inspect unsafe calls without valid AST.")
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                issues.extend(alias.name for alias in node.names if alias.name.split(".")[0] in FORBIDDEN_MODULES)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN_MODULES:
                    issues.append(node.module)
            elif isinstance(node, ast.Call):
                name = self._call_name(node.func)
                if name.split(".")[0] in FORBIDDEN_MODULES or name in FORBIDDEN_CALLS:
                    issues.append(name)
        if issues:
            return MetricScore("safety", 0.0, f"Forbidden operations found: {sorted(set(issues))}.")
        return MetricScore("safety", 1.0, "No forbidden calls or imports found.")

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _compile_pass_rate(self, code: str) -> MetricScore:
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            return MetricScore("compile_pass_rate", 0.0, f"Compilation failed: {exc}.")
        return MetricScore("compile_pass_rate", 1.0, "Code compiles successfully.")

    def _ast_edit_distance(self, original_tree: Optional[ast.AST], rewritten_tree: Optional[ast.AST]) -> float:
        if original_tree is None or rewritten_tree is None:
            return 0.0
        original_nodes = [type(node).__name__ for node in ast.walk(original_tree)]
        rewritten_nodes = [type(node).__name__ for node in ast.walk(rewritten_tree)]
        return 1.0 - _normalized_edit_distance(original_nodes, rewritten_nodes)

    def _tree_edit_distance(self, original_tree: Optional[ast.AST], rewritten_tree: Optional[ast.AST]) -> float:
        if original_tree is None or rewritten_tree is None:
            return 0.0
        original_nodes = _tree_tokens(original_tree)
        rewritten_nodes = _tree_tokens(rewritten_tree)
        return 1.0 - _normalized_edit_distance(original_nodes, rewritten_nodes)

    def _codebleu_syntax(self, original_tree: Optional[ast.AST], rewritten_tree: Optional[ast.AST]) -> float:
        if original_tree is None or rewritten_tree is None:
            return 0.0
        original_nodes = [type(node).__name__ for node in ast.walk(original_tree)]
        rewritten_nodes = [type(node).__name__ for node in ast.walk(rewritten_tree)]
        return _jaccard_multiset(original_nodes, rewritten_nodes)

    def _codebleu_dataflow(self, original_tree: Optional[ast.AST], rewritten_tree: Optional[ast.AST]) -> float:
        if original_tree is None or rewritten_tree is None:
            return 0.0
        original_edges = _dataflow_edges(original_tree)
        rewritten_edges = _dataflow_edges(rewritten_tree)
        return _jaccard_multiset(original_edges, rewritten_edges)

    def _unit_test_score(self, record: CodeRecord, candidate: RewriteCandidate) -> MetricScore:
        if not record.tests:
            return MetricScore("unit_test_pass_rate", 1.0, "No tests supplied; unit-test gate skipped.")
        original_result = self._run_tests(record.code, record.tests)
        rewritten_result = self._run_tests(candidate.code, record.tests)
        if original_result[0] and rewritten_result[0]:
            return MetricScore("unit_test_pass_rate", 1.0, "Original and rewritten code pass supplied tests.")
        return MetricScore(
            "unit_test_pass_rate",
            0.0,
            "Unit tests failed. "
            f"original_ok={original_result[0]}, rewritten_ok={rewritten_result[0]}, "
            f"rewritten_output={rewritten_result[1][-500:]}",
        )

    def _run_tests(self, code: str, tests: List[str]) -> Tuple[bool, str]:
        digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]
        script = code.rstrip() + "\n\n\ndef __run_tests__():"
        if not tests:
            script += "\n    return True\n\n__run_tests__()\n"
        else:
            for test in tests:
                script += "\n" + textwrap.indent(test.rstrip(), "    ")
            script += "\n    return True\n\n__run_tests__()\n"
        with tempfile.TemporaryDirectory(prefix=f"code_rewrite_{digest}_") as tmpdir:
            path = Path(tmpdir) / "candidate_test.py"
            path.write_text(script, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(path)],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    cwd=tmpdir,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, "Timed out."
        output = (completed.stdout or "") + (completed.stderr or "")
        return completed.returncode == 0, output

    def _feedback(self, scores: List[MetricScore], passed: bool) -> str:
        if passed:
            return "Semantic equivalence passed. Keep preserving behavior in later rewrites."
        weak = [
            score
            for score in scores
            if score.score < 0.99 and score.name in {"ast_parse", "signature_consistency", "safety", "compile_pass_rate", "unit_test_pass_rate"}
        ]
        if not weak:
            weak = [score for score in scores if score.score < 0.5]
        details = "; ".join(f"{score.name}={score.score:.2f}: {score.explanation}" for score in weak)
        return f"Semantic equivalence failed or is risky. Fix these issues: {details}"


def _tree_tokens(node: ast.AST) -> List[str]:
    tokens = [f"({type(node).__name__}"]
    for child in ast.iter_child_nodes(node):
        tokens.extend(_tree_tokens(child))
    tokens.append(")")
    return tokens


def _dataflow_edges(tree: ast.AST) -> List[str]:
    edges: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)]
            uses = [item.id for item in ast.walk(node.value) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)]
            for target in targets:
                for used in uses:
                    edges.append(f"{used}->{target}")
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            uses = [item.id for item in ast.walk(node.value) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)]
            for used in uses + [node.target.id]:
                edges.append(f"{used}->{node.target.id}")
        elif isinstance(node, ast.Return):
            uses = [item.id for item in ast.walk(node.value) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)] if node.value else []
            for used in uses:
                edges.append(f"{used}->return")
    return edges or ["no_explicit_dataflow"]


def _normalized_edit_distance(left: List[str], right: List[str]) -> float:
    if not left and not right:
        return 0.0
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
    distance = previous[-1]
    return distance / max(1, max(len(left), len(right)))


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
