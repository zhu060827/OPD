"""用公开定义和固定第三方实现评估四个 code-only Teacher。

本模块不计算项目自定义综合分。每个领域报告一个主指标及注册表中列出的辅助指标。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from .domains import DOMAINS, require_domain
from .io_utils import read_jsonl


CODE_BLOCK = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)


def extract_code(text: str) -> str:
    match = CODE_BLOCK.search(str(text or ""))
    return (match.group(1) if match else str(text or "")).strip()


def normalized_code(code: str) -> str:
    try:
        return ast.unparse(ast.parse(code)).strip()
    except SyntaxError:
        return code.strip()


def subtokens(name: str) -> list[str]:
    pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().split("_")
    return [piece for piece in pieces if piece]


def subtoken_f1(prediction: str, target: str) -> float:
    predicted, gold = Counter(subtokens(prediction)), Counter(subtokens(target))
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((predicted & gold).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def identifiers(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }


def recovered_identifier(source: str, prediction: str, old_name: str) -> str:
    """按 DOBF 的同结构恢复任务，在对应 AST 标识符位置读取预测名称。"""
    try:
        source_tree, prediction_tree = ast.parse(source), ast.parse(prediction)
    except SyntaxError:
        return ""
    source_nodes = [
        node for node in ast.walk(source_tree) if isinstance(node, (ast.Name, ast.arg))
    ]
    prediction_nodes = [
        node for node in ast.walk(prediction_tree) if isinstance(node, (ast.Name, ast.arg))
    ]
    if len(source_nodes) != len(prediction_nodes):
        return ""
    recovered = {
        (right.id if isinstance(right, ast.Name) else right.arg)
        for left, right in zip(source_nodes, prediction_nodes)
        if (left.id if isinstance(left, ast.Name) else left.arg) == old_name
    }
    return next(iter(recovered)) if len(recovered) == 1 else ""


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def run_tests(code: str, tests: list[str], timeout: int) -> bool:
    if not tests:
        raise ValueError("Pass@1 要求每条 control_flow 测试样本提供留出测试")
    script = code.rstrip() + "\n\n"
    for test in tests:
        script += textwrap.dedent(str(test)).rstrip() + "\n"
    with tempfile.TemporaryDirectory(prefix="teacher_eval_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)], cwd=directory, capture_output=True,
                text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def function_signatures(code: str) -> list[str]:
    tree = ast.parse(code)
    signatures = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
            keyword_only = [argument.arg for argument in node.args.kwonlyargs]
            signatures.append(json.dumps({
                "name": node.name,
                "positional": positional,
                "keyword_only": keyword_only,
                "vararg": node.args.vararg.arg if node.args.vararg else None,
                "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
                "defaults": [ast.dump(value, include_attributes=False) for value in node.args.defaults],
                "kw_defaults": [
                    ast.dump(value, include_attributes=False) if value is not None else None
                    for value in node.args.kw_defaults
                ],
            }, sort_keys=True))
    return sorted(signatures)


def pycodestyle_count(code: str) -> int:
    try:
        import pycodestyle
    except ImportError as exc:
        raise RuntimeError("Formatting 正式评估需要固定版本 pycodestyle==2.12.1") from exc
    with tempfile.TemporaryDirectory(prefix="style_eval_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        guide = pycodestyle.StyleGuide(quiet=True)
        report = guide.check_files([str(path)])
        return int(report.total_errors)


def codebleu_score(prediction: str, target: str, language: str) -> float:
    try:
        from codebleu import calc_codebleu
    except ImportError as exc:
        raise RuntimeError("CodeBLEU 正式评估需要固定版本 codebleu==0.7.0") from exc
    result = calc_codebleu([target], [prediction], lang=language)
    return float(result["codebleu"])


def bleu_score(prediction: str, target: str) -> float:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("BLEU 正式评估需要固定版本 sacrebleu==2.5.1") from exc
    return float(sacrebleu.corpus_bleu([prediction], [[target]], tokenize="none").score)


def cyclomatic_complexity(code: str) -> float:
    try:
        from radon.complexity import cc_visit
    except ImportError as exc:
        raise RuntimeError("控制流正式评估需要固定版本 radon==6.0.1") from exc
    blocks = cc_visit(code)
    return float(sum(block.complexity for block in blocks))


def evaluate_row(reference: dict[str, Any], prediction: dict[str, Any], timeout: int) -> dict[str, Any]:
    domain = require_domain(str(reference["domain"]))
    predicted = extract_code(prediction.get("lora_output", prediction.get("prediction", "")))
    target = str(reference["target_code"]).strip()
    source = str(reference["source_code"]).strip()
    language = str(reference.get("language", "python")).lower()
    if language != "python":
        raise ValueError(
            f"evaluate_domains 当前固定实现只支持 Python；{reference['source_id']} 的语言为 {language!r}。"
            "其他语言必须使用其论文对应工具链并单独注册实现。"
        )
    result: dict[str, Any] = {
        "source_id": str(reference["source_id"]),
        "domain": domain,
        "parse_pass": False,
        "signature_pass": False,
        "compile_pass": False,
        "test_pass": False,
    }
    try:
        ast.parse(predicted)
        result["parse_pass"] = True
        result["signature_pass"] = function_signatures(predicted) == function_signatures(source)
        compile(predicted, "<teacher_prediction>", "exec")
        result["compile_pass"] = True
    except SyntaxError:
        pass
    tests = list(reference.get("tests") or [])
    if not tests:
        raise ValueError(f"{reference['source_id']} 缺少留出测试，不能计算正式 test pass/Pass@1")
    result["test_pass"] = result["compile_pass"] and run_tests(predicted, tests, timeout)
    if domain == "formatting":
        result["primary_exact_formatting_repair"] = float(
            result["parse_pass"] and predicted.rstrip() == target.rstrip()
        )
        result["pycodestyle_violation_count"] = pycodestyle_count(predicted)
        result["levenshtein_distance_to_target"] = levenshtein_distance(predicted, target)
    elif domain == "identifier":
        old_name = str(reference["old_identifier"])
        target_name = str(reference["target_identifier"])
        predicted_name = recovered_identifier(source, predicted, old_name)
        result["recovered_identifier"] = predicted_name
        result["primary_identifier_exact_match"] = float(predicted_name == target_name)
        result["identifier_subtoken_f1"] = subtoken_f1(predicted_name, target_name)
        result["full_code_exact_match"] = float(normalized_code(predicted) == normalized_code(target))
    elif domain == "local_structure":
        result["primary_codebleu"] = codebleu_score(predicted, target, language)
        result["bleu"] = bleu_score(predicted, target)
        result["exact_match"] = float(normalized_code(predicted) == normalized_code(target))
    elif domain == "control_flow":
        result["primary_pass_at_1"] = float(
            result["test_pass"]
        )
        result["cyclomatic_complexity"] = cyclomatic_complexity(predicted) if result["parse_pass"] else None
        result["source_cyclomatic_complexity"] = cyclomatic_complexity(source)
        result["codebleu"] = codebleu_score(predicted, target, language)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    summary: dict[str, Any] = {}
    for domain, items in sorted(by_domain.items()):
        keys = sorted(set().union(*(item.keys() for item in items)) - {"source_id", "domain"})
        metrics = {}
        for key in keys:
            values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
            if values:
                metrics[key] = {"mean": fmean(values), "n": len(values)}
        summary[domain] = {"samples": len(items), "metrics": metrics}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="按公开指标评估四个 code-only Teacher。")
    parser.add_argument("--predictions", required=True, help="含 source_id 与 lora_output/prediction 的 JSONL")
    parser.add_argument("--references", required=True, help="prepare_data 生成的 test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    predictions = {str(row["source_id"]): row for row in read_jsonl(args.predictions)}
    references = list(read_jsonl(args.references))
    missing = [str(row["source_id"]) for row in references if str(row["source_id"]) not in predictions]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 条预测；示例：{missing[:5]}")
    rows = [evaluate_row(row, predictions[str(row["source_id"])], args.timeout) for row in references]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"protocol": "teacher-metrics-v4", "domains": summarize(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"完成": True, "样本数": len(rows), "输出目录": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
