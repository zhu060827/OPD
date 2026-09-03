from __future__ import annotations

"""使用完整 Math 包对真实数学数据做批量评测。

最终文件只写入 data test/results，报告使用 pipeline 自带的五项评分，
同时记录真实 API 调用次数和 mock 回退次数。
"""

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
DEFAULT_INPUT = BASE_DIR / "datasets" / "math_real_16.jsonl"
METRIC_NAMES = [
    "similarity",
    "logic_consistency",
    "formula_correctness",
    "completeness",
    "reasoning_gain",
]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import math_adapter  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from math_reasoning_expander.visualization import write_quality_report  # noqa: E402


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {line_number} 行不是合法 JSON：{exc}") from exc
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def average(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def structure_issue_count(text: str) -> int:
    """统计不成对的 LaTeX 展示分隔符，只做格式审计，不替代公式验证。"""
    text = text or ""
    return (
        abs(text.count("\\[") - text.count("\\]"))
        + abs(text.count("\\(") - text.count("\\)"))
        + text.count("$$") % 2
    )


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [record for record in records if not record.get("test_meta", {}).get("failed")]
    summary: Dict[str, Any] = {
        "sample_count": len(records),
        "completed_count": len(valid),
        "error_count": len(records) - len(valid),
        "accepted_count": sum(1 for record in valid if record.get("accepted")),
        "accepted_rate": (
            sum(1 for record in valid if record.get("accepted")) / len(valid)
            if valid
            else 0.0
        ),
        "avg_aggregate_score": average(
            safe_float(record.get("verification", {}).get("aggregate_score")) for record in valid
        ),
        "avg_added_node_count": average(
            safe_float(record.get("metrics", {}).get("added_node_count")) for record in valid
        ),
        "avg_attempt_count": average(
            safe_float(record.get("metrics", {}).get("attempt_count")) for record in valid
        ),
        "avg_runtime_seconds": average(
            safe_float(record.get("test_meta", {}).get("runtime_seconds")) for record in valid
        ),
        "real_api_calls": sum(
            int(record.get("test_meta", {}).get("real_api_calls", 0)) for record in records
        ),
        "mock_fallback_calls": sum(
            int(record.get("test_meta", {}).get("mock_fallback_calls", 0)) for record in records
        ),
        "structure_regression_count": sum(
            1
            for record in valid
            if structure_issue_count(str(record.get("generated", {}).get("expanded_cot", "")))
            > structure_issue_count(str(record.get("generated", {}).get("original_cot", "")))
        ),
    }
    for metric in METRIC_NAMES:
        summary[f"avg_{metric}"] = average(
            safe_float(record.get("verification", {}).get(metric)) for record in valid
        )
    return summary


def dataset_breakdown(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        dataset = str(record.get("input_summary", {}).get("dataset", "unknown"))
        groups.setdefault(dataset, []).append(record)
    return [{"dataset": name, **summarize(items)} for name, items in sorted(groups.items())]


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "sample_count",
        "completed_count",
        "error_count",
        "accepted_count",
        "accepted_rate",
        "avg_aggregate_score",
        "avg_similarity",
        "avg_logic_consistency",
        "avg_formula_correctness",
        "avg_completeness",
        "avg_reasoning_gain",
        "avg_added_node_count",
        "avg_attempt_count",
        "avg_runtime_seconds",
        "real_api_calls",
        "mock_fallback_calls",
        "structure_regression_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def average_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not evaluations:
        return {"accepted": False, "aggregate_score": 0.0, "feedback": "", "scores": []}
    scores = []
    for metric in METRIC_NAMES:
        values = [
            safe_float(score.get("score"))
            for evaluation in evaluations
            for score in evaluation.get("scores", [])
            if score.get("name") == metric
        ]
        scores.append(
            {
                "name": metric,
                "score": average(values),
                "explanation": "Average over retained expansion rounds.",
            }
        )
    return {
        "accepted": any(bool(item.get("accepted")) for item in evaluations),
        "aggregate_score": average(
            safe_float(item.get("aggregate_score")) for item in evaluations
        ),
        "feedback": "Average over retained expansion rounds.",
        "scores": scores,
    }


def flat_evaluation(record: Dict[str, Any]) -> Dict[str, Any]:
    verification = record.get("verification", {})
    return {
        "accepted": bool(record.get("accepted")),
        "aggregate_score": safe_float(verification.get("aggregate_score")),
        "feedback": str(record.get("feedback", "")),
        "scores": [
            {
                "name": metric,
                "score": safe_float(verification.get(metric)),
                "explanation": metric,
            }
            for metric in METRIC_NAMES
        ],
    }


def visualization_record(record: Dict[str, Any]) -> Dict[str, Any]:
    retained = record.get("generated", {}).get("retained_expansions", [])
    if retained:
        original_evaluation = average_evaluations(
            [item.get("original_evaluation", {}) for item in retained]
        )
        evaluation = average_evaluations(
            [item.get("evaluation", {}) for item in retained]
        )
    else:
        original_evaluation = flat_evaluation(record)
        evaluation = dict(original_evaluation)
    return {
        "accepted": bool(record.get("accepted")),
        "expansion_stats": record.get("metrics", {}),
        "original_evaluation": original_evaluation,
        "evaluation": evaluation,
    }


def write_report(
    path: Path,
    model: str,
    summary: Dict[str, Any],
    breakdown: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> None:
    api_mode = (
        "全部为真实 API 调用"
        if summary.get("mock_fallback_calls", 0) == 0
        else "存在 mock 回退，不能全部视为真实模型成绩"
    )
    lines = [
        "# Math 真实数据测试报告",
        "",
        "本次测试使用主工程中的 `Math/math_reasoning_expander` 完整双层循环，",
        "不是旧版单轮 fallback，也不再使用旧版 data test 外部替代评分。",
        "",
        "## 测试配置",
        "",
        f"- 模型：`{model}`",
        f"- 样本数：{summary.get('sample_count', 0)}",
        "- 外层：最多 10 个遮盖点",
        "- 内层：每个遮盖点最多反馈优化 3 次",
        "- 保留规则：生成节点相对原节点 `gain > 0`",
        f"- API 状态：{api_mode}",
        "",
        "## 总体结果",
        "",
        f"- 完成数：{summary.get('completed_count', 0)}",
        f"- 错误数：{summary.get('error_count', 0)}",
        f"- 接受数：{summary.get('accepted_count', 0)}",
        f"- 接受率：{summary.get('accepted_rate', 0):.3f}",
        f"- 平均综合分：{summary.get('avg_aggregate_score', 0):.3f}",
        f"- 平均推理增益：{summary.get('avg_reasoning_gain', 0):.3f}",
        f"- 平均新增节点数：{summary.get('avg_added_node_count', 0):.3f}",
        f"- 平均尝试次数：{summary.get('avg_attempt_count', 0):.3f}",
        f"- 平均单题耗时：{summary.get('avg_runtime_seconds', 0):.3f} 秒",
        f"- 真实 API 调用：{summary.get('real_api_calls', 0)}",
        f"- mock 回退调用：{summary.get('mock_fallback_calls', 0)}",
        f"- LaTeX 结构退化样本：{summary.get('structure_regression_count', 0)}",
        "",
        "## 五项平均分",
        "",
        "| 指标 | 平均分 |",
        "|---|---:|",
        f"| 去重复性 | {summary.get('avg_similarity', 0):.3f} |",
        f"| 逻辑一致性 | {summary.get('avg_logic_consistency', 0):.3f} |",
        f"| 公式正确性 | {summary.get('avg_formula_correctness', 0):.3f} |",
        f"| 完整性 | {summary.get('avg_completeness', 0):.3f} |",
        f"| 推理增益 | {summary.get('avg_reasoning_gain', 0):.3f} |",
        "",
        "## 分数据集结果",
        "",
        "| 数据集 | 样本 | 接受率 | 综合分 | 推理增益 | 真实调用 | mock 回退 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in breakdown:
        lines.append(
        f"| {row.get('dataset')} | {row.get('sample_count')} | "
            f"{row.get('accepted_rate', 0):.3f} | {row.get('avg_aggregate_score', 0):.3f} | "
            f"{row.get('avg_reasoning_gain', 0):.3f} | {row.get('real_api_calls', 0)} | "
            f"{row.get('mock_fallback_calls', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 结果解读与限制",
            "",
            "- `accepted=true` 表示候选节点相对原遮盖节点产生正质量增益，不等于完成了形式化数学证明。",
            "- 公式正确性是本次五项指标中相对较弱的一项，复杂符号推导仍需要更严格的 SymPy/规则验证。",
            "- 个别原始 CoT 含有噪声或不规范 LaTeX；报告单独统计了扩充后结构变差的样本，没有把它们隐藏。",
            "- 本次结果适合作为工程闭环和阶段性质量对比，不应表述为 100% 数学正确率。",
            "",
            "## 样例",
            "",
            "这里只截取前 3 条，完整过程见 `results/math_real_records.jsonl`。",
            "",
        ]
    )
    for index, record in enumerate(records[:3], start=1):
        generated = record.get("generated", {})
        verification = record.get("verification", {})
        question = str(record.get("input_summary", {}).get("question", "")).replace("\n", " ")
        masked = str(generated.get("masked_node", "")).replace("\n", " ")
        recovered = str(generated.get("recovered_node", "")).replace("\n", " ")
        lines.extend(
            [
                f"### 样例 {index}",
                "",
                f"- 题目：{question[:260]}",
                f"- 是否保留：{record.get('accepted')}",
                f"- 综合分：{safe_float(verification.get('aggregate_score')):.3f}",
                f"- 被遮盖节点：{masked[:300]}",
                f"- 恢复节点：{recovered[:300]}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def failed_record(row: Dict[str, Any], exc: Exception, runtime: float) -> Dict[str, Any]:
    return {
        "id": str(row.get("id", "")),
        "modality": "math",
        "input_summary": {
            "dataset": row.get("dataset", "unknown"),
            "source": row.get("source", ""),
        },
        "generated": {},
        "verification": {},
        "feedback": "",
        "accepted": False,
        "round": 0,
        "errors": [f"{type(exc).__name__}: {exc}"],
        "metrics": {},
        "saved_path": "",
        "test_meta": {
            "failed": True,
            "runtime_seconds": round(runtime, 4),
            "real_api_calls": 0,
            "mock_fallback_calls": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="不调用 API，只根据已有 JSONL 重建汇总和报告。")
    args = parser.parse_args()

    if not math_adapter.MATH_MODULE_AVAILABLE:
        raise RuntimeError(f"完整 Math 包不可用：{math_adapter.IMPORT_ERROR}")
    if not args.input.exists():
        raise FileNotFoundError(f"找不到测试数据：{args.input}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = RESULT_DIR / "math_real_records.jsonl"
    summary_path = RESULT_DIR / "math_real_summary.csv"
    svg_path = RESULT_DIR / "math_quality.svg"
    report_path = BASE_DIR / "MATH_REAL_TEST_REPORT.md"

    if args.report_only:
        if not records_path.exists():
            raise FileNotFoundError(f"找不到已有结果：{records_path}")
        result_rows = read_jsonl(records_path, 0)
        model = str(result_rows[0].get("test_meta", {}).get("model", "unknown")) if result_rows else "unknown"
        summary = summarize(result_rows)
        breakdown = dataset_breakdown(result_rows)
        write_summary_csv(summary_path, [{"dataset": "ALL", **summary}] + breakdown)
        write_quality_report(
            str(svg_path),
            [visualization_record(record) for record in result_rows if not record.get("test_meta", {}).get("failed")],
        )
        write_report(report_path, model, summary, breakdown, result_rows)
        print("已根据现有结果重建汇总、质量图和报告。")
        return

    rows = read_jsonl(args.input, args.limit)
    if not rows:
        raise ValueError("测试数据为空。")

    client = LLMClient()
    if args.mock:
        client.api_key = ""
    elif not client.available:
        raise RuntimeError("未读取到真实 API 配置，请设置 OPENAI_API_KEY、OPENAI_BASE_URL 和 OPENAI_MODEL。")

    result_rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="math_pipeline_", dir=BASE_DIR) as temp_dir:
        math_adapter.OUTPUT_DIR = Path(temp_dir)
        for index, row in enumerate(rows, start=1):
            dataset = str(row.get("dataset", "unknown"))
            print(f"[{index}/{len(rows)}] {dataset} 开始测试", flush=True)
            before_real = client.real_call_count
            before_mock = client.mock_fallback_count
            started = time.perf_counter()
            try:
                result = math_adapter.run_math_pipeline(record=row, llm_client=client)
                data = result.to_dict()
                runtime = time.perf_counter() - started
                real_calls = client.real_call_count - before_real
                mock_calls = client.mock_fallback_count - before_mock
                data.setdefault("input_summary", {})["dataset"] = dataset
                data["input_summary"]["source"] = row.get("source", "")
                data["saved_path"] = str(records_path)
                data["test_meta"] = {
                    "failed": False,
                    "model": client.model,
                    "runtime_seconds": round(runtime, 4),
                    "real_api_calls": real_calls,
                    "mock_fallback_calls": mock_calls,
                    "math_engine": data["input_summary"].get("math_engine"),
                }
                result_rows.append(data)
                print(
                    f"[{index}/{len(rows)}] 完成 accepted={data.get('accepted')} "
                    f"score={safe_float(data.get('verification', {}).get('aggregate_score')):.3f} "
                    f"real_calls={real_calls} mock_calls={mock_calls} time={runtime:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                runtime = time.perf_counter() - started
                result_rows.append(failed_record(row, exc, runtime))
                print(f"[{index}/{len(rows)}] 失败 {type(exc).__name__}: {exc}", flush=True)
            time.sleep(max(0.0, args.sleep))

    write_jsonl(records_path, result_rows)
    summary = summarize(result_rows)
    breakdown = dataset_breakdown(result_rows)
    write_summary_csv(summary_path, [{"dataset": "ALL", **summary}] + breakdown)
    write_quality_report(
        str(svg_path),
        [visualization_record(record) for record in result_rows if not record.get("test_meta", {}).get("failed")],
    )
    write_report(report_path, client.model, summary, breakdown, result_rows)

    print("全部测试完成。", flush=True)
    print(f"结果：{records_path}", flush=True)
    print(f"汇总：{summary_path}", flush=True)
    print(f"报告：{report_path}", flush=True)


if __name__ == "__main__":
    main()
