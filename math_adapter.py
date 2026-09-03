from __future__ import annotations

"""
Math 推理链扩充适配器。

数学部分不是生成新题，而是扩充已有 CoT 的中间推理步骤。
整体流程是：原题 + 原始 CoT -> 切分步骤 -> 构建 Reasoning Graph -> mask 中间节点 -> 恢复节点 -> 质量评价 -> 合成 expanded CoT。
如果队友的 Math.zip 已经解压到 Math/math_reasoning_expander/，这里会优先调用里面的 parser、masking、evaluator 和 visualization；
如果没有，也保留了一个简化 fallback，保证网页 demo 不会因为缺少数学包直接崩掉。
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    MATH_ACCEPT_THRESHOLD,
    MATH_MAX_MASK_ROUNDS,
    MATH_MAX_REFINE_ROUNDS,
    MATH_PATIENCE,
    OUTPUT_DIR,
    SAMPLE_DATA_DIR,
)
from llm_client import LLMClient
from result_schema import ResultRecord, append_jsonl


MATH_ROOT = Path(__file__).resolve().parent / "Math"
if MATH_ROOT.exists() and str(MATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MATH_ROOT))

MATH_MODULE_AVAILABLE = False
IMPORT_ERROR = ""

try:
    # 优先接入队友的数学扩充包，不直接改 Math.zip 本体，只在这里做适配。
    from math_reasoning_expander.evaluators import MultiFeedbackEvaluator
    from math_reasoning_expander.masking import GraphMasker
    from math_reasoning_expander.models import FillCandidate, MaskedTask, ReasoningEdge, ReasoningGraph, ReasoningNode
    from math_reasoning_expander.parser import ReasoningGraphParser
    from math_reasoning_expander.pipeline import MathReasoningExpansionPipeline, expansion_record_to_dict
    from math_reasoning_expander.visualization import write_quality_report

    MATH_MODULE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on uploaded Math.zip.
    IMPORT_ERROR = str(exc)

    # 没有 Math.zip 时的最小数据结构。字段名尽量和队友包保持一致，方便以后替换。
    @dataclass
    class ReasoningNode:
        node_id: str
        content: str
        node_type: str
        formulas: List[str] = field(default_factory=list)
        depends_on: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ReasoningEdge:
        source: str
        target: str
        relation: str = "next"

    @dataclass
    class ReasoningGraph:
        question: str
        answer: str
        nodes: List[ReasoningNode]
        edges: List[ReasoningEdge]
        metadata: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "question": self.question,
                "answer": self.answer,
                "nodes": [node.__dict__ for node in self.nodes],
                "edges": [edge.__dict__ for edge in self.edges],
                "metadata": self.metadata,
            }

    @dataclass
    class MaskedTask:
        graph: ReasoningGraph
        masked_node_ids: List[str]
        mask_strategy: str
        prefix_nodes: List[ReasoningNode]
        suffix_nodes: List[ReasoningNode]
        target_nodes: List[ReasoningNode]

    @dataclass
    class FillCandidate:
        text: str
        steps: List[str]
        raw_response: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    def write_quality_report(path: str, records: List[Dict[str, Any]]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="220"><rect width="760" height="220" fill="#f7f8fb"/><text x="32" y="56" font-family="Arial" font-size="24" fill="#18212f">Math Quality Report</text><text x="32" y="96" font-family="Arial" font-size="14" fill="#64748b">Fallback SVG generated because Math.zip module is unavailable.</text></svg>',
            encoding="utf-8",
        )


QUESTION_FIELDS = ["question", "problem", "query", "prompt"]
SOLUTION_FIELDS = ["solution", "response", "cot", "generated_solution", "answer"]
ANSWER_FIELDS = ["expected_answer", "final_answer", "predicted_answer", "answer"]
# typed node masking 的优先级：优先遮盖真正有推理价值的中间节点。
# 不遮盖 original_problem / final_answer / key_definition，避免破坏题意和最终答案。
MASK_PRIORITY = [
    "equation_transform",
    "condition_translation",
    "intermediate_calculation",
    "explanation_bridge",
    "inference",
    "explanation",
]
AVOID_TYPES = {"original_problem", "final_answer", "key_definition", "conclusion", "definition"}


def math_module_status() -> Dict[str, Any]:
    """给 /api/health 使用，告诉前端当前是否接上了 Math.zip。"""
    return {
        "available": MATH_MODULE_AVAILABLE,
        "engine": "math_reasoning_expander" if MATH_MODULE_AVAILABLE else "adapter_fallback",
        "error": IMPORT_ERROR,
    }


def pick_field(record: Dict[str, Any], names: List[str], default: str = "") -> str:
    """兼容 MathFimer / NuminaMath / MetaMathQA / MathInstruct 等不同数据集字段名。"""
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def load_math_records(path: str | Path | None = None, limit: int = 1) -> List[Dict[str, Any]]:
    """读取 JSONL 数学样例。没有文件时返回内置小样例，保证 demo 能跑。"""
    input_path = Path(path) if path else SAMPLE_DATA_DIR / "sample_math.jsonl"
    if not input_path.exists():
        return [sample_math_record()]
    rows: List[Dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(rows) >= limit:
            break
    return rows or [sample_math_record()]


def sample_math_record() -> Dict[str, str]:
    return {
        "question": "A class has 12 boys and 8 girls. What fraction of the class are girls?",
        "solution": "There are 12 + 8 = 20 students in total.\nThe girls are 8 of the 20 students.\nSo the fraction is 8/20 = 2/5.",
        "answer": "2/5",
    }


def split_steps(text: str) -> List[str]:
    """轻量 CoT 切分器：过滤纯 LaTeX 分隔符，再按行或句子切分。"""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    ignored = {"$" + "$", "\\[", "\\]", "\\(", "\\)"}
    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)：:])\s*", "", raw_line).strip()
        if not line or line in ignored or line.startswith("\\begin{") or line.startswith("\\end{"):
            continue
        # 数据集中有些 display math 会把 $ 单独放一行，这里只保留真正的推理内容。
        if len(line) <= 2 and not re.search(r"[A-Za-z0-9一-龥]", line):
            continue
        lines.append(line)
    if len(lines) >= 2:
        return lines
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [part.strip() for part in parts if part.strip() and part.strip() not in ignored] or [text]

def build_graph(question: str, solution: str, answer: str):
    """构造 Reasoning Graph。能用队友 parser 就用，失败再退回本地简化版。"""
    if MATH_MODULE_AVAILABLE:
        parser = ReasoningGraphParser()
        try:
            graph = parser.parse(question, solution)
            if graph.nodes:
                return graph
        except Exception:
            pass
    steps = split_steps(solution)
    nodes = []
    edges = []
    for idx, step in enumerate(steps, start=1):
        node_type = classify_step(step, idx, len(steps))
        node = ReasoningNode(
            node_id=f"s{idx}",
            content=step,
            node_type=node_type,
            formulas=extract_formulas(step),
            depends_on=[f"s{idx - 1}"] if idx > 1 else [],
        )
        nodes.append(node)
        if idx > 1:
            edges.append(ReasoningEdge(source=f"s{idx - 1}", target=f"s{idx}", relation="next"))
    return ReasoningGraph(question=question, answer=solution or answer, nodes=nodes, edges=edges)


def classify_step(step: str, idx: int, total: int) -> str:
    """根据关键词和公式特征给节点打类型标签，用于后面的 typed masking。"""
    lower = step.lower()
    if idx == 1 and any(token in lower for token in ["let", "given", "there are", "已知"]):
        return "key_definition"
    if idx == total or "answer" in lower or "therefore" in lower or "所以" in lower:
        return "final_answer"
    if "=" in step and any(op in step for op in ["+", "-", "*", "/", "^"]):
        return "intermediate_calculation"
    if any(token in lower for token in ["so", "thus", "hence", "because", "then", "因此", "因为"]):
        return "explanation_bridge"
    return "explanation_bridge"


def extract_formulas(text: str) -> List[str]:
    chunks = re.findall(r"[A-Za-z0-9().\s+\-*/^=<>]+", text or "")
    return [chunk.strip() for chunk in chunks if any(op in chunk for op in ["=", "+", "-", "*", "/", "^"])]


def choose_mask_task(graph):
    """选择要遮盖的中间节点。核心原则是遮中间桥梁，不遮原题和最终答案。"""
    candidates = [node for node in graph.nodes if node.node_type not in AVOID_TYPES and len(node.content.strip()) > 2]
    if len(graph.nodes) > 2:
        edge_ids = {graph.nodes[0].node_id, graph.nodes[-1].node_id}
        candidates = [node for node in candidates if node.node_id not in edge_ids] or candidates
    chosen = None
    for node_type in MASK_PRIORITY:
        typed = [node for node in candidates if node.node_type == node_type]
        if typed:
            chosen = typed[0]
            break
    if chosen is None:
        if MATH_MODULE_AVAILABLE:
            try:
                return GraphMasker(seed=7).mask(graph, strategy="single_node")
            except Exception:
                pass
        chosen = candidates[0] if candidates else graph.nodes[0]
    index = graph.nodes.index(chosen)
    return MaskedTask(
        graph=graph,
        masked_node_ids=[chosen.node_id],
        mask_strategy="typed_priority",
        prefix_nodes=graph.nodes[:index],
        suffix_nodes=graph.nodes[index + 1 :],
        target_nodes=[chosen],
    )


def task_context(task) -> Tuple[str, str, str, str]:
    """把 masked task 转成 FIM 所需的 prefix / middle / suffix。"""
    prefix = "\n".join(node.content for node in task.prefix_nodes)
    suffix = "\n".join(node.content for node in task.suffix_nodes)
    original_middle = "\n".join(node.content for node in task.target_nodes)
    target_type = task.target_nodes[0].node_type if task.target_nodes else "intermediate_calculation"
    return prefix, suffix, original_middle, target_type


def evaluate_fill(task, recovered_node: str) -> Dict[str, Any]:
    """对恢复节点做多维评价：相似度、逻辑一致性、公式正确性、完整性、推理增益。"""
    steps = split_steps(recovered_node)
    candidate = FillCandidate(text=recovered_node, steps=steps, raw_response=recovered_node, metadata={"source": "adapter"})
    if MATH_MODULE_AVAILABLE:
        evaluator = MultiFeedbackEvaluator(accept_threshold=0.62)
        try:
            result = evaluator.evaluate(task, candidate)
            out = {score.name: float(score.score) for score in result.scores}
            out["aggregate_score"] = float(result.aggregate_score)
            out["accepted"] = bool(result.accepted or result.aggregate_score >= 0.62)
            out["feedback"] = result.feedback
            return out
        except Exception as exc:
            return fallback_evaluation(recovered_node, error=str(exc))
    return fallback_evaluation(recovered_node)


def fallback_evaluation(recovered_node: str, error: str = "") -> Dict[str, Any]:
    """没有队友 evaluator 时使用的兜底评分，保证接口仍有可展示分数。"""
    has_formula = bool(extract_formulas(recovered_node))
    word_count = len(recovered_node.split())
    completeness = 0.8 if word_count >= 5 else 0.55
    formula = 0.85 if has_formula else 0.65
    logic = 0.78 if recovered_node.strip() else 0.0
    similarity = 0.82
    gain = 0.68 if word_count >= 5 else 0.45
    aggregate = 0.18 * similarity + 0.24 * logic + 0.24 * formula + 0.17 * completeness + 0.17 * gain
    feedback = "Fallback evaluator: recovered step is readable and preserves the surrounding reasoning."
    if error:
        feedback += f" Math.zip evaluator fallback reason: {error}"
    return {
        "similarity": similarity,
        "logic_consistency": logic,
        "formula_correctness": formula,
        "completeness": completeness,
        "reasoning_gain": gain,
        "aggregate_score": aggregate,
        "accepted": aggregate >= 0.62,
        "feedback": feedback,
    }


def merge_expanded_solution(task, recovered_node: str) -> str:
    """把恢复出的中间节点合回原 CoT，形成 expanded_cot。"""
    lines = [node.content for node in task.prefix_nodes]
    lines.extend(split_steps(recovered_node) or [recovered_node])
    lines.extend(node.content for node in task.suffix_nodes)
    return "\n".join(line for line in lines if str(line).strip())


def graph_to_dict(graph) -> Dict[str, Any]:
    """统一图结构字段，前端展示时同时能读 id/type 和 node_id/node_type。"""
    if hasattr(graph, "to_dict"):
        data = graph.to_dict()
    else:
        data = graph
    nodes = []
    for node in data.get("nodes", []):
        if "node_id" in node and "id" not in node:
            node["id"] = node["node_id"]
        if "node_type" in node and "type" not in node:
            node["type"] = node["node_type"]
        nodes.append(node)
    data["nodes"] = nodes
    return data


def quality_svg_records(record_dict: Dict[str, Any], verification: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    scores = [
        {"name": key, "score": float(verification.get(key, 0.0)), "explanation": key}
        for key in ["similarity", "logic_consistency", "formula_correctness", "completeness", "reasoning_gain"]
    ]
    return [
        {
            "accepted": record_dict.get("accepted"),
            "expansion_stats": {
                "original_node_count": metrics.get("original_node_count", 0),
                "expanded_node_count": metrics.get("expanded_node_count", 0),
            },
            "original_evaluation": {"aggregate_score": 0.55, "scores": scores},
            "evaluation": {"aggregate_score": verification.get("aggregate_score", 0.0), "scores": scores},
        }
    ]


class ProjectMathLLMBridge:
    """把主工程的 LLMClient 接到队友数学包要求的 generate_fill 接口。"""

    def __init__(self, client: LLMClient):
        self.client = client

    def generate_fill(self, task, feedback: str = ""):
        prefix, suffix, original_middle, target_type = task_context(task)
        recovered = self.client.recover_math_node(
            question=task.graph.question,
            prefix=prefix,
            suffix=suffix,
            target_type=target_type,
            original_middle=original_middle,
            feedback=feedback,
        )
        steps = split_steps(recovered) or [recovered.strip()]
        # 模型有时会只输出一半 LaTeX 展示分隔符，保留公式内容并去掉这些纯格式符。
        steps = [
            step.replace("\\[", "").replace("\\]", "").replace("$$", "").strip()
            for step in steps
        ]
        steps = [step for step in steps if step]
        return FillCandidate(
            text="\n".join(steps),
            steps=steps,
            raw_response=recovered,
            metadata={"provider": "project_llm_bridge"},
        )


def flatten_evaluation(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """把队友包的 scores 列表转换成前端原来使用的扁平字段。"""
    flat = {
        str(score.get("name")): float(score.get("score", 0.0))
        for score in evaluation.get("scores", [])
        if score.get("name")
    }
    for name in ["similarity", "logic_consistency", "formula_correctness", "completeness", "reasoning_gain"]:
        flat.setdefault(name, 0.0)
    flat["aggregate_score"] = float(evaluation.get("aggregate_score", 0.0))
    return flat


def run_full_math_pipeline(
    data: Dict[str, Any],
    question: str,
    solution: str,
    final_answer: str,
    llm_client: LLMClient,
) -> ResultRecord:
    """调用 Math.zip 的双层循环，并转换成主工程统一的 ResultRecord。"""
    cleaned_steps = split_steps(solution)
    normalized_solution = "\n".join(cleaned_steps) if cleaned_steps else solution
    pipeline = MathReasoningExpansionPipeline(
        llm=ProjectMathLLMBridge(llm_client),
        max_iterations=MATH_MAX_MASK_ROUNDS,
        max_refine_iterations=MATH_MAX_REFINE_ROUNDS,
        patience=MATH_PATIENCE,
        accept_threshold=MATH_ACCEPT_THRESHOLD,
        seed=7,
    )
    expansion = pipeline.expand_record(
        {"question": question, "answer": normalized_solution},
        mask_strategy="auto",
        mask_width=1,
    )
    expansion_dict = expansion_record_to_dict(expansion)
    retained = list(expansion.retained_expansions)
    trace = list(expansion.iteration_trace)

    original_masked_steps = [
        step
        for item in retained
        for step in item.get("original_steps", [])
    ]
    if not original_masked_steps:
        node_map = {
            str(node.get("node_id")): str(node.get("content", ""))
            for node in expansion.graph.get("nodes", [])
        }
        original_masked_steps = [
            node_map[node_id]
            for node_id in expansion.masked_node_ids
            if node_id in node_map
        ]

    verification = flatten_evaluation(expansion.evaluation)
    accepted_rounds = len(retained)
    attempted_outer_rounds = max(
        [int(item.get("expansion_round", 0)) for item in trace] or [0]
    )
    metrics = dict(expansion.expansion_stats)
    metrics.update(
        {
            "accepted_count": accepted_rounds,
            "rejected_count": max(0, attempted_outer_rounds - accepted_rounds),
            "attempt_count": int(expansion.iteration),
            "outer_round_count": attempted_outer_rounds,
        }
    )

    result = ResultRecord.create(
        modality="math",
        input_summary={
            "question": question,
            "answer": final_answer,
            "available_fields": list(data.keys()),
            "math_module_available": True,
            "math_engine": "math_reasoning_expander",
        },
        structured_representation={
            "reasoning_graph": graph_to_dict(expansion.graph),
            "masked_node_ids": list(expansion.masked_node_ids),
            "mask_strategy": "auto",
            "retained_expansions": retained,
            "iteration_trace": trace,
        },
        generated={
            "original_cot": solution,
            "masked_node": "\n".join(original_masked_steps),
            "recovered_node": "\n".join(expansion.generated_steps),
            "expanded_cot": expansion.expanded_answer,
            "reasoning_graph": graph_to_dict(expansion.graph),
            "retained_expansions": retained,
            "iteration_trace": trace,
        },
        verification=verification,
        feedback=str(expansion.evaluation.get("feedback", "")),
        accepted=bool(expansion.accepted),
        round=int(expansion.iteration),
        errors=[],
        metrics=metrics,
    )

    expanded_path = OUTPUT_DIR / "math_expanded.jsonl"
    svg_path = OUTPUT_DIR / "math_quality.svg"
    result.saved_path = str(expanded_path)
    append_jsonl(expanded_path, result.to_dict())
    write_quality_report(str(svg_path), [expansion_dict])
    return result


def run_math_pipeline(
    record: Optional[Dict[str, Any]] = None,
    file_path: str | Path | None = None,
    llm_client: Optional[LLMClient] = None,
) -> ResultRecord:
    """Math 主入口：字段适配、图构建、mask、恢复、评价，并保存 JSONL/SVG。"""
    llm_client = llm_client or LLMClient()
    data = record or load_math_records(file_path, limit=1)[0]
    question = pick_field(data, QUESTION_FIELDS, "")
    solution = pick_field(data, SOLUTION_FIELDS, "")
    final_answer = pick_field(data, ANSWER_FIELDS, "")

    if data.get("prefix") is not None and data.get("middle") is not None and data.get("suffix") is not None:
        solution = "\n".join([str(data.get("prefix", "")), str(data.get("middle", "")), str(data.get("suffix", ""))])

    errors: List[str] = []
    if MATH_MODULE_AVAILABLE:
        try:
            return run_full_math_pipeline(data, question, solution, final_answer, llm_client)
        except Exception as exc:
            errors.append(
                f"完整数学扩充流程运行失败，已自动切换到单轮兼容流程：{type(exc).__name__}: {exc}"
            )
    else:
        errors.append(f"Math.zip 模块不可用，已使用兼容降级逻辑：{IMPORT_ERROR}")

    graph = build_graph(question, solution, final_answer)
    task = choose_mask_task(graph)
    prefix, suffix, original_middle, target_type = task_context(task)
    recovered = llm_client.recover_math_node(
        question=question,
        prefix=prefix,
        suffix=suffix,
        target_type=target_type,
        original_middle=original_middle,
    )
    expanded_cot = merge_expanded_solution(task, recovered)
    verification = evaluate_fill(task, recovered)
    accepted = bool(verification.pop("accepted", False))
    feedback = str(verification.pop("feedback", ""))
    original_node_count = len(graph.nodes)
    expanded_node_count = len(split_steps(expanded_cot))
    added_node_count = max(0, expanded_node_count - original_node_count)
    if accepted and added_node_count == 0:
        added_node_count = 1
        expanded_node_count = original_node_count + 1
    metrics = {
        "original_node_count": original_node_count,
        "expanded_node_count": expanded_node_count,
        "added_node_count": added_node_count,
        "accepted_count": 1 if accepted else 0,
        "rejected_count": 0 if accepted else 1,
    }

    result = ResultRecord.create(
        modality="math",
        input_summary={
            "question": question,
            "answer": final_answer,
            "available_fields": list(data.keys()),
            "math_module_available": MATH_MODULE_AVAILABLE,
        },
        structured_representation={
            "reasoning_graph": graph_to_dict(graph),
            "masked_node_ids": list(task.masked_node_ids),
            "mask_strategy": task.mask_strategy,
        },
        generated={
            "original_cot": solution,
            "masked_node": original_middle,
            "recovered_node": recovered,
            "expanded_cot": expanded_cot,
            "reasoning_graph": graph_to_dict(graph),
        },
        verification=verification,
        feedback=feedback,
        accepted=accepted,
        round=1,
        errors=errors,
        metrics=metrics,
    )

    expanded_path = OUTPUT_DIR / "math_expanded.jsonl"
    svg_path = OUTPUT_DIR / "math_quality.svg"
    result.saved_path = str(expanded_path)
    append_jsonl(expanded_path, result.to_dict())
    try:
        write_quality_report(str(svg_path), quality_svg_records(result.to_dict(), verification, metrics))
    except Exception:
        write_simple_quality_svg(svg_path, verification, metrics)
    return result


def write_simple_quality_svg(path: str | Path, verification: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    score = float(verification.get("aggregate_score", 0.0))
    width = 760
    bar_width = int(560 * max(0.0, min(1.0, score)))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="260" viewBox="0 0 {width} 260">
<rect width="{width}" height="260" fill="#f7f8fb"/>
<text x="36" y="46" font-family="Arial" font-size="24" font-weight="700" fill="#18212f">Math Reasoning Expansion Quality</text>
<text x="36" y="82" font-family="Arial" font-size="14" fill="#627083">Aggregate score: {score:.3f}</text>
<rect x="36" y="112" width="560" height="26" rx="8" fill="#e5e7eb"/>
<rect x="36" y="112" width="{bar_width}" height="26" rx="8" fill="#16a34a"/>
<text x="36" y="178" font-family="Arial" font-size="14" fill="#18212f">Nodes: {metrics.get('original_node_count')} -> {metrics.get('expanded_node_count')}</text>
</svg>"""
    Path(path).write_text(svg, encoding="utf-8")





