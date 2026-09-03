from __future__ import annotations

import re
from typing import Iterable, List

from .models import ReasoningEdge, ReasoningGraph, ReasoningNode


FORMULA_PATTERN = re.compile(r"[A-Za-z0-9().\s+\-*/^=<>]{3,}")


class ReasoningGraphParser:
    """Converts a natural language CoT answer into a lightweight reasoning DAG."""

    def parse(self, question: str, answer: str) -> ReasoningGraph:
        steps = self.split_steps(answer)
        nodes: List[ReasoningNode] = []
        edges: List[ReasoningEdge] = []

        for idx, step in enumerate(steps, start=1):
            node_id = f"s{idx}"
            depends_on = [f"s{idx - 1}"] if idx > 1 else []
            formulas = self.extract_formulas(step)
            node = ReasoningNode(
                node_id=node_id,
                content=step,
                node_type=self.classify_step(step, idx, len(steps)),
                formulas=formulas,
                depends_on=depends_on,
            )
            nodes.append(node)
            if idx > 1:
                edges.append(ReasoningEdge(source=f"s{idx - 1}", target=node_id))

        return ReasoningGraph(question=question, answer=answer, nodes=nodes, edges=edges)

    def split_steps(self, answer: str) -> List[str]:
        cleaned = answer.replace("\r\n", "\n").strip()
        if not cleaned:
            return []

        line_steps = self._split_lines(cleaned)
        if len(line_steps) >= 2:
            return line_steps

        sentence_steps = self._split_sentences(cleaned)
        return sentence_steps if sentence_steps else [cleaned]

    def extract_formulas(self, text: str) -> List[str]:
        formulas = []
        chunks = re.split(
            r"\b(?:so|thus|therefore|then|hence|and|which implies|this gives|we have)\b|[,;.]",
            text,
            flags=re.I,
        )
        for chunk in chunks:
            for match in FORMULA_PATTERN.finditer(chunk):
                formula = match.group(0).strip(" .,:;")
                formula = re.sub(r"\s+", " ", formula)
                formula = re.sub(r"^(?:[A-Za-z]+\s+){1,4}(?=[A-Za-z0-9(])", "", formula).strip()
                if any(op in formula for op in ["=", "+", "-", "*", "/", "^"]) and not formula.isalpha():
                    formulas.append(formula)
        return formulas

    def classify_step(self, step: str, idx: int, total: int) -> str:
        lower = step.lower()
        if idx == total or "answer" in lower or "therefore" in lower or "boxed" in lower:
            return "conclusion"
        if any(token in lower for token in ["let ", "define", "suppose", "assume"]):
            return "definition"
        if "=" in step and any(op in step for op in ["+", "-", "*", "/", "^"]):
            return "equation_transform"
        if any(token in lower for token in ["so", "thus", "hence", "because", "therefore"]):
            return "inference"
        return "explanation"

    def _split_lines(self, text: str) -> List[str]:
        raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
        steps = []
        for line in raw_lines:
            line = re.sub(r"^\s*(?:step\s*)?\d+[\).:-]\s*", "", line, flags=re.I)
            line = re.sub(r"^\s*[-*]\s*", "", line)
            if line:
                steps.append(line)
        return steps

    def _split_sentences(self, text: str) -> List[str]:
        protected = text.replace("e.g.", "eg").replace("i.e.", "ie")
        pieces = re.split(r"(?<=[.!?])\s+|(?<=。)\s*", protected)
        steps = [piece.strip() for piece in pieces if piece.strip()]
        return [self._restore_abbrev(step) for step in steps]

    def _restore_abbrev(self, step: str) -> str:
        return step.replace("eg", "e.g.").replace("ie", "i.e.")


def parse_jsonl_records(records: Iterable[dict]) -> List[ReasoningGraph]:
    parser = ReasoningGraphParser()
    graphs = []
    for record in records:
        question = record.get("question", "")
        answer = record.get("answer") or record.get("cot") or record.get("solution", "")
        graphs.append(parser.parse(question, answer))
    return graphs
