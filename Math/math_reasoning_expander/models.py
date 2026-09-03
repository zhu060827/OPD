from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

    def node_by_id(self, node_id: str) -> Optional[ReasoningNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def visible_nodes(self, masked_ids: List[str]) -> List[ReasoningNode]:
        masked = set(masked_ids)
        return [node for node in self.nodes if node.node_id not in masked]

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

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "question": self.graph.question,
            "answer": self.graph.answer,
            "mask_strategy": self.mask_strategy,
            "masked_node_ids": self.masked_node_ids,
            "prefix": [node.content for node in self.prefix_nodes],
            "suffix": [node.content for node in self.suffix_nodes],
            "target_node_types": [node.node_type for node in self.target_nodes],
            "target_formulas": [formula for node in self.target_nodes for formula in node.formulas],
        }


@dataclass
class FillCandidate:
    text: str
    steps: List[str]
    raw_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScore:
    name: str
    score: float
    explanation: str


@dataclass
class EvaluationResult:
    scores: List[MetricScore]
    accepted: bool
    aggregate_score: float
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "aggregate_score": self.aggregate_score,
            "feedback": self.feedback,
            "scores": [score.__dict__ for score in self.scores],
        }


@dataclass
class ExpansionRecord:
    question: str
    original_answer: str
    expanded_answer: str
    graph: Dict[str, Any]
    masked_node_ids: List[str]
    generated_steps: List[str]
    expansion_stats: Dict[str, Any]
    original_evaluation: Dict[str, Any]
    evaluation: Dict[str, Any]
    retained_expansions: List[Dict[str, Any]]
    iteration_trace: List[Dict[str, Any]]
    iteration: int
    accepted: bool
