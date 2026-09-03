from __future__ import annotations

import random
from typing import List

from .models import MaskedTask, ReasoningGraph, ReasoningNode


class GraphMasker:
    def __init__(self, seed: int = 7):
        self.random = random.Random(seed)

    def mask(self, graph: ReasoningGraph, strategy: str = "auto", width: int = 1) -> MaskedTask:
        if not graph.nodes:
            raise ValueError("Cannot mask an empty reasoning graph.")

        if strategy == "auto":
            strategy = self._choose_strategy(graph)

        if strategy == "formula_node":
            masked = self._mask_formula_node(graph)
        elif strategy == "path":
            masked = self._mask_path(graph, width=width)
        else:
            masked = self._mask_single_node(graph)

        masked_ids = [node.node_id for node in masked]
        first_index = graph.nodes.index(masked[0])
        last_index = graph.nodes.index(masked[-1])
        return MaskedTask(
            graph=graph,
            masked_node_ids=masked_ids,
            mask_strategy=strategy,
            prefix_nodes=graph.nodes[:first_index],
            suffix_nodes=graph.nodes[last_index + 1 :],
            target_nodes=masked,
        )

    def _choose_strategy(self, graph: ReasoningGraph) -> str:
        return "single_node"

    def _mask_single_node(self, graph: ReasoningGraph) -> List[ReasoningNode]:
        candidates = graph.nodes[1:-1] if len(graph.nodes) > 2 else graph.nodes
        return [self.random.choice(candidates)]

    def _mask_formula_node(self, graph: ReasoningGraph) -> List[ReasoningNode]:
        candidates = [node for node in graph.nodes[1:-1] if node.formulas]
        if not candidates:
            return self._mask_single_node(graph)
        return [self.random.choice(candidates)]

    def _mask_path(self, graph: ReasoningGraph, width: int = 2) -> List[ReasoningNode]:
        if len(graph.nodes) <= 2:
            return self._mask_single_node(graph)
        width = max(1, min(width, len(graph.nodes) - 2))
        start = self.random.randint(1, len(graph.nodes) - width - 1)
        return graph.nodes[start : start + width]
