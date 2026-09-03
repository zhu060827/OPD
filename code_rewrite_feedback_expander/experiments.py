from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

from .models import CodeRecord
from .pipeline import DEFAULT_STRATEGIES, CodeRewriteExpansionPipeline, expansion_record_to_dict
from .reporting import build_paper_report


def run_comparison_experiments(
    records: Iterable[CodeRecord],
    pipeline_factory: Callable[..., CodeRewriteExpansionPipeline],
) -> Dict[str, Any]:
    records = list(records)
    comparisons = {
        policy: _run(records, pipeline_factory(selection_policy=policy))
        for policy in ("adaptive", "random", "fixed")
    }
    ablations = {
        f"without_{excluded}": _run(
            records,
            pipeline_factory(strategies=[item for item in DEFAULT_STRATEGIES if item != excluded]),
        )
        for excluded in DEFAULT_STRATEGIES
    }
    return {"strategy_comparisons": comparisons, "method_ablations": ablations}


def _run(records: List[CodeRecord], pipeline: CodeRewriteExpansionPipeline) -> Dict[str, Any]:
    outputs = [expansion_record_to_dict(pipeline.expand_record(record)) for record in records]
    return build_paper_report(outputs)
