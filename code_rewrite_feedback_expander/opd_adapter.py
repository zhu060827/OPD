from __future__ import annotations

import importlib
import os
from typing import Any

from .llm import OPDDistributionScorer


def load_opd_scorer(spec: str | None = None, **kwargs: Any) -> OPDDistributionScorer:
    """Load a production OPD scorer from ``module:function``.

    The external factory must return an OPDDistributionScorer. This keeps the
    code expander independent from the layout of the deployed OPD repository.
    """
    target = spec or os.getenv("CODE_EXPANDER_OPD_SCORER_FACTORY")
    if not target or ":" not in target:
        raise ValueError(
            "Set --opd-scorer-factory module:function or "
            "CODE_EXPANDER_OPD_SCORER_FACTORY."
        )
    module_name, factory_name = target.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    scorer = factory(**kwargs)
    if not isinstance(scorer, OPDDistributionScorer):
        raise TypeError(f"OPD factory {target} did not return OPDDistributionScorer")
    return scorer
