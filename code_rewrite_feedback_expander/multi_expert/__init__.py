"""Auditable Stage-1 multi-expert routing for code OPD."""

from .config import Stage1Config, load_config
from .fusion import build_expert_weight_matrix, route_aligned_teacher_logprobs
from .pipeline import MultiExpertStage1Pipeline

from .opd_reward import compute_topk_opd_reward, gap_following_scales, refresh_topk_opd_reward

__all__ = [
    "Stage1Config",
    "load_config",
    "MultiExpertStage1Pipeline",
    "build_expert_weight_matrix",
    "route_aligned_teacher_logprobs",
    "compute_topk_opd_reward",
    "gap_following_scales",
    "refresh_topk_opd_reward",
]
