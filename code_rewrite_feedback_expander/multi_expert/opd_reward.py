"""Reference implementation of the token-level Open-MOPD reward.

The functions in this module are framework agnostic.  They operate on the
log-probability tensors produced by the rollout/teacher workers and make the
reward contract explicit and unit-testable:

    r[b,t,k] = stopgrad(T[b,t,k] - S[b,t,k]) * p_hat_S[b,t,k]

where ``p_hat_S`` is Student's normalized probability over the selected top-k
support.  Padding and prompt tokens are masked out by ``response_mask``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


def compute_topk_opd_reward(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gap_clip: tuple[float, float] | None = (-20.0, 20.0),
    detach_gap: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Open-MOPD's probability-weighted top-k token reward.

    ``student_log_probs`` and ``teacher_log_probs`` must be ``[B,T,K]`` and
    refer to the same token IDs (the usual ``only_stu`` top-k support).  The
    output keeps the top-k dimension because the policy loss can consume the
    support directly.  The reward is zero outside valid response tokens.
    """

    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher log-prob tensors must have identical shapes")
    if student_log_probs.ndim != 3:
        raise ValueError("top-k log-prob tensors must have shape [batch, tokens, k]")
    if response_mask.shape != student_log_probs.shape[:2]:
        raise ValueError("response_mask must have shape [batch, tokens]")
    if student_log_probs.shape[-1] < 1:
        raise ValueError("top-k support must be non-empty")

    gap = teacher_log_probs - student_log_probs
    if gap_clip is not None:
        low, high = gap_clip
        if low > high:
            raise ValueError("gap_clip lower bound must not exceed upper bound")
        gap = gap.clamp(min=float(low), max=float(high))
    if detach_gap:
        gap = gap.detach()

    # Normalize Student probability on the available top-k support.  This is
    # the p-tilde factor used by Open-MOPD rather than an unweighted gap.
    student_topk_prob = torch.softmax(student_log_probs, dim=-1)
    reward = gap * student_topk_prob
    mask = response_mask.to(dtype=reward.dtype).unsqueeze(-1)
    reward = reward * mask

    valid = mask.expand_as(reward).bool()
    valid_rewards = reward[valid]
    metrics = {
        "valid_token_count": float(response_mask.sum().item()),
        "topk_reward_mean": float(valid_rewards.mean().item()) if valid_rewards.numel() else 0.0,
        "topk_reward_abs_mean": float(valid_rewards.abs().mean().item()) if valid_rewards.numel() else 0.0,
        "teacher_student_gap_mean": float(gap[valid].mean().item()) if valid_rewards.numel() else 0.0,
    }
    return reward, metrics


def refresh_topk_opd_reward(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gap_clip: tuple[float, float] | None = (-20.0, 20.0),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Recompute a detached reward after each Student policy refresh."""

    return compute_topk_opd_reward(
        student_log_probs,
        teacher_log_probs,
        response_mask,
        gap_clip=gap_clip,
        detach_gap=True,
    )


def gap_following_scales(
    domain_gaps: Mapping[str, float],
    *,
    alpha: float = 0.5,
    clip: tuple[float, float] = (0.05, 20.0),
) -> dict[str, float]:
    """Return Open-MOPD-style relative gap scales for routed domains.

    The scale is ``clip((m_d / mean(m)) ** alpha)``.  Absolute values are used
    for robustness because signed mean gaps can cancel across tokens.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if clip[0] <= 0 or clip[0] > clip[1]:
        raise ValueError("clip must satisfy 0 < low <= high")
    if not domain_gaps:
        return {}
    magnitudes = {name: max(abs(float(value)), 1e-12) for name, value in domain_gaps.items()}
    reference = sum(magnitudes.values()) / len(magnitudes)
    return {
        name: max(float(clip[0]), min(float(clip[1]), (value / reference) ** alpha))
        for name, value in magnitudes.items()
    }

