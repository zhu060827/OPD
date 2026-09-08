# Copyright 2026 zhu060827/OPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Effective-token balancing for hard-routed multi-Teacher OPD."""

from __future__ import annotations

import numpy as np
import torch


def apply_domain_token_balance(batch, config, reward_ema=None) -> dict[str, float]:
    """Balance routed-domain gradient budgets by tokens and OPD reward scale.

    The resulting per-domain weight is
    ``target_share / token_share * (reward_magnitude / reference) ** alpha``.
    With ``alpha=0`` this reduces to token-share balancing.  The multiplicative
    reward term follows the remaining Teacher-Student-gap interpretation: a
    domain with a larger OPD advantage receives more of the shared Student's
    update budget.
    """

    balance = config.get("domain_balance", None)
    if balance is None or not bool(balance.get("enabled", False)):
        return {}
    if "domain" not in batch.non_tensor_batch:
        raise ValueError("domain_balance requires a domain field in every training sample")

    domains = tuple(str(value) for value in balance.get("domains", ()))
    raw_targets = [float(value) for value in balance.get("target_shares", ())]
    if len(domains) != len(raw_targets) or any(value <= 0 for value in raw_targets):
        raise ValueError("domain_balance domains and positive target_shares must have equal length")
    target_total = sum(raw_targets)
    targets = {domain: value / target_total for domain, value in zip(domains, raw_targets, strict=True)}
    lower = float(balance.get("min_weight", 0.05))
    upper = float(balance.get("max_weight", 20.0))
    alpha = float(balance.get("reward_scale_alpha", 0.0))
    ema_decay = float(balance.get("reward_ema_decay", 0.9))
    token_share_enabled = bool(balance.get("token_share_enabled", True))
    reward_scale_enabled = bool(balance.get("reward_scale_enabled", True))
    if not 0 < lower <= upper:
        raise ValueError("domain_balance requires 0 < min_weight <= max_weight")
    if not 0.0 <= alpha <= 2.0:
        raise ValueError("reward_scale_alpha must be in [0, 2]")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("reward_ema_decay must be in [0, 1)")

    labels = np.asarray(batch.non_tensor_batch["domain"], dtype=object)
    response_mask = batch.batch["response_mask"].float()
    token_counts = response_mask.sum(dim=-1)
    observed = {
        domain: token_counts[torch.as_tensor(labels == domain, device=token_counts.device)].sum()
        for domain in domains
    }
    present = tuple(domain for domain in domains if observed[domain].item() > 0)
    if not present:
        raise ValueError("generation batch contains no valid response tokens")
    total_tokens = sum(observed.values())
    advantages = batch.batch["advantages"]
    token_advantages = advantages.abs()
    if token_advantages.dim() == 3:
        token_advantages = token_advantages.mean(dim=-1)
    reward_magnitudes = {}
    for domain in present:
        mask = torch.as_tensor(labels == domain, device=token_counts.device)
        domain_mask = response_mask[mask]
        magnitude = (token_advantages[mask] * domain_mask).sum() / domain_mask.sum().clamp_min(1.0)
        current = max(float(magnitude.item()), 1e-12)
        if reward_ema is not None:
            previous = float(reward_ema.get(domain, current))
            current = ema_decay * previous + (1.0 - ema_decay) * current
            reward_ema[domain] = current
        reward_magnitudes[domain] = current
    if reward_ema is not None:
        for domain in domains:
            if domain not in reward_magnitudes and domain in reward_ema:
                reward_magnitudes[domain] = float(reward_ema[domain])
    reference = float(np.median([reward_magnitudes[domain] for domain in present]))
    reference = max(reference, 1e-12)
    raw_domain_weights = {}
    present_target_total = sum(targets[domain] for domain in present)
    for domain in present:
        present_target = targets[domain] / present_target_total
        token_correction = (
            present_target / (observed[domain].item() / total_tokens.item())
            if token_share_enabled
            else 1.0
        )
        reward_correction = (
            (reward_magnitudes[domain] / reference) ** alpha
            if reward_scale_enabled
            else 1.0
        )
        raw_domain_weights[domain] = token_correction * reward_correction
    raw_sample_weights = torch.as_tensor(
        [raw_domain_weights[str(label)] for label in labels],
        device=response_mask.device,
        dtype=response_mask.dtype,
    )
    # Find a common scale whose clipped weights preserve total token mass.
    # This keeps the overall learning-rate scale stable while also guaranteeing
    # the final (not merely pre-normalization) weights remain in [lower, upper].
    low_scale, high_scale = 0.0, upper / max(lower, 1e-12) * 1e6
    for _ in range(80):
        scale = (low_scale + high_scale) / 2.0
        candidate = torch.clamp(raw_sample_weights * scale, min=lower, max=upper)
        if (candidate * token_counts).sum() < total_tokens:
            low_scale = scale
        else:
            high_scale = scale
    sample_weights = torch.clamp(
        raw_sample_weights * ((low_scale + high_scale) / 2.0), min=lower, max=upper
    )
    shape = [len(sample_weights)] + [1] * (advantages.dim() - 1)
    batch.batch["advantages"] = advantages * sample_weights.reshape(shape)
    batch.batch["domain_loss_weight"] = sample_weights

    metrics: dict[str, float] = {}
    weighted_counts = sample_weights * token_counts
    weighted_total = weighted_counts.sum().clamp_min(1e-12)
    for domain in domains:
        mask = torch.as_tensor(labels == domain, device=token_counts.device)
        metrics[f"domain_balance/{domain}/prompt_share"] = mask.float().mean().item()
        metrics[f"domain_balance/{domain}/token_share"] = (observed[domain] / total_tokens).item()
        metrics[f"domain_balance/{domain}/effective_share"] = (weighted_counts[mask].sum() / weighted_total).item()
        metrics[f"domain_balance/{domain}/loss_weight"] = (
            sample_weights[mask].mean().item() if mask.any() else 0.0
        )
        metrics[f"domain_balance/{domain}/reward_magnitude"] = reward_magnitudes.get(domain, 0.0)
        metrics[f"domain_balance/{domain}/reward_scale"] = (
            (reward_magnitudes[domain] / reference) ** alpha
            if domain in reward_magnitudes and reward_scale_enabled
            else 1.0
        )
    return metrics
