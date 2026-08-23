from __future__ import annotations

from collections import Counter
import statistics
from typing import Dict, List, Mapping, Sequence


def build_expert_weight_matrix(
    labels: Sequence[str | None],
    expert_order: Sequence[str],
    unknown_policy: str = "uniform",
) -> List[List[float]]:
    """Build the per-sample `[batch, experts]` matrix used by MT-OPD.

    Known labels are hard-routed one-hot rows. Unknown labels can either use a
    uniform diagnostic fallback or raise, which is preferable for formal runs.
    """
    if not expert_order or len(set(expert_order)) != len(expert_order):
        raise ValueError("expert_order must contain unique expert labels")
    if unknown_policy not in {"uniform", "error"}:
        raise ValueError("unknown_policy must be uniform or error")
    index = {label: position for position, label in enumerate(expert_order)}
    matrix: List[List[float]] = []
    for label in labels:
        row = [0.0] * len(expert_order)
        position = index.get(label) if label is not None else None
        if position is None:
            if unknown_policy == "error":
                raise ValueError(f"Unknown expert label: {label!r}")
            row = [1.0 / len(expert_order)] * len(expert_order)
        else:
            row[position] = 1.0
        matrix.append(row)
    return matrix


def route_aligned_teacher_logprobs(
    teacher_logprobs: Mapping[str, List[List[List[float]]]],
    expert_order: Sequence[str],
    weight_matrix: Sequence[Sequence[float]],
) -> List[List[List[float]]]:
    """Select/combine aligned `[B,T,K]` Teacher log-probs with `[B,N]` weights.

    All Teachers must be evaluated on the same Student token IDs. Hard one-hot
    weights select one Teacher exactly. Non-one-hot weights implement the same
    weighted log-prob target used by a soft-routing ablation.
    """
    missing = [expert for expert in expert_order if expert not in teacher_logprobs]
    if missing:
        raise ValueError(f"Missing Teacher tensors: {missing}")
    tensors = [teacher_logprobs[expert] for expert in expert_order]
    shape = _shape_3d(tensors[0])
    if any(_shape_3d(tensor) != shape for tensor in tensors[1:]):
        raise ValueError("All Teacher log-prob tensors must share [B,T,K] shape")
    batch, tokens, support = shape
    if len(weight_matrix) != batch:
        raise ValueError("Weight matrix batch size does not match Teacher tensors")
    result = [[[0.0 for _ in range(support)] for _ in range(tokens)] for _ in range(batch)]
    for sample in range(batch):
        row = list(weight_matrix[sample])
        if len(row) != len(expert_order):
            raise ValueError("Weight matrix expert dimension mismatch")
        if any(weight < 0.0 for weight in row) or abs(sum(row) - 1.0) > 1e-6:
            raise ValueError("Each weight row must be non-negative and sum to one")
        for expert_index, weight in enumerate(row):
            if weight == 0.0:
                continue
            tensor = tensors[expert_index]
            for token in range(tokens):
                for candidate in range(support):
                    result[sample][token][candidate] += weight * tensor[sample][token][candidate]
    return result


def compute_expert_share_metrics(
    labels: Sequence[str],
    valid_token_counts: Sequence[int],
    reward_abs_means: Sequence[float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Report prompt, token, and effective optimization-budget shares."""
    if len(labels) != len(valid_token_counts):
        raise ValueError("labels and valid_token_counts must have the same length")
    if reward_abs_means is not None and len(labels) != len(reward_abs_means):
        raise ValueError("reward_abs_means must match labels")
    total_prompts = max(len(labels), 1)
    total_tokens = max(sum(valid_token_counts), 1)
    budget_values = [
        valid_token_counts[index]
        * (reward_abs_means[index] if reward_abs_means is not None else 1.0)
        for index in range(len(labels))
    ]
    total_budget = max(sum(budget_values), 1e-12)
    prompt_counts = Counter(labels)
    result: Dict[str, Dict[str, float]] = {}
    for label in sorted(prompt_counts):
        indices = [index for index, value in enumerate(labels) if value == label]
        tokens = sum(valid_token_counts[index] for index in indices)
        budget = sum(budget_values[index] for index in indices)
        result[label] = {
            "prompt_count": float(len(indices)),
            "prompt_share": len(indices) / total_prompts,
            "valid_token_count": float(tokens),
            "token_share": tokens / total_tokens,
            "effective_budget_share": budget / total_budget,
            "mean_response_length": tokens / max(len(indices), 1),
        }
    return result


def compute_token_share_loss_weights(
    labels: Sequence[str],
    valid_token_counts: Sequence[int],
    target_shares: Mapping[str, float] | None = None,
) -> List[float]:
    """Return sample weights that make weighted token shares match the target."""
    if len(labels) != len(valid_token_counts):
        raise ValueError("labels and valid_token_counts must have the same length")
    present = sorted(set(labels))
    if not present:
        return []
    raw_target = {
        label: float(target_shares.get(label, 0.0)) if target_shares else 1.0
        for label in present
    }
    target_total = sum(raw_target.values())
    if target_total <= 0.0:
        raise ValueError("Target shares must have positive mass for present labels")
    target = {label: raw_target[label] / target_total for label in present}
    total_tokens = max(sum(valid_token_counts), 1)
    observed = {
        label: sum(
            count for current, count in zip(labels, valid_token_counts) if current == label
        )
        / total_tokens
        for label in present
    }
    weights = [target[label] / max(observed[label], 1e-12) for label in labels]
    weighted_tokens = sum(weight * count for weight, count in zip(weights, valid_token_counts))
    normalization = total_tokens / max(weighted_tokens, 1e-12)
    return [weight * normalization for weight in weights]


def compute_teacher_conflict_metrics(
    aligned_top1_logprobs: Mapping[str, Sequence[Sequence[float]]],
    valid_mask: Sequence[Sequence[bool]],
    threshold_nats: float = 1.0,
) -> Dict[str, float]:
    """Measure cross-Teacher spread on the same Student-preferred token."""
    if len(aligned_top1_logprobs) < 2:
        return {}
    tensors = list(aligned_top1_logprobs.values())
    shape = (len(tensors[0]), len(tensors[0][0]) if tensors[0] else 0)
    if any((len(tensor), len(tensor[0]) if tensor else 0) != shape for tensor in tensors):
        raise ValueError("All aligned Teacher tensors must share [B,T] shape")
    if (len(valid_mask), len(valid_mask[0]) if valid_mask else 0) != shape:
        raise ValueError("valid_mask shape mismatch")
    spreads: List[float] = []
    for sample in range(shape[0]):
        for token in range(shape[1]):
            if not valid_mask[sample][token]:
                continue
            values = [float(tensor[sample][token]) for tensor in tensors]
            spreads.append(max(values) - min(values))
    if not spreads:
        return {"valid_token_count": 0.0, "mean_spread_nats": 0.0, "conflict_fraction": 0.0}
    return {
        "valid_token_count": float(len(spreads)),
        "mean_spread_nats": statistics.fmean(spreads),
        "median_spread_nats": statistics.median(spreads),
        "max_spread_nats": max(spreads),
        "conflict_threshold_nats": float(threshold_nats),
        "conflict_fraction": sum(value > threshold_nats for value in spreads) / len(spreads),
    }


def _shape_3d(tensor: List[List[List[float]]]) -> tuple[int, int, int]:
    batch = len(tensor)
    tokens = len(tensor[0]) if tensor else 0
    support = len(tensor[0][0]) if tensor and tensor[0] else 0
    if any(len(sample) != tokens for sample in tensor):
        raise ValueError("Ragged token dimension")
    if any(len(row) != support for sample in tensor for row in sample):
        raise ValueError("Ragged support dimension")
    return batch, tokens, support
