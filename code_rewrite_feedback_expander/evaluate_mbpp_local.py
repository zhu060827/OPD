from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from code_rewrite_feedback_expander.mbpp_reward import extract_python_code, reward_func
from code_rewrite_feedback_expander.mbpp_to_opd_parquet import (
    PROMPT_SCHEMA,
    required_interface_spec,
)

LOGGER = logging.getLogger("mbpp_local_eval")
RESULT_FILES = ("predictions.jsonl", "failures.jsonl", "metrics.json", "run_config.json", "evaluation.log")
GENERATION_STRATEGIES = ("independent", "execution_feedback")
EXECUTION_FEEDBACK_MODES = ("summary", "full")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    hit_token_limit: bool


class BatchGenerator(Protocol):
    def generate(self, conversations: Sequence[list[dict[str, str]]]) -> list[list[GenerationResult]]: ...


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object in {path} at line {line_number}")
            rows.append(value)
    return rows


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(model_path: Path) -> str:
    manifest: dict[str, Any] = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "SHA256SUMS"):
        path = model_path / name
        if path.is_file():
            manifest[name] = _file_sha256(path)
    weight_files = sorted(model_path.glob("*.safetensors"))
    manifest["weights"] = [(path.name, path.stat().st_size) for path in weight_files]
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found in model directory: {model_path}")
    return _stable_hash(manifest)


def _setup_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def load_mbpp_rows(dataset_path: Path, start_index: int = 0, max_samples: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path)
    required = {"data_source", "prompt", "reward_model", "extra_info"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"Dataset {dataset_path} is missing columns: {sorted(missing)}")
    rows = table.to_pylist()
    if start_index < 0 or start_index > len(rows):
        raise ValueError(f"start_index must be between 0 and {len(rows)}, got {start_index}")
    selected = rows[start_index:]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        selected = selected[:max_samples]
    if not selected:
        raise ValueError("No MBPP rows selected for evaluation")
    for offset, row in enumerate(selected, start=start_index):
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        if extra_info.get("prompt_schema") != PROMPT_SCHEMA:
            raise ValueError(
                f"MBPP row {offset} uses prompt_schema={extra_info.get('prompt_schema')!r}; "
                f"expected {PROMPT_SCHEMA!r}. Regenerate all splits with train_mbpp_opd.sh prepare."
            )
    return selected


def _normalize_conversation(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("MBPP prompt must be a non-empty list of chat messages")
    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError(f"Invalid chat message: {message!r}")
        messages.append({"role": str(message["role"]), "content": str(message["content"])})
    return messages


def _task_id(row: dict[str, Any], dataset_index: int) -> str:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict) and extra_info.get("task_id") is not None:
        return str(extra_info["task_id"])
    return str(dataset_index)


def _syntax_valid(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _top_level_definition_names(code: str) -> list[str]:
    if not code:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"pass@{k} requires at least {k} samples, got {n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize_results(
    results: Sequence[dict[str, Any]],
    *,
    target_task_ids: Sequence[str],
    samples_per_task: int,
    model_path: str,
    dataset_path: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["task_id"])].append(result)

    target_ids = list(dict.fromkeys(str(task_id) for task_id in target_task_ids))
    complete_groups = {
        task_id: values
        for task_id, values in grouped.items()
        if task_id in target_ids and len({int(item["sample_id"]) for item in values}) >= samples_per_task
    }
    complete_values = [
        item
        for task_id in target_ids
        for item in sorted(complete_groups.get(task_id, []), key=lambda value: int(value["sample_id"]))[
            :samples_per_task
        ]
    ]

    error_counts = Counter(str(item.get("error_type", "unknown")) for item in complete_values)
    passed_completions = sum(bool(item.get("passed")) for item in complete_values)
    completions = len(complete_values)
    task_success_counts = {
        task_id: sum(bool(item.get("passed")) for item in values[:samples_per_task])
        for task_id, values in complete_groups.items()
    }

    pass_metrics: dict[str, float] = {}
    for k in range(1, samples_per_task + 1):
        estimates = [_pass_at_k(samples_per_task, count, k) for count in task_success_counts.values()]
        pass_metrics[f"pass@{k}"] = statistics.fmean(estimates) if estimates else 0.0

    def mean_of(key: str) -> float:
        values = [float(item[key]) for item in complete_values if item.get(key) is not None]
        return statistics.fmean(values) if values else 0.0

    def median_of(key: str) -> float:
        values = [float(item[key]) for item in complete_values if item.get(key) is not None]
        return statistics.median(values) if values else 0.0

    generation_seconds = sum(float(item.get("generation_seconds", 0.0)) for item in complete_values)
    grading_seconds = sum(float(item.get("grading_seconds", 0.0)) for item in complete_values)
    generated_tokens = sum(int(item.get("generated_tokens", 0)) for item in complete_values)
    denominator = completions or 1
    task_denominator = len(complete_groups) or 1

    return {
        "status": "complete" if len(complete_groups) == len(target_ids) else "in_progress",
        "model_path": model_path,
        "dataset_path": dataset_path,
        "tasks_target": len(target_ids),
        "tasks_with_any_result": sum(task_id in grouped for task_id in target_ids),
        "tasks_complete": len(complete_groups),
        "samples_per_task": samples_per_task,
        "completions_evaluated": completions,
        "passed_completions": passed_completions,
        "sample_pass_rate": passed_completions / denominator,
        **pass_metrics,
        "solved_at_least_once": sum(count > 0 for count in task_success_counts.values()),
        "solved_at_least_once_rate": sum(count > 0 for count in task_success_counts.values()) / task_denominator,
        "solve_none": sum(count == 0 for count in task_success_counts.values()),
        "solve_all": sum(count == samples_per_task for count in task_success_counts.values()),
        "code_extraction_rate": sum(bool(item.get("code_extracted")) for item in complete_values) / denominator,
        "syntax_valid_rate": sum(bool(item.get("syntax_valid")) for item in complete_values) / denominator,
        "interface_match_rate": sum(bool(item.get("interface_match")) for item in complete_values) / denominator,
        "max_token_clip_ratio": sum(bool(item.get("hit_token_limit")) for item in complete_values) / denominator,
        "error_counts": dict(sorted(error_counts.items())),
        "error_rates": {key: value / denominator for key, value in sorted(error_counts.items())},
        "avg_prompt_tokens": mean_of("prompt_tokens"),
        "avg_generated_tokens": mean_of("generated_tokens"),
        "median_generated_tokens": median_of("generated_tokens"),
        "avg_generation_seconds": mean_of("generation_seconds"),
        "median_generation_seconds": median_of("generation_seconds"),
        "avg_grading_seconds": mean_of("grading_seconds"),
        "total_generation_seconds": generation_seconds,
        "total_grading_seconds": grading_seconds,
        "generation_tokens_per_second": generated_tokens / generation_seconds if generation_seconds > 0 else 0.0,
    }


def _ordered_attempts(values: Sequence[dict[str, Any]], max_attempts: int) -> list[dict[str, Any]]:
    by_attempt: dict[int, dict[str, Any]] = {}
    for value in values:
        attempt_id = int(value["sample_id"])
        if 0 <= attempt_id < max_attempts:
            by_attempt[attempt_id] = value
    return [by_attempt[index] for index in sorted(by_attempt)]


def _is_iterative_task_complete(values: Sequence[dict[str, Any]], max_attempts: int) -> bool:
    attempts = _ordered_attempts(values, max_attempts)
    if any(bool(item.get("passed")) for item in attempts):
        return True
    return [int(item["sample_id"]) for item in attempts] == list(range(max_attempts))


def summarize_iterative_results(
    results: Sequence[dict[str, Any]],
    *,
    target_task_ids: Sequence[str],
    max_attempts: int,
    model_path: str,
    dataset_path: str,
    execution_feedback_mode: str,
) -> dict[str, Any]:
    """Summarize conditional repair attempts without mislabeling them as independent pass@k."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["task_id"])].append(result)

    target_ids = list(dict.fromkeys(str(task_id) for task_id in target_task_ids))
    complete_groups: dict[str, list[dict[str, Any]]] = {}
    for task_id in target_ids:
        attempts = _ordered_attempts(grouped.get(task_id, []), max_attempts)
        if _is_iterative_task_complete(attempts, max_attempts):
            complete_groups[task_id] = attempts

    complete_values = [item for task_id in target_ids for item in complete_groups.get(task_id, [])]
    task_count = len(complete_groups)
    task_denominator = task_count or 1
    attempt_denominator = len(complete_values) or 1

    first_success_attempt: dict[str, int | None] = {}
    for task_id, attempts in complete_groups.items():
        success = next((int(item["sample_id"]) + 1 for item in attempts if bool(item.get("passed"))), None)
        first_success_attempt[task_id] = success

    initial_passes = sum(attempt == 1 for attempt in first_success_attempt.values())
    initial_failures = task_count - initial_passes
    repaired = sum(attempt is not None and attempt > 1 for attempt in first_success_attempt.values())
    solve_rates: dict[str, float] = {}
    for k in range(1, max_attempts + 1):
        solved = sum(attempt is not None and attempt <= k for attempt in first_success_attempt.values())
        rate = solved / task_denominator
        solve_rates[f"iterative_solve_rate@{k}"] = rate
        solve_rates[f"iterative_pass@{k}"] = rate

    error_counts = Counter(str(item.get("error_type", "unknown")) for item in complete_values)
    passed_attempts = sum(bool(item.get("passed")) for item in complete_values)
    generation_seconds = sum(float(item.get("generation_seconds", 0.0)) for item in complete_values)
    grading_seconds = sum(float(item.get("grading_seconds", 0.0)) for item in complete_values)
    generated_tokens = sum(int(item.get("generated_tokens", 0)) for item in complete_values)

    def mean_of(key: str) -> float:
        values = [float(item[key]) for item in complete_values if item.get(key) is not None]
        return statistics.fmean(values) if values else 0.0

    def median_of(key: str) -> float:
        values = [float(item[key]) for item in complete_values if item.get(key) is not None]
        return statistics.median(values) if values else 0.0

    solved_attempt_numbers = [attempt for attempt in first_success_attempt.values() if attempt is not None]
    final_rate = solve_rates.get(f"iterative_solve_rate@{max_attempts}", 0.0)
    initial_rate = initial_passes / task_denominator
    return {
        "status": "complete" if task_count == len(target_ids) else "in_progress",
        "model_path": model_path,
        "dataset_path": dataset_path,
        "generation_strategy": "execution_feedback",
        "execution_feedback_mode": execution_feedback_mode,
        "metric_protocol": "conditional_iterative_repair",
        "standard_pass_at_k_applicable": False,
        "metric_note": (
            "iterative_pass@k is a cumulative conditional-repair solve rate, not the standard unbiased pass@k "
            "estimator for independent samples"
        ),
        "tasks_target": len(target_ids),
        "tasks_with_any_result": sum(task_id in grouped for task_id in target_ids),
        "tasks_complete": task_count,
        "max_attempts": max_attempts,
        "attempts_evaluated": len(complete_values),
        "passed_attempts": passed_attempts,
        "attempt_pass_rate": passed_attempts / attempt_denominator,
        "pass@1": initial_rate,
        **solve_rates,
        f"repair_gain@{max_attempts}": final_rate - initial_rate,
        "solved_on_first_attempt": initial_passes,
        "repaired_after_initial_failure": repaired,
        "repair_success_rate_after_initial_failure": repaired / initial_failures if initial_failures else 0.0,
        "solved_at_least_once": sum(attempt is not None for attempt in first_success_attempt.values()),
        "solved_at_least_once_rate": final_rate,
        "failed_after_max_attempts": sum(attempt is None for attempt in first_success_attempt.values()),
        "avg_attempts_per_task": len(complete_values) / task_denominator,
        "avg_attempts_to_solve": statistics.fmean(solved_attempt_numbers) if solved_attempt_numbers else 0.0,
        "code_extraction_rate": sum(bool(item.get("code_extracted")) for item in complete_values)
        / attempt_denominator,
        "syntax_valid_rate": sum(bool(item.get("syntax_valid")) for item in complete_values) / attempt_denominator,
        "interface_match_rate": sum(bool(item.get("interface_match")) for item in complete_values)
        / attempt_denominator,
        "max_token_clip_ratio": sum(bool(item.get("hit_token_limit")) for item in complete_values)
        / attempt_denominator,
        "error_counts": dict(sorted(error_counts.items())),
        "error_rates": {key: value / attempt_denominator for key, value in sorted(error_counts.items())},
        "avg_prompt_tokens": mean_of("prompt_tokens"),
        "avg_generated_tokens": mean_of("generated_tokens"),
        "median_generated_tokens": median_of("generated_tokens"),
        "avg_generation_seconds": mean_of("generation_seconds"),
        "median_generation_seconds": median_of("generation_seconds"),
        "avg_grading_seconds": mean_of("grading_seconds"),
        "total_generation_seconds": generation_seconds,
        "total_grading_seconds": grading_seconds,
        "generation_tokens_per_second": generated_tokens / generation_seconds if generation_seconds > 0 else 0.0,
    }


class TransformersBatchGenerator:
    def __init__(
        self,
        *,
        model_path: Path,
        samples_per_task: int,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
        device: str,
        dtype: str,
        attn_implementation: str,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if samples_per_task > 1 and temperature <= 0:
            raise ValueError("samples_per_task > 1 requires temperature > 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        self.torch = torch
        self.samples_per_task = samples_per_task
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking

        LOGGER.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=trust_remote_code
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map[dtype]
        if device == "auto":
            device_map: str | dict[str, str] = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device {device!r} requested, but CUDA is not available")
            device_map = {"": device}

        LOGGER.info(
            "Loading model from %s (device=%s, dtype=%s, attention=%s)",
            model_path,
            device,
            dtype,
            attn_implementation,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            dtype=resolved_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        LOGGER.info("Model ready; input device=%s", self.input_device)

    def _format(self, conversation: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate(self, conversations: Sequence[list[dict[str, str]]]) -> list[list[GenerationResult]]:
        formatted = [self._format(conversation) for conversation in conversations]
        encoded = self.tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [int(length) for length in prompt_lengths if int(length) > self.max_prompt_tokens]
        if too_long:
            raise ValueError(
                f"Prompt length {max(too_long)} exceeds max_prompt_tokens={self.max_prompt_tokens}; "
                "refusing to truncate evaluation input"
            )
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": self.samples_per_task,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=self.temperature, top_p=self.top_p)

        started = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)
        elapsed = time.perf_counter() - started
        generated_ids = output_ids[:, input_width:].detach().cpu()
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        groups: list[list[GenerationResult]] = []
        result_index = 0
        per_completion_seconds = elapsed / max(len(decoded), 1)
        for prompt_length in prompt_lengths:
            task_results: list[GenerationResult] = []
            for _ in range(self.samples_per_task):
                token_ids = generated_ids[result_index]
                generated_length = int((token_ids != self.tokenizer.pad_token_id).sum().item())
                task_results.append(
                    GenerationResult(
                        text=decoded[result_index],
                        prompt_tokens=int(prompt_length),
                        generated_tokens=generated_length,
                        latency_seconds=per_completion_seconds,
                        hit_token_limit=generated_length >= self.max_new_tokens,
                    )
                )
                result_index += 1
            groups.append(task_results)
        return groups


def _prepare_output_dir(output_dir: Path, run_config: dict[str, Any], overwrite: bool) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    predictions_path = output_dir / "predictions.jsonl"
    if overwrite:
        for filename in RESULT_FILES:
            path = output_dir / filename
            if path.exists():
                path.unlink()
    elif config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != run_config["config_hash"]:
            raise ValueError(
                f"Output directory {output_dir} belongs to a different evaluation configuration. "
                "Use a new --output-dir or pass --overwrite explicitly."
            )
    elif predictions_path.exists():
        raise ValueError(f"Found {predictions_path} without run_config.json; use a new directory or --overwrite")

    if not config_path.exists():
        _write_json_atomic(config_path, run_config)
    return _load_jsonl(predictions_path)


def _score_generation(
    *,
    row: dict[str, Any],
    dataset_index: int,
    task_id: str,
    sample_id: int,
    generation: GenerationResult,
    execution_timeout: float,
) -> dict[str, Any]:
    code = extract_python_code(generation.text)
    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    required_entrypoints = [str(name) for name in extra_info.get("required_entrypoints", [])]
    required_interfaces = [str(interface) for interface in extra_info.get("required_interfaces", [])]
    if not required_entrypoints:
        required_entrypoints, required_interfaces = required_interface_spec(
            str(extra_info.get("canonical_code", "")),
            [str(test) for test in extra_info.get("tests", [])],
            str(extra_info.get("setup_code", "")),
        )
    defined_entrypoints = _top_level_definition_names(code)
    interface_match = bool(required_entrypoints) and set(required_entrypoints).issubset(defined_entrypoints)
    started = time.perf_counter()
    outcome = reward_func(
        str(row.get("data_source", "")),
        generation.text,
        ground_truth,
        extra_info,
        timeout=execution_timeout,
    )
    grading_seconds = time.perf_counter() - started
    error_type = str(outcome.get("error_type", "unknown"))
    if error_type == "test_failure" and outcome.get("test_output") == "Timeout":
        error_type = "timeout"
    return {
        "dataset_index": dataset_index,
        "task_id": task_id,
        "sample_id": sample_id,
        "prompt": _normalize_conversation(row["prompt"]),
        "response": generation.text,
        "extracted_code": code,
        "code_extracted": bool(code),
        "syntax_valid": _syntax_valid(code),
        "required_entrypoints": required_entrypoints,
        "required_interfaces": required_interfaces,
        "defined_entrypoints": defined_entrypoints,
        "interface_match": interface_match,
        "passed": bool(outcome.get("passed", False)),
        "score": float(outcome.get("score", 0.0)),
        "error_type": error_type,
        "test_output": str(outcome.get("test_output") or outcome.get("error") or ""),
        "test_count": len(extra_info.get("tests", [])),
        "prompt_tokens": generation.prompt_tokens,
        "generated_tokens": generation.generated_tokens,
        "generation_seconds": generation.latency_seconds,
        "grading_seconds": grading_seconds,
        "hit_token_limit": generation.hit_token_limit,
    }


def _execution_feedback_message(
    previous_result: dict[str, Any],
    *,
    mode: str,
    max_chars: int,
) -> str:
    if mode not in EXECUTION_FEEDBACK_MODES:
        raise ValueError(f"Unknown execution feedback mode: {mode}")
    if max_chars <= 0:
        raise ValueError("feedback max_chars must be positive")

    error_type = str(previous_result.get("error_type", "unknown"))
    required_interfaces = [str(value) for value in previous_result.get("required_interfaces", [])]
    defined_entrypoints = [str(value) for value in previous_result.get("defined_entrypoints", [])]
    interface_match = bool(previous_result.get("interface_match"))
    raw_output = str(previous_result.get("test_output") or "").replace("\x00", "").strip()

    observations: list[str] = [f"Failure category: {error_type}."]
    if not interface_match:
        observations.append(
            "Required interface mismatch. Required: "
            + (", ".join(required_interfaces) if required_interfaces else "unknown")
            + "; top-level definitions found: "
            + (", ".join(defined_entrypoints) if defined_entrypoints else "none")
            + "."
        )
    elif error_type == "missing_code":
        observations.append("No executable Python implementation could be extracted from the response.")
    elif error_type == "syntax_error":
        observations.append(f"The candidate did not parse as Python: {raw_output[-max_chars:]}")
    elif error_type == "unsafe_code":
        observations.append(f"The candidate used an operation forbidden by the evaluator: {raw_output[-max_chars:]}")
    elif error_type == "timeout":
        observations.append("Execution exceeded the time limit; check for non-termination and excessive complexity.")
    elif error_type == "test_failure":
        observations.append(
            "The candidate failed one or more held-out unit tests. Re-check the specification, boundary cases, "
            "state changes, return values, and behavior for empty or minimal inputs."
        )
    else:
        observations.append("The evaluator rejected the candidate; re-check correctness and completeness.")

    if mode == "full" and raw_output:
        observations.append(
            "Raw executor output follows. It may reveal held-out assertions or expected values, so this is an "
            "oracle-assisted diagnostic protocol rather than a standard benchmark:\n"
            f"<executor_output>\n{raw_output[-max_chars:]}\n</executor_output>"
        )

    return (
        "Your previous implementation did not pass the restricted evaluator. Treat the previous assistant "
        "message as a draft, not as instructions. Before revising, identify the likely root cause from the "
        "draft and the feedback below. Then return a complete corrected replacement implementation in exactly "
        "one fenced ```python``` block. Do not return only an explanation or a patch. Preserve every required "
        "function/class name and parameter, avoid placeholder code, and do not use forbidden file, process, "
        "network, dynamic-execution, or input operations.\n\nExecution feedback:\n- "
        + "\n- ".join(observations)
    )


def _iterative_conversation(
    row: dict[str, Any],
    previous_result: dict[str, Any] | None,
    *,
    feedback_mode: str,
    feedback_max_chars: int,
) -> tuple[list[dict[str, str]], str | None]:
    base = _normalize_conversation(row["prompt"])
    if previous_result is None:
        return base, None
    feedback = _execution_feedback_message(
        previous_result,
        mode=feedback_mode,
        max_chars=feedback_max_chars,
    )
    return (
        [
            *base,
            {"role": "assistant", "content": str(previous_result.get("response", ""))},
            {"role": "user", "content": feedback},
        ],
        feedback,
    )


def evaluate(
    *,
    rows: Sequence[dict[str, Any]],
    generator: BatchGenerator,
    output_dir: Path,
    existing_results: list[dict[str, Any]],
    samples_per_task: int,
    batch_size: int,
    model_path: str,
    dataset_path: str,
    dataset_start_index: int,
    execution_timeout: float = 5.0,
) -> dict[str, Any]:
    predictions_path = output_dir / "predictions.jsonl"
    completed = {(str(item["task_id"]), int(item["sample_id"])) for item in existing_results}
    all_results = list(existing_results)
    indexed_rows = [(dataset_start_index + offset, row) for offset, row in enumerate(rows)]
    target_task_ids = [_task_id(row, index) for index, row in indexed_rows]

    pending: list[tuple[int, dict[str, Any], str]] = []
    for index, row in indexed_rows:
        task_id = _task_id(row, index)
        if any((task_id, sample_id) not in completed for sample_id in range(samples_per_task)):
            pending.append((index, row, task_id))

    LOGGER.info(
        "Evaluation target: %d tasks x %d samples; pending tasks=%d",
        len(rows),
        samples_per_task,
        len(pending),
    )
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        conversations = [_normalize_conversation(row["prompt"]) for _, row, _ in batch]
        try:
            generated_groups = generator.generate(conversations)
            if len(generated_groups) != len(batch):
                raise RuntimeError(
                    f"Generator returned {len(generated_groups)} groups for a batch of {len(batch)} tasks"
                )
        except Exception as exc:
            LOGGER.exception("Generation failed for batch beginning at dataset index %d", batch[0][0])
            raise RuntimeError(
                "Generation stopped. Completed predictions remain on disk and the same command will resume them."
            ) from exc
        else:
            for (index, row, task_id), generations in zip(batch, generated_groups, strict=True):
                if len(generations) != samples_per_task:
                    raise RuntimeError(
                        f"Generator returned {len(generations)} samples for task {task_id}; "
                        f"expected {samples_per_task}"
                    )
                for sample_id, generation in enumerate(generations):
                    if (task_id, sample_id) in completed:
                        continue
                    result = _score_generation(
                        row=row,
                        dataset_index=index,
                        task_id=task_id,
                        sample_id=sample_id,
                        generation=generation,
                        execution_timeout=execution_timeout,
                    )
                    _append_jsonl(predictions_path, result)
                    all_results.append(result)
                    completed.add((task_id, sample_id))

        metrics = summarize_results(
            all_results,
            target_task_ids=target_task_ids,
            samples_per_task=samples_per_task,
            model_path=model_path,
            dataset_path=dataset_path,
        )
        _write_json_atomic(output_dir / "metrics.json", metrics)
        LOGGER.info(
            "Progress %d/%d complete tasks; pass@1=%.4f",
            metrics["tasks_complete"],
            metrics["tasks_target"],
            metrics.get("pass@1", 0.0),
        )

    metrics = summarize_results(
        all_results,
        target_task_ids=target_task_ids,
        samples_per_task=samples_per_task,
        model_path=model_path,
        dataset_path=dataset_path,
    )
    failures = [item for item in all_results if not bool(item.get("passed"))]
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
    _write_json_atomic(output_dir / "metrics.json", metrics)
    return metrics


def evaluate_with_execution_feedback(
    *,
    rows: Sequence[dict[str, Any]],
    generator: BatchGenerator,
    output_dir: Path,
    existing_results: list[dict[str, Any]],
    max_attempts: int,
    batch_size: int,
    model_path: str,
    dataset_path: str,
    dataset_start_index: int,
    execution_feedback_mode: str,
    feedback_max_chars: int,
    execution_timeout: float = 5.0,
) -> dict[str, Any]:
    """Generate one candidate at a time and condition repairs on the previous failed attempt."""
    predictions_path = output_dir / "predictions.jsonl"
    all_results = list(existing_results)
    indexed_rows = [(dataset_start_index + offset, row) for offset, row in enumerate(rows)]
    target_task_ids = [_task_id(row, index) for index, row in indexed_rows]

    def attempts_for(task_id: str) -> list[dict[str, Any]]:
        return _ordered_attempts(
            [item for item in all_results if str(item["task_id"]) == task_id],
            max_attempts,
        )

    pending: list[tuple[int, dict[str, Any], str]] = []
    for index, row in indexed_rows:
        task_id = _task_id(row, index)
        attempts = attempts_for(task_id)
        attempt_ids = [int(item["sample_id"]) for item in attempts]
        if attempt_ids != list(range(len(attempt_ids))):
            raise ValueError(f"Task {task_id} has non-contiguous iterative attempt ids: {attempt_ids}")
        if not _is_iterative_task_complete(attempts, max_attempts):
            pending.append((index, row, task_id))

    LOGGER.info(
        "Iterative evaluation target: %d tasks x at most %d attempts; pending tasks=%d; feedback=%s",
        len(rows),
        max_attempts,
        len(pending),
        execution_feedback_mode,
    )
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        while True:
            active: list[tuple[int, dict[str, Any], str, int, list[dict[str, str]], str | None]] = []
            for index, row, task_id in batch:
                attempts = attempts_for(task_id)
                if any(bool(item.get("passed")) for item in attempts) or len(attempts) >= max_attempts:
                    continue
                previous = attempts[-1] if attempts else None
                conversation, feedback_used = _iterative_conversation(
                    row,
                    previous,
                    feedback_mode=execution_feedback_mode,
                    feedback_max_chars=feedback_max_chars,
                )
                active.append((index, row, task_id, len(attempts), conversation, feedback_used))

            if not active:
                break

            conversations = [item[4] for item in active]
            try:
                generated_groups = generator.generate(conversations)
                if len(generated_groups) != len(active):
                    raise RuntimeError(
                        f"Generator returned {len(generated_groups)} groups for {len(active)} iterative tasks"
                    )
            except Exception as exc:
                LOGGER.exception("Iterative generation failed near dataset index %d", active[0][0])
                raise RuntimeError(
                    "Generation stopped. Completed attempts remain on disk and the same command will resume them."
                ) from exc

            for (index, row, task_id, attempt_id, _conversation, feedback_used), generations in zip(
                active, generated_groups, strict=True
            ):
                if len(generations) != 1:
                    raise RuntimeError(
                        f"Iterative generator returned {len(generations)} candidates for task {task_id}; expected 1"
                    )
                result = _score_generation(
                    row=row,
                    dataset_index=index,
                    task_id=task_id,
                    sample_id=attempt_id,
                    generation=generations[0],
                    execution_timeout=execution_timeout,
                )
                result.update(
                    generation_strategy="execution_feedback",
                    attempt_id=attempt_id,
                    attempt_number=attempt_id + 1,
                    previous_attempt_id=attempt_id - 1 if attempt_id > 0 else None,
                    execution_feedback_mode=execution_feedback_mode,
                    feedback_used=feedback_used,
                )
                _append_jsonl(predictions_path, result)
                all_results.append(result)

            metrics = summarize_iterative_results(
                all_results,
                target_task_ids=target_task_ids,
                max_attempts=max_attempts,
                model_path=model_path,
                dataset_path=dataset_path,
                execution_feedback_mode=execution_feedback_mode,
            )
            _write_json_atomic(output_dir / "metrics.json", metrics)
            LOGGER.info(
                "Iterative progress %d/%d complete tasks; pass@1=%.4f; iterative_solve_rate@%d=%.4f",
                metrics["tasks_complete"],
                metrics["tasks_target"],
                metrics.get("pass@1", 0.0),
                max_attempts,
                metrics.get(f"iterative_solve_rate@{max_attempts}", 0.0),
            )

    metrics = summarize_iterative_results(
        all_results,
        target_task_ids=target_task_ids,
        max_attempts=max_attempts,
        model_path=model_path,
        dataset_path=dataset_path,
        execution_feedback_mode=execution_feedback_mode,
    )
    failures = [item for item in all_results if not bool(item.get("passed"))]
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
    _write_json_atomic(output_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and execution-grade MBPP with a local HF model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of MBPP tasks, not rollouts.")
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Independent samples, or the maximum attempts under --generation-strategy execution_feedback.",
    )
    parser.add_argument(
        "--generation-strategy",
        choices=GENERATION_STRATEGIES,
        default="independent",
        help="independent computes standard pass@k; execution_feedback conditionally repairs failed attempts.",
    )
    parser.add_argument(
        "--execution-feedback",
        choices=EXECUTION_FEEDBACK_MODES,
        default="summary",
        help="summary hides held-out assertions; full includes raw executor output and is oracle-assisted.",
    )
    parser.add_argument(
        "--feedback-max-chars",
        type=int,
        default=1000,
        help="Maximum raw executor characters included by full feedback.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Defaults to 512 for independent generation and 4096 for execution-feedback history.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--execution-timeout", type=float, default=5.0, help="Seconds allowed per generated program.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects deterministic greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace known result files in output-dir.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if args.generation_strategy == "execution_feedback" and args.samples_per_task < 2:
        raise ValueError("execution_feedback requires --samples-per-task >= 2")
    if args.feedback_max_chars <= 0:
        raise ValueError("feedback-max-chars must be positive")
    max_prompt_tokens = args.max_prompt_tokens
    if max_prompt_tokens is None:
        max_prompt_tokens = 4096 if args.generation_strategy == "execution_feedback" else 512
    if max_prompt_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")
    if args.execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    model_path = str(args.model_path.resolve())
    dataset_path = str(args.dataset_path.resolve())
    output_dir = args.output_dir.resolve()
    generation_config = {
        "model_path": model_path,
        "model_fingerprint": _model_fingerprint(args.model_path),
        "dataset_path": dataset_path,
        "dataset_sha256": _file_sha256(args.dataset_path),
        "prompt_schema": PROMPT_SCHEMA,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "samples_per_task": args.samples_per_task,
        "generation_strategy": args.generation_strategy,
        "execution_feedback": args.execution_feedback,
        "feedback_max_chars": args.feedback_max_chars,
        "batch_size": args.batch_size,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "execution_timeout": args.execution_timeout,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    run_config = {
        **generation_config,
        "config_hash": _stable_hash(generation_config),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    existing_results = _prepare_output_dir(output_dir, run_config, args.overwrite)
    _setup_logging(output_dir)

    random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_mbpp_rows(args.dataset_path, args.start_index, args.max_samples)
    LOGGER.info("Loaded %d MBPP tasks from %s", len(rows), dataset_path)
    generator = TransformersBatchGenerator(
        model_path=args.model_path,
        samples_per_task=1 if args.generation_strategy == "execution_feedback" else args.samples_per_task,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if args.generation_strategy == "execution_feedback":
        metrics = evaluate_with_execution_feedback(
            rows=rows,
            generator=generator,
            output_dir=output_dir,
            existing_results=existing_results,
            max_attempts=args.samples_per_task,
            batch_size=args.batch_size,
            model_path=model_path,
            dataset_path=dataset_path,
            dataset_start_index=args.start_index,
            execution_feedback_mode=args.execution_feedback,
            feedback_max_chars=args.feedback_max_chars,
            execution_timeout=args.execution_timeout,
        )
    else:
        metrics = evaluate(
            rows=rows,
            generator=generator,
            output_dir=output_dir,
            existing_results=existing_results,
            samples_per_task=args.samples_per_task,
            batch_size=args.batch_size,
            model_path=model_path,
            dataset_path=dataset_path,
            dataset_start_index=args.start_index,
            execution_timeout=args.execution_timeout,
        )
    LOGGER.info("Evaluation finished: %s", json.dumps(metrics, ensure_ascii=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
