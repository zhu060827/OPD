"""Validated five-Teacher, no-SFT launcher for the project Open-MOPD backend.

This module does not emulate MT-OPD.  It validates five real local Teacher
checkpoints and delegates execution to a pinned checkout of Open-MOPD's
``scripts/local/mt_opd.sh`` entry point. Domain balancing is wired through
real verl sampler and trainer extension points rather than unused ``mt_opd``
Hydra placeholders.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence


EXPECTED_DOMAINS = ("cot", "style", "ast", "variable", "control_flow")
PINNED_OPEN_MOPD_COMMIT = "4809a96cf85a869106ff0ff3f37d0a51e12010ae"


@dataclass(frozen=True)
class ExpertTeacher:
    domain: str
    teacher_path: Path


@dataclass(frozen=True)
class Stage2OpenMOPDConfig:
    open_mopd_root: Path
    expected_open_mopd_commit: str
    student_path: Path
    teachers: tuple[ExpertTeacher, ...]
    train_file: Path
    val_file: Path
    output_dir: Path
    gpus: int
    nodes: int
    label_policy: str
    target_gradient_shares: tuple[float, ...]
    gap_following_alpha: float
    reward_refresh: bool
    conflict_policy: str
    extra_overrides: tuple[str, ...]
    use_routing_loss_weight: bool
    use_sft: bool
    token_balance_min_weight: float
    token_balance_max_weight: float
    reward_ema_decay: float
    prompt_sampling_balance: bool
    token_share_balance: bool
    reward_scale_balance: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Stage2OpenMOPDConfig":
        backend = raw.get("open_mopd_backend", {})
        model = raw.get("models", {})
        data = raw.get("data", {})
        runtime = raw.get("runtime", {})
        method = raw.get("method", {})
        teachers = tuple(
            ExpertTeacher(str(item["domain"]), Path(item["teacher_path"]))
            for item in model.get("teachers", [])
        )
        target = method.get("target_gradient_shares", {})
        config = cls(
            open_mopd_root=Path(backend.get("root", "third_party/Open-MOPD")),
            expected_open_mopd_commit=str(
                backend.get("expected_commit", PINNED_OPEN_MOPD_COMMIT)
            ),
            student_path=Path(model.get("student_path", "")),
            teachers=teachers,
            train_file=Path(data.get("train_file", "")),
            val_file=Path(data.get("val_file", "")),
            output_dir=Path(raw.get("output_dir", "outputs/stage2_open_mopd")),
            gpus=int(runtime.get("gpus", 1)),
            nodes=int(runtime.get("nodes", 1)),
            label_policy=str(method.get("label_policy", "recorded_method")),
            target_gradient_shares=tuple(
                float(target.get(domain, 0.0)) for domain in EXPECTED_DOMAINS
            ),
            gap_following_alpha=float(method.get("gap_following_alpha", 1.0)),
            reward_refresh=bool(method.get("reward_refresh", True)),
            conflict_policy=str(method.get("conflict_policy", "none")),
            extra_overrides=tuple(str(value) for value in raw.get("extra_overrides", [])),
            use_routing_loss_weight=bool(method.get("use_routing_loss_weight", False)),
            use_sft=bool(method.get("use_sft", False)),
            token_balance_min_weight=float(method.get("token_balance_min_weight", 0.05)),
            token_balance_max_weight=float(method.get("token_balance_max_weight", 20.0)),
            reward_ema_decay=float(method.get("reward_ema_decay", 0.9)),
            prompt_sampling_balance=bool(method.get("prompt_sampling_balance", True)),
            token_share_balance=bool(method.get("token_share_balance", True)),
            reward_scale_balance=bool(method.get("reward_scale_balance", True)),
        )
        config.validate_static()
        return config

    def validate_static(self) -> None:
        domains = tuple(item.domain for item in self.teachers)
        if domains != EXPECTED_DOMAINS:
            raise ValueError(
                "Exactly five Teachers are required in canonical order: "
                + ",".join(EXPECTED_DOMAINS)
            )
        paths = [str(item.teacher_path) for item in self.teachers]
        if len(set(paths)) != len(paths):
            raise ValueError("The five Teacher checkpoint paths must be distinct")
        if self.label_policy not in {
            "recorded_method",
            "stage1_handoff",
            "pseudo_router_ablation",
        }:
            raise ValueError(
                "label_policy must be recorded_method, stage1_handoff, "
                "or pseudo_router_ablation"
            )
        if self.gpus < 1 or self.nodes != 1:
            raise ValueError("The local launcher requires gpus >= 1 and nodes == 1")
        if len(self.target_gradient_shares) != len(EXPECTED_DOMAINS):
            raise ValueError("A target gradient share is required for every method")
        if any(value < 0 for value in self.target_gradient_shares):
            raise ValueError("Target gradient shares must be non-negative")
        if sum(self.target_gradient_shares) <= 0:
            raise ValueError("Target gradient shares must have positive mass")
        if not 0.0 <= self.gap_following_alpha <= 2.0:
            raise ValueError("gap_following_alpha must be in [0, 2]")
        if self.conflict_policy not in {"none", "mask", "consensus"}:
            raise ValueError("conflict_policy must be none, mask, or consensus")
        if self.use_sft:
            raise ValueError("This project is OPD-only: method.use_sft must be false")
        if not 0.0 < self.token_balance_min_weight <= self.token_balance_max_weight:
            raise ValueError("token balance requires 0 < min_weight <= max_weight")
        if not 0.0 <= self.reward_ema_decay < 1.0:
            raise ValueError("reward_ema_decay must be in [0, 1)")


def load_stage2_config(path: str | Path) -> Stage2OpenMOPDConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage-2 config must be a JSON object")
    if "extends" in raw:
        parent_path = Path(str(raw["extends"]))
        if not parent_path.is_absolute():
            parent_path = source.parent / parent_path
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if not isinstance(parent, dict):
            raise ValueError("Extended Stage-2 config must be a JSON object")
        merged = dict(parent)
        merged["method"] = {
            **dict(parent.get("method", {})),
            **dict(raw.get("method_overrides", {})),
        }
        for key, value in raw.items():
            if key not in {"extends", "method_overrides", "description"}:
                merged[key] = value
        raw = merged
    return Stage2OpenMOPDConfig.from_dict(raw)


def validate_real_run(config: Stage2OpenMOPDConfig) -> dict[str, Any]:
    launcher = config.open_mopd_root / "scripts" / "local" / "mt_opd.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"Official Open-MOPD launcher not found: {launcher}")
    _validate_open_mopd_commit(config)
    _validate_checkpoint(config.student_path, "Student")
    for teacher in config.teachers:
        _validate_checkpoint(teacher.teacher_path, f"Teacher[{teacher.domain}]")
    _validate_tokenizer_family(
        [config.student_path, *(teacher.teacher_path for teacher in config.teachers)]
    )
    for label, path in (("training data", config.train_file), ("validation data", config.val_file)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    train_counts = inspect_dataset_domains(config.train_file)
    val_counts = inspect_dataset_domains(config.val_file)
    _validate_domain_counts(train_counts, "training")
    if set(val_counts) - set(EXPECTED_DOMAINS):
        raise ValueError(f"Unknown validation domains: {sorted(set(val_counts) - set(EXPECTED_DOMAINS))}")
    handoff_report = None
    if config.label_policy == "stage1_handoff":
        handoff_report = validate_stage1_handoff(config.train_file)
    return {
        "open_mopd_launcher": str(launcher),
        "teacher_count": len(config.teachers),
        "teacher_domains": list(EXPECTED_DOMAINS),
        "training_domain_counts": dict(train_counts),
        "validation_domain_counts": dict(val_counts),
        "label_policy": config.label_policy,
        "stage1_handoff": handoff_report,
    }


def build_open_mopd_command(
    config: Stage2OpenMOPDConfig, execute: bool = False
) -> list[str]:
    launcher = config.open_mopd_root / "scripts" / "local" / "mt_opd.sh"
    domains = ",".join(EXPECTED_DOMAINS)
    shares = _hydra_list(config.target_gradient_shares)
    quoted_domains = "[" + ",".join(EXPECTED_DOMAINS) + "]"
    command = [
        "bash",
        str(launcher),
        "--model",
        str(config.student_path),
    ]
    for teacher in config.teachers:
        command.extend(["--teacher", str(teacher.teacher_path)])
    command.extend(
        [
            "--domains",
            domains,
            "--train",
            str(config.train_file),
            "--val",
            str(config.val_file),
            "--output",
            str(config.output_dir),
            "--gpus",
            str(config.gpus),
            "--nodes",
            str(config.nodes),
            "--extra",
            "+mt_opd.domain_weighting=domain_routing",
            "--extra",
            f"+algorithm.domain_balance.enabled={str(config.token_share_balance or config.reward_scale_balance).lower()}",
            "--extra",
            f"+algorithm.domain_balance.domains={quoted_domains}",
            "--extra",
            f"+algorithm.domain_balance.target_shares={shares}",
            "--extra",
            f"+algorithm.domain_balance.min_weight={config.token_balance_min_weight:g}",
            "--extra",
            f"+algorithm.domain_balance.max_weight={config.token_balance_max_weight:g}",
            "--extra",
            f"+algorithm.domain_balance.reward_scale_alpha={config.gap_following_alpha:g}",
            "--extra",
            f"+algorithm.domain_balance.reward_ema_decay={config.reward_ema_decay:g}",
            "--extra",
            f"+algorithm.domain_balance.token_share_enabled={str(config.token_share_balance).lower()}",
            "--extra",
            f"+algorithm.domain_balance.reward_scale_enabled={str(config.reward_scale_balance).lower()}",
            "--extra",
            "actor_rollout_ref.actor.opd_refresh_advantage="
            + str(config.reward_refresh).lower(),
        ]
    )
    if config.prompt_sampling_balance:
        command.extend(
            [
                "--extra",
                "data.shuffle=false",
                "--extra",
                "data.dataloader_num_workers=0",
                "--extra",
                "data.sampler.class_path=pkg://verl.experimental.dataset.domain_balanced_sampler",
                "--extra",
                "data.sampler.class_name=DomainBalancedSampler",
                "--extra",
                "+data.sampler.domain_key=domain",
                "--extra",
                f"+data.sampler.domains={quoted_domains}",
                "--extra",
                f"+data.sampler.target_shares={shares}",
            ]
        )
    else:
        command.extend(["--extra", "data.shuffle=true"])
    for override in config.extra_overrides:
        command.extend(["--extra", override])
    if config.use_routing_loss_weight:
        raise ValueError(
            "routing confidence weighting is intentionally disabled for the first "
            "formal experiment; domain token balancing is the only loss reweighting"
        )
    if execute:
        command.append("--run")
    return command


def inspect_dataset_domains(path: str | Path) -> Counter[str]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        records = _read_json_records(source)
        return Counter(_domain_of(record) for record in records)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to validate Parquet domain labels"
            ) from exc
        table = pq.read_table(source, columns=["domain"])
        return Counter(str(value) for value in table.column("domain").to_pylist())
    raise ValueError(f"Unsupported dataset format: {source}")


def validate_stage1_handoff(path: str | Path) -> dict[str, Any]:
    """Validate the explicit Stage-1-to-Open-MOPD routing contract.

    Verification status controls augmentation handling, while routing fields
    control which frozen Teacher scores the next on-policy Student rollout.
    They are intentionally validated as independent axes.
    """

    source = Path(path)
    records = _read_handoff_records(source)
    verification_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    expected_experts = {f"expert_{domain}" for domain in EXPECTED_DOMAINS}
    expected_actions = {
        "semantic_pass": "positive_augmentation",
        "semantic_fail": "repair_or_negative",
        "semantic_unverified": "unverified_pool",
    }
    for index, record in enumerate(records, start=1):
        prompt = record.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(prompt[0], dict)
            or not str(prompt[0].get("content", "")).strip()
        ):
            raise ValueError(f"Stage-1 record {index} has an empty or invalid prompt")
        domain = _domain_of(record)
        teacher_id = record.get("teacher_id")
        if teacher_id != f"expert_{domain}":
            raise ValueError(
                f"Stage-1 record {index} has inconsistent domain/teacher_id: "
                f"{domain!r}/{teacher_id!r}"
            )
        weights = record.get("teacher_weights")
        if not isinstance(weights, dict) or set(weights) != expected_experts:
            raise ValueError(f"Stage-1 record {index} has an invalid Teacher weight schema")
        numeric_weights = {key: float(value) for key, value in weights.items()}
        if not math.isclose(sum(numeric_weights.values()), 1.0, abs_tol=1e-8):
            raise ValueError(f"Stage-1 record {index} Teacher weights must sum to one")
        if not math.isclose(numeric_weights[teacher_id], 1.0, abs_tol=1e-8) or any(
            not math.isclose(value, 0.0, abs_tol=1e-8)
            for key, value in numeric_weights.items()
            if key != teacher_id
        ):
            raise ValueError(f"Stage-1 record {index} must use a one-hot Teacher route")
        if not isinstance(record.get("routing_source"), str) or not record["routing_source"]:
            raise ValueError(f"Stage-1 record {index} is missing routing_source")
        confidence = float(record.get("routing_confidence", float("nan")))
        if not math.isfinite(confidence) or confidence < 0.0:
            raise ValueError(f"Stage-1 record {index} has invalid routing_confidence")
        loss_weight = float(record.get("routing_loss_weight", float("nan")))
        if not math.isfinite(loss_weight) or not 0.0 < loss_weight <= 1.0:
            raise ValueError(f"Stage-1 record {index} has invalid routing_loss_weight")
        verification = record.get("verification_status")
        action = record.get("downstream_action")
        if verification not in expected_actions:
            raise ValueError(f"Stage-1 record {index} has invalid verification_status")
        if action != expected_actions[verification]:
            raise ValueError(
                f"Stage-1 record {index} has inconsistent verification/action fields"
            )
        if record.get("opd_training_eligible") is not True:
            raise ValueError(f"Stage-1 record {index} is not OPD-training eligible")
        verification_counts[verification] += 1
        action_counts[action] += 1
    return {
        "record_count": len(records),
        "verification_status_distribution": dict(sorted(verification_counts.items())),
        "downstream_action_distribution": dict(sorted(action_counts.items())),
    }


def convert_stage1_handoff_to_parquet(
    input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Validate Stage-1 JSONL and write the Parquet expected by Open-MOPD."""

    source = Path(input_path)
    destination = Path(output_path)
    if source.suffix.lower() not in {".jsonl", ".json"}:
        raise ValueError("Stage-1 conversion input must be JSON or JSONL")
    report = validate_stage1_handoff(source)
    records = _read_json_records(source)
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to create Open-MOPD Parquet data") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), destination)
    validate_stage1_handoff(destination)
    return {**report, "output_parquet": str(destination)}


def _read_handoff_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        return _read_json_records(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to validate Parquet handoff data") from exc
        records = pq.read_table(path).to_pylist()
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"Dataset must contain records: {path}")
        return records
    raise ValueError(f"Unsupported Stage-1 handoff format: {path}")


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Dataset must contain JSON objects: {path}")
    return records


def _domain_of(record: dict[str, Any]) -> str:
    domain = record.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("Every Stage-2 record must contain a non-empty domain label")
    return domain


def _validate_domain_counts(counts: Counter[str], split: str) -> None:
    unknown = set(counts) - set(EXPECTED_DOMAINS)
    missing = set(EXPECTED_DOMAINS) - set(counts)
    if unknown:
        raise ValueError(f"Unknown {split} domains: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing {split} domains: {sorted(missing)}")


def _validate_checkpoint(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint directory not found: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label} is missing config.json: {path}")
    weight_patterns = ("*.safetensors", "pytorch_model*.bin", "*.safetensors.index.json")
    if not any(any(path.glob(pattern)) for pattern in weight_patterns):
        raise FileNotFoundError(f"{label} has no model weight files: {path}")


def _validate_tokenizer_family(paths: Sequence[Path]) -> None:
    configs = [json.loads((path / "config.json").read_text(encoding="utf-8")) for path in paths]
    signatures = {(item.get("model_type"), item.get("vocab_size")) for item in configs}
    if len(signatures) != 1:
        raise ValueError("Student and all Teachers must share model_type and vocab_size")
    fingerprints = [_tokenizer_fingerprint(path) for path in paths]
    present = [value for value in fingerprints if value is not None]
    if present and len(present) != len(fingerprints):
        raise ValueError("Tokenizer artifacts must be present for every model or none")
    if len(set(present)) > 1:
        raise ValueError("Student and all Teachers must use identical tokenizer artifacts")


def _tokenizer_fingerprint(path: Path) -> str | None:
    names = ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")
    files = [path / name for name in names if (path / name).is_file()]
    if not files:
        return None
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _validate_open_mopd_commit(config: Stage2OpenMOPDConfig) -> None:
    if not config.expected_open_mopd_commit:
        return
    result = subprocess.run(
        ["git", "-C", str(config.open_mopd_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != config.expected_open_mopd_commit:
        raise ValueError(
            f"Open-MOPD commit mismatch: expected {config.expected_open_mopd_commit}, got {actual}"
        )


def _hydra_list(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:g}" for value in values) + "]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", action="store_true", help="Execute after strict preflight")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_stage2_config(args.config)
    report = validate_real_run(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return 0
    command = build_open_mopd_command(config, execute=args.run)
    print("Open-MOPD command:")
    print(" ".join(command))
    if args.run:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
