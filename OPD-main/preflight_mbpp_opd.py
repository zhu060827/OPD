from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RUNTIME_MODULES = (
    "torch",
    "transformers",
    "safetensors",
    "pyarrow",
    "hydra",
    "omegaconf",
    "ray",
    "vllm",
    "verl",
    "flash_attn",
)
EXPECTED_MBPP_PROMPT_SCHEMA = "mbpp_required_interfaces_v1"


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"ERROR: {message}")


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def model_shards(model_dir: Path, errors: list[str]) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        fail(f"Missing model index: {index_path}", errors)
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        fail(f"Invalid model index {index_path}: {exc}", errors)
        return []
    for shard in shards:
        shard_path = model_dir / shard
        if not shard_path.is_file():
            fail(f"Missing model shard: {shard_path}", errors)
        elif shard_path.stat().st_size == 0:
            fail(f"Empty model shard: {shard_path}", errors)
    incomplete = sorted(model_dir.glob("*.incomplete"))
    if incomplete:
        fail(f"Incomplete model files found in {model_dir}: {incomplete}", errors)
    if importlib.util.find_spec("safetensors") is not None and all((model_dir / shard).is_file() for shard in shards):
        try:
            from safetensors import safe_open

            header_keys: set[str] = set()
            for shard in shards:
                with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
                    header_keys.update(handle.keys())
            if header_keys != set(weight_map):
                fail(f"Safetensors headers do not match the model index in {model_dir}", errors)
        except Exception as exc:
            fail(f"Cannot validate safetensors headers in {model_dir}: {exc}", errors)
    return shards


def parquet_rows(path: Path, errors: list[str]) -> int:
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(path).names)
        required = {"data_source", "prompt", "reward_model", "extra_info"}
        missing = required - schema_names
        if missing:
            fail(f"{path} is missing verl columns: {sorted(missing)}", errors)
            return pq.ParquetFile(path).metadata.num_rows
        table = pq.read_table(path, columns=["prompt", "extra_info"])
        for row_index, row in enumerate(table.to_pylist()):
            extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            schema = extra_info.get("prompt_schema")
            if schema != EXPECTED_MBPP_PROMPT_SCHEMA:
                fail(
                    f"{path} row {row_index} has prompt_schema={schema!r}; expected "
                    f"{EXPECTED_MBPP_PROMPT_SCHEMA!r}. Re-run train_mbpp_opd.sh prepare.",
                    errors,
                )
                break
            entrypoints = extra_info.get("required_entrypoints") or []
            interfaces = extra_info.get("required_interfaces") or []
            prompt = row.get("prompt") or []
            prompt_text = str(prompt[0].get("content", "")) if prompt and isinstance(prompt[0], dict) else ""
            if not entrypoints or not interfaces or not all(str(interface) in prompt_text for interface in interfaces):
                fail(f"{path} row {row_index} has incomplete required-interface prompt metadata", errors)
                break
        return table.num_rows
    except Exception as exc:
        fail(f"Cannot read parquet {path}: {exc}", errors)
        return 0


def gpu_memory_mib() -> list[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return []


def system_memory_mib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight checks for MBPP Qwen3 OPD training.")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    errors: list[str] = []
    student = Path(args.student_model).expanduser().resolve()
    teacher = Path(args.teacher_model).expanduser().resolve()
    train_data = Path(args.train_data).expanduser().resolve()
    val_data = Path(args.val_data).expanduser().resolve()

    expected_batch = args.per_device_batch_size * args.gradient_accumulation_steps * args.n_gpus
    if args.global_batch_size != expected_batch:
        fail(
            "Batch mismatch: global_batch_size must equal per_device_batch_size * "
            f"gradient_accumulation_steps * n_gpus ({expected_batch})",
            errors,
        )

    for model_dir, label in ((student, "student"), (teacher, "teacher")):
        if not model_dir.is_dir():
            fail(f"Missing {label} model directory: {model_dir}", errors)
        else:
            shards = model_shards(model_dir, errors)
            print(f"{label}_model={model_dir} shards={len(shards)}")

    for data_path, label in ((train_data, "train"), (val_data, "validation")):
        if not data_path.is_file():
            fail(f"Missing {label} parquet: {data_path}", errors)
        else:
            print(f"{label}_rows={parquet_rows(data_path, errors)} path={data_path}")

    if student.is_dir() and teacher.is_dir() and importlib.util.find_spec("transformers"):
        try:
            from transformers import AutoTokenizer

            student_tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=True)
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
            if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
                fail("Student and teacher tokenizers do not share the same vocabulary", errors)
            else:
                print(f"shared_tokenizer_vocab={len(student_tokenizer)}")
        except Exception as exc:
            fail(f"Tokenizer compatibility check failed: {exc}", errors)

    if not args.skip_runtime:
        for module in RUNTIME_MODULES:
            if importlib.util.find_spec(module) is None:
                fail(f"Missing runtime module in {sys.executable}: {module}", errors)
        try:
            from packaging.version import Version

            numpy_version = Version(version("numpy"))
            if numpy_version >= Version("2.0.0"):
                fail(
                    f"Incompatible numpy={numpy_version}; this verl checkout requires numpy<2.0.0",
                    errors,
                )
        except (PackageNotFoundError, ImportError, ValueError) as exc:
            fail(f"Cannot validate the NumPy version: {exc}", errors)
        try:
            from packaging.version import Version

            transformers_version = Version(version("transformers"))
            if not (Version("4.51.0") <= transformers_version < Version("5.0.0")):
                fail(
                    f"Incompatible transformers={transformers_version}; this verl checkout requires >=4.51.0,<5.0.0",
                    errors,
                )
        except (PackageNotFoundError, ImportError, ValueError) as exc:
            fail(f"Cannot validate the transformers version: {exc}", errors)
        try:
            import torch

            if not torch.cuda.is_available():
                fail(f"CUDA is unavailable in {sys.executable} (torch={torch.__version__})", errors)
        except Exception as exc:
            fail(f"PyTorch CUDA check failed: {exc}", errors)

    gpu_memory = gpu_memory_mib()
    ram_mib = system_memory_mib()
    print(f"gpu_memory_mib={gpu_memory or 'unavailable'} system_memory_mib={ram_mib or 'unavailable'}")
    low_memory = bool(gpu_memory and max(gpu_memory) < 16 * 1024) or bool(ram_mib and ram_mib < 24 * 1024)
    if low_memory:
        message = (
            "This host is below the practical memory floor for full-parameter Qwen3-1.7B/Qwen3-4B OPD "
            "(recommended: >=16GB GPU and >=24GB RAM; 24GB GPU/32GB RAM is safer)."
        )
        if args.allow_low_memory:
            warn(message)
        else:
            fail(message + " Re-run with --allow-low-memory only if you accept likely OOM/very slow offload.", errors)

    if errors:
        raise SystemExit(f"Preflight failed with {len(errors)} error(s).")
    print("Preflight passed.")


if __name__ == "__main__":
    main()
