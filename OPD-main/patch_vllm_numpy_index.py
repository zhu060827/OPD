from __future__ import annotations

import argparse
import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ORIGINAL = "logit_indices = np.cumsum(num_scheduled_tokens) - 1"
PATCHED = (
    "logit_indices = torch.tensor((np.cumsum(num_scheduled_tokens) - 1).tolist(), "
    "device=hidden_states.device, dtype=torch.long)"
)


def installed_runner_path() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vllm package")
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package_dir / "v1" / "worker" / "gpu_model_runner.py"


def patch_runner(target: Path) -> str:
    target = target.expanduser().resolve()
    if not target.is_file():
        raise RuntimeError(f"Cannot find the vLLM GPU model runner: {target}")

    source = target.read_text(encoding="utf-8")
    original_count = source.count(ORIGINAL)
    patched_count = source.count(PATCHED)

    if original_count == 0 and patched_count == 1:
        return "already patched"
    if original_count != 1 or patched_count != 0:
        raise RuntimeError(
            "The installed vLLM runner does not match the supported source layout "
            f"(original_count={original_count}, patched_count={patched_count}): {target}"
        )

    patched_source = source.replace(ORIGINAL, PATCHED, 1)
    compile(patched_source, str(target), "exec")

    backup = target.with_name(f"{target.name}.opd-numpy-index.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
        backup.chmod(target.stat().st_mode)

    temporary = target.with_name(f"{target.name}.opd-numpy-index.tmp")
    temporary.write_text(patched_source, encoding="utf-8")
    temporary.chmod(target.stat().st_mode)
    os.replace(temporary, target)
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch vLLM's NumPy-array CUDA indexing for the OPD runtime."
    )
    parser.add_argument("--target", type=Path, help="Override the installed gpu_model_runner.py path.")
    args = parser.parse_args()

    target = args.target or installed_runner_path()
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = "unknown"
    result = patch_runner(target)
    print(f"vLLM NumPy index compatibility: {result} (vllm={vllm_version}, path={target})")


if __name__ == "__main__":
    main()
