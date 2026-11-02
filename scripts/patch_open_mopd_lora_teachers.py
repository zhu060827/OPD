"""Patch pinned Open-MOPD to load frozen PEFT adapters for reward Teachers.

The patch is deliberately narrow and idempotent.  It leaves the official
multi-Teacher routing and token-reward implementation unchanged; only the
reward-model construction path gains an optional ``lora_adapter_path``.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "4809a96cf85a869106ff0ff3f37d0a51e12010ae"
MARKER = "# four-expert-lora-teacher-patch-v1"


def patch_checkout(root: Path) -> Path:
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != PINNED_COMMIT:
        raise ValueError(f"Open-MOPD commit mismatch: expected {PINNED_COMMIT}, got {actual}")

    target = root / "training/verl/verl/workers/fsdp_workers.py"
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return target

    function_anchor = """    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForCausalLM
"""
    function_start = source.find(function_anchor)
    if function_start < 0 or source.find(function_anchor, function_start + 1) >= 0:
        raise RuntimeError("Unexpected Open-MOPD reward loader function anchor")

    anchor_path = "        local_path = copy_to_local(config.model.path, use_shm=use_shm)\n"
    replacement_path = anchor_path + (
        f"        {MARKER}\n"
        "        adapter_path = config.model.get(\"lora_adapter_path\", None)\n"
        "        local_adapter_path = (\n"
        "            copy_to_local(adapter_path, use_shm=use_shm) if adapter_path else None\n"
        "        )\n"
    )
    path_index = source.find(anchor_path, function_start)
    if path_index < 0:
        raise RuntimeError("Unexpected Open-MOPD reward loader path anchor")
    source = source[:path_index] + replacement_path + source[path_index + len(anchor_path) :]

    anchor_model = """            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=model_dtype,
                attn_implementation=\"flash_attention_2\",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
"""
    replacement_model = """            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=model_dtype,
                attn_implementation=\"flash_attention_2\",
                trust_remote_code=trust_remote_code,
            )
            if local_adapter_path:
                from peft import PeftModel

                reward_module = PeftModel.from_pretrained(
                    reward_module,
                    local_adapter_path,
                    is_trainable=False,
                )
                reward_module.eval()

            apply_monkey_patch(
"""
    model_index = source.find(anchor_model, path_index)
    if model_index < 0:
        raise RuntimeError("Unexpected Open-MOPD reward loader model anchor")
    source = source[:model_index] + replacement_model + source[model_index + len(anchor_model) :]
    target.write_text(source, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-mopd-root", required=True, type=Path)
    args = parser.parse_args()
    target = patch_checkout(args.open_mopd_root.resolve())
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
