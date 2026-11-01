"""Merge one teacher_training LoRA adapter into the Qwen base model."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="auto", device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
