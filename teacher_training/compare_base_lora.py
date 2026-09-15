from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .io_utils import read_jsonl, write_jsonl


def _configure_font() -> bool:
    """选择中文字体；服务器未安装中文字体时改用英文标签，避免方框乱码。"""
    candidates = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


def _generate(model, tokenizer, messages, max_new_tokens: int, thinking: bool) -> str:
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


REASONING_PATTERNS = {
    "algorithm": re.compile(r"算法|algorithm|approach|method", re.IGNORECASE),
    "steps": re.compile(r"步骤|第一|其次|最后|step|first|then|finally", re.IGNORECASE),
    "complexity": re.compile(r"复杂度|time complexity|space complexity|\bO\s*\(", re.IGNORECASE),
    "boundaries": re.compile(r"边界|特殊情况|空输入|edge case|boundary|empty input", re.IGNORECASE),
}


def _reasoning_evidence(text: str) -> dict[str, bool]:
    return {name: bool(pattern.search(text)) for name, pattern in REASONING_PATTERNS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 Qwen3 基座模型与领域 LoRA 在同一测试集上的输出。")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--plot", help="输出基座/LoRA 对比图 PNG")
    args = parser.parse_args()
    has_chinese_font = _configure_font()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, quantization_config=quantization, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base.eval()
    rows = list(read_jsonl(args.test_file))[: args.limit]
    base_outputs = [_generate(base, tokenizer, row["messages"], args.max_new_tokens, args.thinking) for row in rows]
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    adapted.eval()
    results = []
    for row, base_output in zip(rows, base_outputs):
        lora_output = _generate(adapted, tokenizer, row["messages"], args.max_new_tokens, args.thinking)
        base_evidence = _reasoning_evidence(base_output)
        lora_evidence = _reasoning_evidence(lora_output)
        results.append({"source_id": row["source_id"], "domain": row["domain"], "base_output": base_output, "lora_output": lora_output, "reference_output": row["messages"][-1]["content"], "base_chars": len(base_output), "lora_chars": len(lora_output), "base_reasoning_markers": sum(base_evidence.values()), "lora_reasoning_markers": sum(lora_evidence.values()), "base_reasoning_evidence": base_evidence, "lora_reasoning_evidence": lora_evidence, "qwen3_thinking": args.thinking})
    count = write_jsonl(args.output, results)
    if args.plot and results:
        labels = [row["source_id"] for row in results]
        positions = list(range(len(labels)))
        figure, axes = plt.subplots(2, 1, figsize=(max(8, len(labels) * 1.5), 8), constrained_layout=True)
        width = 0.38
        base_label, lora_label = ("基座模型", "LoRA Teacher") if has_chinese_font else ("Base model", "LoRA Teacher")
        length_title, length_ylabel = (("输出长度对比（字符数）", "字符数") if has_chinese_font else ("Output length comparison (characters)", "Characters"))
        marker_title, marker_ylabel = (("推理要素覆盖对比（算法/步骤/复杂度/边界）", "覆盖项数量（0-4）") if has_chinese_font else ("Reasoning-element coverage (algorithm/steps/complexity/boundaries)", "Covered items (0-4)"))
        axes[0].bar([p - width / 2 for p in positions], [row["base_chars"] for row in results], width, label=base_label)
        axes[0].bar([p + width / 2 for p in positions], [row["lora_chars"] for row in results], width, label=lora_label)
        axes[0].set_title(length_title)
        axes[0].set_ylabel(length_ylabel)
        axes[0].set_xticks(positions, labels, rotation=45, ha="right")
        axes[0].legend()
        axes[1].bar([p - width / 2 for p in positions], [row["base_reasoning_markers"] for row in results], width, label=base_label)
        axes[1].bar([p + width / 2 for p in positions], [row["lora_reasoning_markers"] for row in results], width, label=lora_label)
        axes[1].set_title(marker_title)
        axes[1].set_ylabel(marker_ylabel)
        axes[1].set_xticks(positions, labels, rotation=45, ha="right")
        axes[1].set_ylim(0, 4.5)
        axes[1].legend()
        if not any(row["base_reasoning_markers"] or row["lora_reasoning_markers"] for row in results):
            message = "未检测到中英文推理关键词，请查看 JSONL 原始输出" if has_chinese_font else "No Chinese/English reasoning markers detected; inspect the JSONL outputs"
            axes[1].text(0.5, 0.5, message, transform=axes[1].transAxes, ha="center", va="center")
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=160)
        plt.close(figure)
    payload = {"完成": True, "样本数": count, "输出": str(Path(args.output).resolve())}
    if args.plot:
        payload["图表"] = str(Path(args.plot).resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
