from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """你是代码推理数据生成专家。
请根据编程题、正确代码和测试用例生成高质量推理说明。
要求：说明问题目标、算法及选择原因、关键执行步骤、边界情况、时间复杂度和空间复杂度；
推理必须与代码严格一致；不得修改或重新生成代码；只输出推理文本。"""


def main() -> None:
    parser = argparse.ArgumentParser(description="从原始 MBPP 生成 CoT Teacher 候选数据。")
    parser.add_argument("--dataset", required=True, help="datasets.save_to_disk 保存的 MBPP 目录")
    parser.add_argument("--model", required=True, help="Qwen3-4B 本地模型目录")
    parser.add_argument("--output", required=True, help="输出 JSONL")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as writer:
        for row in dataset:
            if written >= args.limit:
                break
            task = str(row.get("text", "")).strip()
            code = str(row.get("code", "")).strip()
            tests = list(row.get("test_list") or [])
            if not task or not code or not tests:
                continue
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"题目：\n{task}\n\n正确代码：\n{code}\n\n测试用例：\n{json.dumps(tests, ensure_ascii=False)}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            reasoning = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            result = {
                "source_id": f"mbpp_{row['task_id']}",
                "source_dataset": "MBPP",
                "source_paper": "Austin et al. (2021), Program Synthesis with Large Language Models",
                "domain": "cot",
                "language": "python",
                "task": task,
                "source_code": code,
                "target_code": code,
                "target_reasoning": reasoning,
                "tests": tests,
                "semantic_pass": True,
                "generation_model": args.model,
                "qwen3_thinking": False,
            }
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()
            written += 1
            print(f"[{written}/{args.limit}] 已生成 {result['source_id']}", flush=True)
    print(f"完成：共生成 {written} 条，文件位于 {output}")


if __name__ == "__main__":
    main()
