from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

import torch

from .io_utils import read_jsonl


def _configure_font() -> bool:
    """选择中文字体；服务器未安装中文字体时改用英文标签，避免方框乱码。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

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


def _generation_limit(tokenizer, messages, configured_limit: int) -> int:
    """代码改写输出应接近 gold 长度；保留 20% 余量和标签余量，避免固定生成 1024 token。"""
    reference = str(messages[-1].get("content", "")) if messages else ""
    reference_tokens = len(tokenizer(reference, add_special_tokens=False)["input_ids"])
    return min(configured_limit, max(64, int(reference_tokens * 1.2) + 32))


def _generate(model, tokenizer, messages, max_new_tokens: int, thinking: bool) -> tuple[str, int, int, bool]:
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_limit = _generation_limit(tokenizer, messages, max_new_tokens)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=generation_limit,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    generated_tokens = int(generated.shape[0])
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated_tokens, generation_limit, generated_tokens >= generation_limit


CODE_BLOCK = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
MARKDOWN_BLOCK = re.compile(r"```(?:python|py)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> tuple[str, bool, str]:
    """提取模型代码，同时单独报告是否遵守项目的 <code> 输出协议。"""
    stripped = text.strip()
    match = CODE_BLOCK.fullmatch(stripped)
    if match:
        return match.group(1).strip(), True, "code_tag"
    match = MARKDOWN_BLOCK.fullmatch(stripped)
    if match:
        return match.group(1).strip(), False, "markdown_fence"
    return stripped, False, "plain_text"


def _code_contract(text: str, language: str) -> tuple[bool, bool, str, str]:
    code, protocol_pass, extraction_method = _extract_code(text)
    if language.lower() != "python":
        return protocol_pass, False, code, extraction_method
    try:
        ast.parse(code)
    except SyntaxError:
        return protocol_pass, False, code, extraction_method
    return protocol_pass, True, code, extraction_method


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_manifest_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(path.name + ".resume.json")


def _run_spec(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": "base-lora-generation-v3",
        "base_model": str(Path(args.base_model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "test_file": str(Path(args.test_file).resolve()),
        "test_file_sha256": _file_sha256(args.test_file),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
    }


def _prepare_resume(output: str | Path, spec: dict[str, object], restart: bool = False) -> tuple[list[dict], set[str]]:
    """创建或校验断点信息，并返回已完整写入的样本。"""
    output_path = Path(output)
    manifest_path = _resume_manifest_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if restart:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if output_path.exists() and not manifest_path.exists():
        raise ValueError(
            f"输出已存在但缺少断点 manifest：{output_path}。"
            "若要放弃旧结果并重新生成，请显式使用 --restart。"
        )
    if manifest_path.exists():
        existing_spec = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_spec != spec:
            raise ValueError(
                f"断点参数或测试集已变化，拒绝混合续跑：{manifest_path}。"
                "请更换 --output，或确认后使用 --restart。"
            )
    else:
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)

    results = list(read_jsonl(output_path)) if output_path.exists() else []
    required_fields = {"source_id", "base_output", "lora_output", "reference_output"}
    for line_number, row in enumerate(results, 1):
        missing = sorted(required_fields - set(row))
        if missing:
            raise ValueError(
                f"断点输出第 {line_number} 行不是完整样本，缺少：{', '.join(missing)}"
            )
    source_ids = [str(row.get("source_id", "")) for row in results]
    if not all(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError(f"断点输出包含空或重复 source_id：{output_path}")
    return results, set(source_ids)


def _append_jsonl_durable(path: str | Path, row: dict) -> None:
    """每条样本单独落盘，SSH 断开时最多损失当前未完成样本。"""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _result_row(row: dict, base_result: tuple[str, int, int, bool], lora_result: tuple[str, int, int, bool], thinking: bool) -> dict:
    base_output, base_tokens, base_limit, base_limit_reached = base_result
    lora_output, lora_tokens, lora_limit, lora_limit_reached = lora_result
    language = str(row.get("language", "python"))
    base_tag, base_parse, base_code, base_method = _code_contract(base_output, language)
    lora_tag, lora_parse, lora_code, lora_method = _code_contract(lora_output, language)
    reference_output = row["messages"][-1]["content"]
    reference_code, _, _ = _extract_code(reference_output)
    return {"source_id": row["source_id"], "problem_id": row.get("problem_id", row["source_id"]), "domain": row["domain"], "language": language, "base_output": base_output, "lora_output": lora_output, "reference_output": reference_output, "base_extracted_code": base_code, "lora_extracted_code": lora_code, "reference_extracted_code": reference_code, "base_extraction_method": base_method, "lora_extraction_method": lora_method, "base_raw_chars": len(base_output), "lora_raw_chars": len(lora_output), "base_chars": len(base_code), "lora_chars": len(lora_code), "base_generated_tokens": base_tokens, "lora_generated_tokens": lora_tokens, "base_generation_limit": base_limit, "lora_generation_limit": lora_limit, "base_max_tokens_reached": base_limit_reached, "lora_max_tokens_reached": lora_limit_reached, "base_suspected_truncation": base_limit_reached and not base_tag, "lora_suspected_truncation": lora_limit_reached and not lora_tag, "base_exact_code_match": base_code.rstrip() == reference_code.rstrip(), "lora_exact_code_match": lora_code.rstrip() == reference_code.rstrip(), "base_code_tag_complete": base_tag, "lora_code_tag_complete": lora_tag, "base_protocol_pass": base_tag, "lora_protocol_pass": lora_tag, "base_parse_pass": base_parse, "lora_parse_pass": lora_parse, "qwen3_thinking": thinking}


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser(description="比较 Qwen3 基座模型与领域 LoRA 在同一测试集上的输出。")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--plot", help="输出基座/LoRA 对比图 PNG")
    parser.add_argument("--restart", action="store_true", help="显式丢弃该 output 的旧断点并从头生成")
    args = parser.parse_args()
    has_chinese_font = _configure_font()
    rows = list(read_jsonl(args.test_file))[: args.limit]
    expected_ids = [str(row.get("source_id", "")) for row in rows]
    if not all(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("测试集在 limit 范围内存在空或重复 source_id")
    results, completed_ids = _prepare_resume(args.output, _run_spec(args), args.restart)
    unexpected = completed_ids - set(expected_ids)
    if unexpected:
        raise ValueError(f"断点包含当前测试集之外的 source_id：{sorted(unexpected)[:5]}")
    pending_rows = [row for row in rows if row["source_id"] not in completed_ids]
    print(f"断点检查完成：已完成 {len(completed_ids)}/{len(rows)}，待生成 {len(pending_rows)}。", flush=True)
    if not pending_rows:
        print("所有样本已在断点输出中，跳过模型加载。", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, quantization_config=quantization, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base.eval()
        adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
        adapted.eval()
        for row in pending_rows:
            index = expected_ids.index(row["source_id"]) + 1
            limit = _generation_limit(tokenizer, row["messages"], args.max_new_tokens)
            print(f"[基座生成 {index}/{len(rows)}] source_id={row['source_id']} max_new_tokens={limit}", flush=True)
            with adapted.disable_adapter():
                base_result = _generate(adapted, tokenizer, row["messages"], args.max_new_tokens, args.thinking)
            print(f"[LoRA 生成 {index}/{len(rows)}] source_id={row['source_id']} max_new_tokens={limit}", flush=True)
            lora_result = _generate(adapted, tokenizer, row["messages"], args.max_new_tokens, args.thinking)
            result = _result_row(row, base_result, lora_result, args.thinking)
            _append_jsonl_durable(args.output, result)
            results.append(result)
            completed_ids.add(row["source_id"])
            print(f"[已落盘 {len(completed_ids)}/{len(rows)}] source_id={row['source_id']}", flush=True)
    order = {source_id: index for index, source_id in enumerate(expected_ids)}
    results.sort(key=lambda row: order[row["source_id"]])
    count = len(results)
    mapping_path = None
    if args.plot and results:
        import matplotlib.pyplot as plt

        labels = [f"S{index}" for index in range(1, len(results) + 1)]
        id_map = [{"sample": label, "source_id": row["source_id"]} for label, row in zip(labels, results)]
        mapping_path = str(Path(args.plot).with_name(Path(args.plot).stem + "_sample_map.json"))
        Path(mapping_path).write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
        positions = list(range(len(labels)))
        figure, axes = plt.subplots(4, 1, figsize=(max(8, len(labels) * 1.2), 14), constrained_layout=True)
        width = 0.38
        base_label, lora_label = ("基座模型", "LoRA Teacher") if has_chinese_font else ("Base model", "LoRA Teacher")
        length_title, length_ylabel = (("抽取后代码长度对比（字符数）", "字符数") if has_chinese_font else ("Extracted code length (characters)", "Characters"))
        parse_title, parse_ylabel = (("抽取后 Python 语法解析", "通过（0/1）") if has_chinese_font else ("Python parse after extraction", "Pass (0/1)"))
        protocol_title = "<code> 输出协议遵循" if has_chinese_font else "<code> protocol compliance"
        exact_title = "Formatting 主指标：代码精确匹配" if has_chinese_font else "Formatting primary metric: exact code match"
        axes[0].bar([p - width / 2 for p in positions], [row["base_chars"] for row in results], width, label=base_label)
        axes[0].bar([p + width / 2 for p in positions], [row["lora_chars"] for row in results], width, label=lora_label)
        axes[0].set_title(length_title)
        axes[0].set_ylabel(length_ylabel)
        axes[0].set_xticks(positions, labels, rotation=45, ha="right")
        axes[0].legend()
        axes[1].bar([p - width / 2 for p in positions], [int(row["base_code_tag_complete"]) for row in results], width, label=base_label)
        axes[1].bar([p + width / 2 for p in positions], [int(row["lora_code_tag_complete"]) for row in results], width, label=lora_label)
        axes[1].set_title(protocol_title)
        axes[1].set_ylabel(parse_ylabel)
        axes[1].set_xticks(positions, labels, rotation=45, ha="right")
        axes[1].set_ylim(0, 1.15)
        axes[1].legend()
        axes[2].bar([p - width / 2 for p in positions], [int(row["base_exact_code_match"]) for row in results], width, label=base_label)
        axes[2].bar([p + width / 2 for p in positions], [int(row["lora_exact_code_match"]) for row in results], width, label=lora_label)
        axes[2].set_title(exact_title)
        axes[2].set_ylabel(parse_ylabel)
        axes[2].set_xticks(positions, labels, rotation=45, ha="right")
        axes[2].set_ylim(0, 1.15)
        axes[2].legend()
        axes[3].bar([p - width / 2 for p in positions], [int(row["base_parse_pass"]) for row in results], width, label=base_label)
        axes[3].bar([p + width / 2 for p in positions], [int(row["lora_parse_pass"]) for row in results], width, label=lora_label)
        axes[3].set_title(parse_title)
        axes[3].set_ylabel(parse_ylabel)
        axes[3].set_xticks(positions, labels, rotation=45, ha="right")
        axes[3].set_ylim(0, 1.15)
        axes[3].legend()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=160)
        plt.close(figure)
    payload = {"完成": True, "样本数": count, "输出": str(Path(args.output).resolve())}
    if args.plot and mapping_path:
        payload["图表"] = str(Path(args.plot).resolve())
        payload["样本编号映射"] = str(Path(mapping_path).resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
