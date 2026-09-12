"""将 routing_labels.jsonl（974条完整证据）转换为 mt_opd_handoff 格式。"""
import json
import sys
from pathlib import Path

def convert_record(record: dict) -> dict | None:
    routing = record.get("routing", {})
    selected_expert_id = routing.get("selected_expert_id")
    pseudo_label = routing.get("pseudo_method_label")
    
    # 只保留可用于训练的样本
    if not routing.get("usable_for_training") or not selected_expert_id:
        return None
    
    selected = next(
        (item for item in record.get("expert_assessments", [])
         if item.get("expert_id") == selected_expert_id),
        None,
    )
    if selected is None:
        return None
    
    return {
        "data_source": "code_multi_expert_stage1",
        "task_id": record.get("task_id"),
        "prompt": [{"role": "user", "content": record.get("prompt", "")}],
        "ability": "code",
        "response": selected.get("candidate", {}).get("code", ""),
        "domain": pseudo_label,
        "teacher_id": selected_expert_id,
        "teacher_weights": routing.get("expert_weights", {}),
        "routing_source": routing.get("routing_source", ""),
        "routing_confidence": routing.get("margin", 0.0),
        "routing_loss_weight": routing.get("opd_sample_weight", 1.0),
        "verification_status": record.get("verification_status", "semantic_unverified"),
        "downstream_action": record.get("downstream_action", "unverified_pool"),
        "opd_training_eligible": True,
        "requires_stage2_rescore": routing.get("status") == "fallback_missing_trajectory",
        "positive_augmentation_eligible": record.get("downstream_action") == "positive_augmentation",
        "reward_model": {"style": "rule", "ground_truth": {"tests": record.get("tests", [])}},
        "extra_info": {
            "tests": record.get("tests", []),
            "original_code": record.get("original_code", ""),
            "routing_status": routing.get("status", "unknown"),
            "routing_margin": routing.get("margin", 0.0),
            "top_k": routing.get("top_k", []),
            "routing_source": routing.get("routing_source", ""),
            "formal_training_result": False,
            "verification_status": record.get("verification_status", "semantic_unverified"),
            "downstream_action": record.get("downstream_action", "unverified_pool"),
        },
    }

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    converted = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            handoff = convert_record(record)
            if handoff:
                converted.append(handoff)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 转换完成：{len(converted)} 条可用训练样本 → {output_path}")

if __name__ == "__main__":
    main()
