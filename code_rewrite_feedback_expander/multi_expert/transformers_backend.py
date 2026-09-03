from __future__ import annotations

import gc
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..attribution import structural_token_weights
from ..llm import build_rewrite_prompt
from ..models import CodeRecord, RewriteCandidate, TokenDistribution
from .base import ExpertTrajectoryScorer
from .config import ExpertConfig, Stage1Config


class LocalTransformersMultiExpertScorer(ExpertTrajectoryScorer):
    def __init__(
        self,
        student_model_path: str,
        teacher_model_paths: Dict[str, str],
        top_k: int = 16,
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.top_k = top_k
        self.device = device

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quant_config = None

        model_kwargs = {
            "trust_remote_code": True,
            "quantization_config": quant_config if load_in_4bit else None,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        # 加载 Student（1.5B）
        print(f"加载 Student: {student_model_path}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_path, trust_remote_code=True
        )
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_path, **model_kwargs
        ).eval()
        for param in self.student.parameters():
            param.requires_grad_(False)
        torch.cuda.empty_cache()
        gc.collect()

        # 独立加载每个 Teacher（不同模型）
        self.teachers = {}
        self.teacher_tokenizers = {}
        for expert_id, model_path in teacher_model_paths.items():
            print(f"加载 Teacher: {expert_id} -> {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            teacher = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            ).eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            self.teachers[expert_id] = teacher
            self.teacher_tokenizers[expert_id] = tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"✅ 加载完成: Student + {len(self.teachers)} 个独立 Teacher")
        print(f"   显存已分配 {allocated:.1f}GB, 预留 {reserved:.1f}GB")

    def get_shared_model_and_tokenizer(self) -> Tuple:
        """供生成器复用 Student 模型"""
        return self.student, self.student_tokenizer

    def score(self, expert, record, candidate):
        teacher = self.teachers.get(expert.expert_id)
        if teacher is None:
            return self._mock_score(candidate)
        return self._compute_score(teacher, expert, record, candidate)

    def _compute_score(self, teacher, expert, record, candidate):
        prefix = build_rewrite_prompt(record, expert.strategy, feedback="") + "\nCandidate code:\n"
        
        prefix_ids = self.student_tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"]
        response_ids = self.student_tokenizer(candidate.code, add_special_tokens=False, return_tensors="pt")["input_ids"]

        student_device = next(self.student.parameters()).device
        student_input = torch.cat([prefix_ids, response_ids], dim=1).to(student_device)

        with torch.inference_mode():
            student_logits = self.student(
                input_ids=student_input,
                attention_mask=torch.ones_like(student_input),
            ).logits
            teacher_device = next(teacher.parameters()).device
            teacher_input = student_input.to(teacher_device)
            teacher_logits = teacher(
                input_ids=teacher_input,
                attention_mask=torch.ones_like(teacher_input),
            ).logits

        start = prefix_ids.shape[1] - 1
        length = response_ids.shape[1]
        student_logits = student_logits[:, start:start + length].float().cpu()
        teacher_logits = teacher_logits[:, start:start + length].float().cpu()
        actual_ids = response_ids[0].cpu()
        lexical_weights = structural_token_weights(candidate.code, candidate.reasoning)

        profiles: List[TokenDistribution] = []
        for index in range(length):
            student_lp = torch.log_softmax(student_logits[0, index], dim=-1)
            teacher_lp = torch.log_softmax(teacher_logits[0, index], dim=-1)
            k = min(self.top_k, student_lp.numel())
            student_top = torch.topk(student_lp, k=k).indices
            teacher_top = torch.topk(teacher_lp, k=k).indices
            union_ids = torch.unique(torch.cat([student_top, teacher_top])).tolist()
            generated_id = int(actual_ids[index])
            aspect_index = (
                min(len(lexical_weights) - 1, int(index * len(lexical_weights) / max(length, 1)))
                if lexical_weights else 0
            )
            profiles.append(
                TokenDistribution(
                    token=self.student_tokenizer.decode([generated_id]),
                    student_logprobs={f"id:{token_id}": float(student_lp[token_id]) for token_id in union_ids},
                    teacher_logprobs={f"id:{token_id}": float(teacher_lp[token_id]) for token_id in union_ids},
                    student_token_logprob=float(student_lp[generated_id]),
                    teacher_token_logprob=float(teacher_lp[generated_id]),
                    aspect_weights=lexical_weights[aspect_index] if lexical_weights else {"ast": 1.0},
                    attribution_source="ast_cfg_def_use_alignment",
                )
            )
        return profiles

    def _mock_score(self, candidate):
        return [
            TokenDistribution(
                token="return",
                student_logprobs={"return": -0.8, "pass": -1.2},
                teacher_logprobs={"return": -0.8, "pass": -1.2},
                student_token_logprob=-0.8,
                teacher_token_logprob=-0.8,
                aspect_weights={"style": 1.0},
            )
        ]


def create_trajectory_scorer(config: Stage1Config) -> LocalTransformersMultiExpertScorer:
    student_model = os.getenv("STAGE1_STUDENT_MODEL")
    if not student_model:
        raise ValueError("Set STAGE1_STUDENT_MODEL")

    teacher_paths: Dict[str, str] = {}
    for expert in config.enabled_experts:
        env_name = expert.scoring.get("teacher_model_env")
        path = os.getenv(str(env_name)) if env_name else None
        if path:
            teacher_paths[expert.expert_id] = str(path)

    return LocalTransformersMultiExpertScorer(
        student_model_path=student_model,
        teacher_model_paths=teacher_paths,
        top_k=int(os.getenv("DISTILLATION_TOPK", "16")),
        device=os.getenv("STAGE1_MODEL_DEVICE", "cuda"),
        load_in_4bit=os.getenv("STAGE1_LOAD_IN_4BIT", "true").lower() == "true",
    )
