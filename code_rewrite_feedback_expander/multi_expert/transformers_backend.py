from __future__ import annotations

import importlib
import os
from typing import Dict, List

from ..attribution import structural_token_weights
from ..llm import build_rewrite_prompt
from ..models import CodeRecord, RewriteCandidate, TokenDistribution
from .backends import ExpertTrajectoryScorer
from .config import ExpertConfig, Stage1Config


class LocalTransformersMultiExpertScorer(ExpertTrajectoryScorer):
    """Score candidate tokens with one shared Student and five frozen Teachers.

    This backend is deliberately local-only. It never sends code or logits to an
    external service. Models are loaded once, Teacher forward passes run under
    inference mode, and every comparison uses the exact same aligned token IDs.
    """

    def __init__(
        self,
        student_model_path: str,
        teacher_model_paths: Dict[str, str],
        top_k: int = 16,
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LocalTransformersMultiExpertScorer requires torch and transformers"
            ) from exc
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model_path, trust_remote_code=True
        )
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_path, **model_kwargs
        ).eval()
        if device != "auto":
            self.student.to(device)
        self.teachers = {}
        for expert_id, model_path in teacher_model_paths.items():
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self.tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
                raise ValueError(
                    f"Student and {expert_id} tokenizers must share the same vocabulary"
                )
            teacher = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
            if device != "auto":
                teacher.to(device)
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self.teachers[expert_id] = teacher

    def score(
        self,
        expert: ExpertConfig,
        record: CodeRecord,
        candidate: RewriteCandidate,
    ) -> List[TokenDistribution]:
        teacher = self.teachers.get(expert.expert_id)
        if teacher is None:
            raise KeyError(f"No loaded Teacher for {expert.expert_id}")
        prefix = build_rewrite_prompt(record, expert.strategy, feedback="") + "\nCandidate code:\n"
        prefix_ids = self.tokenizer(
            prefix, add_special_tokens=True, return_tensors="pt"
        )["input_ids"]
        response_ids = self.tokenizer(
            candidate.code, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        student_device = next(self.student.parameters()).device
        student_input = self.torch.cat([prefix_ids, response_ids], dim=1).to(student_device)
        with self.torch.inference_mode():
            student_logits = self.student(
                input_ids=student_input,
                attention_mask=self.torch.ones_like(student_input),
            ).logits
            teacher_device = next(teacher.parameters()).device
            teacher_input = student_input.to(teacher_device)
            teacher_logits = teacher(
                input_ids=teacher_input,
                attention_mask=self.torch.ones_like(teacher_input),
            ).logits

        start = prefix_ids.shape[1] - 1
        length = response_ids.shape[1]
        student_logits = student_logits[:, start : start + length].float().cpu()
        teacher_logits = teacher_logits[:, start : start + length].float().cpu()
        actual_ids = response_ids[0].cpu()
        lexical_weights = structural_token_weights(candidate.code, candidate.reasoning)
        profiles: List[TokenDistribution] = []
        for index in range(length):
            student_lp = self.torch.log_softmax(student_logits[0, index], dim=-1)
            teacher_lp = self.torch.log_softmax(teacher_logits[0, index], dim=-1)
            k = min(self.top_k, student_lp.numel())
            student_top = self.torch.topk(student_lp, k=k).indices
            teacher_top = self.torch.topk(teacher_lp, k=k).indices
            union_ids = self.torch.unique(self.torch.cat([student_top, teacher_top])).tolist()
            generated_id = int(actual_ids[index])
            aspect_index = (
                min(
                    len(lexical_weights) - 1,
                    int(index * len(lexical_weights) / max(length, 1)),
                )
                if lexical_weights
                else 0
            )
            profiles.append(
                TokenDistribution(
                    token=self.tokenizer.decode([generated_id]),
                    student_logprobs={f"id:{token_id}": float(student_lp[token_id]) for token_id in union_ids},
                    teacher_logprobs={f"id:{token_id}": float(teacher_lp[token_id]) for token_id in union_ids},
                    student_token_logprob=float(student_lp[generated_id]),
                    teacher_token_logprob=float(teacher_lp[generated_id]),
                    aspect_weights=(
                        lexical_weights[aspect_index] if lexical_weights else {"ast": 1.0}
                    ),
                    attribution_source="ast_cfg_def_use_alignment",
                )
            )
        return profiles


def create_trajectory_scorer(config: Stage1Config) -> LocalTransformersMultiExpertScorer:
    student_model = os.getenv("STAGE1_STUDENT_MODEL")
    if not student_model:
        raise ValueError("Set STAGE1_STUDENT_MODEL to the local Student checkpoint")
    teacher_paths: Dict[str, str] = {}
    for expert in config.enabled_experts:
        direct_path = expert.scoring.get("teacher_model_path")
        env_name = expert.scoring.get("teacher_model_env")
        path = direct_path or (os.getenv(str(env_name)) if env_name else None)
        if not path:
            raise ValueError(
                f"Missing Teacher model path for {expert.expert_id}; "
                "set scoring.teacher_model_path or scoring.teacher_model_env"
            )
        teacher_paths[expert.expert_id] = str(path)
    return LocalTransformersMultiExpertScorer(
        student_model_path=student_model,
        teacher_model_paths=teacher_paths,
        top_k=int(os.getenv("DISTILLATION_TOPK", "16")),
        device=os.getenv("STAGE1_MODEL_DEVICE", "auto"),
        torch_dtype=os.getenv("STAGE1_MODEL_DTYPE", "auto"),
    )
