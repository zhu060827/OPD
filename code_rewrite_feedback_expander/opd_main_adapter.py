from __future__ import annotations

import importlib
import os
from typing import List

from .attribution import structural_token_weights
from .llm import OPDDistributionScorer, build_rewrite_prompt
from .models import CodeRecord, RewriteCandidate, TokenDistribution


class OPDMainTransformersScorer(OPDDistributionScorer):
    """Teacher-force student and teacher models on the same candidate trajectory.

    This standalone adapter follows OPD-main's union top-k comparison. It is
    intended for evaluation and strategy selection, not distributed training.
    """

    def __init__(
        self,
        student_model_path: str,
        teacher_model_path: str,
        top_k: int = 16,
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError("OPDMainTransformersScorer requires torch and transformers") from exc

        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer

        self.torch = torch
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_path, trust_remote_code=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
        if self.tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("Student and teacher tokenizers must share the same vocabulary for token-level KL")
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.student = AutoModelForCausalLM.from_pretrained(student_model_path, **model_kwargs).eval()
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_model_path, **model_kwargs).eval()
        if device != "auto":
            self.student.to(device)
            self.teacher.to(device)
        self.device = next(self.student.parameters()).device

    def score_student_trajectory(
        self,
        record: CodeRecord,
        candidate: RewriteCandidate,
        strategy: str,
    ) -> List[TokenDistribution]:
        prefix = build_rewrite_prompt(record, strategy, feedback="") + "\nCandidate code:\n"
        prefix_ids = self.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"]
        response = candidate.code
        response_encoding = self.tokenizer(response, add_special_tokens=False, return_tensors="pt")
        response_ids = response_encoding["input_ids"]
        input_ids = self.torch.cat([prefix_ids, response_ids], dim=1).to(self.device)
        attention_mask = self.torch.ones_like(input_ids)

        with self.torch.inference_mode():
            student_logits = self.student(input_ids=input_ids, attention_mask=attention_mask).logits
            teacher_input = input_ids.to(next(self.teacher.parameters()).device)
            teacher_mask = attention_mask.to(teacher_input.device)
            teacher_logits = self.teacher(input_ids=teacher_input, attention_mask=teacher_mask).logits

        start = prefix_ids.shape[1] - 1
        length = response_ids.shape[1]
        student_logits = student_logits[:, start : start + length, :].float().cpu()
        teacher_logits = teacher_logits[:, start : start + length, :].float().cpu()
        actual_ids = response_ids[0].cpu()
        lexical_weights = structural_token_weights(response, candidate.reasoning)
        profiles: List[TokenDistribution] = []

        for index in range(length):
            student_lp = self.torch.log_softmax(student_logits[0, index], dim=-1)
            teacher_lp = self.torch.log_softmax(teacher_logits[0, index], dim=-1)
            k = min(self.top_k, student_lp.numel())
            student_top_ids = self.torch.topk(student_lp, k=k).indices
            teacher_top_ids = self.torch.topk(teacher_lp, k=k).indices
            union_ids = self.torch.unique(self.torch.cat([student_top_ids, teacher_top_ids])).tolist()
            labels = {token_id: f"id:{token_id}" for token_id in union_ids}
            generated_id = int(actual_ids[index])
            aspect_index = min(
                len(lexical_weights) - 1,
                int(index * len(lexical_weights) / max(1, length)),
            ) if lexical_weights else 0
            profiles.append(
                TokenDistribution(
                    token=self.tokenizer.decode([generated_id]),
                    student_logprobs={labels[token_id]: float(student_lp[token_id]) for token_id in union_ids},
                    teacher_logprobs={labels[token_id]: float(teacher_lp[token_id]) for token_id in union_ids},
                    student_token_logprob=float(student_lp[generated_id]),
                    teacher_token_logprob=float(teacher_lp[generated_id]),
                    aspect_weights=lexical_weights[aspect_index] if lexical_weights else {"ast": 1.0},
                    attribution_source="ast_cfg_def_use_alignment",
                )
            )
        return profiles


def create_opd_scorer() -> OPDMainTransformersScorer:
    student = os.getenv("STUDENT_MODEL") or os.getenv("CODE_EXPANDER_STUDENT_MODEL")
    teacher = os.getenv("TEACHER_MODEL") or os.getenv("CODE_EXPANDER_TEACHER_MODEL")
    if not student or not teacher:
        raise ValueError("Set STUDENT_MODEL and TEACHER_MODEL to OPD-main model paths")
    return OPDMainTransformersScorer(
        student_model_path=student,
        teacher_model_path=teacher,
        top_k=int(os.getenv("DISTILLATION_TOPK", "16")),
        device=os.getenv("CODE_EXPANDER_OPD_DEVICE", "auto"),
        torch_dtype=os.getenv("CODE_EXPANDER_OPD_DTYPE", "auto"),
    )
