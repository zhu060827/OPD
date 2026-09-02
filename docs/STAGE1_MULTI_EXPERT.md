# Stage-1 Code Multi-Expert Routing

> Experimental ablation only. This module discovers pseudo method labels; it is
> not the formal Open-MOPD training path. Its fixed utility weights are project
> hypotheses, not formulas published by Open-MOPD.

## Purpose

This stage discovers an auditable pseudo method label for each code sample before
formal multi-teacher OPD training. It does not update a Student model and its mock
run is not a training result.

The five experts are:

- `cot`
- `style`
- `ast`
- `variable`
- `control_flow`

The implementation extends the existing code rewrite, semantic verification,
quality evaluation, and aligned token-distribution code. It does not replace the
original project.

## Design

```text
one immutable source sample
  -> five independent expert candidates
  -> compile/signature/safety/unit-test hard gate
  -> one shared deterministic routing reward
  -> Teacher-vs-Student token NLL advantage
  -> normalized evidence fusion
  -> Top-1 pseudo method label + Top-2 diagnostic weights
  -> MT-OPD handoff JSONL
```

Correctness is a hard constraint. A candidate that fails the gate receives zero
routing weight regardless of its reward or NLL advantage. If every candidate
fails, the sample is marked `no_valid_expert`; no label is fabricated.

### Shared routing utility

The reward is computed only to rank feasible candidates:

```text
R = 0.40 * method_alignment
  + 0.30 * general_quality
  + 0.20 * non_regression
  + 0.10 * changed_candidate
```

All components and metric deltas are retained. The concrete weights have no
direct literature provenance and require Reward-only, Advantage-only,
equal-weight, and sensitivity ablations. This is explicitly marked as a
project-level routing utility that requires ablation. It is not the Open-MOPD
token reward or a universal code quality score.

### Teacher advantage

For every valid candidate trajectory:

```text
advantage_t = log p_teacher(y_t | prefix) - log p_student(y_t | prefix)
```

The router uses mean token advantage, while also recording median advantage,
Teacher win fraction, forward KL, Total Variation, and aligned token count.

### Route fusion

Reward and NLL advantage are independently min-max normalized among feasible
experts for the same sample:

```text
route_score = 0.55 * normalized_reward
            + 0.45 * normalized_nll_advantage
```

Top-1 is the hard pseudo label used by the first MT-OPD experiment. Top-2 softmax
weights are diagnostics and a later ablation. Low-margin labels are marked
`low_confidence`; the canonical config keeps them but can be changed to abstain.

## Relationship to Open-MOPD

Open-MOPD assumes an oracle domain label and builds a per-sample one-hot Teacher
weight matrix. It then scores the Student's own token trajectory with the routed
frozen Teacher. This project preserves that separation:

1. Stage 1 creates a code-derived routing label because rewrite-method labels are
   not present in the source dataset.
2. Stage 2 uses that label to route a Student rollout to one frozen Teacher and
   performs real OPD/PPO training.

Stage 1 is therefore a routing-label discovery layer, not a replacement for the
official OPD loss.

The `multi_expert.fusion` module provides the Stage-2 bridge now, without
requiring a GPU: hard-route weight matrices, routed aligned Teacher log-probs,
prompt/token/effective-budget shares, token-share loss weights, and cross-Teacher
conflict metrics. The fusion functions reject ragged or unaligned tensors instead
of silently combining incompatible trajectories.

## CPU smoke run

The canonical config uses deterministic mock candidates and mock token
distributions only to exercise all five paths:

```bash
python -m code_rewrite_feedback_expander.multi_expert validate-config \
  --config configs/stage1_multi_expert.json

python -m code_rewrite_feedback_expander.multi_expert run \
  --config configs/stage1_multi_expert.json \
  --input code_rewrite_feedback_expander/data/code_real_16.jsonl \
  --output-dir outputs/stage1_multi_expert_smoke \
  --limit 3
```

The output manifest sets `formal_training_result=false`. Mock results must never
be reported as real MOPD evidence.

## Outputs

- `routing_labels.jsonl`: complete evidence for all five experts.
- `mt_opd_handoff.jsonl`: only samples usable for later training.
- `summary.json`: routing, gate, and per-expert aggregates.
- `resolved_config.json`: exact canonical configuration.
- `run_manifest.json`: backend and formal-result status.

The handoff records include `domain`, `teacher_id`, and `teacher_weights`, matching
the routing concepts expected by Open-MOPD.

## Local-model GPU handoff

Use `configs/stage1_multi_expert.gpu.example.json` as a template. It expects five
local OpenAI-compatible generation endpoints and local Hugging Face model paths
for aligned log-probability scoring. These endpoints can be local vLLM services;
no paid API is required.

Required environment variables:

```bash
export STAGE1_STUDENT_MODEL=/path/to/Qwen3-1.7B-Base
export STAGE1_TEACHER_COT_MODEL=/path/to/cot-teacher
export STAGE1_TEACHER_STYLE_MODEL=/path/to/style-teacher
export STAGE1_TEACHER_AST_MODEL=/path/to/ast-teacher
export STAGE1_TEACHER_VARIABLE_MODEL=/path/to/variable-teacher
export STAGE1_TEACHER_CONTROL_FLOW_MODEL=/path/to/control-flow-teacher
export DISTILLATION_TOPK=16
```

Student and Teacher tokenizers must have identical vocabularies. Teacher models
are frozen and loaded once. Formal Stage-1 runs should use Base checkpoints from
the same model family.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover configuration consistency, hard-gate behavior, immutable shared
inputs, Top-1 selection, Top-2 normalization, no-valid-expert handling, and the
MT-OPD handoff schema. The handoff Student prompt is also checked for Teacher,
route, and reference-code leakage.

The built-in subprocess test runner is appropriate for the current trusted
HumanEval/MBPP fixtures. Before accepting arbitrary third-party tests on a GPU
server, run them inside a hardened container or Firejail-compatible sandbox.

## References

- MOPD: <https://arxiv.org/abs/2606.30406>
- Open-MOPD: <https://github.com/BytedTsinghua-SIA/Open-MOPD>
- OPD: <https://github.com/thunlp/OPD>
