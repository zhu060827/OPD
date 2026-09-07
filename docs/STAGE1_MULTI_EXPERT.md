# Stage-1 Code Multi-Expert Routing

> This is the optional routing layer before formal Open-MOPD training. Recorded
> labels are preferred. Pseudo-routing is used only for unlabeled data.

## Purpose

This stage implements a three-tier policy. It does not update a Student model and
its mock run is not a training result.

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
1. recorded method/domain label -> direct one-hot Teacher route
2. unlabeled sample -> one shared completion scored by all five Teachers
3. low calibrated Top-1 margin -> abstain (or explicit configured fallback)
```

For unlabeled records, all Teachers receive the identical prompt prefix, token
IDs, and completion. The system does not generate five candidates. It computes
the aligned mean OPD advantage for each Teacher, then applies per-Teacher robust
calibration fitted on a held-out split:

```text
raw_advantage_e = mean_t[log p_teacher_e(y_t|s_t) - log p_student(y_t|s_t)]
calibrated_advantage_e = (raw_advantage_e - median_e) / (1.4826 * MAD_e)
route = argmax_e calibrated_advantage_e
```

Calibration matters because raw log-probability scales can differ between
Teachers. A large uncalibrated gap is not automatically evidence that a Teacher
is the best route. The Top-1/Top-2 margin is recorded; the canonical policy
abstains below the configured threshold instead of inventing a label.

The CPU smoke config uses the record's existing code as the shared completion.
The GPU config sets `shared_completion_source=student_generate`: the current
Student generates exactly once, then all five frozen Teachers score that same
on-policy rollout. The Student model is reused by the scorer rather than loaded
twice. Exact generated prompt/completion token IDs are retained and reused;
decoded text is never re-tokenized as a substitute for the on-policy trajectory.

Routing and verification are independent axes in the canonical three-tier path.
An incorrect Student completion is still a valid on-policy state, so semantic or
unit-test failure does not suppress aligned five-Teacher scoring. Verification is
recorded as `semantic_pass`, `semantic_fail`, or `semantic_unverified`; it never
enters the calibrated advantage formula. Only a missing, non-finite, or
alignment-invalid token trajectory blocks routing.

Downstream handling remains conservative: `semantic_pass` can enter positive
augmentation, `semantic_fail` goes to repair/negative-data handling, and samples
without executable tests go to an unverified pool. Thus an incorrect completion
can teach the OPD Student without being mislabeled as a correct augmentation.

The legacy `heuristic_ablation` still treats semantic equivalence as a hard gate,
because that policy compares independently generated rewrite candidates rather
than scoring one on-policy Student trajectory.

### Legacy heuristic ablation

The reward is computed only to rank feasible candidates:

```text
R = 0.40 * method_alignment
  + 0.30 * general_quality
  + 0.20 * non_regression
  + 0.10 * changed_candidate
```

All components and metric deltas remain available under
`routing.policy=heuristic_ablation`. The concrete weights have no
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

Its legacy route fusion is:

Reward and NLL advantage are independently min-max normalized among feasible
experts for the same sample:

```text
route_score = 0.55 * normalized_reward
            + 0.45 * normalized_nll_advantage
```

This is no longer the canonical router. It remains solely for controlled
comparison. Top-2 softmax weights are diagnostics and are never used as a
parameter or logits average.

## Relationship to Open-MOPD

Open-MOPD assumes an oracle domain label and builds a per-sample one-hot Teacher
weight matrix. It then scores the Student's own token trajectory with the routed
frozen Teacher. This project preserves that separation:

1. Stage 1 uses the recorded augmentation method whenever it exists.
2. Only unlabeled data enters calibrated same-trajectory pseudo-routing.
3. Stage 2 uses that label to route a Student rollout to one frozen Teacher and
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

The handoff records include `domain`, `teacher_id`, `teacher_weights`,
`routing_source`, `verification_status`, and `downstream_action`, matching
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

The identity calibration values in the example configs are smoke-test
placeholders. Before a real unlabeled-routing run, score a held-out split and fit
robust Teacher-specific statistics:

```bash
python -m code_rewrite_feedback_expander.multi_expert fit-calibration \
  --config configs/stage1_multi_expert.gpu.example.json \
  --input outputs/calibration_pass/routing_labels.jsonl \
  --output outputs/teacher_advantage_calibration.json \
  --min-samples 20
```

Copy the resulting `location` and `scale` values into the formal config. Do not
fit calibration on the evaluation split.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover configuration consistency, recorded-label priority, one shared
completion across five Teachers, robust calibration, low-confidence abstention,
semantic-failure routing without positive-augmentation leakage, missing-test
handling, legacy hard-gate behavior, and the
MT-OPD handoff schema. The handoff Student prompt is also checked for Teacher,
route, and reference-code leakage.

The built-in subprocess test runner is appropriate for the current trusted
HumanEval/MBPP fixtures. Before accepting arbitrary third-party tests on a GPU
server, run them inside a hardened container or Firejail-compatible sandbox.

## References

- MOPD: <https://arxiv.org/abs/2606.30406>
- Open-MOPD: <https://github.com/BytedTsinghua-SIA/Open-MOPD>
- OPD: <https://github.com/thunlp/OPD>
