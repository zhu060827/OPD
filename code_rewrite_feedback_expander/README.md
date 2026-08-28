# Code Rewrite Feedback Expander

This folder implements a feedback-driven code data expansion loop:

```text
Original Code
  -> Student Rewrites
  -> Semantic Equivalence Check
  -> Quality Evaluation
  -> Teacher Scores the Same Student Token Trajectory
  -> Hierarchical Token-KL Strategy Selection
  -> Feedback
  -> Rewrite Again
```

The pipeline does not create new programming problems. It rewrites existing solutions while preserving behavior, and the source data is expected to contain code reasoning steps.

## Input

Use JSONL:

```json
{
  "task_id": "factorial",
  "prompt": "Return n factorial.",
  "reasoning": [
    "Initialize result to 1.",
    "Multiply by every integer from 2 through n.",
    "Return the final product."
  ],
  "code": "def factorial(n):\n    ...",
  "tests": ["assert factorial(5) == 120"],
  "language": "python"
}
```

## Run Offline

```bash
python3 -m code_rewrite_feedback_expander.cli run \
  --input data/sample_code.jsonl \
  --output outputs/code_expanded.jsonl \
  --provider mock
```

The command also writes:

```text
outputs/code_expanded.jsonl.quality.svg
```

## Run With OpenAI-Compatible API

```bash
export CODE_EXPANDER_API_KEY="..."
export CODE_EXPANDER_BASE_URL="https://api.gpt.ge/v1"
export CODE_EXPANDER_MODEL="gpt-3.5-turbo"

python3 -m code_rewrite_feedback_expander.cli run \
  --input data/sample_code.jsonl \
  --output outputs/code_expanded.real.jsonl \
  --provider openai_compatible \
  --base-url "$CODE_EXPANDER_BASE_URL" \
  --model "$CODE_EXPANDER_MODEL"
```

## Output

Each output record contains:

- `original_code`: original solution.
- `original_reasoning`: original code reasoning chain.
- `expanded_code`: final retained rewrite.
- `expanded_reasoning`: reasoning chain aligned with the final retained rewrite.
- `retained_rewrites`: every rewrite that passed semantic checks and improved quality.
- `iteration_trace`: every rewrite attempt, with semantic result, independent quality metrics, token KL, Total Variation, distribution overlap, token NLL reduction, and strategy ranking.
- `expansion_stats`: line-count expansion, reasoning-step expansion, and retained rewrite count.

## Checks

Semantic equivalence:

- Python AST parse.
- Function signature consistency.
- Dangerous call/import filtering.
- Compile pass rate.
- AST edit distance and tree edit distance.
- CodeBLEU-style syntax and data-flow overlap.
- Supplied unit-test pass rate.

Quality:

- The five paper-level primary metrics use literature or specification terminology: `cot -> maintainability_index`, `style -> style_violation_rate`, `ast -> codebleu_syntax_match`, `variable -> naming_convention_compliance`, and `control_flow -> cyclomatic_complexity`.
- `style_violation_rate` is a PEP 8/pycodestyle-compatible metric: violations are counted and normalized by logical lines. Maintainability reports the SEI derivative raw MI and its 0-100 normalization. Cyclomatic complexity reports the raw McCabe number and a direction-normalized comparison score.
- `naming_convention_compliance` follows the PEP 8 "Naming Conventions" terminology and is computed as the proportion of checked identifiers that conform.
- `codebleu_syntax_match` follows the Syntax Match component name in CodeBLEU. The current implementation is still labelled `codebleu_compatible_fallback` until the official tree-sitter backend is installed.
- `ast` is labelled `codebleu_compatible_fallback` in the output unless an official CodeBLEU/tree-sitter implementation is integrated. Fallback results must not be reported as official CodeBLEU scores.
- No weighted aggregate quality score is calculated.
- A candidate is retained only when it is semantically valid, differs from the working code, improves the primary metric mapped to its rewrite method, and does not regress other metrics beyond their per-metric tolerance. AST rewrites use CodeBLEU Syntax Match as a minimum preservation constraint rather than requiring improvement above the identity baseline of 1.0.
- McCabe and MI provide established risk scales, but the literature does not provide universal before/after delta thresholds for readability, identifier quality, or CodeBLEU. Such thresholds are marked `validation_parameter` in every trace and must be selected on a validation set with sensitivity analysis.
- The CLI writes `<output>.paper.json` containing per-method retention rates, per-metric before/after improvements, semantic pass rate, and unit-test pass rate.
- The paper report separates final-output correctness from candidate-level semantic and unit-test pass rates. This prevents unchanged original outputs from masking failures among generated candidates.
- The CLI uses `MockOPDDistributionScorer` only with `--opd-scorer mock`. For formal OPD runs use `--opd-scorer external --opd-scorer-factory module:function`; the factory must return an `OPDDistributionScorer` backed by the deployed teacher/student models.
- `opd_main_adapter.create_opd_scorer` is the built-in OPD-main adapter. It loads `STUDENT_MODEL` and `TEACHER_MODEL`, teacher-forces both models on the same candidate trajectory, and compares the union of their top-k token supports, matching OPD-main's top-k distillation data flow.
- For strict on-policy evaluation, the rewrite provider must serve the same checkpoint configured by `STUDENT_MODEL`; using a different API model for generation would make the reported student distribution off-policy.

Example on the deployed server:

```bash
export PYTHONPATH="/root/code_rewrite_feedback_expander:/root/OPD/verl:$PYTHONPATH"
export STUDENT_MODEL="/path/to/student-checkpoint"
export TEACHER_MODEL="/path/to/teacher-checkpoint"
export DISTILLATION_TOPK=16

python3 -m code_rewrite_feedback_expander.cli run \
  --input data/test_imperfect.jsonl \
  --output outputs/formal_opd.jsonl \
  --provider openai_compatible \
  --opd-scorer external \
  --opd-scorer-factory code_rewrite_feedback_expander.opd_main_adapter:create_opd_scorer \
  --max-iterations 3
```
- `experiments.run_comparison_experiments` runs adaptive, random, and fixed-order selection plus five leave-one-method-out ablations.

OPD strategy selection:

- The existing code expander is treated as the student model.
- The teacher does not generate a separate rewrite. The OPD platform scores the exact student-generated trajectory under both teacher and student distributions.
- Production integration implements `OPDDistributionScorer.score_student_trajectory(...)` and returns aligned teacher/student top-k log-probabilities for each generated token.
- The first OPD step tries all five rewrite strategies.
- After each OPD step, all candidate records are retained in `iteration_trace`; a feedback LLM converts those structured records into natural-language instructions for the next step.
- Token attribution is structural: variable tokens use AST identifier and definition-use evidence; control-flow tokens use branch/loop keywords; AST tokens use structural operators; style tokens use formatting tokens; CoT tokens use comments/docstrings and reasoning alignment. Tokens may receive multiple normalized aspect weights.
- Feedback is generated through a fixed schema `{preserve, fix, avoid, evidence}`. Invalid model output falls back to rule-based content. The trace stores the structured source records, rendered feedback, and the feedback input used by the next student call.
- The next student call receives the previous round's `feedback_used`, and the generated text is stored as `generated_feedback` for auditability.
- Token KL is `D_KL(P_teacher(.|x,y_<t) || P_student(.|x,y_<t))` on the same student prefix.
- Token contributions are attributed to `cot`, `style`, `ast`, `variable`, and `control_flow`, producing one KL value per aspect.
- Semantic validity is a hard feasibility filter and is not multiplied into a numerical utility score.
- Each aspect independently reports forward KL, Total Variation distance, distribution overlap (`1 - TV`), and teacher-versus-student NLL reduction on the generated token.
- Adaptive selection uses a transparent lexicographic order: larger forward KL, then larger token NLL reduction, then smaller Total Variation. No custom weighted or multiplicative score is calculated.
- This order defines the fallback queue. If the highest-ranked method fails semantic validation, the pipeline tries the next method until one passes or all methods are exhausted.
- Any semantic-passing candidate may determine the next OPD direction; a rewrite updates the working code and is retained only when it also improves the unified quality score.
- Iteration stops when `max_iterations` is reached, no semantic-passing candidate exists in the current OPD step, or no aspect has positive KL on a valid student trajectory.
- `unit_test_pass_rate` is treated as a semantic gate, not a quality metric.

Feedback LLM integration:

- Offline mode uses `MockIterationFeedbackLLM`.
- Production mode can use `--feedback-provider openai_compatible`, with `--feedback-model`, `--feedback-base-url`, and `--feedback-api-key-env`.

References used for the unified quality metrics:

- [McCabe, 1976](https://doi.org/10.1109/TSE.1976.233837) for cyclomatic complexity.
- [Oman & Hagemeister, 1992](https://doi.org/10.1109/ICSM.1992.242525) for Maintainability Index.
- [Buse & Weimer, 2010](https://doi.org/10.1109/TSE.2009.70) for readability.
- [Ren et al., 2020](https://arxiv.org/abs/2009.10297) for CodeBLEU AST match.

References used for the OPD framing:

- [THUNLP OPD repository](https://github.com/thunlp/OPD) for the student/teacher distillation setup and the NPU dependency profile you referenced.
- [Agarwal et al., 2024, Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) for on-policy distillation on student-generated trajectories.
- [Hinton et al., 2015, Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) for teacher-student distribution matching and soft-target distillation.
- [Gibbs and Su, 2002, On Choosing and Bounding Probability Metrics](https://doi.org/10.1111/1467-9868.00362) for Total Variation and relationships among probability-distribution distances.

Forward KL, Total Variation, distribution overlap, and token negative log-likelihood are established quantities. Their lexicographic use for rewrite-strategy selection, and the five-way token-to-aspect attribution, are project-level design choices that must be validated through ablation and attribution studies; they are not claimed as standard OPD formulas.

Additional references used by the semantic checks:

- [Chen et al., 2021](https://arxiv.org/abs/2107.03374) for execution-based code generation evaluation.
- [Zhang & Shasha, 1989](https://doi.org/10.1137/0218082) for tree edit distance.
- [Ren et al., 2020](https://arxiv.org/abs/2009.10297) for syntax and data-flow overlap.

Implementation note: some metrics are lightweight pure-Python approximations of the standard definitions so the project stays dependency-free.

## MBPP OPD Training

The root `OPD-main` integration includes a dedicated Qwen3 teacher/student training path. It converts the local
Hugging Face MBPP parquet files to verl format, supports an explicit Qwen3 thinking-mode setting, evaluates generated
Python with the supplied MBPP assertions, and uses OPD-main's teacher top-k token rewards on the student's own
rollout.

Use a dedicated CUDA training environment; the `llm_reviewer` environment used for ModelScope downloads is not
a verl training environment. On a host with sufficient GPU and system memory, install the repository's pinned
vLLM/Transformers stack as follows:

```bash
conda create -n verl python=3.12 -y
conda activate verl
cd OPD-main/verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
cd ../..
```

Prepare the full MBPP splits:

```bash
OPD_PYTHON=/path/to/verl/python bash OPD-main/train_mbpp_opd.sh prepare
```

The converter derives every test-visible function/class interface from the canonical code and the supplied
tests/setup, then inserts implementation-free signatures into every train, validation, and test prompt. Conversion
fails rather than emitting an underspecified row if no test/setup call can be matched to a canonical top-level
definition. The same `prepare` command must therefore be run after pulling data-conversion changes and before a
strict train/evaluation-distribution-consistent experiment.

Compose the exact Hydra configuration without loading either model:

```bash
OPD_PYTHON=/path/to/verl/python bash OPD-main/train_mbpp_opd.sh config
```

Run all environment, model, tokenizer, dataset, batch, GPU, and RAM checks:

```bash
OPD_PYTHON=/path/to/verl/python bash OPD-main/train_mbpp_opd.sh preflight
```

Start training:

```bash
OPD_PYTHON=/path/to/verl/python bash OPD-main/train_mbpp_opd.sh train
```

The defaults are:

- student: `student_model` (Qwen3-1.7B)
- teacher: `teacher_model` (Qwen3-4B)
- `trainer.total_training_steps=200`
- global batch size `32`
- per-device actor micro-batch size `8`
- gradient accumulation `4` on one data-parallel GPU
- one student rollout per MBPP prompt
- Qwen3 thinking mode controlled by `ENABLE_THINKING` (default: `true`)
- union top-k (`k=16`) with teacher-probability weighting
- teacher inference under `torch.no_grad()` with no teacher optimizer

These models require substantially more memory than an 8GB GPU plus 8GB system RAM for full-parameter OPD.
The preflight check therefore stops on undersized hosts. `ALLOW_LOW_MEMORY=1` only bypasses that guard; it does
not make an otherwise impossible allocation fit. A 24GB GPU and 32GB system RAM are the practical target for
this configuration, with larger memory preferred.

### Local MBPP evaluation

After merging the trained actor to Hugging Face format, run a deterministic 10-task smoke evaluation first:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m code_rewrite_feedback_expander.evaluate_mbpp_local \
  --model-path /home/asus/OPD/cloud_exports \
  --dataset-path OPD-main/datasets/mbpp_opd/test.parquet \
  --output-dir /home/asus/OPD/cloud_exports/mbpp_eval_smoke \
  --max-samples 10 \
  --batch-size 1 \
  --max-new-tokens 1024 \
  --enable-thinking \
  --seed 42
```

If the smoke run succeeds, evaluate all 500 test tasks by omitting `--max-samples` and selecting a fresh output
directory:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m code_rewrite_feedback_expander.evaluate_mbpp_local \
  --model-path /home/asus/OPD/cloud_exports \
  --dataset-path OPD-main/datasets/mbpp_opd/test.parquet \
  --output-dir /home/asus/OPD/cloud_exports/mbpp_eval_step200 \
  --batch-size 1 \
  --max-new-tokens 1024 \
  --enable-thinking \
  --seed 42
```

The command appends each completion to `predictions.jsonl`, can resume an interrupted run with the
same arguments, writes failed completions to `failures.jsonl`, and continuously refreshes `metrics.json`.
Deterministic one-sample evaluation reports execution accuracy as `pass@1`. For sampled pass@k evaluation, set
`--samples-per-task K --temperature 1.0`; pass@1 through pass@K use the standard unbiased estimator.

### Iterative execution-feedback repair

Standard `pass@k` draws `k` candidates independently from the same problem prompt. It does not let a later candidate
observe an earlier failure. The optional execution-feedback strategy instead evaluates one candidate, retains its code
and failure signal as short-lived task memory, and asks the same model for a corrected replacement only when that
candidate fails:

```text
Problem -> Attempt 1 -> Restricted execution
                     -> failure category + previous draft
                     -> Attempt 2 -> Restricted execution
                                  -> failure category + previous draft
                                  -> Attempt 3
```

Run a three-attempt, hidden-test-safe smoke evaluation as follows:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m code_rewrite_feedback_expander.evaluate_mbpp_local \
  --model-path /path/to/merged_hf_model \
  --dataset-path OPD-main/datasets/mbpp_opd/test.parquet \
  --output-dir /path/to/mbpp_iterative_feedback_smoke \
  --max-samples 10 \
  --samples-per-task 3 \
  --generation-strategy execution_feedback \
  --execution-feedback summary \
  --max-prompt-tokens 4096 \
  --batch-size 1 \
  --max-new-tokens 1024 \
  --temperature 1.0 \
  --top-p 1.0 \
  --disable-thinking \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --seed 42
```

`summary` feedback exposes syntax, safety, timeout, missing-code, required-interface, and generic held-out-test failure
categories, but deliberately withholds hidden assertion source and expected values. `--execution-feedback full` also
injects the raw executor output; because that output can reveal held-out assertions, the run is labelled
`oracle-assisted` and must not be compared directly with a standard benchmark result.

Conditional repair attempts are not independent samples, so the evaluator does not claim that their cumulative result
is the standard unbiased `pass@3`. It reports `pass@1` for the first attempt and
`iterative_solve_rate@1` through `iterative_solve_rate@3`, with `iterative_pass@k` retained only as an explicitly
labelled conditional-repair alias. It additionally reports `repair_gain@3`, repaired-task count, repair success after
an initial failure, attempts-to-solve, and total inference cost. Generation stops for a task immediately after its first
passing attempt. Every prediction stores `attempt_id`, `previous_attempt_id`, `feedback_used`, and the feedback mode so
the trajectory is auditable and resumable.

This strategy is test-time in-context repair: it does not update model parameters. Existing OPD checkpoints are valid
for measuring emergent repair ability and should be retained as controls. Training a model to internalize repair
behavior requires a separate trajectory-training experiment containing `(problem, failed code, execution feedback,
corrected code)` examples, as in CYCLE or reinforcement learning from execution feedback; it must not overwrite the
single-turn OPD checkpoints.

References used for iterative execution-feedback repair:

- [Shinn et al., 2023, Reflexion (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) for retaining verbal feedback as episodic context across attempts.
- [Madaan et al., 2023, Self-Refine (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html) for the iterative feedback-refinement-stopping loop without parameter updates.
- [Chen et al., 2024, Teaching Large Language Models to Self-Debug (ICLR)](https://openreview.net/pdf?id=KuPixIqPiq) for reusing failed predictions and execution results in MBPP code repair.
- [Le et al., 2022, CodeRL (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html) for unit-test/critic feedback and feedback-aware regeneration.
- [Ding et al., 2024, CYCLE (PACMPL/OOPSLA)](https://doi.org/10.1145/3649825) for training code models on developing logs of faulty generations, execution feedback, and iterative correction.
- [Gehring et al., 2024, RLEF](https://arxiv.org/abs/2410.02089) for reinforcement learning that teaches code models to condition future generations on execution feedback; this is cited as a preprint rather than a journal publication.
- [Dou et al., 2024, Re-ReST](https://arxiv.org/abs/2406.01495) for reflection-reinforced self-training from low-quality attempts and environmental feedback; this is cited as a preprint.

The summary also reports task/completion counts, sample pass rate, solved-at-least-once rate, code extraction and
syntax validity and required-interface match rates, max-token clipping, error counts/rates, average and median token
counts and latencies, and generation throughput. Each JSONL prediction retains the full response, extracted code,
required and defined entrypoints, execution result, error type, truncated test output, token counts, and timings for
failure analysis.

For strict distribution consistency, do not reuse a checkpoint trained on an older prompt schema. Regenerate all
three splits, select a new experiment/checkpoint directory, train from the original student model, merge the final
actor, and evaluate that merged checkpoint on the regenerated test parquet. Using a new experiment name prevents
verl's automatic checkpoint resume from loading the previous run.

The earlier 1024-token thinking-mode smoke run clipped most generations before a complete implementation. For the
signature-v1 retraining experiment, use `ENABLE_THINKING=false` during OPD training and `--disable-thinking` during
both smoke and full test evaluation. Keep `--max-new-tokens 1024`; use `--temperature 1.0 --top-p 1.0` when strict
rollout-policy matching with training is required, or deterministic temperature 0 only as a separately named
greedy-decoding report.

Generated code is checked for forbidden operations and executed in an isolated child Python process with CPU,
address-space, file-size, file-descriptor, and wall-time limits. This is defense in depth, not a hardened security
sandbox; only evaluate models and datasets you trust.
