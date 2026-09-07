# Stage 2: Five Real Teachers with Open-MOPD

## Status

This is the formal multi-Teacher path. It requires five distinct local Teacher
checkpoints and one trainable Student checkpoint. It delegates to the official
Open-MOPD implementation pinned at commit
`4809a96cf85a869106ff0ff3f37d0a51e12010ae`.

No GPU training result is claimed yet. The launcher, configuration contract,
strict preflight, and CPU integration tests are implemented so the same path can
be executed later on the rented GPU.

## What "fusion" means

The five Teachers are not parameter-merged. Each training record contains one
known `domain` label:

```text
cot | style | ast | variable | control_flow
```

The Student generates the on-policy trajectory. The label creates a one-hot
Teacher route, the routed frozen Teacher scores the aligned Student top-k token
support, and every sample updates the same Student. This is the hard-routing
design used by Open-MOPD.

## Official distillation reward

For aligned Student candidate token `v`, Open-MOPD constructs a detached dense
reward proportional to:

```text
(log p_teacher(v | state) - log p_student(v | state))
    * normalized_student_topk_probability(v)
```

The Teacher-minus-Student log-probability gap is the OPD advantage. It is not the
Stage-1 code-quality utility. The formal Stage-2 route does not use the project's
`0.40/0.30/0.20/0.10` utility or `0.55/0.45` router weights.

Open-MOPD then applies:

1. hard domain-to-Teacher routing;
2. token-share loss balancing toward configured target gradient shares;
3. gap-following allocation using the remaining reward magnitude;
4. Student-dependent advantage refresh for repeated inner updates;
5. one shared Student update.

## Setup

Fetch the pinned official backend:

```bash
bash scripts/fetch_open_mopd.sh /root/autodl-tmp/Open-MOPD
```

Install its requirements according to the official repository, then copy and
edit the configuration:

```bash
cp configs/stage2_open_mopd_five_teacher.example.json \
   configs/stage2_open_mopd_five_teacher.json
```

Every model directory must contain `config.json` and real model weight files.
The five Teacher paths must be distinct. Student and Teachers must share the
same model/tokenizer family so aligned token scoring is valid.

Training and validation records must contain `domain`. The training split must
contain all five canonical labels. Prefer labels recorded by the augmentation
pipeline for the first formal baseline. Pseudo labels from Stage 1 are an
explicit later ablation.

When `method.label_policy=stage1_handoff`, preflight additionally validates the
Stage-1 contract: `domain` must match `teacher_id`, Teacher weights must be
one-hot, the routing source must be present, and verification status must agree
with its downstream action. Semantic failure does not invalidate an OPD prompt;
it only prevents the current completion from entering positive augmentation.
This keeps routing and data-quality decisions separate without duplicating the
five-expert implementation inside the official backend.
Use `configs/stage2_open_mopd_from_stage1.example.json` for this path.

Convert the validated Stage-1 JSONL to the Parquet consumed by the official
launcher before preflight:

```bash
python scripts/prepare_stage2_handoff.py \
  --input outputs/stage1_multi_expert/mt_opd_handoff.jsonl \
  --output /root/autodl-tmp/data/mt_opd_handoff.parquet
```

The upstream route should follow this precedence: recorded augmentation label,
then calibrated same-trajectory five-Teacher scoring for unlabeled records,
then abstention/fallback for low-confidence cases. Open-MOPD itself still trains
from the resulting hard domain label; it does not consume the router's Top-2
diagnostic weights.

## Preflight and execution

Strict validation only:

```bash
bash scripts/run_stage2_open_mopd.sh \
  configs/stage2_open_mopd_five_teacher.json --preflight-only
```

Print the official Open-MOPD command without training:

```bash
bash scripts/run_stage2_open_mopd.sh \
  configs/stage2_open_mopd_five_teacher.json --dry-run
```

Execute real training:

```bash
bash scripts/run_stage2_open_mopd.sh \
  configs/stage2_open_mopd_five_teacher.json --run
```

The generated command passes five repeated `--teacher` arguments and enables
`domain_routing`, target gradient shares, gap-following (`multiply` direction),
and `opd_refresh_advantage` in the official trainer.

## Required formal comparisons

1. single-Teacher OPD;
2. five-Teacher hard routing without budget balancing;
3. five-Teacher Open-MOPD with token-share balancing;
4. full Open-MOPD with gap following and reward refresh;
5. full Open-MOPD plus the Stage-1 pseudo-router ablation.

The fixed Stage-1 routing utility weights are project hypotheses and require
Reward-only, Advantage-only, equal-weight, and sensitivity ablations before any
scientific claim.
