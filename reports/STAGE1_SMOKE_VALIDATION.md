# Stage-1 Multi-Expert Smoke Validation

Date: 2026-08-23

## Result

The Stage-1 five-expert routing pipeline completed end to end without a GPU,
model download, or external paid API.

- Unit and integration tests: **18/18 passed**
- Existing project samples processed: **16/16**
- Hard-gate and trajectory-scoring completion: **16/16 records routed**
- External paid API calls: **0**
- GPU training: **not performed**
- Formal MOPD result: **no**

The deterministic mock run produced 12 normal routes and 4 low-confidence
routes. Its pseudo-label distribution was `style=10`, `ast=5`, `variable=1`.
These labels only verify control flow and schemas; they are not evidence about
real expert quality.

## Verified behavior

- Every expert receives the same immutable source record.
- Compile, signature, safety, and unit tests are hard constraints.
- Failed candidates receive zero Teacher weight.
- No valid expert produces `no_valid_expert`, never a fabricated label.
- Top-1 handoff weights are one-hot; Top-2 softmax weights are diagnostics.
- Student handoff prompts contain no Teacher ID, route, or reference code.
- Teacher log-prob fusion rejects unaligned shapes.
- Token share and effective optimization-budget share are measured separately.
- Cross-Teacher conflict is measured only on aligned valid tokens.

## Existing bugs found and fixed

1. HumanEval stored its complete test harness as a string. The old loader split
   it into characters, and the test runner indented only its first line.
2. MBPP stored the correct implementation in `reference_code`. The old loader
   ignored it and passed an empty solution to the rewrite pipeline.

Regression tests now cover both formats.

## Artifacts

The ignored local smoke artifacts are in `outputs/stage1_multi_expert_smoke_full/`:

- `routing_labels.jsonl`
- `mt_opd_handoff.jsonl`
- `summary.json`
- `resolved_config.json`
- `run_manifest.json`

## Next formal gate

Replace both mock backends with five local expert generation endpoints and the
local Transformers trajectory scorer. The first GPU-backed Stage-1 validation
must confirm that the five experts produce distinguishable candidate behavior,
that all tokenizers share one vocabulary, and that label balance remains usable.
Only after that validation should the handoff JSONL enter MT-OPD training.
