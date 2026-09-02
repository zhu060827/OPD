# Stage-1 Three-Tier Router Validation

## Scope

This validation checks control flow and data contracts only. It is not a model
quality result and did not use GPU training.

## Result

- Full repository test suite: **30/30 passed**.
- Three unlabeled HumanEval records completed the mock smoke path.
- Two records abstained because their calibrated Top-1/Top-2 margin was below
  `0.05`; one record produced a hard route.
- Recorded labels bypass five-way discovery and select exactly one Teacher.
- Unlabeled records reuse one identical completion for all five Teacher scores;
  the tested GPU contract generates that Student completion exactly once.
- Low-confidence records produce no MT-OPD handoff row unless an explicit
  fallback is configured.
- The formal five-checkpoint Open-MOPD Stage-2 launcher remains unchanged.

## Interpretation

The new policy removes arbitrary code-quality weights from the canonical route,
reduces candidate-generation work from five candidates to one shared completion,
and prevents forced low-confidence labels. These are structural improvements.
Accuracy, routing precision, and downstream pass@1 cannot be claimed until real
Teacher checkpoints are calibrated on a held-out split and compared on GPU.

The identity calibration in the smoke config is a placeholder and must not be
reported as a real calibrated router.
