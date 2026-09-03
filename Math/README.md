# Math Reasoning Expander

This project implements a runnable prototype for graph-masked mathematical reasoning data expansion.

It follows this loop:

```text
Raw math CoT
  -> Reasoning graph parser
  -> Node/path/formula masking
  -> LLM graph fill
  -> Multi-feedback evaluator
  -> Natural language feedback
  -> Reflection/regeneration
  -> Adaptive stop
```

The default provider is `mock`, so the full pipeline runs offline. For a real model, use an OpenAI-compatible endpoint.

## Input format

Use JSONL. Each row needs `question` and one of `answer`, `cot`, or `solution`.

```json
{"question": "...", "answer": "step 1\nstep 2\nstep 3"}
```

## Run

```bash
python3 -m math_reasoning_expander.cli run \
  --input data/sample_math.jsonl \
  --output outputs/expanded.jsonl \
  --provider mock
```

By default, the pipeline attempts at most 10 randomly selected mask points and accepts a generated fill only when the aggregate quality score reaches 0.80. For each selected point, it performs up to 3 feedback-regeneration attempts before moving on. Whenever a generated fill improves over the masked original node, that fill is immediately retained and merged into the final expanded CoT, then the next random point is selected. The output JSONL records all retained rounds in `retained_expansions`, and records node-count expansion in `expansion_stats` so you can see how many reasoning points existed before and after expansion. It also writes a before/after SVG quality comparison report next to the JSONL output:

```text
outputs/expanded.jsonl.quality.svg
```

You can choose a custom report path:

```bash
python3 -m math_reasoning_expander.cli run \
  --input data/sample_math.jsonl \
  --output outputs/expanded.jsonl \
  --quality-report outputs/quality.svg \
  --provider mock
```

Inspect the parsed reasoning graph:

```bash
python3 -m math_reasoning_expander.cli inspect \
  --input data/sample_math.jsonl \
  --index 0
```

## Real LLM provider

Set these environment variables for an OpenAI-compatible API. Do not hard-code API keys in source files.

```bash
export MATH_EXPANDER_API_KEY="..."
export MATH_EXPANDER_BASE_URL="https://api.gpt.ge/v1"
export MATH_EXPANDER_MODEL="gpt-3.5-turbo"
```

Then run:

```bash
python3 -m math_reasoning_expander.cli run \
  --input data/sample_math.jsonl \
  --output outputs/expanded.real.jsonl \
  --provider openai_compatible \
  --base-url "$MATH_EXPANDER_BASE_URL" \
  --model "$MATH_EXPANDER_MODEL"
```

## Main modules

- `parser.py`: splits CoT into steps and builds a lightweight reasoning graph.
- `masking.py`: masks single nodes, formula nodes, or short paths.
- `llm.py`: mock and OpenAI-compatible fill clients.
- `evaluators.py`: similarity, logic consistency, formula correctness, completeness, and reasoning gain.
- `pipeline.py`: iterative feedback loop with adaptive stopping.
- `visualization.py`: SVG before/after line charts for aggregate score and average metric scores.

## Suggested next steps

1. Replace `MockLLMClient` with MathFimer-style local inference or a stronger API model.
2. Add a PRM scorer as another metric in `MultiFeedbackEvaluator`.
3. Add symbolic verification with SymPy if the runtime includes it.
4. Add MathAgent-style constraint graph generation before this expansion pipeline.
