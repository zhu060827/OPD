from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .llm import create_llm_client
from .pipeline import MathReasoningExpansionPipeline, expansion_record_to_dict
from .visualization import write_quality_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand mathematical reasoning data with graph masking and feedback.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the expansion pipeline on a JSONL dataset.")
    run.add_argument("--input", required=True, help="Input JSONL with question and answer/cot/solution fields.")
    run.add_argument("--output", required=True, help="Output JSONL path.")
    run.add_argument("--provider", default="mock", choices=["mock", "openai", "openai_compatible"])
    run.add_argument("--model", default=None, help="gpt-3.5-turbo")
    run.add_argument("--base-url", default=None, help="https://api.gpt.ge/v1")
    run.add_argument(
        "--api-key-env",
        default="MATH_EXPANDER_API_KEY",
        help="Environment variable containing the provider API key.",
    )
    run.add_argument("--max-iterations", type=int, default=10)
    run.add_argument(
        "--max-refine-iterations",
        type=int,
        default=3,
        help="Maximum feedback-regeneration attempts for each selected masked node.",
    )
    run.add_argument("--patience", type=int, default=2)
    run.add_argument("--accept-threshold", type=float, default=0.80)
    run.add_argument("--mask-strategy", default="auto", choices=["auto", "single_node", "formula_node", "path"])
    run.add_argument("--mask-width", type=int, default=1)
    run.add_argument("--limit", type=int, default=0, help="Optional maximum number of records to process.")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument(
        "--quality-report",
        default=None,
        help="SVG quality report path. Defaults to '<output>.quality.svg'.",
    )
    run.add_argument(
        "--disable-quality-report",
        action="store_true",
        help="Disable SVG quality report generation.",
    )

    inspect = subparsers.add_parser("inspect", help="Parse a single record and print its reasoning graph.")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--index", type=int, default=0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_pipeline(args)
    elif args.command == "inspect":
        inspect_record(args)


def run_pipeline(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    api_key = None
    if args.provider != "mock":
        import os

        api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    llm = create_llm_client(args.provider, model=args.model, base_url=args.base_url, api_key=api_key)
    pipeline = MathReasoningExpansionPipeline(
        llm=llm,
        max_iterations=args.max_iterations,
        max_refine_iterations=args.max_refine_iterations,
        patience=args.patience,
        accept_threshold=args.accept_threshold,
        seed=args.seed,
    )
    expanded = []
    for idx, record in enumerate(records, start=1):
        result = pipeline.expand_record(record, mask_strategy=args.mask_strategy, mask_width=args.mask_width)
        expanded.append(expansion_record_to_dict(result))
        print(
            f"[{idx}/{len(records)}] accepted={result.accepted} "
            f"score={result.evaluation['aggregate_score']:.3f} "
            f"nodes={result.expansion_stats['original_node_count']}->{result.expansion_stats['expanded_node_count']} "
            f"(+{result.expansion_stats['added_node_count']}) "
            f"masked={result.masked_node_ids}"
        )
    write_jsonl(args.output, expanded)
    print(f"Wrote {len(expanded)} expanded records to {Path(args.output).resolve()}")
    if not args.disable_quality_report:
        report_path = args.quality_report or f"{args.output}.quality.svg"
        write_quality_report(report_path, expanded)
        print(f"Wrote quality report to {Path(report_path).resolve()}")


def inspect_record(args: argparse.Namespace) -> None:
    from .parser import ReasoningGraphParser

    records = read_jsonl(args.input)
    record = records[args.index]
    parser = ReasoningGraphParser()
    graph = parser.parse(record.get("question", ""), record.get("answer") or record.get("cot") or record.get("solution", ""))
    print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
