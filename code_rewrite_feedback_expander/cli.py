from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .llm import (
    MockIterationFeedbackLLM,
    MockOPDDistributionScorer,
    OpenAICompatibleIterationFeedbackClient,
    create_rewrite_client,
)
from .opd_adapter import load_opd_scorer
from .pipeline import CodeRewriteExpansionPipeline, expansion_record_to_dict
from .reporting import build_paper_report
from .visualization import write_quality_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feedback-driven semantic-preserving code rewrite expansion.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run code rewrite expansion on a JSONL dataset.")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--provider", default="mock", choices=["mock", "openai", "openai_compatible"])
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument("--api-key-env", default="CODE_EXPANDER_API_KEY")
    run.add_argument("--max-iterations", type=int, default=10)
    run.add_argument("--accept-threshold", type=float, default=0.78)
    run.add_argument("--strategies", default="cot,style,ast,variable,control_flow")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--quality-report", default=None)
    run.add_argument("--paper-report", default=None)
    run.add_argument("--selection-policy", default="adaptive", choices=["adaptive", "random", "fixed"])
    run.add_argument("--random-seed", type=int, default=0)
    run.add_argument("--disable-quality-report", action="store_true")
    run.add_argument("--feedback-provider", default="mock", choices=["mock", "openai", "openai_compatible"])
    run.add_argument("--feedback-model", default=None)
    run.add_argument("--feedback-base-url", default=None)
    run.add_argument("--feedback-api-key-env", default="CODE_EXPANDER_FEEDBACK_API_KEY")
    run.add_argument("--opd-scorer", choices=["mock", "external"], default="mock")
    run.add_argument("--opd-scorer-factory", default=None, help="module:function factory for the deployed OPD scorer")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run(args)


def run(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    api_key = None
    if args.provider != "mock":
        import os

        api_key = os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
    client = create_rewrite_client(args.provider, model=args.model, base_url=args.base_url, api_key=api_key)
    if args.opd_scorer == "mock":
        print(
            "WARNING: OPD scoring uses MockOPDDistributionScorer. "
            "Do not use KL strategy results as formal OPD results."
        )
        opd_scorer = MockOPDDistributionScorer()
    else:
        opd_scorer = load_opd_scorer(args.opd_scorer_factory)
    if args.feedback_provider == "mock":
        feedback_llm = MockIterationFeedbackLLM()
    else:
        import os

        feedback_key = os.getenv(args.feedback_api_key_env) or os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY")
        feedback_llm = OpenAICompatibleIterationFeedbackClient(
            model=args.feedback_model or args.model,
            base_url=args.feedback_base_url or args.base_url,
            api_key=feedback_key,
        )
    pipeline = CodeRewriteExpansionPipeline(
        llm=client,
        opd_scorer=opd_scorer,
        feedback_llm=feedback_llm,
        selection_policy=args.selection_policy,
        random_seed=args.random_seed,
        max_iterations=args.max_iterations,
        accept_threshold=args.accept_threshold,
        strategies=[item.strip() for item in args.strategies.split(",") if item.strip()],
    )
    expanded = []
    for idx, record in enumerate(records, start=1):
        result = pipeline.expand_record(record)
        data = expansion_record_to_dict(result)
        expanded.append(data)
        stats = result.expansion_stats
        print(
            f"[{idx}/{len(records)}] accepted={result.accepted} "
            f"lines={stats['original_line_count']}->{stats['expanded_line_count']} "
            f"(+{stats['added_line_count']}) "
            f"reasoning={stats['original_reasoning_steps']}->{stats['expanded_reasoning_steps']} "
            f"(+{stats['added_reasoning_steps']}) "
            f"rewrites={stats['rewrite_count']}"
        )
    write_jsonl(args.output, expanded)
    print(f"Wrote {len(expanded)} expanded records to {Path(args.output).resolve()}")
    if not args.disable_quality_report:
        report_path = args.quality_report or f"{args.output}.quality.svg"
        write_quality_report(report_path, expanded)
        print(f"Wrote quality report to {Path(report_path).resolve()}")
    paper_report_path = args.paper_report or f"{args.output}.paper.json"
    Path(paper_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(paper_report_path).write_text(
        json.dumps(build_paper_report(expanded), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote paper report to {Path(paper_report_path).resolve()}")


if __name__ == "__main__":
    main()
