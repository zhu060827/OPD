"""按数据注册表下载原始数据快照，并记录可复现清单。

本脚本只下载，不把原始数据直接标成某个 Teacher 领域。许可证需由实验者在下载前确认。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "commitpackft": "bigcode/commitpackft",
    "codesearchnet": "code_search_net",
    "code_contests": "deepmind/code_contests",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Teacher 数据构造所需的公开原始数据。")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="python")
    parser.add_argument("--revision", help="Hugging Face 数据集 revision/commit；正式实验必须固定")
    parser.add_argument("--limit", type=int, help="仅用于冒烟测试")
    args = parser.parse_args()
    if not args.revision and not args.limit:
        parser.error("正式下载必须通过 --revision 固定数据版本；冒烟测试可使用 --limit")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("请先安装 teacher_training/requirements.txt") from exc
    name = DATASETS[args.dataset]
    kwargs = {"revision": args.revision} if args.revision else {}
    configuration = args.language if args.dataset == "codesearchnet" else None
    dataset = load_dataset(name, configuration, split=args.split, **kwargs)
    if args.language and "language" in dataset.column_names:
        dataset = dataset.filter(lambda row: str(row.get("language", "")).lower() == args.language.lower())
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    output = Path(args.output_dir) / args.dataset / args.language / args.split
    output.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output / "dataset"))
    manifest = {
        "dataset": name,
        "configuration": configuration,
        "split": args.split,
        "language": args.language,
        "revision": args.revision,
        "samples": len(dataset),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "raw_source_only_requires_domain_construction",
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
