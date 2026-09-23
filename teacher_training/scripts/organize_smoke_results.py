"""将旧版散落的 Teacher smoke 结果安全整理到统一目录。

默认只预览；显式传入 --apply 才移动文件。不覆盖已有目标。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def planned_moves(results_root: Path, domain: str) -> list[tuple[Path, Path]]:
    prefix = f"{domain}_smoke"
    bundle = results_root / prefix
    candidates = [
        (results_root / f"{prefix}_predictions.jsonl", bundle / "predictions.jsonl"),
        (results_root / f"{prefix}_generation.png", bundle / "generation" / "generation.png"),
        (results_root / f"{prefix}_base_eval", bundle / "evaluation" / "base"),
        (results_root / f"{prefix}_lora_eval", bundle / "evaluation" / "lora"),
        (results_root / f"{prefix}_comparison_artifacts", bundle / "figures" / "comparison_artifacts"),
        (results_root / f"{prefix}_formal_eval", bundle / "evaluation" / "formal"),
        (results_root / f"{prefix}_eval", bundle / "evaluation" / "combined"),
    ]
    # 兼容历史上以 formatting_smoke_comparison_* 散落保存的单文件。
    for source in sorted(results_root.glob(f"{prefix}_comparison.*")):
        candidates.append((source, bundle / "figures" / "legacy" / source.name))
    for source in sorted(results_root.glob(f"{prefix}_comparison_*")):
        if source.name != f"{prefix}_comparison_artifacts":
            candidates.append((source, bundle / "figures" / "legacy" / source.name))
    return [(source, target) for source, target in candidates if source.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 Teacher smoke 结果；默认仅预览。")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--domain", required=True, choices=("formatting", "identifier", "local_structure", "control_flow"))
    parser.add_argument("--apply", action="store_true", help="执行移动；不传时只打印计划")
    args = parser.parse_args()
    root = Path(args.results_root).expanduser().resolve()
    moves = planned_moves(root, args.domain)
    conflicts = [{"source": str(s), "target": str(t)} for s, t in moves if t.exists()]
    report = {
        "mode": "apply" if args.apply else "preview",
        "bundle": str(root / f"{args.domain}_smoke"),
        "moves": [{"source": str(s), "target": str(t)} for s, t in moves],
        "conflicts": conflicts,
    }
    if conflicts:
        raise FileExistsError("目标已存在，为防止覆盖已中止：" + json.dumps(conflicts, ensure_ascii=False))
    if args.apply:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        manifest = root / f"{args.domain}_smoke" / "organization_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["manifest"] = str(manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
