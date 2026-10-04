"""论文启发的多分类数据集研究入口：python tabular_data/research.py。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabular_data.paths import DATASET_DIR, OUTPUT_DIR
from tabular_data.run_logging import atomic_json, training_output_dir


from tabular_data.dataset_registry import DATASETS, RARE_CLASS_DATASETS, RELATIONAL_DATASETS, ROUND3_DATASETS, SPECS
RESEARCH_DATASETS = DATASETS


def experiment_plan(seeds=(42,), train_fraction=1., profile="weak"):
    return [dict(dataset=dataset, dataset_view="full", feature_mode="reasoned", profile=profile,
                 seed=seed, train_fraction=train_fraction, iterations=5, candidates_per_round=5)
            for dataset in DATASETS for seed in seeds]


def write_summary(results, directory):
    metrics = ("accuracy", "balanced_accuracy", "f1", "f1_macro", "auc", "log_loss")
    lines = ["# 特征生成研究结果", "", "以下为已完成运行；所有数值来自各运行的独立测试报告。",
             "同一行内 baseline 与 optimized 的数据、模型配置相同。跨数据集的绝对分数不能当作统一难度标尺。",
             "主指标为 Macro F1；Log Loss 越低越好，其余越高越好。分数显示四位小数，增益用百分点。", "",
             "| 数据集/视图 | 模型 | 生成器 | 种子 | 训练行数 | 接受数 | 测试 Macro F1：基线 → 优化 | Δ百分点 | 实际 LLM 调用 |",
             "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for r in results:
        b, o = r["test"]["baseline"], r["test"]["optimized"]
        generation = r["feature_generation"]
        source = "LLM" if generation["api_calls_succeeded"] else "离线"
        lines.append(f"| {r['dataset']}/{r['experiment']['dataset_view']} | {r['model']['profile']} | "
            f"{generation['mode']} / {generation.get('version', '未记录')} ({source}) | {r['model']['parameters']['random_state']} | {r['rows']['train']} | "
            f"{len(r['accepted_features'])} | {b['f1_macro']:.4f} → {o['f1_macro']:.4f} | "
            f"{100*(o['f1_macro']-b['f1_macro']):+.4f} | {r['feature_generation']['api_calls_succeeded']} |")
    for r in results:
        name = f"{r['dataset']}/{r['experiment']['dataset_view']}/{r['model']['profile']}/{r['feature_generation']['mode']}/seed_{r['model']['parameters']['random_state']}"
        generation = r["feature_generation"]
        name += "/LLM" if generation["api_calls_succeeded"] else "/offline"
        link = Path(os.path.relpath(Path(r["output_dir"]), directory)).as_posix()
        lines += ["", f"## {name}", "", f"[完整日志]({link}/training.log) · [机器报告]({link}/training_results.json)", "",
                  "| 指标 | baseline | optimized | 改善量（正数更好） |", "|---|---:|---:|---:|"]
        for metric in metrics:
            lines.append(f"| {metric} | {r['test']['baseline'][metric]:.4f} | {r['test']['optimized'][metric]:.4f} | {r['test']['improvement'][metric]:+.4f} |")
        lines += ["", "接受特征：" + (", ".join(r["accepted_features"]) or "无"),
                  f"实际训练预算：{r['experiment']['training_budget']['used_rows']}/{r['experiment']['training_budget']['available_rows']}。"]
        lines += [f"训练流程版本：{r.get('training_version', '未记录')}；生成器版本：{r['feature_generation'].get('version', '未记录')}。"]
        lines += [f"真实响应次数：{generation['api_calls_succeeded']}；本地规则/回退次数：{generation.get('local_rule_calls', 0)}。格式修复重试计入真实响应次数，因此可能超过轮数。"]
        if generation.get("llm_model"):
            lines += [f"LLM 模型：`{generation['llm_model']}`；允许回退：{generation.get('fallback_allowed', False)}。"]
        if r.get("test_class_diagnostics"):
            counts = r["experiment"]["training_budget"]["class_counts"]
            lines += ["", "| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |",
                      "|---|---:|---:|---|---|---|"]
            for label, count in sorted(counts.items()):
                b, o = (r["test_class_diagnostics"][s][label] for s in ("baseline", "optimized"))
                lines.append(f"| {label} | {count} / {100*count/sum(counts.values()):.2f}% | {int(b['support'])} | "
                             f"{b['precision']:.4f} → {o['precision']:.4f} | {b['recall']:.4f} → {o['recall']:.4f} | "
                             f"{b['f1-score']:.4f} → {o['f1-score']:.4f} |")
        if r["dataset"] in (*RARE_CLASS_DATASETS, *RELATIONAL_DATASETS, *ROUND3_DATASETS):
            spec = SPECS[r["dataset"]]
            lines += ["", f"[论文依据]({spec['benchmark_reference']})：{spec['literature_scope']}", spec["scope"]]
        if r["ablation"]["feature_rows"]:
            lines += ["", "| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |",
                      "|---|---:|---:|---:|---|"]
            for row in r["ablation"]["feature_rows"]:
                lines.append(f"| {row['feature']} | {row['split_count']} | {row['importance_score']:.4f} ± {row['importance_std']:.4f} | "
                             f"{row['drop_retrain_score_drop']['f1_macro']:+.4f} | {'通过' if row['supported_predictive_contribution'] else '未通过'} |")
            lines += ["", "存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。"]
    lines += ["", "## 解释边界", "", "CART 和残差预筛只使用训练集；正式接受仍要求验证主指标改善且其他指标不退化。测试退化照实保留。",
              "这是固定外层划分的实验。多个 seed 改变模型/CART/预算采样，不代表多个独立数据划分；同一测试集反复评估也不构成独立重复。",
              "局部模板与 CART 生成结果不等于真实 LLM 成绩。训练残差预筛不是 OpenFE FeatureBoost 的原样复现。",
              "离线与 LLM 对照还需核对生成器版本。当前 Jungle Chess 离线报告为 v5、LLM 为 v6，不能把两者差异完全归因于 LLM；各自相对 baseline 的增益仍来自对应独立测试报告。",
              "当前生成器包含候选池、CART 路径与预筛策略；不能将收益单独归因于任一组件。CART 是从训练标签学习的监督式特征构造。",
              "当前是固定划分的三个已筛选任务，不能推广为所有表格任务的有效性证据；棋盘任务的随机划分不证明对未见棋子组合的泛化。", ""]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest_summary.md").write_text("\n".join(lines), encoding="utf-8")


def sparse_labels(result, maximum_training_fraction=.05):
    """由训练计数定义诊断组；不根据测试表现选类别。"""
    counts = result["experiment"]["training_budget"]["class_counts"]
    return sorted(label for label, count in counts.items() if count / sum(counts.values()) <= maximum_training_fraction)


def sparse_class_scores(result):
    labels = sparse_labels(result)
    if not labels:
        return None
    return {stage: sum(result["test_class_diagnostics"][stage][label]["f1-score"] for label in labels) / len(labels)
            for stage in ("baseline", "optimized")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("weak", "reference"), default="weak", help="同一配置用于全部数据集")
    parser.add_argument("--offline", action="store_true", help="明确使用本地候选，不调用 LLM")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="模型/CART/预算种子；外层数据划分固定")
    parser.add_argument("--train-fraction", type=float, default=1., help="可选受限训练预算；0.5 使用原训练集一半")
    args = parser.parse_args()
    import math
    if any(seed < 0 for seed in args.seeds) or not math.isfinite(args.train_fraction) or not 0 < args.train_fraction <= 1:
        parser.error("种子必须非负；训练预算必须在 (0,1] 内")
    plan = experiment_plan(args.seeds, args.train_fraction, args.profile)
    missing = [c["dataset"] for c in plan if not (DATASET_DIR/c["dataset"]/"manifest.json").exists()]
    if missing:
        parser.error(f"数据尚未准备：{sorted(set(missing))}；请先运行 python -m tabular_data.prepare_datasets")
    directory = OUTPUT_DIR / "research"
    directory.mkdir(parents=True, exist_ok=True)
    # 新入口使用自己的输出根目录，完整保留用户已完成的历史结果。
    runs_root = directory / "runs"
    state = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
             "outer_split_seed": 42, "test_used_for_plan": False, "plan": plan, "completed": []}
    atomic_json(directory / "experiment_plan.json", state)
    results = []
    import os
    env = dict(os.environ, TABULAR_OUTPUT_DIR=str(runs_root))
    for config in plan:
        command = [sys.executable, "-u", str(Path(__file__).with_name("train.py"))]
        for key, value in config.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        if args.offline:
            command.append("--offline")
        code = subprocess.call(command, env=env)
        if code:
            state.update(status="failed", failed_configuration=config, exit_code=code)
            atomic_json(directory / "experiment_plan.json", state)
            return code
        output = training_output_dir(runs_root, **config)
        result = json.loads((output/"training_results.json").read_text(encoding="utf-8"))
        results.append(result)
        state["completed"].append({"configuration": config, "run_id": result["run_id"], "output_dir": str(output)})
        atomic_json(directory / "experiment_plan.json", state)
        write_summary(results, directory)
    state["status"] = "completed"
    atomic_json(directory / "experiment_plan.json", state)
    from tabular_data.refresh_summary import refresh_summary
    refresh_summary(OUTPUT_DIR)
    print(f"研究汇总：{directory / 'latest_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
