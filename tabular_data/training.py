"""正式训练：20% 训练、40% 验证筛选特征、40% 独立测试。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import build_info, __version__ as XGBOOST_VERSION

from llm_client import LLMClient
from result_schema import write_json
from .evaluation import (
    PROFILES, SELECTION_POLICY, accept_candidate, applicable_metrics, compute_metrics,
    make_model, metric_improvement, predict_model,
)
from .feature_generation import OfflineFeatureClient
from .paths import DATASET_DIR, OUTPUT_DIR
from .tabular_octree import check_feature_robustness, evaluate_feature_round, load_table_file
from .feature_proposals import GENERATOR_VERSION
from .feature_feedback import DEFAULT_CANDIDATES, DEFAULT_ROUNDS
from .ablation import run_ablation, ablation_table
from .run_logging import TRAINING_VERSION, atomic_json, training_output_dir, utc_now
from .split_validation import audit_class_counts, label_counts, load_split_policy, print_distribution, write_distribution
from .feature_explanations import display_values
from .research_protocol import training_budget
from .dataset_registry import DATASETS, RELATIONAL_DATASETS, ROUND3_DATASETS
from .model_guidance import build_guidance


SPLIT_PROTOCOL = "holdout_20_40_40"
NOISE_FEATURES = {"jungle_chess": (), "pendigits": None, "gesture_phase": None,
                  "thyroid_ann": (), "balance_scale": (), "connect4": (),
                  **{key: () for key in (*RELATIONAL_DATASETS, *ROUND3_DATASETS)}}


class TrainPreprocessor:
    """只在训练集学习缺失值填充、类别编码和标签映射，数值保留原始尺度。"""

    def fit(self, frame, task_type, *, categorical_encoding="ordinal"):
        if categorical_encoding not in {"ordinal", "onehot"}:
            raise ValueError("未知类别编码策略。")
        self.categorical_encoding = categorical_encoding
        self.task_type = task_type
        self.columns = [col for col in frame if col != "target"]
        self.numeric = [col for col in self.columns if pd.api.types.is_numeric_dtype(frame[col])]
        self.medians = {}
        self.categories = {}
        for col in self.columns:
            if col in self.numeric:
                values = frame[col].replace([np.inf, -np.inf], np.nan)
                median = values.median()
                self.medians[col] = float(median) if pd.notna(median) else 0.0
            else:
                self.categories[col] = sorted(frame[col].fillna("__missing__").astype(str).unique().tolist())
        self.onehot_columns = {}
        if categorical_encoding == "onehot":
            import re
            for col, values in self.categories.items():
                for index, value in enumerate(values):
                    suffix = value if re.fullmatch(r"[a-zA-Z0-9_]+", value) else f"category_{index}"
                    name = f"{col}__is_{suffix}"
                    if name in self.columns or name in self.onehot_columns:
                        raise ValueError(f"独热编码列名冲突：{name}")
                    self.onehot_columns[name] = {"column": col, "category": value}
        self.generator_columns = self.numeric + list(self.onehot_columns)
        if frame.target.isna().any():
            raise ValueError("训练标签不能缺失。")
        self.classes = sorted(frame.target.astype(str).unique().tolist()) if task_type == "classification" else []
        if task_type == "classification" and len(self.classes) < 2:
            raise ValueError("分类训练集至少需要两个类别。")
        return self

    def transform(self, frame):
        if set(frame.columns) != set(self.columns + ["target"]):
            raise ValueError("训练、验证和测试文件的列不一致。")
        if frame.target.isna().any():
            raise ValueError("标签不能缺失。")
        values_by_column = {}
        for col in self.columns:
            if col in self.medians:
                values_by_column[col] = pd.to_numeric(frame[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).fillna(self.medians[col]).astype(float)
            elif self.categorical_encoding == "ordinal":
                mapping = {value: index for index, value in enumerate(self.categories[col])}
                values_by_column[col] = frame[col].fillna("__missing__").astype(str).map(mapping).fillna(-1).astype(float)
        for name, source in self.onehot_columns.items():
            values_by_column[name] = (frame[source["column"]].fillna("__missing__").astype(str)
                                     == source["category"]).astype(float)
        result = pd.DataFrame(values_by_column, index=frame.index)
        if self.task_type == "classification":
            mapping = {value: index for index, value in enumerate(self.classes)}
            labels = frame.target.astype(str).map(mapping)
            if labels.isna().any():
                raise ValueError("验证/测试集存在训练集中未见的目标类别。")
            result["target"] = labels.astype(int)
        else:
            result["target"] = pd.to_numeric(frame.target, errors="raise").astype(float)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("预处理后含有非有限数值。")
        return result

    def to_dict(self):
        return {"fit_split": "train", "scaling": "none", "task_type": self.task_type,
                "columns": self.columns, "numeric_columns": self.numeric, "medians": self.medians,
                "categories": self.categories, "categorical_encoding": self.categorical_encoding,
                "onehot_columns": self.onehot_columns, "generator_columns": self.generator_columns,
                "unknown_category_value": "all_zero" if self.categorical_encoding == "onehot" else -1,
                "target_classes": self.classes}


class FeatureNoise:
    """可复现的测量噪声压力实验；训练集决定尺度，三份数据使用独立随机流。"""

    def __init__(self, frame, dataset, strength, seed):
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("噪声强度必须是有限的非负数。")
        self.strength, self.seed = strength, seed
        columns = NOISE_FEATURES[dataset]
        if columns is None:
            columns = [col for col in frame if col != "target" and pd.api.types.is_numeric_dtype(frame[col])]
        self.scales = {}
        if strength:
            for col in columns:
                scale = float(pd.to_numeric(frame[col], errors="coerce").std(ddof=0))
                if np.isfinite(scale) and scale > 0:
                    self.scales[col] = scale

    def transform(self, frame, split):
        result = frame.copy()
        split_id = {"train": 0, "validation": 1, "test": 2}[split]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, split_id]))
        for col, scale in self.scales.items():
            result[col] = result[col].astype(float) + rng.normal(0, self.strength * scale, len(result))
        return result

    def to_dict(self):
        return {"enabled": self.strength > 0, "kind": "additive_gaussian_feature_noise",
                "strength_in_train_std": self.strength, "seed": self.seed,
                "scale_fit_split": "train", "train_feature_std": self.scales,
                "affected_splits": ["train", "validation", "test"] if self.strength else [],
                "target_unchanged": True, "source_csv_unchanged": True}


def _load_split(directory, manifest, split):
    path = directory / f"{split}.csv"
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != manifest["files"][split]["sha256"]:
        raise ValueError(f"数据文件校验失败：{path}，请重新准备数据集。")
    frame = load_table_file(path)
    if len(frame) != manifest["files"][split]["rows"]:
        raise ValueError(f"{split} 实际行数与划分记录不一致。")
    if manifest["task_type"] == "classification" and label_counts(frame) != manifest["class_counts"][split]:
        raise ValueError(f"{split} 实际类别数量与划分记录不一致，请重新准备数据。")
    return frame


def _fit(frame, task_type, n_classes, profile, seed):
    model = make_model(task_type, n_classes, profile, seed)
    model.fit(frame.drop(columns="target"), frame.target.to_numpy())
    return model


def _evaluate(model, frame, task_type):
    predictions, probabilities = predict_model(model, frame.drop(columns="target"), task_type)
    return applicable_metrics(compute_metrics(frame.target.to_numpy(), predictions, probabilities, task_type)), predictions


def _dummy_metrics(train, evaluated, task_type, n_classes):
    if task_type == "regression":
        predictions = np.full(len(evaluated), train.target.mean())
        probabilities = None
    else:
        counts = np.bincount(train.target.to_numpy(dtype=int), minlength=n_classes)
        predictions = np.full(len(evaluated), counts.argmax())
        probabilities = np.tile(counts / counts.sum(), (len(evaluated), 1))
    return applicable_metrics(compute_metrics(evaluated.target.to_numpy(), predictions, probabilities, task_type))


def _comparison(baseline, optimized):
    return {"baseline": baseline, "optimized": optimized,
            "improvement": applicable_metrics(metric_improvement(baseline, optimized))}


def run_training(dataset, *, iterations=DEFAULT_ROUNDS, profile="weak", noise_std=0.0, seed=42,
                 offline=False, save_outputs=True, llm_client=None, feature_mode="reasoned", candidates_per_round=DEFAULT_CANDIDATES,
                 permutation_repeats=10, importance_threshold=.001, dataset_view="full", train_fraction=1.):
    if iterations < 0 or seed < 0 or profile not in PROFILES:
        raise ValueError("训练参数无效。")
    if feature_mode != "reasoned" or candidates_per_round < 1 or permutation_repeats < 2:
        raise ValueError("特征生成或消融参数无效。")
    if not np.isfinite(importance_threshold) or importance_threshold < 0:
        raise ValueError("重要性阈值必须为有限非负数。")
    if dataset not in DATASETS or dataset_view != "full":
        raise ValueError("只支持当前多分类数据集的完整特征视图。")
    directory = DATASET_DIR / dataset
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["preparation"]["split"].get("protocol") != SPLIT_PROTOCOL:
        raise ValueError("数据划分不是 20/40/40，请先运行 python -m tabular_data.prepare_datasets --offline")
    task_type = manifest["task_type"]
    policy = load_split_policy()
    if manifest["preparation"]["split"].get("seed") != policy["seed"]:
        raise ValueError("数据划分种子与 split_policy.json 不一致，请先离线重建数据。")
    distribution = (audit_class_counts(manifest["class_counts"], policy=policy,
        original_counts=manifest["preparation"].get("class_distribution", {}).get("original_class_counts"))
        if task_type == "classification" else None)
    if distribution:
        print_distribution(distribution)
    raw_train = _load_split(directory, manifest, "train")
    raw_validation = _load_split(directory, manifest, "validation")
    original_columns = list(raw_train.columns[:-1])
    raw_train, budget = training_budget(raw_train, train_fraction, seed)
    experiment = {"dataset_view": dataset_view, "original_features": original_columns,
                  "used_features": list(raw_train.columns[:-1]), "training_budget": budget,
                  "outer_split_unchanged": True, "source_csv_unchanged": True, "labels_unchanged": True,
                  "effective_fraction_of_all_rows": len(raw_train)/sum(v["rows"] for v in manifest["files"].values())}
    noise = FeatureNoise(raw_train, dataset, noise_std, seed)
    noisy_train = noise.transform(raw_train, "train")
    noisy_validation = noise.transform(raw_validation, "validation")
    preprocessor = TrainPreprocessor().fit(noisy_train, task_type,
        categorical_encoding=manifest.get("categorical_encoding", "ordinal"))
    train = preprocessor.transform(noisy_train)
    validation = preprocessor.transform(noisy_validation)
    n_classes = len(preprocessor.classes)
    experiment["preprocessing"] = {"categorical_encoding": preprocessor.categorical_encoding,
        "fit_split": "train", "shared_by_baseline_and_optimized": True,
        "model_input_feature_count": len(train.columns) - 1,
        "onehot_columns": preprocessor.onehot_columns}
    variant = "clean" if noise_std == 0 else f"noise_{noise_std:g}"
    output_dir = training_output_dir(OUTPUT_DIR, dataset, profile=profile, noise_std=noise_std, feature_mode=feature_mode,
                                      iterations=iterations, candidates_per_round=candidates_per_round, seed=seed,
                                      dataset_view=dataset_view, train_fraction=train_fraction)
    run_id = os.environ.get("OPD_TRAINING_RUN_ID")
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        if distribution:
            write_distribution(output_dir, distribution)
        atomic_json(output_dir / "experiment.json", experiment)

    def save_progress(stage, completed_rounds, accepted=None):
        if save_outputs:
            atomic_json(output_dir / "progress.json", {"run_id": run_id, "training_version": TRAINING_VERSION,
                "status": "running", "stage": stage, "updated_at_utc": utc_now(), "dataset": dataset,
                "requested_rounds": iterations, "completed_rounds": len(completed_rounds),
                "iterations": completed_rounds, "accepted_features": accepted or []})
            atomic_json(output_dir / "feature_explanations.json", [
                {"round": row["round"], "candidates": row["candidates"]} for row in completed_rounds])

    save_progress("baseline_training", [])
    print(f"实验视图：{dataset_view}，使用 {len(raw_train.columns)-1}/{len(original_columns)} 列；"
          f"训练预算 {len(raw_train)}/{budget['available_rows']}，实际类别数量 {budget['class_counts']}。", flush=True)
    print(f"开始训练：{dataset}，train={len(train):,}，validation={len(validation):,}，"
          f"test={manifest['files']['test']['rows']:,}；GPU {profile}，{feature_mode} 特征 {iterations} 轮、"
          f"每轮 {candidates_per_round} 个候选，{variant}。", flush=True)
    if preprocessor.onehot_columns:
        print("共有预处理：训练集拟合独热编码，基线及优化模型共用；映射如下：\n"
              + json.dumps(experiment["preprocessing"], ensure_ascii=False, indent=2), flush=True)

    baseline_model = _fit(train, task_type, n_classes, profile, seed)
    baseline_validation, _ = _evaluate(baseline_model, validation, task_type)
    print(f"基线验证指标：{display_values(baseline_validation)}", flush=True)
    best_model, current_train, current_validation = baseline_model, train, validation
    best_metrics = baseline_validation.copy()
    client = llm_client or (OfflineFeatureClient() if offline else LLMClient())
    history, rounds, codes, accepted_features, accepted_proposals = [], [], [], [], []
    seen = set()
    for round_index in range(1, iterations + 1):
        print(f"\n阶段：特征生成与验证，第 {round_index}/{iterations} 轮。", flush=True)
        eligible = [col for col in current_train if col in preprocessor.generator_columns or col in accepted_features]
        predictions, probabilities = predict_model(best_model, current_train.drop(columns="target"), task_type)
        guidance = build_guidance(current_train, eligible, predictions=predictions, probabilities=probabilities,
                                  task_type=task_type, seed=seed, round_index=round_index)
        print("训练 CART 分支（仅作生成依据）：\n" + guidance["cart_rules"], flush=True)
        print("当前 XGBoost 训练类别诊断：" + str(display_values(guidance["class_diagnostics"])), flush=True)
        def evaluate(candidate_train, candidate_validation):
            model = _fit(candidate_train, task_type, n_classes, profile, seed)
            metrics, _ = _evaluate(model, candidate_validation, task_type)
            return {"model": model, "metrics": metrics}
        current_train, current_validation, report, selected, iteration = evaluate_feature_round(
            current_train, current_validation, evaluate, best_metrics, task_type, client, round_index, history, seen,
            candidate_count=candidates_per_round, feature_mode=feature_mode, eligible_columns=eligible, guidance=guidance)
        best_metrics = iteration["metrics_after"].copy()
        if selected:
            best_model = report["model"]
            codes.append(selected.code)
            accepted_features.append(selected.name)
            accepted_proposals.append(selected.to_dict())
        history.append(iteration["learning_feedback"])
        rounds.append(iteration)
        save_progress("feature_selection", rounds, accepted_features)

    # 特征与模型已冻结。测试集现在才读取，测试表现不参与回退、筛选或重训。
    print("\n阶段：特征与模型已冻结，开始独立测试集评估。", flush=True)
    save_progress("test_evaluation", rounds, accepted_features)
    raw_test = _load_split(directory, manifest, "test")
    noisy_test = noise.transform(raw_test, "test")
    test = preprocessor.transform(noisy_test)
    optimized_test_frame = test.copy()
    for code in codes:
        valid, optimized_test_frame, reason, _ = check_feature_robustness(optimized_test_frame, code, allow_constant=True)
        if not valid:
            raise ValueError(f"已冻结特征无法应用到独立测试集：{reason}")
    baseline_test, baseline_predictions = _evaluate(baseline_model, test, task_type)
    optimized_test, optimized_predictions = _evaluate(best_model, optimized_test_frame, task_type)
    print(f"独立测试基线指标：{display_values(baseline_test)}", flush=True)
    print(f"独立测试优化指标：{display_values(optimized_test)}", flush=True)
    print("阶段：执行特征重要性与消融检查。", flush=True)
    save_progress("ablation", rounds, accepted_features)
    ablation = run_ablation(best_model, baseline_model, current_train, optimized_test_frame, accepted_proposals,
                            task_type=task_type, profile=profile, seed=seed, repeats=permutation_repeats,
                            importance_threshold=importance_threshold)
    result = {
        "success": True, "dataset": dataset, "task_type": task_type,
        "run_id": run_id, "training_version": TRAINING_VERSION, "class_distribution": distribution,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": SPLIT_PROTOCOL, "split": manifest["preparation"]["split"],
        "rows": {**{split: item["rows"] for split, item in manifest["files"].items()}, "train": len(train)},
        "experiment": experiment,
        "source_files": {split: item["sha256"] for split, item in manifest["files"].items()},
        "model": {"backend": "xgboost", "device": "cuda:0", "profile": profile,
                  "parameters": dict(PROFILES[profile], random_state=seed, tree_method="hist"),
                  "xgboost_version": XGBOOST_VERSION,
                  "cuda_version": ".".join(map(str, build_info().get("CUDA_VERSION", [])))},
        "selection_policy": SELECTION_POLICY.copy(), "noise": noise.to_dict(),
        "feature_generation": {"client": type(client).__name__, "version": GENERATOR_VERSION,
                               "llm_model": getattr(client, "model", None),
                               "fallback_allowed": getattr(client, "use_mock_when_fails", False),
                               "mode": feature_mode, "rounds": iterations, "candidates_per_round": candidates_per_round,
                               "feedback_learning": "prior_round_validation_only",
                               "evaluated_candidates": sum(len(row["candidates"]) for row in rounds),
                               "api_calls_succeeded": getattr(client, "real_call_count", 0),
                               "local_rule_calls": getattr(client, "mock_fallback_count", 0)},
        "validation": _comparison(baseline_validation, best_metrics),
        "test": _comparison(baseline_test, optimized_test),
        "dummy": {"strategy": "majority_and_class_priors" if task_type == "classification" else "train_mean",
                  "validation": _dummy_metrics(train, validation, task_type, n_classes),
                  "test": _dummy_metrics(train, test, task_type, n_classes)},
        "iterations": rounds, "accepted_features": accepted_features, "accepted_feature_codes": codes,
        "learning_history": history,
        "accepted_feature_proposals": accepted_proposals, "ablation": ablation,
        "output_dir": str(output_dir),
        "log_file": str(output_dir / "training.log") if run_id else None,
    }
    if task_type == "classification":
        from sklearn.metrics import classification_report
        result["test_class_diagnostics"] = {
            stage: classification_report(test.target.to_numpy(), predictions, target_names=preprocessor.classes,
                                         labels=np.arange(n_classes), output_dict=True, zero_division=0)
            for stage, predictions in (("baseline", baseline_predictions), ("optimized", optimized_predictions))}
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "training_results.json", result)
        write_json(output_dir / "preprocessor.json", preprocessor.to_dict())
        write_json(output_dir / "noise.json", noise.to_dict())
        write_json(output_dir / "feature_explanations.json", [{"round": row["round"], "candidates": row["candidates"]} for row in rounds])
        write_json(output_dir / "ablation.json", ablation)
        ablation_table(ablation).to_csv(output_dir / "ablation.csv", index=False)
        baseline_model.save_model(output_dir / "baseline_model.json")
        best_model.save_model(output_dir / "optimized_model.json")
        current_train.to_csv(output_dir / "tabular_augmented.csv", index=False)
        rows = [{"split": split, "stage": stage, **metrics}
                for split in ("validation", "test") for stage, metrics in result[split].items()]
        rows.extend({"split": split, "stage": "dummy", **result["dummy"][split]} for split in ("validation", "test"))
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
        if task_type == "classification":
            labels = np.asarray(preprocessor.classes)
            baseline_predictions = labels[baseline_predictions.astype(int)]
            optimized_predictions = labels[optimized_predictions.astype(int)]
        pd.DataFrame({"target": raw_test.target, "baseline_prediction": baseline_predictions,
                      "optimized_prediction": optimized_predictions}).to_csv(output_dir / "test_predictions.csv", index=False)
        if noise_std:
            for split, frame in (("train", noisy_train), ("validation", noisy_validation), ("test", noisy_test)):
                frame.to_csv(output_dir / f"noisy_{split}.csv", index=False)
        save_progress("completed", rounds, accepted_features)
        progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
        progress["status"] = "completed"
        atomic_json(output_dir / "progress.json", progress)
        print(f"结果已保存到：{output_dir}", flush=True)
    return result
