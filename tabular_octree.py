from __future__ import annotations

"""Tabular OCTree 主线。

这一部分是中期工作的主体：LLM 不是直接改标签，而是生成可执行的新特征代码。
每个候选特征都要先过沙盒检查，再用模型和多指标评估决定是否保留。
整体思路保留 test_advanced / test_final 的闭环，只是整理成 Flask 可以调用的函数。
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from config import MAX_ITERATIONS, OUTPUT_DIR, SAMPLE_DATA_DIR
from llm_client import LLMClient
from result_schema import ResultRecord, write_json

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


TARGET_CANDIDATES = ["target", "label", "class", "y", "is_phishing", "phishing", "quality"]


def get_data() -> pd.DataFrame:
    """没有上传文件时，优先使用 sample_data/sample_tabular.csv，否则使用 sklearn wine。"""
    sample_path = SAMPLE_DATA_DIR / "sample_tabular.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    if "target" not in df.columns:
        df["target"] = data.target
    return df


def get_data_with_llm(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """兼容旧代码函数名，实际预处理交给 preprocess_dataframe。"""
    return preprocess_dataframe(df, target_col=target_col)


def infer_target_column(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """自动识别目标列。用户指定优先，其次匹配常见名字，最后退回最后一列。"""
    if target_col and target_col in df.columns:
        return target_col
    lowered = {str(col).lower(): col for col in df.columns}
    for name in TARGET_CANDIDATES:
        if name in lowered:
            return str(lowered[name])
    for col in df.columns:
        lower = str(col).lower()
        if any(key in lower for key in ["target", "label", "class", "是否", "类别", "结果"]):
            return str(col)
    return str(df.columns[-1])


def preprocess_dataframe(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """清洗数据并修复 target leakage。

    关键点：先识别 target_col，再从特征矩阵中 drop 掉原始目标列，
    不能让目标列既当标签又当特征。
    """
    raw = df.copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    detected_target = infer_target_column(raw, target_col)
    y_raw = raw[detected_target].copy()
    feature_df = raw.drop(columns=[detected_target]).copy()

    metadata: Dict[str, Any] = {
        "target_col": detected_target,
        "original_columns": list(raw.columns),
        "feature_columns": list(feature_df.columns),
        "rows": int(len(raw)),
        "target_leakage_fixed": True,
        "encoders": {},
    }

    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            median = feature_df[col].median()
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(median if pd.notna(median) else 0)
        else:
            feature_df[col] = feature_df[col].fillna("missing").astype(str)
            encoder = LabelEncoder()
            feature_df[col] = encoder.fit_transform(feature_df[col])
            metadata["encoders"][col] = list(map(str, encoder.classes_[:20]))

    if len(feature_df.columns) > 0:
        scaler = MinMaxScaler()
        feature_df[feature_df.columns] = scaler.fit_transform(feature_df[feature_df.columns])

    numeric_target = pd.to_numeric(y_raw, errors="coerce")
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > max(20, int(math.sqrt(max(len(y_raw), 1)))):
        y = numeric_target.fillna(numeric_target.median())
        task_type = "regression"
    else:
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y_raw.fillna("missing").astype(str)), index=raw.index)
        task_type = "classification"
        metadata["target_classes"] = list(map(str, target_encoder.classes_))

    numeric = feature_df.copy()
    numeric["target"] = y
    metadata["task_type"] = task_type
    metadata["processed_columns"] = list(numeric.columns)
    return numeric, detected_target, metadata


def dataframe_stats(df: pd.DataFrame, max_cols: int = 20) -> Dict[str, Any]:
    """给 LLM 的简要数据画像，只传必要统计量，避免 prompt 太长。"""
    stats: Dict[str, Any] = {"shape": list(df.shape), "columns": list(df.columns)}
    desc = df.drop(columns=["target"], errors="ignore").describe().T.head(max_cols)
    stats["describe"] = json.loads(desc.to_json(orient="index"))
    return stats


def generate_domain_knowledge(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """生成轻量领域知识，模拟原项目里的知识库提示。"""
    cols = ", ".join(metadata.get("feature_columns", [])[:20])
    return (
        f"数据共有 {metadata.get('rows')} 行，目标列为 {metadata.get('target_col')}。"
        f"可用特征包括：{cols}。OCTree 优先尝试比例、差值、平方、交互项等可解释特征。"
    )


def check_feature_robustness(df: pd.DataFrame, feature_code: str) -> Tuple[bool, pd.DataFrame, str, List[str]]:
    """执行 LLM 特征代码，并检查所有新增列是否有效。"""
    before = df.copy()
    before_columns = list(before.columns)
    safe_df = before.copy()
    namespace = {
        "__builtins__": {},
        "df": safe_df,
        "np": np,
        "pd": pd,
        "math": math,
    }
    try:
        exec(feature_code, namespace, namespace)
    except Exception as exc:
        return False, before, f"特征代码执行失败：{type(exc).__name__}: {exc}", []

    new_columns = [col for col in safe_df.columns if col not in before_columns]
    if not new_columns:
        return False, before, "特征代码没有生成任何新列。", []

    for col in new_columns:
        series = safe_df[col]
        if len(series) != len(before):
            return False, before, f"新特征 {col} 长度不一致。", new_columns
        if series.isna().any():
            return False, before, f"新特征 {col} 存在 NaN。", new_columns
        if not pd.api.types.is_numeric_dtype(series):
            return False, before, f"新特征 {col} 不是数值类型。", new_columns
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            return False, before, f"新特征 {col} 无法稳定转成数值。", new_columns
        if np.isinf(values.to_numpy()).any():
            return False, before, f"新特征 {col} 存在 Inf，可能有除零风险。", new_columns
        if values.nunique(dropna=False) <= 1:
            return False, before, f"新特征 {col} 是常数列。", new_columns
        for old_col in before_columns:
            if old_col == "target":
                continue
            old_values = pd.to_numeric(before[old_col], errors="coerce")
            if values.reset_index(drop=True).equals(old_values.reset_index(drop=True)):
                return False, before, f"新特征 {col} 与已有列 {old_col} 完全重复。", new_columns
        safe_df[col] = values
    return True, safe_df, f"通过沙盒检查，新增列：{', '.join(map(str, new_columns))}", new_columns


def _classification_model(n_classes: int):
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=42,
        )
    return RandomForestClassifier(n_estimators=80, random_state=42)


def _regression_model():
    if XGBRegressor is not None:
        return XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42)
    return RandomForestRegressor(n_estimators=80, random_state=42)


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        return None


def _safe_log_loss(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    try:
        return float(log_loss(y_true, proba, labels=np.unique(y_true)))
    except Exception:
        return None


def evaluate_and_reason(df: pd.DataFrame, task_type: Optional[str] = None) -> Dict[str, Any]:
    """五折交叉验证评估当前特征集合，并导出 CART 规则作为自然语言反馈基础。"""
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()
    if task_type is None:
        task_type = "classification" if len(np.unique(y)) <= max(20, int(math.sqrt(max(len(y), 1)))) else "regression"

    if task_type == "classification":
        n_classes = len(np.unique(y))
        min_count = int(pd.Series(y).value_counts().min())
        folds = max(2, min(5, min_count))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        model = _classification_model(n_classes)
        pred = cross_val_predict(clone(model), X, y, cv=cv, method="predict")
        try:
            proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
        except Exception:
            proba = np.eye(n_classes)[pred]
        metrics = {
            "accuracy": float(accuracy_score(y, pred)),
            "auc": _safe_auc(y, proba),
            "f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "log_loss": _safe_log_loss(y, proba),
            "mse": float(mean_squared_error(y, pred)),
        }
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X, y)
    else:
        folds = max(2, min(5, len(df)))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        model = _regression_model()
        pred = cross_val_predict(clone(model), X, y, cv=cv)
        metrics = {
            "accuracy": None,
            "auc": None,
            "f1": None,
            "log_loss": None,
            "mse": float(mean_squared_error(y, pred)),
        }
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(X, y)

    try:
        cart_rules = export_text(tree, feature_names=list(X.columns))
    except Exception:
        cart_rules = "决策树规则导出失败。"
    return {"metrics": metrics, "cart_rules": cart_rules, "feature_names": list(X.columns)}


def _metric_improvement(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    improvement: Dict[str, Any] = {}
    for key in ["accuracy", "auc", "f1"]:
        b, o = baseline.get(key), optimized.get(key)
        improvement[key] = None if b is None or o is None else float(o - b)
    for key in ["log_loss", "mse"]:
        b, o = baseline.get(key), optimized.get(key)
        improvement[key] = None if b is None or o is None else float(b - o)
    return improvement


def get_strategy(history_log: List[str], baseline_metrics: Dict[str, Any]) -> str:
    if not history_log:
        return "首轮优先生成简单、可解释、低风险的比例或交互特征。"
    return "根据上一轮沙盒和模型反馈，避免常数列、重复列和除零，优先生成能改变树分裂的数值特征。"


def ask_llm_for_feature(
    llm_client: LLMClient,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    history_log: List[str],
) -> str:
    """调用统一 LLMClient，失败时自动走 mock。"""
    return llm_client.generate_feature_code(
        columns=[col for col in df.columns if col != "target"],
        stats=dataframe_stats(df),
        history=history_log + [generate_domain_knowledge(df, metadata)],
    )


def _is_better(baseline: Dict[str, Any], candidate: Dict[str, Any], task_type: str) -> bool:
    improvement = _metric_improvement(baseline, candidate)
    if task_type == "classification":
        gains = [improvement.get("accuracy"), improvement.get("f1"), improvement.get("auc"), improvement.get("log_loss"), improvement.get("mse")]
        return any(g is not None and g > 0.0001 for g in gains)
    mse_gain = improvement.get("mse")
    return mse_gain is not None and mse_gain > 0.0001


def run_octree_analysis(
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
    llm_client: Optional[LLMClient] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Flask 调用的 Tabular 主函数。"""
    llm_client = llm_client or LLMClient()
    raw_df = df.copy() if df is not None else get_data()
    processed, detected_target, metadata = get_data_with_llm(raw_df, target_col=target_col)
    task_type = metadata.get("task_type", "classification")

    baseline_report = evaluate_and_reason(processed, task_type=task_type)
    baseline_metrics = baseline_report["metrics"]
    current_df = processed.copy()
    best_metrics = baseline_metrics.copy()
    history_log: List[str] = []
    iterations: List[Dict[str, Any]] = []
    accepted_features: List[str] = []
    generated_rules: List[str] = []

    for round_idx in range(1, max_iterations + 1):
        strategy = get_strategy(history_log, baseline_metrics)
        feature_code = ask_llm_for_feature(llm_client, current_df, metadata, history_log)
        generated_rules.append(feature_code)
        valid, candidate_df, sandbox_reason, new_cols = check_feature_robustness(current_df, feature_code)
        accepted = False
        candidate_metrics: Dict[str, Any] = best_metrics.copy()
        cart_feedback = sandbox_reason
        if valid:
            candidate_report = evaluate_and_reason(candidate_df, task_type=task_type)
            candidate_metrics = candidate_report["metrics"]
            accepted = _is_better(best_metrics, candidate_metrics, task_type)
            cart_feedback = f"{sandbox_reason}\n决策树反馈：\n{candidate_report['cart_rules'][:1200]}"
            if accepted:
                current_df = candidate_df
                best_metrics = candidate_metrics
                accepted_features.extend(new_cols)
        feedback = (
            f"第 {round_idx} 轮策略：{strategy}\n"
            f"沙盒结果：{sandbox_reason}\n"
            f"是否接受：{'是' if accepted else '否'}"
        )
        history_log.append(feedback)
        iterations.append(
            {
                "round": round_idx,
                "feature_code": feature_code,
                "new_columns": new_cols,
                "valid": valid,
                "accepted": accepted,
                "reason": sandbox_reason,
                "metrics": candidate_metrics,
                "feedback": cart_feedback,
            }
        )

    optimized_report = evaluate_and_reason(current_df, task_type=task_type)
    optimized_metrics = optimized_report["metrics"]
    improvement = _metric_improvement(baseline_metrics, optimized_metrics)
    accepted = len(accepted_features) > 0
    feedback = "\n\n".join(history_log) or "未生成反馈。"

    record = ResultRecord.create(
        modality="tabular",
        input_summary={
            "rows": int(len(raw_df)),
            "columns": list(map(str, raw_df.columns)),
            "target_col": detected_target,
            "task_type": task_type,
        },
        structured_representation={
            "processed_columns": list(map(str, current_df.columns)),
            "domain_knowledge": generate_domain_knowledge(current_df, metadata),
            "decision_tree_rules": optimized_report["cart_rules"],
        },
        generated={
            "feature_codes": generated_rules,
            "accepted_features": accepted_features,
            "iterations": iterations,
        },
        verification={
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "improvement": improvement,
        },
        feedback=feedback,
        accepted=accepted,
        round=max_iterations,
        metrics={"baseline": baseline_metrics, "optimized": optimized_metrics, "improvement": improvement},
    )

    result = {
        "success": True,
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "improvement": improvement,
        "iterations": iterations,
        "generated_rules": generated_rules,
        "accepted_features": accepted_features,
        "decision_tree_explanation": optimized_report["cart_rules"],
        "history_log": history_log,
        "record": record.to_dict(),
    }

    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        augmented_path = OUTPUT_DIR / "tabular_augmented.csv"
        results_path = OUTPUT_DIR / "tabular_results.json"
        metrics_path = OUTPUT_DIR / "metrics.csv"
        current_df.to_csv(augmented_path, index=False, encoding="utf-8-sig")
        metrics_rows = []
        for stage, values in [("baseline", baseline_metrics), ("optimized", optimized_metrics), ("improvement", improvement)]:
            row = {"stage": stage}
            row.update(values)
            metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        record.saved_path = str(results_path)
        result["record"] = record.to_dict()
        write_json(results_path, result)
    return result


def run_tabular_demo() -> ResultRecord:
    result = run_octree_analysis()
    return ResultRecord(**result["record"])


def load_table_file(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    return pd.read_csv(input_path)
