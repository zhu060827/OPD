from __future__ import annotations

"""结题版 Flask 后端入口。

这个文件相当于整个系统的总控制台。前端 index.html 负责展示和发请求，
真正的 Tabular / Math / Code 三类扩充逻辑都从这里分发到对应 pipeline。
继续使用 Flask，是为了兼容原来的网页、上传接口和模拟接口。
"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from code_expander import run_code_pipeline
from config import BASE_DIR, OUTPUT_DIR, UPLOAD_DIR
from llm_client import LLMClient
from math_adapter import MATH_MODULE_AVAILABLE, math_module_status, run_math_pipeline
from result_schema import read_json_if_exists, read_jsonl_tail
from safe_exec import run_python_tests
from tabular_octree import load_table_file, run_octree_analysis


app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


def safe_json(value: Any) -> Any:
    """把 numpy / Path 等对象转成前端能识别的普通 JSON。"""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def save_upload(file_storage) -> Path:
    """所有上传文件先落到 uploads/，再交给对应 pipeline。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(file_storage.filename or "upload.data").name
    path = UPLOAD_DIR / filename
    file_storage.save(path)
    return path


@app.route("/")
def index():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return send_file(index_path)
    return "index.html not found", 404


@app.route("/outputs/<path:filename>")
def download_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/api/health", methods=["GET"])
def api_health():
    """前端启动时检查后端、LLM、Math 模块和 Code 沙盒是否可用。"""
    sandbox = run_python_tests("def _x():\n    return 1", "assert _x() == 1", timeout=2)
    return jsonify(
        safe_json(
            {
                "status": "ok",
                "llm_available": LLMClient().available,
                "math_module_available": MATH_MODULE_AVAILABLE,
                "math_module": math_module_status(),
                "code_sandbox_available": bool(sandbox.get("passed")),
            }
        )
    )


@app.route("/api/results", methods=["GET"])
def api_results():
    """读取 outputs/ 下最近一次结果，给前端和队友快速查看。"""
    tabular = read_json_if_exists(OUTPUT_DIR / "tabular_results.json")
    math_rows = read_jsonl_tail(OUTPUT_DIR / "math_expanded.jsonl", limit=5)
    code_rows = read_jsonl_tail(OUTPUT_DIR / "code_repair_traces.jsonl", limit=5)
    return jsonify(
        safe_json(
            {
                "tabular": tabular,
                "math": math_rows,
                "code": code_rows,
                "files": {
                    "tabular_results": str(OUTPUT_DIR / "tabular_results.json"),
                    "tabular_augmented": str(OUTPUT_DIR / "tabular_augmented.csv"),
                    "math_expanded": str(OUTPUT_DIR / "math_expanded.jsonl"),
                    "math_quality": str(OUTPUT_DIR / "math_quality.svg"),
                    "code_repair_traces": str(OUTPUT_DIR / "code_repair_traces.jsonl"),
                    "code_quality": str(OUTPUT_DIR / "code_quality.svg"),
                    "metrics": str(OUTPUT_DIR / "metrics.csv"),
                },
            }
        )
    )


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """兼容原来的 CSV/Excel 上传功能，默认运行 Tabular OCTree 分析。"""
    try:
        file_storage = request.files.get("file")
        if not file_storage:
            return jsonify({"success": False, "error": "请上传 CSV 或 Excel 文件。"}), 400
        path = save_upload(file_storage)
        target_col = request.form.get("target_col") or request.args.get("target_col")
        df = load_table_file(path)
        result = run_octree_analysis(df=df, target_col=target_col)
        return jsonify(safe_json(result))
    except Exception as exc:
        return jsonify({"success": False, "error": f"上传分析失败：{type(exc).__name__}: {exc}"}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """保留原模拟接口，旧前端按钮仍然可用。"""
    simulated = {
        "success": True,
        "message": "OCTree 模拟完成：LLM 生成候选特征，沙盒检查，模型评估，并输出自然语言反馈。",
        "baseline": {"accuracy": 0.86, "auc": 0.91, "f1": 0.85, "log_loss": 0.34, "mse": 0.14},
        "optimized": {"accuracy": 0.91, "auc": 0.95, "f1": 0.90, "log_loss": 0.24, "mse": 0.09},
        "improvement": {"accuracy": 0.05, "auc": 0.04, "f1": 0.05, "log_loss": 0.10, "mse": 0.05},
        "generated_rules": [
            "df['feature_ratio'] = df['feature_a'] / (df['feature_b'] + 1e-6)",
            "df['feature_interaction'] = df['feature_a'] * df['feature_c']",
        ],
        "history_log": ["第 1 轮接受比例特征。", "第 2 轮接受交互特征。"],
    }
    return jsonify(simulated)


@app.route("/api/run/tabular", methods=["POST"])
def api_run_tabular():
    try:
        file_storage = request.files.get("file")
        target_col = request.form.get("target_col") or request.args.get("target_col")
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        if payload.get("target_col"):
            target_col = payload.get("target_col")
        if file_storage:
            path = save_upload(file_storage)
            df = load_table_file(path)
            result = run_octree_analysis(df=df, target_col=target_col)
        else:
            result = run_octree_analysis(target_col=target_col)
        return jsonify(safe_json(result["record"]))
    except Exception as exc:
        return jsonify({"success": False, "error": f"Tabular demo 失败：{type(exc).__name__}: {exc}"}), 500


@app.route("/api/run/math", methods=["POST"])
def api_run_math():
    try:
        file_storage = request.files.get("file")
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        if file_storage:
            path = save_upload(file_storage)
            record = run_math_pipeline(file_path=path)
        elif payload:
            record = run_math_pipeline(record=payload)
        else:
            record = run_math_pipeline()
        return jsonify(safe_json(record.to_dict()))
    except Exception as exc:
        return jsonify({"success": False, "error": f"Math demo 失败：{type(exc).__name__}: {exc}"}), 500


@app.route("/api/run/code", methods=["POST"])
def api_run_code():
    try:
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        record = run_code_pipeline(task=payload if payload else None)
        return jsonify(safe_json(record.to_dict()))
    except Exception as exc:
        return jsonify({"success": False, "error": f"Code demo 失败：{type(exc).__name__}: {exc}"}), 500


@app.route("/api/run/all_demo", methods=["POST"])
def api_run_all_demo():
    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    try:
        results["tabular"] = run_octree_analysis()["record"]
    except Exception as exc:
        errors["tabular"] = f"{type(exc).__name__}: {exc}"
    try:
        results["math"] = run_math_pipeline().to_dict()
    except Exception as exc:
        errors["math"] = f"{type(exc).__name__}: {exc}"
    try:
        results["code"] = run_code_pipeline().to_dict()
    except Exception as exc:
        errors["code"] = f"{type(exc).__name__}: {exc}"
    return jsonify(safe_json({"success": not errors, "results": results, "errors": errors}))


if __name__ == "__main__":
    print("OCTree Flask backend is running.")
    print("Open http://127.0.0.1:5000 in your browser.")
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug)
