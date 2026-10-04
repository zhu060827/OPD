"""真实 LLM 实验入口；不允许回退，保留离线实验用于比较。"""
from __future__ import annotations

import os
import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from tabular_data.dataset_registry import DATASETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    args = parser.parse_args()
    from llm_client import LLMClient

    client = LLMClient(use_mock_when_fails=False)
    if not client.available:
        print("真实 LLM 配置缺失。请在项目 .env 配置 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL；密钥不要写入代码或日志。", file=sys.stderr)
        return 2
    root = Path(os.environ.get("TABULAR_OUTPUT_DIR", str(project / "tabular_data" / "outputs"))).resolve()
    env = os.environ.copy()
    env["USE_MOCK_WHEN_LLM_FAILS"] = "0"
    env["REQUEST_TIMEOUT"] = "180"
    env["TABULAR_OUTPUT_DIR"] = str(root / "llm")
    env["TABULAR_LLM_AUDIT_DIR"] = str(root / "llm" / "api_calls")
    print(f"真实 LLM 模型：{client.model}；禁止本地回退；结果目录：{root / 'llm'}", flush=True)
    code = subprocess.call([sys.executable, "-u", str(project / "tabular_data" / "train.py"),
                            "--profile", "reference", "--dataset", args.dataset], cwd=project, env=env)
    # 已完成任务即使后续失败也纳入汇总；失败运行由 run_id/status 检查过滤。
    from tabular_data.refresh_summary import collect_reports
    from tabular_data.research import write_summary

    write_summary(collect_reports(root), root)
    print(f"离线/LLM 对比汇总：{root / 'latest_summary.md'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
