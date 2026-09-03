# data test 使用说明

这里专门保存真实数据测试，不复制主工程算法代码，避免出现两个不同版本。

Math 测试直接调用根目录：

```text
math_adapter.py
Math/math_reasoning_expander/
llm_client.py
```

## 当前固定测试数据

`datasets/math_real_16.jsonl` 共 16 条：

- NuminaMath-CoT：12 条
- MetaMathQA：4 条

## 运行真实 Math 测试

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe ".\data test\run_math_batch_eval.py" --limit 16
```

真实 API 配置来自根目录 `.env` 或当前终端环境变量。测试器会记录真实调用次数和 mock 回退次数。

只测试程序流程、不调用真实 API：

```powershell
.\.venv\Scripts\python.exe ".\data test\run_math_batch_eval.py" --limit 16 --mock
```

## Math 输出

- `results/math_real_records.jsonl`：16 条完整结果和每轮轨迹
- `results/math_real_summary.csv`：总体和分数据集指标
- `results/math_quality.svg`：扩充前后质量图
- `MATH_REAL_TEST_REPORT.md`：中文实验报告

## 少量 Code 测试

Code 数据目前只有 3 条 HumanEval 样例，只做流程检查：

```powershell
.\.venv\Scripts\python.exe ".\data test\run_code_quick_eval.py" --limit 3
```

输出：

- `results/code_quick_records.jsonl`
- `results/code_quick_summary.csv`

## 如何判断本次测试可信

先看 `math_real_summary.csv`：

- `error_count` 应为 0
- `real_api_calls` 应大于 0
- `mock_fallback_calls` 应为 0
- `accepted_rate` 表示有多少题真正产生了正质量增益
- `avg_reasoning_gain` 表示恢复节点提供新推理信息的平均程度

`accepted=false` 不等于程序失败。它表示生成步骤没有超过原节点质量，因此被 pipeline 正确拒绝。

如果只调整报告格式或审计规则，不想再次消耗 API，可以运行：

```powershell
.\.venv\Scripts\python.exe ".\data test\run_math_batch_eval.py" --report-only
```

汇总中的 `structure_regression_count` 用于统计扩充后新增不成对 LaTeX 展示符的样本。
