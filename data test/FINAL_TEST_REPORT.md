# 结题版真实数据测试总报告

## 测试范围

本轮主要测试 Math，Code 只做少量流程验证。

- 模型：`gpt-5.6-terra`
- Math 数据：16 条
- Code 数据：3 条 HumanEval
- Math 引擎：`Math/math_reasoning_expander`
- Code 引擎：`code_rewrite_feedback_expander`

## Math 测试结果

| 指标 | 结果 |
|---|---:|
| 完成数 | 16 / 16 |
| 错误数 | 0 |
| 接受数 | 11 |
| 接受率 | 68.75% |
| 平均综合分 | 0.820 |
| 平均去重复性 | 0.946 |
| 平均逻辑一致性 | 0.813 |
| 平均公式正确性 | 0.719 |
| 平均完整性 | 0.900 |
| 平均推理增益 | 0.759 |
| 平均新增节点数 | 3.125 |
| 平均尝试次数 | 12.250 |
| 真实 API 调用 | 196 |
| mock 回退 | 0 |

分数据集结果：

- NuminaMath-CoT：12 条，接受率 75.0%，平均综合分 0.811。
- MetaMathQA：4 条，接受率 50.0%，平均综合分 0.847。

这说明完整的“遮盖节点、恢复、评价、反馈、再生成、正增益保留”流程已经跑通。
`accepted=false` 的样本也有较高综合分，原因是生成节点没有超过原节点质量，而不是程序报错。

## Math 结果限制

公式正确性 0.719，是五项指标中相对较弱的一项。原始数据和模型输出都存在少量不规范
LaTeX，首轮真实结果中有 4 条出现扩充后结构分隔符变差的情况。

根据这次测试，主工程已经增加两项修正：

1. Prompt 明确禁止输出单独的 `\[`、`\]`、`$$`。
2. Math LLM bridge 会去掉恢复步骤中不成对的展示分隔符，同时保留公式内容。

已有真实结果不做伪造或覆盖，报告中仍保留这项限制。后续如果用于论文式严格评测，
还应增加更强的 SymPy 等价验证和人工抽样复核。

## Code 冒烟测试结果

| 任务 | 是否保留 | 质量分 | 保留 rewrite | 尝试次数 |
|---|---:|---:|---:|---:|
| HumanEval/0 | 是 | 0.938 | 2 | 11 |
| HumanEval/1 | 是 | 0.908 | 2 | 8 |
| HumanEval/2 | 是 | 1.000 | 1 | 7 |

三条结果都通过 AST、函数签名、安全检查和 subprocess 单元测试。

测试过程中修复了一个真实 bug：原代码会把 HumanEval 多行测试逐行 `.strip()`，
导致 `def check(candidate):` 函数体缩进丢失并产生 `IndentationError`。
修复后字符串形式的测试会完整保留多行缩进，3 条原始正确实现和 3 条 rewrite 结果均能正常验证。

由于 Code 只有 3 条数据，这部分只能说明工程流程可用，不能代表整体代码数据集性能。

## 最终文件

- `datasets/math_real_16.jsonl`：固定的 16 条 Math 数据
- `datasets/code_humaneval_3.jsonl`：3 条 Code 冒烟数据
- `results/math_real_records.jsonl`：Math 完整结果和迭代轨迹
- `results/math_real_summary.csv`：Math 汇总指标
- `results/math_quality.svg`：Math 质量对比图
- `MATH_REAL_TEST_REPORT.md`：Math 详细报告
- `results/code_quick_records.jsonl`：Code 完整结果
- `results/code_quick_summary.csv`：Code 汇总指标
- `results/code_quality.svg`：Code 质量图

本轮结论：Math 是主要实验，已经完成真实模型、真实数据和完整反馈循环测试；
Code 完成小规模工程验证；所有结果均保存在 `data test`，不与主工程演示输出混放。
