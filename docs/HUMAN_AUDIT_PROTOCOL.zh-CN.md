# 无主观训练阈值时的人工审计协议

人工审计不用于训练、调参或选择 checkpoint，只用于估计自动过滤器的 precision，并报告数据质量
风险。因此它不会把主观分数混入 Teacher 的自动指标。

## 抽样

- 每个领域从 `accepted.jsonl` 和 `rejected.jsonl` 各使用固定随机种子抽取 50 条；样本量不足时全部抽取。
- 审计者看不到模型名称、训练 split 和自动判定原因（盲审）。
- 两名审计者独立判断；不一致样本由第三人裁决。

## 只判断三个二元问题

1. **领域归属**：是否确实属于该领域的允许操作？
2. **行为保持**：给定提供的测试和静态证据，是否没有观察到功能变化？
3. **数据可追溯**：来源 commit/题目、语言、检测器证据和测试结果是否齐全？

不要对“可读性好不好”“计划是否漂亮”“改写是否有创意”进行 1～5 分主观打分；这些判断没有
统一标准，且不用于模型选择。

## 报告

分别报告：领域归属 precision、行为保持 precision、来源完整率、两名审计者 Cohen's κ，以及
accepted/rejected 的混淆表。Cohen (1960) 的 κ 用于衡量分类者一致性。

若不进行人工审计，论文必须明确写明：

> No human-labeled threshold calibration was used. Acceptance was determined by frozen executable
> tests, parser/compiler checks, and detector rules; the resulting automatically verified subset is a
> limitation rather than proof of complete semantic equivalence.

人工审计结果也不应替换 `pass@1`、BLEU、CodeBLEU 或重构类型准确率等自动指标。
