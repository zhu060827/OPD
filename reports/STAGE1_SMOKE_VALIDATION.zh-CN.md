# 阶段一多专家 Smoke 验证报告

日期：2026-08-23

## 结果

五专家阶段一路由已经在不使用 GPU、不下载模型、不调用外部付费 API 的条件下完整跑通。

- 单元与集成测试：**18/18 通过**
- 处理仓库现有样本：**16/16**
- 完成硬门控、trajectory 评分和路由：**16/16**
- 外部付费 API：**0 次**
- GPU 训练：**未执行**
- 是否属于正式 MOPD 结果：**否**

确定性 Mock 跑通产生 12 条普通路由、4 条低置信度路由；伪标签分布为
`style=10`、`ast=5`、`variable=1`。这些数字只证明流程和 schema 正常，不能用于说明
真实专家能力。

## 已验证行为

- 五个专家收到完全相同、不可变的原始样本；
- 编译、函数签名、安全和单元测试是硬约束；
- 失败候选的 Teacher 权重必为零；
- 没有有效专家时输出 `no_valid_expert`，不会伪造标签；
- Top-1 交接权重是 one-hot，Top-2 softmax 只作诊断；
- Student 交接 prompt 不含 Teacher ID、路由或参考代码；
- Teacher log-prob 形状或 token 未对齐时直接拒绝融合；
- prompt share、token share 与有效优化预算占比分开统计；
- 只在对齐的有效 token 上计算跨 Teacher 冲突。

## 发现并修复的原仓库问题

1. HumanEval 把完整测试 harness 存为字符串，旧读取器把它拆成单个字符，测试 runner
   也只缩进了第一行；
2. MBPP 把正确实现放在 `reference_code`，旧读取器没有识别，导致空代码进入流程。

两个格式都已经增加回归测试。

## 产物

本地 smoke 产物位于被 `.gitignore` 排除的
`outputs/stage1_multi_expert_smoke_full/`，包括完整路由证据、MT-OPD 交接 JSONL、
汇总、resolved config 和运行清单。

## 下一道正式门槛

下一步需要把两个 Mock backend 替换成五个本地专家生成 endpoint 和本地 Transformers
trajectory scorer，确认五个专家行为确实不同、tokenizer vocabulary 完全一致、路由分布
没有退化。只有通过这一步，handoff JSONL 才能进入正式 MT-OPD 训练。
