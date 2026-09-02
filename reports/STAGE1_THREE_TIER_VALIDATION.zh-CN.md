# Stage-1 三级路由验证说明

## 验证范围

本次只验证控制流和数据契约，不是模型质量结果，也没有进行 GPU 训练。

## 结果

- 仓库完整测试：**30/30 通过**。
- 3 条无标签 HumanEval 样本完成 Mock smoke 流程。
- 2 条因校准后 Top-1/Top-2 margin 低于 `0.05` 而拒绝伪标签，1 条生成 hard route。
- 已记录 method 标签会跳过五路推断，只选择一个 Teacher。
- 无标签样本向五个 Teacher 提供完全相同的 completion；测试也确认 GPU 契约只让 Student 生成一次。
- 低置信度样本不会进入 MT-OPD handoff，除非显式设置 fallback。
- 正式五 checkpoint Open-MOPD Stage-2 启动路径没有被改动。

## 评价

新方案将任意的代码质量固定权重移出默认路由，将候选生成从五次降为一条共享
completion，并防止强制产生低置信度标签。这些是已验证的结构性改进。真实路由准确率和
下游 pass@1 仍需使用五个真实 Teacher，在独立校准集拟合统计量后进行 GPU 对照。

Smoke 配置中的单位 calibration 只是占位值，不能当作真实校准结果。
