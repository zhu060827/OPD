# 四专家 Student 融合方案

当前正式方案使用四个真实 LoRA 专家：`formatting`、`identifier`、`local_structure` 和 `control_flow`。

Student 对每个样本只生成一次 rollout。四个冻结 Teacher 在同一组原始 token 上计算
`log p_teacher - log p_student`，对有效 response token 求均值，得到每个专家的样本级
OPD advantage。路由阶段默认使用独立 validation split 拟合的 median/MAD 校准：

```text
z_e = (advantage_e - median_e) / (1.4826 * MAD_e)
```

选择校准分数最高的专家；当 Top-1/Top-2 差距低于 `minimum_margin` 时可以 abstain，避免
强行伪造路由。`raw`、`mean_std` 和 `recal_style` 保留为可比较的校准消融，不能用最终
test split 拟合校准参数。

正式 Open-MOPD 阶段仍使用官方 token reward：Teacher–Student log-probability gap 乘以
Student 归一化 top-k 概率。路由权重是 one-hot 的 Teacher 选择，不是四个模型参数合并，
也不使用早期的 0.4/0.3/0.2/0.1 启发式权重。

配置模板：

- `configs/stage1_multi_expert.four_domain.gpu.example.json`
- `configs/stage2_open_mopd_four_teacher.example.json`
- `configs/stage2_open_mopd_four_teacher_from_stage1.example.json`

在正式运行前，需将当前 LoRA adapter 合并成 Open-MOPD 可加载的完整 Teacher checkpoint，
并把四个合并后的路径写入 Stage-2 配置。Stage-1 的 calibration 输出应来自独立 validation
轨迹，随后才可生成 handoff 并启动 Open-MOPD。
