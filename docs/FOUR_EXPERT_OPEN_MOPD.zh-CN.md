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

正式运行使用一份 Qwen3-4B Teacher base 和四个冻结 LoRA adapter。
`scripts/patch_open_mopd_lora_teachers.py` 为固定版本 Open-MOPD 的 reward worker
增加可选 `lora_adapter_path`：每个官方 reward worker 从共享磁盘路径加载
同一 base，再在内存中加载对应 adapter。该扩展不修改 Open-MOPD 的路由、
token reward 或训练更新公式，也不生成四份完整 Teacher 权重。
Stage-1 的 calibration 输出应来自独立 validation 轨迹，随后才可生成
handoff 并启动 Open-MOPD。
