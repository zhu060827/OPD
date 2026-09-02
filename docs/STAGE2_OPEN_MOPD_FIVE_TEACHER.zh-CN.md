# Stage 2：五个真实 Teacher 的 Open-MOPD 框架

这是正式多教师训练入口。它要求五个互不相同的本地 Teacher checkpoint 和一个可训练
Student，并调用固定 commit 的 Open-MOPD 官方实现。当前没有声称已经完成 GPU 训练，
但启动入口、配置、严格预检和 CPU 集成测试已经建立。

五个 Teacher 不进行参数合并。每条训练数据包含
`cot/style/ast/variable/control_flow` 中的一个 `domain` 标签。Student 先生成轨迹，
标签 one-hot 选择对应 Teacher，Teacher 在完全对齐的 Student token 上评分，所有样本
最终更新同一个 Student。

Open-MOPD 的正式 token reward 来自 Teacher 与 Student 的对数概率差：

```text
(log p_teacher - log p_student) * Student top-k 归一化概率
```

它不是 Stage 1 的代码质量评分。正式 Stage 2 不使用我们自定义的
`0.40/0.30/0.20/0.10` 和 `0.55/0.45` 权重；这些只保留为无标签路由消融。

正式入口会向 Open-MOPD 传入：五个 Teacher、hard routing、token-share balancing、
gap-following allocation 和 reward refresh。原来的 `verl_example/opd.sh` 只保留为
单 Teacher 对照组。

使用顺序：

```bash
bash scripts/fetch_open_mopd.sh /root/autodl-tmp/Open-MOPD
cp configs/stage2_open_mopd_five_teacher.example.json \
   configs/stage2_open_mopd_five_teacher.json
# 修改真实模型、数据和输出路径
bash scripts/run_stage2_open_mopd.sh \
   configs/stage2_open_mopd_five_teacher.json --preflight-only
bash scripts/run_stage2_open_mopd.sh \
   configs/stage2_open_mopd_five_teacher.json --dry-run
# 确认命令后才执行：
bash scripts/run_stage2_open_mopd.sh \
   configs/stage2_open_mopd_five_teacher.json --run
```

第一组正式实验优先使用数据增强过程记录的真实 method 标签。Stage 1 自动生成的伪标签
必须作为后续独立实验，并与 Reward-only、Advantage-only 和不同融合权重进行消融。
