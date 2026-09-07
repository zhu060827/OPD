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

更新后的上游路由顺序是：已记录 method 标签 > 无标签样本的同轨迹五 Teacher
校准评分 > 低置信度拒绝/fallback。Open-MOPD 本身仍只接收最终 hard domain
标签，不使用路由层的 Top-2 诊断权重。

如果设置 `method.label_policy=stage1_handoff`，预检还会严格验证 Stage 1 接口：
`domain` 必须与 `teacher_id` 一致，五 Teacher 权重必须是 one-hot，必须保留
`routing_source`，且 `verification_status` 必须与 `downstream_action` 匹配。
语义失败不会取消样本的 OPD 路由资格，只会阻止当前错误 completion 进入正向扩增。
这样无需复制第二套 5experts 文件夹，就能自然连接 Stage 1 与正式 `multi/mt_opd`。
该路径使用 `configs/stage2_open_mopd_from_stage1.example.json` 模板。
正式启动前先把 Stage 1 JSONL 转换成官方训练入口使用的 Parquet：

```bash
python scripts/prepare_stage2_handoff.py \
  --input outputs/stage1_multi_expert/mt_opd_handoff.jsonl \
  --output /root/autodl-tmp/data/mt_opd_handoff.parquet
```
