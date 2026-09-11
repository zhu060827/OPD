# MBPP 五教师 OPD 端到端框架

主方法不执行 SFT。Stage 1 只生成路由标签；Stage 2 使用五个冻结 Teacher
对一个 Student 执行 Open-MOPD；最终在固定测试集上计算 Pass@1 和反馈修复累计成功率。

Stage 2 对每个 generation batch 做五领域 prompt 比例平衡，并依据真实
`response_mask` 对五领域的有效 token 梯度贡献进行平衡。第一版正式实验不使用
`routing_confidence` 加权，避免与领域 token 平衡重复补偿。

参考 MOPD 的完整预算思想，领域权重为：

```text
weight_d = target_share_d / current_token_share_d
           * (reward_magnitude_d / median_reward_magnitude) ** alpha
```

其中 `reward_magnitude` 是当前领域每个有效 response token 的平均绝对 OPD
advantage，并使用 EMA 降低 MBPP 小 batch 的噪声。MBPP 首测建议 `alpha=0.5`、
`EMA decay=0.9`；`alpha=0` 可作为仅 token-share 平衡的消融基线。合并后的权重
按有效 token 总量归一化，范围限制为 `[0.05, 20.0]`，保持整体学习率尺度基本不变。

### MBPP 平衡对比

### 无微调 Teacher 消融

在不微调 Qwen 的前提下，使用同一个 Qwen 3.5B checkpoint 做两组严格对照：

```text
Teacher-plain：STAGE1_TEACHER_PROMPT_MODE=none
              五个 Teacher 使用完全相同的任务上下文

Teacher-directional：STAGE1_TEACHER_PROMPT_MODE=directional
                     分别使用 COT/Style/AST/Variable/Control-flow 分析提示词
```

两组实验必须固定 Student、Teacher checkpoint、MBPP 顺序、随机种子和 calibration，
唯一改变 Teacher 上下文提示词。结果应分别报告路由分布、Top-1/Top-2 margin、
fallback 比例和 test-500 Pass@1。该实验验证的是“提示词条件化专家路由”，不能宣称
五个 Teacher 具有参数层面的独立领域能力。

三个平衡层均通过配置开关控制，不需要删除采样代码：

| 实验 | prompt sampling | token share | reward scale |
|---|---:|---:|---:|
| A Natural OPD | 关闭 | 关闭 | 关闭 |
| B Loss-balanced OPD | 关闭 | 开启 | 开启 |
| C Full MOPD（主方法） | 开启 | 开启 | 开启 |

自然采样的 batch 可以缺少某些领域。公式只在当前出现的领域之间重新归一化目标份额，
缺失领域本 step 的贡献记为 0，并保留其历史 reward EMA。对应配置片段位于
`configs/stage2_mbpp_ablation_plain_opd.json`、
`configs/stage2_mbpp_ablation_natural.json` 和 `configs/stage2_mbpp_full_mopd.json`。

## 数据流

```text
MBPP 974
  -> Stage 1 共享 Student 轨迹、五 Teacher 校准路由
  -> mt_opd_handoff.jsonl
  -> prompt 1-10 / test 11-510 / validation 511-600 / train 601-974
  -> train/validation Parquet
  -> Open-MOPD 五 Teacher hard routing
  -> 一个 Student 的多个 checkpoint
  -> test 500 Pass@1 与最多三次反馈修复
```

语义状态和路由标签是独立轴。`semantic_fail` 且轨迹有效的记录仍可进入 OPD，
但不能作为正向 SFT/最终扩充答案。所有 Stage-1 轨迹缺失时，正式全覆盖配置使用
`expert_cot` 作为可审计 fallback，并要求 Stage 2 重新评分。

当前 fallback 使用 `task_id` 的 SHA-256 对五个既有 domain 做确定性均衡映射，
不增加 Open-MOPD 未知的第六类标签。`routing_loss_weight` 会写入 handoff 供消融，
默认不影响 Open-MOPD loss；当前正式配置禁止开启该权重。

## 命令骨架

```bash
python -m code_rewrite_feedback_expander.multi_expert run \
  --config configs/stage1_multi_expert.json \
  --input data/mbpp_974.jsonl \
  --output-dir outputs/stage1_mbpp_974

python scripts/prepare_mbpp_workflow.py \
  --input outputs/stage1_mbpp_974/mt_opd_handoff.jsonl \
  --output-dir outputs/mbpp_workflow \
  --parquet

bash scripts/run_stage2_open_mopd.sh \
  configs/stage2_open_mopd_from_stage1.json --preflight-only

bash scripts/run_stage2_open_mopd.sh \
  configs/stage2_open_mopd_from_stage1.json --run

python scripts/evaluate_mbpp_checkpoints.py \
  --test outputs/mbpp_workflow/mbpp_test.jsonl \
  --checkpoint base=/models/student-base \
  --checkpoint step100=/runs/checkpoint-100 \
  --generator-factory your_runtime:create_generator \
  --verifier-factory your_runtime:create_verifier \
  --output outputs/evaluation/results.json
```

评测器通过工厂接口隔离具体推理框架和代码沙箱。`generate(record, feedback)` 返回代码，
`verify(record, code)` 返回 `(passed, feedback)`。同一题第一次通过后停止修复；累计成功率
不会重复计算同一题。
