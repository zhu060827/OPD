# 阶段一：代码多专家路由

> 本模块是正式 Open-MOPD 训练前的可选路由层：优先使用已记录的 method
> 标签，只对无标签数据做伪路由。

## 目标

本阶段实现三级策略，并为每条路由保留可审计证据。
它不更新 Student；Mock 跑通结果也不属于真实训练结果。

五个专家分别对应：`cot`、`style`、`ast`、`variable` 和
`control_flow`。实现直接复用现有代码改写、语义验证、质量评估与 token 分布代码，
没有推翻原项目。

## 核心流程

```text
1. 有 method/domain/rewrite_method 来源标签：直接 one-hot 路由
2. 无标签样本：五个 Teacher 评分同一条 completion
3. 校准后 Top-1/Top-2 差距过小：拒绝伪标签（或使用明确配置的 fallback）
```

无标签路由不再让五个专家各生成一次。五个 Teacher 看到完全相同的 prompt、
token ID 和 completion，因此评分可直接比较，生成成本也从五次降为一条共享轨迹。
本地 smoke 配置复用数据中已有 code；GPU 配置使用
`shared_completion_source=student_generate`，让当前 Student 只 rollout 一次，
五个冻结 Teacher 再评分同一条 on-policy 轨迹，且不会重复加载 Student。
生成时的 prompt/completion token ID 会被原样保留并复用，不会用 decode 后的文本重新
tokenize 来假冒原 on-policy 轨迹。

对每个 Teacher 先在独立 validation split 拟合中位数和 MAD，再路由：

```text
raw_advantage_e = mean_t[log p_teacher_e(y_t|s_t) - log p_student(y_t|s_t)]
calibrated_advantage_e = (raw_advantage_e - median_e) / (1.4826 * MAD_e)
route = argmax_e calibrated_advantage_e
```

校准可防止某个 Teacher 仅因概率尺度或锐度不同而长期占据路由。

正式三级流程将 Teacher 路由与代码验证拆成两个独立维度。Student 当前回答即使未通过
语义检查或单元测试，仍是有价值的 on-policy 错误轨迹；只要 token 对齐有效，五个
Teacher 就会继续评分。验证结果单独记录为 `semantic_pass`、`semantic_fail` 或
`semantic_unverified`，不会与 advantage 加权混合。只有 token 轨迹缺失、非有限或
对齐失败才阻断路由。

后续数据按质量状态分流：通过验证的样本进入 `positive_augmentation`，失败样本进入
`repair_or_negative`，无可执行测试的样本进入 `unverified_pool`。因此错误轨迹可以用于
OPD 学习，但不会被冒充成正确增强答案。旧 `heuristic_ablation` 比较的是五个独立生成
的改写候选，所以仍保留语义硬门控。

## 旧启发式消融

原来的 routing utility 仍然保留在 `routing.policy=heuristic_ablation`：

```text
R = 0.40 * 方法对应指标改善
  + 0.30 * 整体质量改善
  + 0.20 * 无明显指标退化比例
  + 0.10 * 候选确实发生变化
```

所有分量和原始指标差值都会保存。具体比例暂无直接文献依据，必须做 Reward-only、
Advantage-only、等权和权重敏感性消融。代码明确将其标记为“需要消融验证的项目级
路由 utility”；它不是 Open-MOPD 的 token reward，也不是普适的代码质量总分。

对于每个通过门控的候选，计算：

```text
advantage_t = log p_teacher(y_t | prefix) - log p_student(y_t | prefix)
```

路由主要使用平均 token advantage，同时保留中位数、Teacher 胜出 token 比例、
forward KL、Total Variation 和有效 token 数。

默认最终路由分数为：

```text
route_score = 0.55 * 归一化 Reward
            + 0.45 * 归一化 NLL advantage
```

它现在只用于对照，不再是默认路由。Top-2 权重始终只是诊断，不做 Teacher
参数合并或 logits 平均。

## 与 Open-MOPD 的关系

Open-MOPD 已知每条样本的领域标签，因此可以直接使用 one-hot 权重选择领域 Teacher，
再让这个冻结 Teacher 在 Student 自己的 token trajectory 上评分。本项目将问题拆成两层：

1. 已记录的改写方法直接作为 Open-MOPD hard route；
2. 只有无标签数据进入校准后的同轨迹五 Teacher 评分；
3. 阶段二使用最终 hard label 执行真实 Open-MOPD 更新。

因此本阶段是“路由标签发现层”，不会替代正式 OPD loss。

项目同时提供了不依赖 GPU 的 `multi_expert.fusion` 阶段二桥接模块，包括 hard-route
权重矩阵、对齐 Teacher log-prob 路由、prompt/token/有效优化预算占比、token-share
loss 权重和跨 Teacher 冲突诊断。如果不同 Teacher 的 trajectory 形状或 token 对齐不一致，
函数会直接报错，不会静默混合错误位置。

## CPU 跑通

```bash
python -m code_rewrite_feedback_expander.multi_expert validate-config \
  --config configs/stage1_multi_expert.json

python -m code_rewrite_feedback_expander.multi_expert run \
  --config configs/stage1_multi_expert.json \
  --input code_rewrite_feedback_expander/data/code_real_16.jsonl \
  --output-dir outputs/stage1_multi_expert_smoke \
  --limit 3
```

Mock 模式只用于验证五专家流程和输出 schema，所有输出都明确标记
`formal_training_result=false`，禁止作为正式实验结果。

## 输出文件

- `routing_labels.jsonl`：五个专家的完整候选、门控和评分证据；
- `mt_opd_handoff.jsonl`：包含路由可用的样本，并明确记录
  `verification_status`、`downstream_action`、OPD 训练资格和正向扩增资格；
- `summary.json`：标签、门控和各专家汇总；
- `resolved_config.json`：本次实际使用的完整配置；
- `run_manifest.json`：后端类型与结果性质。

## 后续 A800 接入

复制并修改 `configs/stage1_multi_expert.gpu.example.json`。生成端可以使用五个本地
vLLM OpenAI-compatible endpoint，不需要付费 API；评分端使用本地 Hugging Face
checkpoint。Student 与五个 Teacher 必须拥有完全相同的 tokenizer vocabulary。

正式运行前设置 Student 路径、五个 Teacher 路径和 `DISTILLATION_TOPK=16`。代码会
冻结 Teacher，并在完全对齐的 token ID 上计算 Student/Teacher 分布。

配置中的 `location=0, scale=1` 只是 smoke test 占位值。真实无标签实验前，必须用
独立校准集拟合每个 Teacher 的统计量，再写回正式配置：

```bash
python -m code_rewrite_feedback_expander.multi_expert fit-calibration \
  --config configs/stage1_multi_expert.gpu.example.json \
  --input outputs/calibration_pass/routing_labels.jsonl \
  --output outputs/teacher_advantage_calibration.json \
  --min-samples 20
```

不能用最终测试集拟合 calibration。

## 测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

测试覆盖统一配置、已有标签优先、五 Teacher 共享同一 completion、语义失败轨迹仍评分、
失败样本不进入正向扩增、缺少测试时进入 unverified、稳健校准、低置信度拒绝、
旧启发式硬门控、Top-1、Top-2 诊断，
无有效专家处理、Student prompt 中无 Teacher/路由/参考答案泄漏，以及 MT-OPD 交接
schema。

当前 subprocess runner 适合仓库内可信的 HumanEval/MBPP 测试。以后如果接收来源未知的
第三方测试代码，必须先放进加固容器或 Firejail 兼容沙盒，不能直接在 GPU 主机执行。
