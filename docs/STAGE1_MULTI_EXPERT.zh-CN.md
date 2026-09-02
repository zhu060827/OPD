# 阶段一：代码多专家路由

> 本模块仅用于无 method 标签的实验性伪路由，不是正式 Open-MOPD 训练入口。下述固定
> 权重是项目假设，不是 Open-MOPD 发表的公式。

## 目标

本阶段在正式多教师 OPD 训练前，为每条代码样本生成可审计的“改写方法伪标签”。
它不更新 Student；Mock 跑通结果也不属于真实训练结果。

五个专家分别对应：`cot`、`style`、`ast`、`variable` 和
`control_flow`。实现直接复用现有代码改写、语义验证、质量评估与 token 分布代码，
没有推翻原项目。

## 核心流程

```text
同一条不可变的原始样本
  -> 五个专家分别生成候选
  -> 编译/签名/安全/单元测试硬门控
  -> 所有专家共用同一套确定性路由 Reward
  -> 计算 Teacher 相对 Student 的 token NLL advantage
  -> 归一化融合两类证据
  -> Top-1 方法伪标签 + Top-2 诊断权重
  -> 输出 MT-OPD 交接 JSONL
```

正确性是硬约束。候选只要未通过门控，无论 Reward 或 NLL advantage 多高，路由权重
都为零。如果五个候选全部失败，样本标记为 `no_valid_expert`，不会强行制造标签。

## 评分方法

共享 routing utility 只用于在正确候选之间排序：

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

Top-1 是第一轮 MT-OPD 使用的 hard route；Top-2 权重只作为诊断和后续消融。

## 与 Open-MOPD 的关系

Open-MOPD 已知每条样本的领域标签，因此可以直接使用 one-hot 权重选择领域 Teacher，
再让这个冻结 Teacher 在 Student 自己的 token trajectory 上评分。本项目将问题拆成两层：

1. 阶段一解决代码数据没有改写方法标签的问题，生成伪标签；
2. 阶段二使用该标签选择冻结 Teacher，并执行真实 OPD/PPO 更新。

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
- `mt_opd_handoff.jsonl`：仅包含可进入下一阶段训练的样本；
- `summary.json`：标签、门控和各专家汇总；
- `resolved_config.json`：本次实际使用的完整配置；
- `run_manifest.json`：后端类型与结果性质。

## 后续 A800 接入

复制并修改 `configs/stage1_multi_expert.gpu.example.json`。生成端可以使用五个本地
vLLM OpenAI-compatible endpoint，不需要付费 API；评分端使用本地 Hugging Face
checkpoint。Student 与五个 Teacher 必须拥有完全相同的 tokenizer vocabulary。

正式运行前设置 Student 路径、五个 Teacher 路径和 `DISTILLATION_TOPK=16`。代码会
冻结 Teacher，并在完全对齐的 token ID 上计算 Student/Teacher 分布。

## 测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

测试覆盖统一配置、正确性硬门控、五专家输入一致、Top-1、Top-2 权重归一化、
无有效专家处理、Student prompt 中无 Teacher/路由/参考答案泄漏，以及 MT-OPD 交接
schema。

当前 subprocess runner 适合仓库内可信的 HumanEval/MBPP 测试。以后如果接收来源未知的
第三方测试代码，必须先放进加固容器或 Firejail 兼容沙盒，不能直接在 GPU 主机执行。
