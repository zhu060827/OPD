# Four-Expert Code Specialization and Open-MOPD Fusion

本分支实现四个代码改写专家的训练、评测与 Student 融合。四个专家共享
Qwen3-4B 基座，分别以 LoRA adapter 表示领域能力；Student 阶段使用固定版本
Open-MOPD 在同一学生模型中融合四类能力。

## 当前状态

- 四个专家已完成正式训练、最佳 checkpoint 选择和独立 test split 评测。
- Base/LoRA 推理、主指标、辅助指标和 hard-gate 图已生成。
- Stage-2 已接入固定 commit 的 Open-MOPD 启动器。
- Teacher reward worker 使用一份 Qwen3-4B base 和四个冻结 LoRA adapter，不生成四份完整合并模型。
- Student 正式训练尚未开始；当前阻塞项是构造与专家训练集无泄漏的独立 Student 数据。

## 四个专家

| 专家 | 能力 | 数据来源 | 最佳 checkpoint | Base | LoRA |
|---|---|---|---|---:|---:|
| Formatting | 精确格式修复 | CommitPackFT | checkpoint-900 | 38.26% | 88.60% |
| Identifier | 标识符精确恢复 | CodeSearchNet | checkpoint-700 | 24.73% | 51.79% |
| Local Structure | 局部结构等价改写 | CodeSearchNet | checkpoint-600 | 84.42% | 99.78% |
| Control Flow | 控制流语义保持改写 | DeepMind Code Contests | checkpoint-500 | 63.48% | 97.38% |

主指标依次为 Formatting Exact Repair、Identifier Exact Match、Local Structure
Exact Match 和 Control Flow Pass@1。这些是 held-out test-set 生成指标，不是训练 token accuracy。

```text
Qwen3-4B base
├── Formatting LoRA
├── Identifier LoRA
├── Local Structure LoRA
└── Control Flow LoRA
```

训练使用 assistant-only causal language-modeling loss，禁用 Qwen3 thinking。
LoRA 参数为 `r=16`、`alpha=32`、`dropout=0.05`。

## Student 融合方法

Student 对每个 prompt 只 rollout 一次。冻结 Teacher 对 Student 实际生成的同一组
token 计算 log-probability，token 级基础信号为：

```text
log p_teacher(token | prefix) - log p_student(token | prefix)
```

正式更新使用 Open-MOPD 的 Student top-k 概率加权 token reward。本项目额外保留：

- prompt/domain sampling balance；
- response-token share balance；
- reward-scale/gap-following balance；
- repeated-update reward refresh；
- `raw`、`mean/std`、`median/MAD` 路由校准消融。

Open-MOPD 官方实验使用已知 domain 标签路由。本项目的 median/MAD 异构 Teacher
校准是额外扩展，必须通过 validation 和消融实验评估，不宣称为 Open-MOPD 原始公式。

## 共享 base + LoRA Teacher 扩展

官方 `mt_opd.sh` 原生接受多个完整 Teacher 模型路径。本项目为固定版本
Open-MOPD 的 reward worker 增加可选 `model.lora_adapter_path`：

1. `model.path` 指向共享 Qwen3-4B base；
2. reward worker 加载自己的冻结 PEFT adapter；
3. 保留官方 multi-Teacher worker、domain routing 和 token reward 逻辑；
4. 不向磁盘写出四份约 8 GB 的合并权重。

应用补丁：

```bash
python scripts/patch_open_mopd_lora_teachers.py \
  --open-mopd-root third_party/Open-MOPD
```

补丁脚本仅允许应用到 commit：

```text
4809a96cf85a869106ff0ff3f37d0a51e12010ae
```

## 目录

```text
code_rewrite_feedback_expander/multi_expert/
  calibration.py              # robust Teacher 校准
  config.py                   # Stage-1 配置
  pipeline.py                 # 路由标签/handoff 流程
  router.py                   # 专家选择
  opd_reward.py               # top-k token reward 参考实现
  stage2_open_mopd.py         # Stage-2 配置、预检和启动器

teacher_training/             # 专家数据、训练与评测
configs/                      # 四专家 Stage-1/Stage-2 配置模板
scripts/                      # Open-MOPD 获取、补丁和转换脚本
tests/                        # 单元测试
docs/FOUR_EXPERT_OPEN_MOPD.zh-CN.md
```

## 安装与验证

环境需包含 PyTorch、Transformers、PEFT、Datasets 和 PyArrow。

```bash
python -m compileall -q code_rewrite_feedback_expander teacher_training scripts tests
PYTHONPATH=. python tests/multi_expert/test_opd_reward.py
PYTHONPATH=. python tests/multi_expert/test_stage2_lora_teachers.py
```

获取并固定 Open-MOPD：

```bash
bash scripts/fetch_open_mopd.sh third_party/Open-MOPD
python scripts/patch_open_mopd_lora_teachers.py \
  --open-mopd-root third_party/Open-MOPD
```

## 配置

模板：

- `configs/stage1_multi_expert.four_domain.gpu.example.json`
- `configs/stage2_open_mopd_four_teacher.example.json`
- `configs/stage2_open_mopd_four_teacher_from_stage1.example.json`

Stage-2 关键字段：

```json
{
  "models": {
    "student_path": "/path/to/Qwen3-4B",
    "teacher_base_path": "/path/to/Qwen3-4B",
    "teachers": [
      {
        "domain": "formatting",
        "adapter_path": "/path/to/formatting-adapter"
      }
    ]
  }
}
```

实际配置必须按以下顺序提供四个领域：

```text
formatting, identifier, local_structure, control_flow
```

## 预检和运行

Student 数据准备完成后：

```bash
PYTHONPATH=. python -m \
  code_rewrite_feedback_expander.multi_expert.stage2_open_mopd \
  --config configs/stage2_open_mopd_four_teacher.server.json \
  --preflight-only
```

只在预检通过、GPU/内存/磁盘充足时启动：

```bash
PYTHONPATH=. python -u -m \
  code_rewrite_feedback_expander.multi_expert.stage2_open_mopd \
  --config configs/stage2_open_mopd_four_teacher.server.json \
  --run
```

## Student 数据要求

Student 原始数据必须独立于专家数据来源。已用于专家训练的来源为：

```text
Formatting       CommitPackFT
Identifier       CodeSearchNet
Local Structure  CodeSearchNet
Control Flow     DeepMind Code Contests
```

候选正式数据为 APPS train split；MBPP train 仅用于小规模端到端 smoke test，
MBPP test 保留为外部评测。数据必须对四专家全部 split 执行：

- `problem_id` 去重；
- normalized problem-text hash 去重；
- source/target code hash 去重；
- Student train/validation/test 隔离；
- 四领域分布和 response-token share 审计。

## 论文实验要求

至少报告：

- Base Student；
- 单专家蒸馏；
- 四专家 raw advantage；
- mean/std 与 median/MAD 校准；
- recorded-domain Open-MOPD；
- dynamic/pseudo-router 消融；
- token-share、reward-scale 和 reward-refresh 消融；
- 95% 置信区间、paired win/loss/tie 和失败案例。

不得将未完成的 Student GPU 训练或 mock 输出写成正式实验结果。
