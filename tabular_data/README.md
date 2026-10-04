# 表格特征生成与多分类实验

当前仅保留 `jungle_chess`、`balance_scale`、`chess_krk` 三个数据集。数据注册表、下载准备入口、默认训练任务、结果汇总和网页结果快照同步使用这三个任务。

## 运行

本模块在仓库根目录运行，复用根目录已有的 `config.py`、`llm_client.py` 和 `result_schema.py`。首次使用时，在 `llm_reviewer` 环境安装模块依赖：

```bash
pip install -r tabular_data/requirements.txt
```

GPU 训练需要可用的 NVIDIA 驱动及 CUDA 12 兼容环境。此目录的依赖文件固定实验使用的 XGBoost 版本，不需要修改根目录依赖文件。

在仓库根目录、激活 `llm_reviewer` 后，按当前结果的 reference 配置运行：

```bash
python tabular_data/train.py --profile reference --offline
```

共 5 轮、每轮 5 个候选，使用本地 reasoned 生成器和 GPU XGBoost；`reference` 为深度 3、60 个 boosting 轮次。仅训练一个任务可加 `--dataset jungle_chess`、`--dataset balance_scale` 或 `--dataset chess_krk`。直接运行不带参数的 `train.py` 仍保留原有 weak 模型默认值，模型方法没有因清理而改变。

数据来源见 [datasets/README.md](datasets/README.md)，结果见 [latest_summary.md](outputs/latest_summary.md)。本次仅清理和刷新索引，没有重新训练这三个任务。

| 数据集 | 原始行数 | 特征数 | 类别数 | train / validation / test |
|---|---:|---:|---:|---|
| Jungle Chess | 44,819 | 6 | 3 | 8,963 / 17,928 / 17,928 |
| Balance Scale | 625 | 4 | 3 | 125 / 250 / 250 |
| Chess KRK | 28,056 | 6 | 18 | 5,611 / 11,222 / 11,223 |

固定 seed=42，按原始类别比例分层 20%/40%/40%。训练集拟合预处理、CART、残差与模型；验证集选择特征；冻结后才读取测试集。主指标为 Macro F1，同时报告 Accuracy、Balanced Accuracy、Weighted F1、OVR Macro AUC、Log Loss。接受仍要求验证 Macro F1 改善且其他可用指标不退化，测试回落照实保留。

原有分批入口只处理仍保留的任务：`train_rare.py` 为 Balance Scale，`train_relations.py` 为 Chess KRK；`train_round3.py` 已停用并提示当前入口，避免空运行伪报成功。`research.py` 也只遍历当前三个任务。

## reasoned 的生成依据与解释

### 真实 LLM 实验

在仓库根目录的 `.env` 中配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 后运行：

```bash
python tabular_data/train_llm.py
```

当前真实实验使用配置模型 `gpt-6-luna`；入口给 API 请求设置 180 秒超时。单独重跑可加 `--dataset balance_scale` 等参数，避免重跑其他已完成任务。

入口使用与离线对照相同的 reference 模型、固定划分、5 轮 × 5 个候选和消融设置。结果单独保存在 `outputs/llm/`，不覆盖离线报告；完成的两类运行汇总至 `outputs/latest_summary.md`。此入口禁止本地规则回退，不合格式的响应最多重试 5 次，保留已验证候选并仅补齐缺额；API 失败或重试耗尽会停止并保存失败日志，不能作为成功的 LLM 成绩。报告中的 `api_calls_succeeded` 是已返回响应的调用计数（含格式修复），候选来源见每轮记录；报告也保留本地回退次数。`outputs/llm/api_calls/` 记录模型、提示及原始响应，不包含密钥。密钥仅放在本地配置中。

每轮用当前训练数据拟合不超过深度 4 的 CART，提取条件路径，并读取当前 XGBoost 的训练残差。候选包括 CART 路径指示列、Jungle 几何关系、简单交互、多列组合、非线性/分段表达式和历史拒绝原因对应的修复。

本地候选预筛分数为：

`score = Σ_b n_b ||mean(r_b) − mean(r)||² / Σ_i ||r_i − mean(r)||²`

分类残差 `r = onehot(y) − p_XGBoost`；按最多 8 个训练分位箱分组，低基数候选直接按取值分组。该分数只是训练内启发式，**不是 OOF、验证增益或 OpenFE FeatureBoost 的完整复现**。候选正式接受仍由验证集判断。保留 CART 探索名额和家族多样性，排除与已有列完全单调重复的变换，并避免本轮候选过度相关。

历史反馈区分无效、未被模型使用、指标退化、增益不足，分别尝试数值保护、简化/换列、限幅或改门槛、新条件组合。修复也要参加预筛，不保证每轮入选。

真实 LLM 获得训练画像、CART 门槛和历史验证反馈，返回带解释的结构化候选；当前残差排序作用于本地候选池。报告保存每轮 CART、公式构造步骤、列统计、阈值边界、每个常数的来源及历史引用。终端展示通常四位小数，执行和 JSON 保留数值精度。未知裸常数、目标访问、跨行聚合、导入及行索引会被拒绝。

## 消融与完整输出

[当前结果索引](outputs/latest_summary.md) 仅包含 Jungle Chess、Balance Scale 和 Chess KRK。其余数据集及相关输出已按用户要求清理，保留任务的原始数据与完整运行文件未改动。

`train.py` 的目录规则：

```text
outputs/<dataset>/holdout_20_40_40/<profile>/clean/features_reasoned/rounds_5_candidates_5/seed_42/
```

`research.py` 的运行及此前保留的 Jungle 结果位于 `outputs/research/runs/`，其后目录层次相同。启动时会打印绝对路径；训练完成自动刷新 `outputs/latest_summary.md` 和网页测试结果快照。此前保留的 Jungle 报告仍标注原版本，未伪装成重新训练。每次运行保留：

- `training.log`：完整 stdout/stderr，含每轮依据、公式构造、候选评估、接受/拒绝、错误堆栈。
- `run_status.json`、`progress.json`：运行状态与逐轮进度；同目录并发保护，失败或中断仍保留已写日志。
- `training_results.json`、`metrics.csv`、`test_predictions.csv`：验证/测试指标、分类型诊断与预测。
- `feature_explanations.json`：全部候选解释、计算参数和反馈引用。
- `baseline_model.json`、`optimized_model.json`、`preprocessor.json`：GPU 模型及冻结的预处理参数。
- `ablation.json/csv`：分裂使用次数、10 次置换重要性、删除后同配置重训；有派生后代时按依赖组处理。
- `experiment.json`、`class_distribution.json/csv`、`tabular_augmented.csv`：训练预算、类别分布和扩充后的训练数据。

重要性参数 `--importance-threshold` 默认 0.001，以置换/删除导致的测试 Macro F1 降幅衡量。报告里的均值−2×标准差判据是工程启发式，不是置信区间或显著性检验。消融不反馈生成器。相同配置重复运行会刷新该目录，旧成功报告如仍存在，必须与运行状态的 `run_id` 对照。

## 代码位置

`tabular_octree.py` 是共用候选检查和选择流程；`training.py` 实现正式划分隔离；`model_guidance.py` 提供 CART 和残差；`feature_proposals.py` 生成候选；`feature_feedback.py` 处理历史反馈；`feature_explanations.py` 管理常数来源；`evaluation.py` 保存 GPU 模型和指标策略；`ablation.py` 负责消融。数据集名单与官方版本统一在 `dataset_registry.py`。

本次集成范围仅为 `tabular_data/`，根目录网站入口及 Math 模块保持目标分支的实现。目录内的 `web.js` 和结果快照供后续网站接入使用；仅新增本目录不会自动切换根目录网站到新版表格流程。当前完整训练入口是上述 `train.py` 命令。

模块的网页演示流程从 Jungle 正式训练集分层抽取 600 行，使用同一 reasoned 生成器和真实 GPU 评估。网页 CV 结果只用于交互演示，正式研究以 `train.py` 的独立测试报告为准。
