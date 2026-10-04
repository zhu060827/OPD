# 特征生成研究结果

以下为已完成运行；所有数值来自各运行的独立测试报告。
同一行内 baseline 与 optimized 的数据、模型配置相同。跨数据集的绝对分数不能当作统一难度标尺。
主指标为 Macro F1；Log Loss 越低越好，其余越高越好。分数显示四位小数，增益用百分点。

| 数据集/视图 | 模型 | 生成器 | 种子 | 训练行数 | 接受数 | 测试 Macro F1：基线 → 优化 | Δ百分点 | 实际 LLM 调用 |
|---|---|---|---:|---:|---:|---|---:|---:|
| jungle_chess/full | reference | reasoned / cart_residual_v6 (LLM) | 42 | 8963 | 3 | 0.5603 → 0.7022 | +14.1829 | 10 |
| balance_scale/full | reference | reasoned / cart_residual_v6 (LLM) | 42 | 125 | 2 | 0.5824 → 0.8099 | +22.7567 | 9 |
| chess_krk/full | reference | reasoned / cart_residual_v6 (LLM) | 42 | 5611 | 3 | 0.3705 → 0.4046 | +3.4130 | 5 |

## jungle_chess/full/reference/reasoned/seed_42/LLM

[完整日志](jungle_chess/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training.log) · [机器报告](jungle_chess/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training_results.json)

| 指标 | baseline | optimized | 改善量（正数更好） |
|---|---:|---:|---:|
| accuracy | 0.7716 | 0.8020 | +0.0305 |
| balanced_accuracy | 0.5753 | 0.6739 | +0.0986 |
| f1 | 0.7374 | 0.7914 | +0.0540 |
| f1_macro | 0.5603 | 0.7022 | +0.1418 |
| auc | 0.9132 | 0.9403 | +0.0270 |
| log_loss | 0.5498 | 0.4845 | +0.0654 |

接受特征：oct_llm_strength_gap_log_r1, oct_llm_oct_piece_strength_rank_context_r2, oct_llm_oct_file_rank_geometry_strength_r3_r3
实际训练预算：8963/8963。
训练流程版本：multiclass_reasoned_v7；生成器版本：cart_residual_v6。
真实响应次数：10；本地规则/回退次数：0。格式修复重试计入真实响应次数，因此可能超过轮数。
LLM 模型：`gpt-6-luna`；允许回退：False。

| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |
|---|---:|---:|---|---|---|
| b | 3484 / 38.87% | 6969 | 0.7669 → 0.7837 | 0.8047 → 0.8208 | 0.7853 → 0.8018 |
| d | 867 / 9.67% | 1734 | 0.6364 → 0.7910 | 0.0363 → 0.3230 | 0.0687 → 0.4586 |
| w | 4612 / 51.46% | 9225 | 0.7762 → 0.8163 | 0.8848 → 0.8779 | 0.8269 → 0.8460 |

| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |
|---|---:|---:|---:|---|
| oct_llm_strength_gap_log_r1 | 244 | 0.1424 ± 0.0031 | +0.0484 | 通过 |
| oct_llm_oct_piece_strength_rank_context_r2 | 68 | 0.0766 ± 0.0014 | +0.0078 | 通过 |
| oct_llm_oct_file_rank_geometry_strength_r3_r3 | 9 | 0.0002 ± 0.0003 | +0.0022 | 未通过 |

存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。

## balance_scale/full/reference/reasoned/seed_42/LLM

[完整日志](balance_scale/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training.log) · [机器报告](balance_scale/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training_results.json)

| 指标 | baseline | optimized | 改善量（正数更好） |
|---|---:|---:|---:|
| accuracy | 0.8280 | 0.8880 | +0.0600 |
| balanced_accuracy | 0.6000 | 0.7674 | +0.1674 |
| f1 | 0.8037 | 0.8818 | +0.0782 |
| f1_macro | 0.5824 | 0.8099 | +0.2276 |
| auc | 0.8412 | 0.9030 | +0.0618 |
| log_loss | 0.5076 | 0.4277 | +0.0800 |

接受特征：oct_llm_bounded_pair_discrepancy_r1, oct_llm_oct_llm_distance_regime_discrepancy_r3_r3
实际训练预算：125/125。
训练流程版本：multiclass_reasoned_v7；生成器版本：cart_residual_v6。
真实响应次数：9；本地规则/回退次数：0。格式修复重试计入真实响应次数，因此可能超过轮数。
LLM 模型：`gpt-6-luna`；允许回退：False。

| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |
|---|---:|---:|---|---|---|
| B | 10 / 8.00% | 20 | 0.0000 → 1.0000 | 0.0000 → 0.4500 | 0.0000 → 0.6207 |
| L | 57 / 45.60% | 115 | 0.8268 → 0.8640 | 0.9130 → 0.9391 | 0.8678 → 0.9000 |
| R | 58 / 46.40% | 115 | 0.8718 → 0.9052 | 0.8870 → 0.9130 | 0.8793 → 0.9091 |

[论文依据](https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf#page=9)：ICML 2023 OpenFE 自动特征生成的直接实验数据集。
Small mechanistic benchmark (625 rows); minority training count is about 10. Not evidence of large-scale real-world generalization.

| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |
|---|---:|---:|---:|---|
| oct_llm_bounded_pair_discrepancy_r1 | 172 | 0.2133 ± 0.0211 | +0.2276 | 通过 |
| oct_llm_oct_llm_distance_regime_discrepancy_r3_r3 | 42 | 0.0014 ± 0.0027 | -0.0056 | 未通过 |

存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。

## chess_krk/full/reference/reasoned/seed_42/LLM

[完整日志](chess_krk/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training.log) · [机器报告](chess_krk/holdout_20_40_40/reference/clean/features_reasoned/rounds_5_candidates_5/seed_42/training_results.json)

| 指标 | baseline | optimized | 改善量（正数更好） |
|---|---:|---:|---:|
| accuracy | 0.4276 | 0.4673 | +0.0397 |
| balanced_accuracy | 0.3579 | 0.3934 | +0.0355 |
| f1 | 0.4097 | 0.4435 | +0.0338 |
| f1_macro | 0.3705 | 0.4046 | +0.0341 |
| auc | 0.9176 | 0.9248 | +0.0072 |
| log_loss | 1.6419 | 1.5453 | +0.0966 |

接受特征：oct_llm_rook_black_file_periodic_tension_r2, oct_llm_oct_llm_king_rank_periodic_phase_r3_r3, oct_llm_rook_blackking_file_root_gap_v5_r5
实际训练预算：5611/5611。
训练流程版本：multiclass_reasoned_v7；生成器版本：cart_residual_v6。
真实响应次数：5；本地规则/回退次数：0。格式修复重试计入真实响应次数，因此可能超过轮数。
LLM 模型：`gpt-6-luna`；允许回退：False。

| 类别 | 训练样本/比例 | 测试样本 | Precision：基线 → 优化 | Recall：基线 → 优化 | F1：基线 → 优化 |
|---|---:|---:|---|---|---|
| 1 | 559 / 9.96% | 1119 | 0.5069 → 0.4689 | 0.4924 → 0.9500 | 0.4995 → 0.6279 |
| 10 | 5 / 0.09% | 11 | 0.6667 → 0.8333 | 0.1818 → 0.4545 | 0.2857 → 0.5882 |
| 11 | 287 / 5.11% | 573 | 0.4047 → 0.4213 | 0.5410 → 0.5323 | 0.4630 → 0.4703 |
| 12 | 571 / 10.18% | 1142 | 0.4348 → 0.4901 | 0.2513 → 0.2391 | 0.3185 → 0.3214 |
| 13 | 433 / 7.72% | 867 | 0.4661 → 0.5522 | 0.4198 → 0.4268 | 0.4417 → 0.4815 |
| 14 | 94 / 1.68% | 188 | 0.5023 → 0.5654 | 0.5745 → 0.5745 | 0.5360 → 0.5699 |
| 15 | 40 / 0.71% | 79 | 0.6250 → 0.6389 | 0.3165 → 0.2911 | 0.4202 → 0.4000 |
| 16 | 911 / 16.24% | 1821 | 0.4195 → 0.4674 | 0.7243 → 0.6656 | 0.5313 → 0.5492 |
| 17 | 342 / 6.10% | 685 | 0.4040 → 0.4600 | 0.2920 → 0.2774 | 0.3390 → 0.3461 |
| 18 | 16 / 0.29% | 31 | 0.0000 → 0.0000 | 0.0000 → 0.0000 | 0.0000 → 0.0000 |
| 2 | 137 / 2.44% | 273 | 0.3000 → 0.4286 | 0.0440 → 0.1319 | 0.0767 → 0.2017 |
| 3 | 118 / 2.10% | 237 | 0.5587 → 0.5873 | 0.5021 → 0.4684 | 0.5289 → 0.5211 |
| 4 | 78 / 1.39% | 156 | 0.8310 → 0.7703 | 0.3782 → 0.3654 | 0.5198 → 0.4957 |
| 5 | 397 / 7.08% | 794 | 0.3640 → 0.4000 | 0.2040 → 0.2393 | 0.2615 → 0.2994 |
| 6 | 839 / 14.95% | 1678 | 0.3761 → 0.4326 | 0.4678 → 0.4607 | 0.4170 → 0.4462 |
| 7 | 16 / 0.29% | 32 | 0.5000 → 0.0000 | 0.0625 → 0.0000 | 0.1111 → 0.0000 |
| 8 | 719 / 12.81% | 1439 | 0.4128 → 0.4498 | 0.2960 → 0.3204 | 0.3448 → 0.3742 |
| 9 | 49 / 0.87% | 98 | 0.4892 → 0.5194 | 0.6939 → 0.6837 | 0.5738 → 0.5903 |

[论文依据](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf#page=5)：JMLR 分类基准；另有 FLAIRS 2012 的距离特征研究，不能称其为 AAAI 主会。
IID endgame positions; 18 nominal labels are retained, not regression on encoded class IDs. Rare classes have very few test cases; not independent-game/trajectory generalization.

| 新特征 | 分裂使用次数 | 置换 Macro F1 降幅 | 删除重训 Macro F1 降幅 | 联合贡献判据 |
|---|---:|---:|---:|---|
| oct_llm_rook_black_file_periodic_tension_r2 | 499 | 0.0461 ± 0.0047 | +0.0230 | 通过 |
| oct_llm_oct_llm_king_rank_periodic_phase_r3_r3 | 464 | 0.0525 ± 0.0027 | +0.0029 | 通过 |
| oct_llm_rook_blackking_file_root_gap_v5_r5 | 308 | 0.0576 ± 0.0082 | -0.0012 | 未通过 |

存在派生后代时，置换/删除作用于依赖组；分裂次数仅统计该列。判据是预设启发式，不是显著性检验。

## 解释边界

CART 和残差预筛只使用训练集；正式接受仍要求验证主指标改善且其他指标不退化。测试退化照实保留。
这是固定外层划分的实验。多个 seed 改变模型/CART/预算采样，不代表多个独立数据划分；同一测试集反复评估也不构成独立重复。
局部模板与 CART 生成结果不等于真实 LLM 成绩。训练残差预筛不是 OpenFE FeatureBoost 的原样复现。
离线与 LLM 对照还需核对生成器版本。当前 Jungle Chess 离线报告为 v5、LLM 为 v6，不能把两者差异完全归因于 LLM；各自相对 baseline 的增益仍来自对应独立测试报告。
当前生成器包含候选池、CART 路径与预筛策略；不能将收益单独归因于任一组件。CART 是从训练标签学习的监督式特征构造。
当前是固定划分的三个已筛选任务，不能推广为所有表格任务的有效性证据；棋盘任务的随机划分不证明对未见棋子组合的泛化。
