# 表格实验可视化

配色参考：`../matplotlib.jpg`；Matplotlib 输出 300 DPI PNG 及矢量 PDF。

复现：在仓库根目录运行 `python tabular_data/plot_results.py`。只读取已完成的 outputs 报告，不重新训练或调用 API。

## 四张图

1. [测试 Macro F1 柱状图](01_test_macro_f1.png)：完整展示 baseline、离线和 LLM 成绩，纵轴从 0 开始。
2. [逐轮验证曲线](02_validation_rounds.png)：展示 0–5 轮的特征选择轨迹；这是验证指标，不是逐轮测试指标。
3. [各类别 F1 变化热图](03_class_f1_changes.png)：展示全部类别相对 baseline 的 F1 变化（百分点），并标明训练比例；蓝色为下降，橙色为提升，粗体为训练比例不超过 10% 的类别。
4. [LLM 特征消融图](04_llm_feature_ablation.png)：展示已接受特征的分裂次数、置换重要性及删除重训影响；保留负值，不将所有接受特征称为有效。

## 数据与解释边界

所有图使用 reference XGBoost、seed 42、20/40/40 固定划分、5 轮 × 5 个候选的原始结果。完整数值和来源 run_id 见 [figure_data.json](figure_data.json)。
置换误差线是 10 次置换的标准差，不是跨随机种子或数据划分的置信区间。依赖组消融不能解释成单列的独立因果贡献。
Jungle 离线生成器为 v5，LLM 为 v6，两条轨迹不是只改变 LLM 的严格对照。LLM 在 Balance Scale 上略优于当前离线结果，在另两个任务上低于离线；不筛掉这些结果。
三项任务只有一次固定划分，不据此声称统计显著性。少数类别的变化在热图中完整展示，不声称全部稀少类别都得到提升。

## 消融图特征编号

### 斗兽棋残局

- 特征 1: `oct_llm_strength_gap_log_r1`；分裂次数 244。
- 特征 2: `oct_llm_oct_piece_strength_rank_context_r2`；分裂次数 68。
- 特征 3: `oct_llm_oct_file_rank_geometry_strength_r3_r3`；分裂次数 9。

### 天平平衡

- 特征 1: `oct_llm_bounded_pair_discrepancy_r1`；分裂次数 172。
- 特征 2: `oct_llm_oct_llm_distance_regime_discrepancy_r3_r3`；分裂次数 42。

### 国际象棋残局

- 特征 1: `oct_llm_rook_black_file_periodic_tension_r2`；分裂次数 499。
- 特征 2: `oct_llm_oct_llm_king_rank_periodic_phase_r3_r3`；分裂次数 464。
- 特征 3: `oct_llm_rook_blackking_file_root_gap_v5_r5`；分裂次数 308。
