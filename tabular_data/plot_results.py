"""从已完成 outputs 报告绘制四张实验图：python tabular_data/plot_results.py。"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "opd_tabular_matplotlib"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from tabular_data.dataset_registry import DATASETS
from tabular_data.refresh_summary import collect_reports

PALETTE = ['#E6F0FF', '#A7C7E7', '#4E79B7', '#FAD7B1', '#F4A261', '#E76F51']
NAMES = {'jungle_chess': '斗兽棋残局', 'balance_scale': '天平平衡', 'chess_krk': '国际象棋残局'}
COLORS = {'baseline': PALETTE[1], 'offline': PALETTE[2], 'llm': PALETTE[4]}
NOTE = '训练／验证／测试：20%／40%／40%；随机种子：42；XGBoost 参考配置。斗兽棋离线生成器 v5，大模型生成器 v6。单次实验，不代表统计显著性。'


def load_runs(root, model):
    selected = {}
    reports = collect_reports(root)
    for dataset in DATASETS:
        selected[dataset] = {}
        for method in ('offline', 'llm'):
            matches = []
            for r in reports:
                g = r['feature_generation']
                if (r['dataset'] != dataset or r['model']['profile'] != 'reference'
                        or r['model']['parameters']['random_state'] != 42
                        or r['noise']['strength_in_train_std'] != 0
                        or g['rounds'] != 5 or g['candidates_per_round'] != 5
                        or r['experiment']['training_budget']['requested_fraction_of_training_split'] != 1):
                    continue
                online = g['api_calls_succeeded'] > 0
                if (method == 'llm') != online:
                    continue
                if online and (g.get('llm_model') != model or g.get('local_rule_calls', 0) != 0):
                    continue
                matches.append(r)
            if len(matches) != 1:
                raise ValueError(f'{dataset}/{method}: expected one declared run, found {len(matches)}; refusing score-based selection.')
            selected[dataset][method] = matches[0]
        offline, llm = selected[dataset].values()
        assert offline['source_files'] == llm['source_files']
        assert offline['model']['parameters'] == llm['model']['parameters']
        assert abs(offline['test']['baseline']['f1_macro'] - llm['test']['baseline']['f1_macro']) < 1e-10
    return selected


def style_axis(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#A2ABB3')
    ax.tick_params(colors='#47545E', length=0, pad=7)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color='#E6E9ED', linewidth=.7)


def save(fig, folder, stem, title, subtitle):
    fig.suptitle(title, x=.065, y=.975, ha='left', va='top', fontsize=17, fontweight='bold', color='#263746')
    fig.text(.065, .90, subtitle, ha='left', va='top', fontsize=10, color='#596775')
    fig.text(.065, .027, NOTE, fontsize=8, color='#6C7883')
    for suffix in ('png', 'pdf'):
        fig.savefig(folder / f'{stem}.{suffix}', dpi=300, facecolor='white')
    plt.close(fig)


def comparison(runs, folder, model):
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.subplots_adjust(left=.09, right=.965, top=.82, bottom=.19)
    x = np.arange(len(DATASETS)); width = .22
    series = [('baseline', '原始特征基线'), ('offline', '离线特征生成'), ('llm', f'大模型特征生成（{model}）')]
    data = {}
    for i, (method, label) in enumerate(series):
        values = [runs[d]['offline' if method == 'baseline' else method]['test']['baseline' if method == 'baseline' else 'optimized']['f1_macro'] for d in DATASETS]
        bars = ax.bar(x + (i-1)*width, values, width, color=COLORS[method], label=label, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4, fontsize=10, color='#344554')
        data[method] = dict(zip(DATASETS, values))
    ax.set_xticks(x, [NAMES[d] for d in DATASETS]); ax.set_ylim(0, 1)
    ax.set_ylabel('测试集宏平均 F1'); style_axis(ax)
    ax.legend(loc='upper left', frameon=False, ncol=3, bbox_to_anchor=(0, 1.075), fontsize=9)
    for i, d in enumerate(DATASETS):
        delta = 100*(data['llm'][d]-data['baseline'][d])
        ax.text(i, .94, f'大模型提升：+{delta:.2f} 个百分点', ha='center', color=PALETTE[5], fontsize=10, fontweight='normal')
    save(fig, folder, '01_test_macro_f1', '特征生成对独立测试集分类效果的提升', '对比原始特征、离线生成与大模型生成；纵轴从零开始，展示真实测试结果。')
    return data


def learning_curves(runs, folder, model):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True)
    fig.subplots_adjust(left=.075, right=.975, top=.79, bottom=.22, wspace=.2)
    data = {}
    for ax, d in zip(axes, DATASETS):
        data[d] = {}
        baseline = runs[d]['llm']['validation']['baseline']['f1_macro']
        ax.axhline(baseline, color='#96A6B4', ls=':', lw=1.4, label='原始特征基线')
        for method, marker in [('offline', 'o'), ('llm', 'D')]:
            r = runs[d][method]
            values = [r['validation']['baseline']['f1_macro']] + [round_['metrics_after']['f1_macro'] for round_ in r['iterations']]
            ax.plot(range(6), values, marker=marker, markersize=5, color=COLORS[method], linewidth=2,
                    label='离线特征生成' if method == 'offline' else f'大模型特征生成（{model}）')
            data[d][method] = values
        ax.set_title(NAMES[d], fontsize=12, pad=12); ax.set_xticks(range(6)); ax.set_ylim(0, 1)
        ax.set_xlabel('特征生成轮次（第 0 轮为基线）'); style_axis(ax)
    axes[0].set_ylabel('验证集宏平均 F1')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .09), ncol=3, frameon=False, fontsize=9)
    save(fig, folder, '02_validation_rounds', '五轮特征生成中的验证指标变化', '曲线仅展示验证集指标，用于选择特征；不使用测试集指标指导逐轮优化。')
    return data


def class_heatmaps(runs, folder):
    fig = plt.figure(figsize=(11.7, 8.8))
    grid = fig.add_gridspec(2, 2, left=.17, right=.86, top=.82, bottom=.13, wspace=.7, hspace=.6)
    axes = {'jungle_chess': fig.add_subplot(grid[0,0]), 'balance_scale': fig.add_subplot(grid[1,0]), 'chess_krk': fig.add_subplot(grid[:,1])}
    data = {}; maximum = 0
    for d in DATASETS:
        counts = runs[d]['llm']['experiment']['training_budget']['class_counts']
        labels = sorted(counts, key=lambda v: int(v) if v.isdigit() else v)
        matrix = []
        for label in labels:
            row = []
            for method in ('offline', 'llm'):
                stats = runs[d][method]['test_class_diagnostics']
                row.append(100*(stats['optimized'][label]['f1-score']-stats['baseline'][label]['f1-score']))
            matrix.append(row)
        data[d] = {'labels': labels, 'training_counts': counts, 'delta_f1_pp': matrix}
        maximum = max(maximum, float(np.max(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list('reference_blue_orange', [PALETTE[2], PALETTE[0], '#FFFFFF', PALETTE[3], PALETTE[5]])
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    for d in DATASETS:
        ax = axes[d]; entry = data[d]; counts = entry['training_counts']; total = sum(counts.values())
        values = np.array(entry['delta_f1_pp'])
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')
        ticks = [f'类别 {label}｜{100*counts[label]/total:.2f}%' for label in entry['labels']]
        ax.set_yticks(range(len(ticks)), ticks, fontsize=9)
        for text, label in zip(ax.get_yticklabels(), entry['labels']):
            if counts[label]/total <= .1:
                text.set_fontweight('bold')
        ax.set_xticks([0,1], ['离线生成', '大模型生成']); ax.set_title(NAMES[d], fontsize=12, pad=12)
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        for i in range(values.shape[0]):
            for j in range(2):
                v = values[i,j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > .7*maximum else '#314151')
    cax = fig.add_axes([.9, .2, .015, .5])
    fig.colorbar(im, cax=cax, label='测试集各类别 F1 相对基线的变化（百分点）')
    save(fig, folder, '03_class_f1_changes', '特征生成对各类别的改善情况', '展示全部类别；标签为类别及训练样本占比，粗体表示占比不超过 10% 的类别。')
    return data


def ablation(runs, folder, model):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=.12, right=.975, top=.78, bottom=.23, wspace=.55)
    data = {}
    for ax, d in zip(axes, DATASETS):
        rows = runs[d]['llm']['ablation']['feature_rows']; y = np.arange(len(rows)); data[d] = rows
        p = [100*r['importance_score'] for r in rows]; std = [100*r['importance_std'] for r in rows]
        drop = [100*r['drop_retrain_score_drop']['f1_macro'] for r in rows]
        ax.barh(y-.17, p, height=.3, xerr=std, capsize=3, color=PALETTE[2], label='置换消融（均值 ± 标准差）')
        ax.barh(y+.17, drop, height=.3, color=PALETTE[4], label='删除特征后重新训练')
        ax.axvline(0, color='#7F8C97', lw=.9)
        ax.set_yticks(y, [f'特征 {i+1}\n分裂 {r["split_count"]} 次' for i,r in enumerate(rows)], fontsize=9)
        ax.invert_yaxis(); ax.set_title(NAMES[d], fontsize=12, pad=12)
        ax.set_xlabel('测试集宏平均 F1 降幅（百分点）'); style_axis(ax, 'x')
        ax.margins(x=.2)
        for i, (pv, dv) in enumerate(zip(p,drop)):
            ax.annotate(f'{pv:+.2f}', (pv + (std[i] if pv >= 0 else -std[i]), i-.17), xytext=(5 if pv >= 0 else -5,0), textcoords='offset points', va='center', ha='left' if pv >= 0 else 'right', fontsize=8)
            ax.annotate(f'{dv:+.2f}', (dv, i+.17), xytext=(5 if dv>=0 else -5,0),
                        textcoords='offset points', va='center', ha='left' if dv>=0 else 'right', fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.1), ncol=2, frameon=False, fontsize=9)
    save(fig, folder, '04_llm_feature_ablation', '大模型生成特征的使用情况与消融贡献', f'模型：{model}；仅展示已接受特征。置换重复 10 次；存在派生特征时按依赖组消融；子图横轴尺度不同。')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=PROJECT/'tabular_data'/'outputs')
    parser.add_argument('--llm-model', default='gpt-6-luna')
    args = parser.parse_args()
    root = args.outputs.resolve(); folder = root/'figures'; folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': ['Droid Sans Fallback', 'DejaVu Sans'], 'font.size': 10, 'axes.labelcolor': '#344554',
                         'pdf.fonttype': 42, 'savefig.dpi': 300, 'axes.unicode_minus': False})
    runs = load_runs(root, args.llm_model)
    source = {d:{m:{'report':str(Path(r['output_dir']).relative_to(root)/'training_results.json'),
                        'run_id':r['run_id'], 'generator_version':r['feature_generation']['version']}
                    for m,r in methods.items()} for d,methods in runs.items()}
    values = {'palette': PALETTE, 'sources': source,
              'test_macro_f1': comparison(runs,folder,args.llm_model),
              'validation_rounds': learning_curves(runs,folder,args.llm_model),
              'class_f1_changes': class_heatmaps(runs,folder),
              'llm_feature_ablation': ablation(runs,folder,args.llm_model)}
    (folder/'figure_data.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 表格实验可视化', '', '配色参考：`../matplotlib.jpg`；Matplotlib 输出 300 DPI PNG 及矢量 PDF。',
             '', '复现：在仓库根目录运行 `python tabular_data/plot_results.py`。只读取已完成的 outputs 报告，不重新训练或调用 API。', '',
             '## 四张图', '',
             '1. [测试 Macro F1 柱状图](01_test_macro_f1.png)：完整展示 baseline、离线和 LLM 成绩，纵轴从 0 开始。',
             '2. [逐轮验证曲线](02_validation_rounds.png)：展示 0–5 轮的特征选择轨迹；这是验证指标，不是逐轮测试指标。',
             '3. [各类别 F1 变化热图](03_class_f1_changes.png)：展示全部类别相对 baseline 的 F1 变化（百分点），并标明训练比例；蓝色为下降，橙色为提升，粗体为训练比例不超过 10% 的类别。',
             '4. [LLM 特征消融图](04_llm_feature_ablation.png)：展示已接受特征的分裂次数、置换重要性及删除重训影响；保留负值，不将所有接受特征称为有效。', '',
             '## 数据与解释边界', '',
             '所有图使用 reference XGBoost、seed 42、20/40/40 固定划分、5 轮 × 5 个候选的原始结果。完整数值和来源 run_id 见 [figure_data.json](figure_data.json)。',
             '置换误差线是 10 次置换的标准差，不是跨随机种子或数据划分的置信区间。依赖组消融不能解释成单列的独立因果贡献。',
             'Jungle 离线生成器为 v5，LLM 为 v6，两条轨迹不是只改变 LLM 的严格对照。LLM 在 Balance Scale 上略优于当前离线结果，在另两个任务上低于离线；不筛掉这些结果。',
             '三项任务只有一次固定划分，不据此声称统计显著性。少数类别的变化在热图中完整展示，不声称全部稀少类别都得到提升。', '', '## 消融图特征编号', '']
    for d in DATASETS:
        lines += [f'### {NAMES[d]}', '']
        for i,r in enumerate(runs[d]['llm']['ablation']['feature_rows']):
            lines.append(f'- 特征 {i+1}: `{r["feature"]}`；分裂次数 {r["split_count"]}。')
        lines.append('')
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Created four figures (PNG + PDF) in {folder}')


if __name__ == '__main__':
    main()
