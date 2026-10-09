"""Render Phase 2 review figures from the small, source-linked plotting tables.

This script draws figures only. It does not load checkpoints, fit models or
recompute correlations from predictions. Paired deltas are descriptive derived
values calculated from matching seeds in the accepted metric records.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data', ROOT / 'exports'
OUT.mkdir(exist_ok=True)
INK, BLUE, TEAL, ORANGE, PURPLE = '#243247', '#3B77A5', '#268C84', '#C7804C', '#7869A7'
GRAY, GRID = '#64748B', '#E8EDF2'
plt.rcParams.update({
    'font.family': 'Microsoft YaHei', 'font.size': 11,
    'axes.unicode_minus': False, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': GRAY, 'ytick.color': INK, 'axes.edgecolor': '#B5BFCA',
    'axes.linewidth': .8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
MODELS = ['uni', 'uni2h', 'virchow2']
MODEL_LABEL = {'uni': 'UNI', 'uni2h': 'UNI2-h', 'virchow2': 'Virchow2'}
ARMS = ['point', 'no_b', 'spatial']
ARM_LABEL = {'point': '单点 point', 'no_b': '去空间分支 no_b', 'spatial': '完整空间 spatial'}
COLORS = {'point': BLUE, 'no_b': ORANGE, 'spatial': TEAL}
MARKERS = {'point': 'o', 'no_b': 's', 'spatial': 'D'}
DENSITY = ['dense', 'stride_2x2', 'random_equal']
DENSITY_LABEL = {'dense': '稠密  ·  9,472', 'stride_2x2': '规则隔点  ·  2,383', 'random_equal': '随机等量  ·  2,383'}
DENSITY_COLORS = {'dense': TEAL, 'stride_2x2': BLUE, 'random_equal': ORANGE}
SPLIT_LABEL = {'internal_val': '内部验证', 'external_test': '外部 XZY · 单患者'}
METRIC_LABEL = {'macro_pcc': '患者—通路等权 PCC', 'pooled_pcc': '整体展平 PCC'}
CHECKS = []


def panel(ax, letter, title):
    ax.set_title(title, fontsize=12, loc='left', pad=14)
    ax.text(-.155, 1.055, letter, transform=ax.transAxes, fontsize=15, fontweight='bold')


def axes_style(ax, lim, step=.05, delta=False):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.tick_params(axis='x', length=3, pad=5, labelsize=10)
    ax.set_xlim(*lim)
    ticks = np.arange(np.ceil(lim[0] / step) * step, lim[1] + step / 100, step)
    ticks[np.abs(ticks) < 1e-9] = 0.
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylim(2.55, -.55)
    if delta:
        ax.axvline(0, color=GRAY, lw=1, zorder=1)


def interval(ax, y, mean, sd, values, color, marker='o', seed_spread=.035):
    values = np.asarray(values, dtype=float)
    assert len(values) == 3 and np.all(np.isfinite(values))
    assert np.isclose(values.mean(), mean, atol=1e-8, rtol=1e-8)
    assert np.isclose(values.std(ddof=1), sd, atol=1e-8, rtol=1e-8)
    # Fixed display offsets separate the three seeds; y offset has no data meaning.
    ax.scatter(values, y + np.array([-seed_spread, 0, seed_spread]),
               s=14, color=color, alpha=.45, zorder=3, edgecolors='none')
    ax.errorbar(mean, y, xerr=sd, fmt=marker, markersize=6,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.6,
                color=color, capsize=3.5, elinewidth=1.4, capthick=1.1, zorder=4)
    CHECKS.append({'mean': float(mean), 'sample_sd': float(sd), 'n_seeds': 3})


def save(fig, stem):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    renderer = fig.canvas.get_renderer()
    outside = []
    hidden_axis_labels = {id(t) for ax in fig.axes if not ax.axison
                          for t in ax.get_xticklabels() + ax.get_yticklabels()}
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in hidden_axis_labels or t.axes and t.get_clip_on():
            continue
        # Tick locators may create invisible-to-axis tick labels outside limits.
        if t.axes and t in t.axes.get_xticklabels():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -2 or b.y0 < -2 or b.x1 > w + 2 or b.y1 > h + 2:
            outside.append(t.get_text())
    if outside:
        raise ValueError(f'Figure text outside canvas: {outside}')
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
    plt.close(fig)


def main_comparison():
    summary = pd.read_csv(DATA / 'main_summary.csv')
    seeds = pd.read_csv(DATA / 'main_per_seed.csv')
    assert len(summary) == 36 and len(seeds) == 108
    assert set(seeds.seed) == {45, 46, 47}
    fig = plt.figure(figsize=(12, 11))
    fig.text(.065, .965, '图 P2-3｜编码器与空间分支的重建表现', fontsize=19, fontweight='bold')
    fig.text(.065, .936, '固定全视野输入与30条通路  |  三编码器 × 三实验臂 × 三种子', fontsize=11, color=GRAY)
    handles = [Line2D([0], [0], marker=MARKERS[a], color=COLORS[a], lw=0,
                      markerfacecolor='white', markeredgewidth=1.6, markersize=7, label=ARM_LABEL[a]) for a in ARMS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.058, .919), ncol=3,
               frameon=False, handletextpad=.5, columnspacing=2.2, fontsize=11)
    gs = fig.add_gridspec(3, 2, left=.14, right=.955, bottom=.13, top=.842,
                         hspace=.66, wspace=.38, height_ratios=[1, 1, .9])
    for row, metric in enumerate(['macro_pcc', 'pooled_pcc']):
        for col, split in enumerate(['internal_val', 'external_test']):
            ax = fig.add_subplot(gs[row, col])
            panel(ax, chr(65 + row * 2 + col), f'{SPLIT_LABEL[split]}\n{METRIC_LABEL[metric]}')
            axes_style(ax, (.49, .71) if metric == 'macro_pcc' else (.61, .835))
            ax.set_yticks(range(3), [MODEL_LABEL[m] for m in MODELS])
            ax.set_xlabel('PCC', labelpad=5, fontsize=10)
            for yi, model in enumerate(MODELS):
                for offset, arm in zip([-.22, 0, .22], ARMS):
                    s = summary.query('model == @model and arm == @arm and split == @split and metric == @metric')
                    v = seeds.query('model == @model and arm == @arm and split == @split and metric == @metric').sort_values('seed')
                    assert len(s) == 1
                    s = s.iloc[0]
                    interval(ax, yi + offset, s['mean'], s.sample_sd, v.value, COLORS[arm], MARKERS[arm])
    # Delta uncertainty comes from the three paired seed differences.
    delta_rows, delta_seed_rows = [], []
    for split in ['internal_val', 'external_test']:
        for model in MODELS:
            for metric in ['macro_pcc', 'pooled_pcc']:
                subset = seeds.query('split == @split and model == @model and metric == @metric')
                wide = subset.pivot(index='seed', columns='arm', values='value').loc[[45, 46, 47]]
                vals = wide.spatial - wide.no_b
                delta_rows.append(dict(split=split, model=model, metric=metric, mean=vals.mean(), sample_sd=vals.std(ddof=1), n_seeds=3))
                for seed, value in vals.items():
                    delta_seed_rows.append(dict(split=split, model=model, metric=metric, seed=seed, value=value))
    ds, dv = pd.DataFrame(delta_rows), pd.DataFrame(delta_seed_rows)
    ds.to_csv(DATA / 'paired_spatial_gain_summary.csv', index=False, encoding='utf-8-sig')
    dv.to_csv(DATA / 'paired_spatial_gain_per_seed.csv', index=False, encoding='utf-8-sig')
    for col, split in enumerate(['internal_val', 'external_test']):
        ax = fig.add_subplot(gs[2, col])
        panel(ax, chr(69 + col), f'{SPLIT_LABEL[split]} · 空间分支增量')
        axes_style(ax, (-.005, .061), step=.02, delta=True)
        ax.set_yticks(range(3), [MODEL_LABEL[m] for m in MODELS])
        ax.set_xlabel('ΔPCC  =  spatial − no_b', labelpad=5, fontsize=10)
        for yi, model in enumerate(MODELS):
            for metric, offset, color, marker in [('macro_pcc', -.14, PURPLE, 'o'), ('pooled_pcc', .14, INK, 's')]:
                s = ds.query('model == @model and split == @split and metric == @metric').iloc[0]
                v = dv.query('model == @model and split == @split and metric == @metric').sort_values('seed')
                interval(ax, yi + offset, s['mean'], s.sample_sd, v.value, color, marker)
    fig.legend(handles=[Line2D([0], [0], marker='o', color=PURPLE, lw=0, markerfacecolor='white', label='增量：患者—通路等权 PCC'),
                        Line2D([0], [0], marker='s', color=INK, lw=0, markerfacecolor='white', label='增量：整体展平 PCC')],
               bbox_to_anchor=(.51, .071), loc='center', ncol=2, frameon=False, fontsize=10)
    fig.text(.065, .035, '空心标记与误差线：均值 ± 样本SD；浅色小点：种子45/46/47。增量按相同种子配对计算。', fontsize=9.2, color=GRAY)
    fig.text(.065, .014, '内部：6名开发患者的1,078个留出点；外部：XZY的1,039点。空间增量对照使用同一配方，point另用单点配方。', fontsize=9.2, color=GRAY)
    save(fig, 'Fig_P2_3_编码器与空间分支')


def density_plot(selector='equal_updates', supplement=False):
    summary = pd.read_csv(DATA / 'density_summary.csv')
    seeds = pd.read_csv(DATA / 'density_per_seed.csv')
    summary = summary[summary.selector == selector]
    seeds = seeds[seeds.selector == selector]
    assert len(summary) == 12 and len(seeds) == 36
    fig = plt.figure(figsize=(12, 9.6))
    title = '图 P2-4｜训练密度消融' if not supplement else '图 P2-4S｜训练密度消融：等50轮敏感性分析'
    fig.text(.065, .96, title, fontsize=19, fontweight='bold')
    subtitle = '固定14,800次更新  |  50次内部选模机会  |  9次从零训练' if not supplement else '同一训练轨迹的另一检查点选择器  |  不新增训练重复'
    fig.text(.065, .929, subtitle, fontsize=11, color=GRAY)
    if not supplement:
        ax = fig.add_axes([.065, .737, .89, .145])
        ax.set_xlim(-.5, 30.5); ax.set_ylim(-3.1, 8); ax.axis('off')
        ax.text(0, 7.7, 'A', fontsize=15, fontweight='bold')
        ax.text(1.0, 7.7, '训练采样方案', fontsize=12)
        xy = np.array([(x, y) for y in range(8) for x in range(8)], dtype=float)
        masks = [np.ones(64, dtype=bool), (xy[:, 0] % 2 == 0) & (xy[:, 1] % 2 == 0), np.zeros(64, dtype=bool)]
        masks[2][np.random.default_rng(81).choice(64, size=16, replace=False)] = True
        for j, arm in enumerate(DENSITY):
            x = xy[:, 0] * .32 + j * 10.2 + .2; y = xy[:, 1] * .65
            ax.scatter(x, y, s=7, color=GRID, zorder=1)
            ax.scatter(x[masks[j]], y[masks[j]], s=11, color=DENSITY_COLORS[arm], zorder=2)
            ax.text(j * 10.2 + 3.2, 3.8, DENSITY_LABEL[arm].split('  ·  ')[0], fontsize=12, fontweight='bold')
            ax.text(j * 10.2 + 3.2, 1.3, f"{9472 if arm == 'dense' else 2383:,} 个训练点", fontsize=11, color=GRAY)
        ax.text(0, -2.5, '点阵仅为采样概念示意，非真实组织坐标；内部和外部评价均使用固定稠密点位。', fontsize=9.5, color=GRAY)
    else:
        fig.text(.065, .877, '选择范围：各臂前50个完整训练轮次；底层轨迹仍为14,800次更新。', fontsize=11, color=GRAY)
    top = .66 if not supplement else .77
    gs = fig.add_gridspec(2, 2, left=.17, right=.955, bottom=.155, top=top, hspace=.7, wspace=.4)
    for row, metric in enumerate(['macro_pcc', 'pooled_pcc']):
        for col, split in enumerate(['internal_val', 'external_test']):
            ax = fig.add_subplot(gs[row, col])
            panel(ax, chr((66 if not supplement else 65) + row * 2 + col), f'{SPLIT_LABEL[split]}\n{METRIC_LABEL[metric]}')
            axes_style(ax, (.515, .70) if metric == 'macro_pcc' else (.635, .83))
            ax.set_yticks(range(3), [DENSITY_LABEL[a] for a in DENSITY])
            ax.set_xlabel('PCC', fontsize=10, labelpad=5)
            for yi, arm in enumerate(DENSITY):
                s = summary.query('arm == @arm and split == @split and metric == @metric').iloc[0]
                v = seeds.query('arm == @arm and split == @split and metric == @metric').sort_values('seed')
                interval(ax, yi, s['mean'], s.sample_sd, v.value, DENSITY_COLORS[arm], ['D', 's', 'o'][yi], seed_spread=.055)
                ax.annotate(f"{s['mean']:.3f}", (s['mean'], yi), xytext=(0, -19), textcoords='offset points',
                            ha='center', fontsize=9.5, color=DENSITY_COLORS[arm])
    fig.text(.065, .079, '空心标记与误差线：均值 ± 样本SD；浅色小点：种子45/46/47。SD描述训练随机性。', fontsize=9.4, color=GRAY)
    fig.text(.065, .049, '内部：6名开发患者的1,078个留出点；外部：XZY单患者1,039点。两种PCC按各自定义分别呈现。', fontsize=9.4, color=GRAY)
    fig.text(.065, .020, '抽稀同时改变训练点数、空间覆盖与图连接；该比较反映固定图参数下的整体表现。', fontsize=9.4, color=GRAY)
    stem = 'Fig_P2_4S_等50轮敏感性分析' if supplement else 'Fig_P2_4_训练密度消融'
    save(fig, stem)


if __name__ == '__main__':
    main_comparison()
    density_plot()
    density_plot(selector='equal_50_epochs', supplement=True)
    (ROOT / 'render_checks.json').write_text(json.dumps({
        'status': 'pass', 'checked_point_intervals': len(CHECKS),
        'checks': ['mean and sample SD agree with all three seed values', 'text remains inside figure canvas'],
        'derived_metric': 'paired spatial-minus-no_b per seed; mean and sample SD with ddof=1',
        'formats': ['PNG at 300 dpi', 'SVG editable text', 'PDF vector'],
        'visual_review': 'See README; numerical and canvas checks alone do not prove visual quality.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Rendered main comparison, density comparison and density sensitivity.')
