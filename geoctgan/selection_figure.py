from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _embedding_key(frame):
    return frame[['dimension', 'walks_per_node', 'walk_length']].astype(int).apply(tuple, axis=1)


def _embedding_label(row):
    return f'{int(row.dimension)} D; {int(row.walks_per_node)} walks\nlength {int(row.walk_length)}'


def _cutoff(frame):
    best = frame.sort_values('selection_score_mean').iloc[0]
    value = float(best['selection_score_mean']) + float(best['selection_score_se'])
    if not np.isfinite(value):
        raise ValueError('The selection figure requires finite composite means and standard errors.')
    return value


def _check_runs(summary, runs, keys):
    required = [*keys, 'seed', 'selection_score']
    if any(column not in runs for column in required):
        raise ValueError('The selection figure requires the candidate-level seed results.')
    for row in summary.itertuples(index=False):
        mask = np.ones(len(runs), dtype=bool)
        for key in keys:
            mask &= runs[key].eq(getattr(row, key)).to_numpy()
        group = runs.loc[mask]
        if set(group['seed'].astype(int)) != {42, 52, 62} or len(group) != 3:
            raise ValueError('Each composite-score candidate requires exactly seeds 42, 52 and 62.')
        values = group['selection_score'].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isclose(values.mean(), row.selection_score_mean, rtol=1e-7, atol=1e-9):
            raise ValueError('Composite-score summary and individual seed values do not agree.')


def _composite_panel(ax, summary, runs, keys, selected_key, labels):
    _check_runs(summary, runs, keys)
    threshold = _cutoff(summary)
    blue = '#0072B2'
    for index, row in enumerate(summary.itertuples(index=False)):
        key = tuple(getattr(row, name) for name in keys)
        chosen = key == selected_key
        color = blue if chosen else '#737373'
        marker = '*' if chosen else 's'
        ax.errorbar(index, row.selection_score_mean, yerr=row.selection_score_sd, fmt=marker, color=color, markersize=11 if chosen else 6, capsize=4, linewidth=1.3, zorder=4)
        mask = np.ones(len(runs), dtype=bool)
        for name in keys:
            mask &= runs[name].eq(getattr(row, name)).to_numpy()
        values = runs.loc[mask].sort_values('seed')['selection_score'].to_numpy(dtype=float)
        ax.scatter(index + np.array([-0.11, 0.0, 0.11]), values, s=28, facecolor='white', edgecolor='#858585', linewidth=1, zorder=3)
        ax.annotate(f'{row.selection_score_mean:.3f}', (index, row.selection_score_mean), xytext=(12, 0), textcoords='offset points', va='center', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .93, 'pad': 1.2}, zorder=5)
    ax.axhline(threshold, color='#D55E00', linestyle='--', linewidth=1.3, zorder=2)
    ax.annotate(f'One-SE limit: {threshold:.4f}', (1, threshold), xycoords=('axes fraction', 'data'), xytext=(-4, 7), textcoords='offset points', ha='right', va='bottom', color='#D55E00', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .93, 'pad': 1.2})
    ax.set_xticks(np.arange(len(summary)), labels)
    ax.set_xlim(-.45, len(summary) - .25)
    ax.margins(y=.24)
    ax.grid(axis='y', color='#DEE2E6', linewidth=.8)
    ax.set_axisbelow(True)


def generate_selection_figure(snap_results, embedding_results, embedding_pilot_results, temporal_results, out_dir, *, snap_selection, embedding_selection, temporal_selection):
    snap = pd.read_csv(snap_results)
    if 'split' in snap:
        snap = snap.loc[snap['split'].eq('inner_validation')]
    if snap.empty:
        raise ValueError('The selection figure requires inner-validation snapping results.')
    snap = snap.sort_values('snap_tolerance_m')
    intrinsic = pd.read_csv(embedding_results).sort_values('screen_score_mean').reset_index(drop=True)
    pilot = pd.read_csv(embedding_pilot_results).sort_values('selection_score_mean').reset_index(drop=True)
    pilot_runs = pd.read_csv(Path(embedding_pilot_results).with_name('embedding_generation_pilot_runs.csv'))
    temporal = pd.read_csv(temporal_results)
    order = ['global', 'year', 'year_quarter', 'year_month']
    if len(temporal) != 4 or set(temporal['scheme']) != set(order):
        raise ValueError('The selection figure requires the four reported temporal-pooling schemes.')
    temporal = temporal.set_index('scheme').loc[order].reset_index()
    temporal_runs = pd.read_csv(Path(temporal_results).with_name('temporal_scheme_sensitivity_runs.csv'))
    if len(intrinsic) != 7 or len(pilot) != 3:
        raise ValueError('The selection figure requires seven intrinsic configurations and three downstream finalists.')
    selected_embedding = tuple(int(embedding_selection[key]) for key in ['dimension', 'walks_per_node', 'walk_length'])
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.1), gridspec_kw={'wspace': .58, 'hspace': .55})
        a, b, c, d = axes.flat
        for letter, ax in zip('abcd', axes.flat):
            ax.text(-.14, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=14, ha='left', va='bottom')
        x = snap['snap_tolerance_m'].to_numpy(dtype=float)
        y = 100 * snap['retention'].to_numpy(dtype=float)
        a.plot(x, y, 'o-', color='#737373', linewidth=1.5, markersize=6)
        selected_snap = float(snap_selection['snap_tolerance_m'])
        selected_row = snap.loc[snap['snap_tolerance_m'].eq(selected_snap)]
        if len(selected_row) != 1:
            raise ValueError('The selected snapping tolerance is absent from the figure data.')
        a.scatter(selected_snap, 100 * float(selected_row.iloc[0]['retention']), marker='*', s=160, color='#0072B2', edgecolor='#333333', linewidth=.7, zorder=4)
        eligibility = 100 * (float(snap['retention'].max()) - .001)
        a.axhline(eligibility, color='#D55E00', linestyle='--', linewidth=1.3)
        a.annotate(f'Eligibility: {eligibility:.3f}%', (0, eligibility), xycoords=('axes fraction', 'data'), xytext=(5, 8), textcoords='offset points', fontsize=10, color='#D55E00', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .95, 'pad': 1})
        for index, (tolerance, retention) in enumerate(zip(x, y)):
            a.annotate(f'{retention:.2f}%', (tolerance, retention), xytext=(10 if index == 0 else 0, 9 if index == 0 else -21), textcoords='offset points', ha='left' if index == 0 else 'center', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .95, 'pad': 1})
        a.set_xticks(x)
        a.set_xlabel('Snapping tolerance (m)', labelpad=7)
        a.set_ylabel('Retained events (%)', labelpad=7)
        a.set_ylim(min(y) - .8, max(y) + 1)
        a.margins(x=.07)
        a.grid(axis='y', color='#DEE2E6', linewidth=.8)
        for index, row in enumerate(intrinsic.itertuples(index=False)):
            chosen = (int(row.dimension), int(row.walks_per_node), int(row.walk_length)) == selected_embedding
            color = '#0072B2' if chosen else '#737373'
            b.hlines(index, 0, row.screen_score_mean, color='#C6C6C6', linewidth=1.2)
            deviation = float(row.screen_score_sd)
            b.errorbar(row.screen_score_mean, index, xerr=deviation if np.isfinite(deviation) else None, fmt='*' if chosen else 'o', color=color, markersize=10 if chosen else 6, capsize=3)
            b.annotate(f'{row.screen_score_mean:.3f}', (row.screen_score_mean, index), xytext=(9, 0), textcoords='offset points', va='center', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .95, 'pad': 1})
        b.set_yticks(np.arange(len(intrinsic)), [_embedding_label(row) for row in intrinsic.itertuples(index=False)])
        b.invert_yaxis()
        b.set_xlim(0, max(float(intrinsic['screen_score_mean'].max()) * 1.3, .01))
        b.set_xlabel('Intrinsic composite score', labelpad=7)
        b.grid(axis='x', color='#DEE2E6', linewidth=.8)
        b.set_axisbelow(True)
        _composite_panel(c, pilot, pilot_runs, ['dimension', 'walks_per_node', 'walk_length'], selected_embedding, [f'{int(row.dimension)} D\n{int(row.walks_per_node)} walks\nlength {int(row.walk_length)}' for row in pilot.itertuples(index=False)])
        c.set_ylabel('Downstream composite loss', labelpad=7)
        names = {'global': 'Global', 'year': 'Year', 'year_quarter': 'Year–\nquarter', 'year_month': 'Year–\nmonth'}
        selected_scheme = str(temporal_selection['scheme'])
        labels = [f'{names[row.scheme]}\n{int(row.model_count)} model' + ('s' if int(row.model_count) != 1 else '') + ('\n(selected)' if row.scheme == selected_scheme else '') for row in temporal.itertuples(index=False)]
        _composite_panel(d, temporal, temporal_runs, ['scheme'], (selected_scheme,), labels)
        d.set_ylabel('Temporal composite loss', labelpad=7)
        fig.subplots_adjust(left=.1, right=.985, bottom=.11, top=.965)
        paths = {}
        for extension in ['png', 'pdf']:
            path = output / f'Supplementary_Fig_S12.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            paths[extension] = str(path.resolve())
        plt.close(fig)
    return paths
