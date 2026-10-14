from __future__ import annotations
from .utils import write_output_csv
import inspect
import re
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
COLORS = {'Real held-out': '#222222', 'GeoCTGAN-Full': '#0072B2', 'GeoCTGAN-Priv': '#D55E00', 'GeoCTGAN-UpperQuota': '#56B4E9'}
DCR_RANDOM_SEEDS = {(42, 'DCRBaselineProtection'): 2671879672, (42, 'DCROverfittingProtection'): 2672010407, (52, 'DCRBaselineProtection'): 1943310416, (52, 'DCROverfittingProtection'): 3558307939, (62, 'DCRBaselineProtection'): 159110251, (62, 'DCROverfittingProtection'): 2498872653}
DPE_RANDOM_SEEDS = {(42, 'Full', 'release_attribute'): 3730491714, (42, 'Full', 'release_category_aware'): 2105110939, (42, 'UpperQuota', 'release_attribute'): 1261055104, (42, 'UpperQuota', 'release_category_aware'): 3850277879, (42, 'Priv', 'release_attribute'): 3030378602, (42, 'Priv', 'release_category_aware'): 3230399155, (52, 'Full', 'release_attribute'): 1012485721, (52, 'Full', 'release_category_aware'): 1189051039, (52, 'UpperQuota', 'release_attribute'): 2775075990, (52, 'UpperQuota', 'release_category_aware'): 1217434993, (52, 'Priv', 'release_attribute'): 3624222994, (52, 'Priv', 'release_category_aware'): 1852219379, (62, 'Full', 'release_attribute'): 1788356628, (62, 'Full', 'release_category_aware'): 1335280657, (62, 'UpperQuota', 'release_attribute'): 1764574341, (62, 'UpperQuota', 'release_category_aware'): 4156577157, (62, 'Priv', 'release_attribute'): 49565638, (62, 'Priv', 'release_category_aware'): 862287341}
CALENDAR = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW', 'OCC_HOUR']
ATTRIBUTES = ['MCI_CATEGO', 'OFFENCE', 'PREMISES_T', 'LOCATION_T']
DPE_KNOWN = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_HOUR', 'PREMISES_T', 'LOCATION_T']
KMAP_SCENARIOS = {'release_core': ['MCI_CATEGO', 'PREMISES_T', 'LOCATION_T', 'OCC_DOW', 'OCC_MONTH']}

def method_seed(label):
    match = re.fullmatch('(.+?)\\s*\\|\\s*seed\\s+(\\d+)', str(label))
    if match is None:
        raise ValueError('Dataset labels must include a generator seed.')
    return (match.group(1), int(match.group(2)))

def save_figure(fig, directory, name):
    fig.savefig(Path(directory) / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def deterministic_sample(values, rows, seed):
    if len(values) <= rows:
        return values.copy()
    return values[np.random.default_rng(seed).choice(len(values), rows, replace=False)]

def categorical_series(series, calendar=False):
    if calendar:
        return pd.to_numeric(series, errors='coerce').astype('Int64').astype('string').fillna('<MISSING>').astype(object)
    return series.astype('string').fillna('<MISSING>').astype(object)

def metric_random_seed(seed, metric, profile='Full', scenario='release_attribute'):
    if metric == 'DisclosureProtectionEstimate':
        key = (int(seed), profile, scenario)
        if key in DPE_RANDOM_SEEDS:
            return DPE_RANDOM_SEEDS[key]
        metric_key = 3
    else:
        key = (int(seed), metric)
        if key in DCR_RANDOM_SEEDS:
            return DCR_RANDOM_SEEDS[key]
        metric_key = {'DCRBaselineProtection': 1, 'DCROverfittingProtection': 2}[metric]
    profile_key = {'Full': 1, 'UpperQuota': 2, 'Priv': 3}[profile] if metric_key == 3 else 0
    scenario_key = {'release_attribute': 1, 'release_category_aware': 2}[scenario] if metric_key == 3 else 0
    return int(np.random.SeedSequence([101, int(seed), metric_key, profile_key, scenario_key]).generate_state(1)[0])

def seeded_metric_call(metric_class, kwargs, seed, require_class_seed=False):
    np.random.seed(int(seed))
    random.seed(int(seed))
    has_class_seed = hasattr(metric_class, '_seed')
    if require_class_seed and (not has_class_seed):
        raise RuntimeError('DCR Baseline Protection requires native random-seed control.')
    previous = getattr(metric_class, '_seed', None)
    if has_class_seed:
        metric_class._seed = int(seed)
    try:
        result = metric_class.compute_breakdown(**kwargs)
    finally:
        if has_class_seed:
            metric_class._seed = previous
    score = float(result['score'] if isinstance(result, dict) else result)
    if not np.isfinite(score):
        raise RuntimeError('The privacy estimator returned a non-finite score.')
    return score

def privacy_metrics(real_train, real_test, synthetic, artifacts, cfg):
    from sklearn.neighbors import KDTree
    from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection, DisclosureProtectionEstimate
    from .evaluation import _coordinates, _coordinate_valid_mask, _balanced_dcr_training_reference, _infer_sdmetrics_metadata
    train_xy = _coordinates(real_train, cfg)
    tree = KDTree(train_xy)
    proximity_rows, kmap_rows, exact_rows, metric_rows = ([], [], [], [])
    train_valid = real_train.loc[_coordinate_valid_mask(real_train, cfg)].copy()
    selected = set(cfg.heavy_privacy_datasets)
    for label, frame in synthetic.items():
        method, seed = method_seed(label)
        profile = method.removeprefix('GeoCTGAN-')
        distances = tree.query(_coordinates(frame, cfg), k=1)[0].ravel()
        row = {'dataset': label, 'synthetic_rows': len(distances), 'training_rows': len(train_xy), 'distance_min_m': float(distances.min())}
        for q in [0.01, 0.05, 0.5, 0.95]:
            row[f'distance_p{round(100 * q):02d}_m'] = float(np.quantile(distances, q))
        row['distance_median_m'] = float(np.median(distances))
        for threshold in cfg.near_real_thresholds_m:
            row[f'fraction_within_{threshold:g}m'] = float(np.mean(distances <= threshold))
        proximity_rows.append(row)
        if profile not in {'Full', 'UpperQuota', 'MinimumSupport', 'Priv'}:
            continue
        generated_valid = frame.loc[_coordinate_valid_mask(frame, cfg)].copy()
        generated_xy = _coordinates(generated_valid, cfg)
        for scenario, columns in KMAP_SCENARIOS.items():
            if scenario != 'release_core':
                continue
            if not set(columns).issubset(real_train.columns) or not set(columns).issubset(frame.columns):
                continue
            left = pd.DataFrame({column: categorical_series(train_valid[column], column in CALENDAR) for column in columns}).reset_index(drop=True)
            right = pd.DataFrame({column: categorical_series(generated_valid[column], column in CALENDAR) for column in columns}).reset_index(drop=True)
            for size in cfg.kmap_grid_sizes_m:
                for ox, oy in cfg.kmap_grid_origins:
                    shift = np.array([ox * size, oy * size])
                    left[['grid_x', 'grid_y']] = np.floor((train_xy - shift) / size).astype(np.int64)
                    right[['grid_x', 'grid_y']] = np.floor((generated_xy - shift) / size).astype(np.int64)
                    keys = columns + ['grid_x', 'grid_y']
                    counts = left.groupby(keys, dropna=False, observed=True).size().rename('real_k')
                    matches = right.merge(counts.reset_index(), on=keys, how='left')['real_k'].fillna(0)
                    matched = matches[matches >= 1]
                    item = {'dataset': label, 'scenario': scenario, 'grid_size_m': size, 'origin_x_fraction': ox, 'origin_y_fraction': oy, 'synthetic_rows': len(matches), 'matched_rows': len(matched), 'match_rate': len(matched) / len(matches)}
                    for k in [1, 2, 5, 10]:
                        item[f'k_le_{k}_conditional'] = float(np.mean(matched <= k)) if len(matched) else np.nan
                        item[f'k_le_{k}_all'] = float(np.mean((matches >= 1) & (matches <= k)))
                    kmap_rows.append(item)
        for panel, columns in ({'complete_release': cfg.heavy_privacy_columns, 'release_core': KMAP_SCENARIOS['release_core']} if cfg.privacy_profile == 'toronto' else {}).items():
            if not set(columns).issubset(real_train.columns) or not set(columns).issubset(frame.columns):
                continue
            left = real_train[columns].copy()
            right = frame[columns].copy()
            for column in columns:
                if column in ATTRIBUTES or panel == 'release_core':
                    left[column] = categorical_series(left[column], column in CALENDAR)
                    right[column] = categorical_series(right[column], column in CALENDAR)
            reference_keys = left.drop_duplicates()
            matched = right.merge(reference_keys.assign(_match=1), on=columns, how='left')['_match'].fillna(0)
            exact_rows.append({'dataset': label, 'panel': panel, 'synthetic_rows': len(right), 'matched_rows': int(matched.sum()), 'match_rate': float(matched.mean())})
        if label not in selected:
            continue
        columns = cfg.heavy_privacy_columns
        for data in [real_train, real_test, frame]:
            if not set(columns).issubset(data.columns):
                raise ValueError('The selected privacy panel is missing required fields.')
        real, test, generated = [data[columns].copy() for data in [real_train, real_test, frame]]
        if cfg.privacy_profile == 'toronto':
            for column in columns:
                for data in [real, test, generated]:
                    data[column] = categorical_series(data[column]) if column in ATTRIBUTES else pd.to_numeric(data[column], errors='coerce')
        metadata = _infer_sdmetrics_metadata(real)
        balanced = _balanced_dcr_training_reference(real, test, maximum_ratio=cfg.dcr_training_to_validation_max_ratio, random_state=101)
        for metric_class in [DCRBaselineProtection, DCROverfittingProtection]:
            metric = metric_class.__name__
            kwargs = {'synthetic_data': generated, 'metadata': metadata, 'num_rows_subsample': min(cfg.privacy_native_subsample_rows, len(real), len(generated)), 'num_iterations': cfg.privacy_native_iterations}
            if metric == 'DCRBaselineProtection':
                kwargs['real_data'] = real
            else:
                kwargs.update(real_training_data=balanced, real_validation_data=test)
                kwargs['num_rows_subsample'] = min(cfg.privacy_native_subsample_rows, len(balanced), len(generated), len(test))
            if cfg.privacy_profile == 'toronto':
                random_seed = metric_random_seed(seed, metric)
                score = seeded_metric_call(metric_class, kwargs, random_seed, metric == 'DCRBaselineProtection')
            else:
                result = metric_class.compute_breakdown(**kwargs)
                score = float(result['score'])
            if not np.isfinite(score):
                raise RuntimeError('The privacy estimator returned a non-finite score.')
            metric_rows.append({'dataset': label, 'metric': metric, 'scenario': 'complete_release', 'score': score})
        scenarios = {'release_attribute': DPE_KNOWN, 'release_category_aware': DPE_KNOWN + ['MCI_CATEGO']} if cfg.privacy_profile == 'toronto' else {'ottawa': ['DIVISION', 'MCI_CATEGO']}
        for scenario, known in scenarios.items():
            columns = known + ['OFFENCE']
            left, right = (real_train[columns].copy(), frame[columns].copy())
            continuous = ['OCC_YEAR', 'OCC_DAY', 'OCC_HOUR'] if cfg.privacy_profile == 'toronto' else []
            for column in columns:
                if column in continuous:
                    left[column], right[column] = (pd.to_numeric(left[column], errors='coerce'), pd.to_numeric(right[column], errors='coerce'))
                elif cfg.privacy_profile == 'toronto':
                    left[column], right[column] = (categorical_series(left[column], column in CALENDAR), categorical_series(right[column], column in CALENDAR))
            kwargs = {'real_data': left, 'synthetic_data': right, 'known_column_names': known, 'sensitive_column_names': ['OFFENCE'], 'continuous_column_names': continuous, 'num_discrete_bins': 10}
            if cfg.privacy_profile == 'toronto':
                kwargs.update(computation_method='cap', num_rows_subsample=min(10000, len(left), len(right)), num_iterations=10, verbose=False)
                score = seeded_metric_call(DisclosureProtectionEstimate, kwargs, metric_random_seed(seed, 'DisclosureProtectionEstimate', profile, scenario))
            else:
                kwargs.update(computation_method='cap',num_rows_subsample=1000,num_iterations=10,verbose=False)
                score = float(DisclosureProtectionEstimate.compute_breakdown(**kwargs)['score'])
            if not np.isfinite(score):
                raise RuntimeError('The privacy estimator returned a non-finite score.')
            metric_rows.append({'dataset': label, 'metric': 'DisclosureProtectionEstimate', 'scenario': scenario, 'score': score})
    tables = {'spatial_proximity_risk.csv': proximity_rows, 'spatial_kmap_risk.csv': kmap_rows, 'exact_key_matching.csv': exact_rows, 'sdmetrics_privacy.csv': metric_rows}
    for name, rows in tables.items():
        if rows:
            write_output_csv(pd.DataFrame(rows), artifacts.tables / name, index=False)
    if cfg.privacy_profile == 'toronto':
        privacy_figure(pd.DataFrame(proximity_rows), pd.DataFrame(metric_rows), artifacts.figures)
    return {'spatial_proximity': 'Tables/spatial_proximity_risk.csv', 'sdmetrics_privacy': 'Tables/sdmetrics_privacy.csv'}

def privacy_figure(proximity, metrics, directory):
    methods = ['GeoCTGAN-Full', 'GeoCTGAN-UpperQuota', 'GeoCTGAN-MinimumSupport', 'GeoCTGAN-Priv']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    datasets = proximity['dataset'].map(lambda value: method_seed(value)[0])
    for index, ax in enumerate(axes.flat):
        ax.text(-0.13, 1.03, 'abcd'[index], transform=ax.transAxes, fontweight='bold', fontsize=15)
        if index == 0:
            selected = proximity.assign(method=datasets)
            column = 'fraction_within_5m'
            chosen = [method for method in methods if method in set(selected['method'])]
            factor = 100
            ax.set_ylabel('Synthetic points within 5 m (%)', fontsize=12)
        else:
            metric, scenario = [('DisclosureProtectionEstimate', 'release_attribute'), ('DCRBaselineProtection', 'complete_release'), ('DCROverfittingProtection', 'complete_release')][index - 1]
            selected = metrics.loc[(metrics['metric'] == metric) & ((metrics['scenario'] == scenario) | (metrics['scenario'] == 'ottawa'))].copy()
            selected['method'] = selected['dataset'].map(lambda value: method_seed(value)[0])
            column, factor = ('score', 1)
            chosen = [method for method in methods if method in set(selected['method'])]
            ax.set_ylabel('Protection score (0–1)', fontsize=12)
            ax.set_ylim(0, 1.04)
        for x, method in enumerate(chosen):
            values = selected.loc[selected['method'] == method, column].to_numpy(float) * factor
            mean = float(np.mean(values))
            color = COLORS.get(method, '#E69F00')
            ax.bar(x, mean, color=color, edgecolor='#333333', width=0.68)
            if len(values) > 1:
                ax.errorbar(x, mean, yerr=np.std(values, ddof=1), color='#444444', capsize=3, lw=1)
                ax.scatter(x + np.linspace(-0.04, 0.04, len(values)), values, s=18, facecolors='white', edgecolors='#444444', zorder=4)
            label = f'{mean:.2f}%' if index == 0 else f'{mean:.3f}'
            y = mean * 0.68
            ax.text(x, y if mean else (0.9 if index == 0 else 0.015), label, ha='center', va='center' if mean else 'bottom', color='white' if mean else '#222222', fontweight='bold', fontsize=10)
        ax.set_xticks(range(len(chosen)), [m.replace('GeoCTGAN-', '').replace('UpperQuota', 'Upper\nQuota').replace('MinimumSupport', 'Minimum\nSupport') for m in chosen], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dfe3e8', lw=0.7)
        if index == 0:
            ax.set_ylim(0, max(27, selected[column].max() * 125))
    fig.tight_layout(h_pad=3, w_pad=2)
    save_figure(fig, directory, 'privacy_indicators')

def normalized_da_count_wape(reference, synthetic):
    reference = np.asarray(reference, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    reference_total = reference.sum()
    synthetic_total = synthetic.sum()
    if reference_total <= 0 or synthetic_total <= 0:
        return np.nan
    scaled = synthetic * (reference_total / synthetic_total)
    return float(np.abs(scaled - reference).sum() / reference_total)

def spatial_metrics(real_train, real_test, synthetic, artifacts, cfg, da_boundaries):
    from . import evaluation as ev
    train_xy, test_xy = (ev._coordinates(real_train, cfg), ev._coordinates(real_test, cfg))
    coordinates = {name: ev._coordinates(frame, cfg) for name, frame in synthetic.items()}
    da = da_boundaries.to_crs(cfg.metric_crs) if da_boundaries is not None else None
    extent = tuple(cfg.spatial_extent_m) if cfg.spatial_extent_m is not None else ev._spatial_extent(train_xy, da)
    nnd_rows, cdf_rows, grid_rows = ([], [], [])
    distances = np.linspace(0, 75, 1501)
    n = min(len(test_xy), cfg.maximum_equal_sample_rows)
    real_nnd = ev._nearest_neighbour_distances(deterministic_sample(test_xy, n, 101))
    cdf = np.searchsorted(np.sort(real_nnd), distances, side='right') / len(real_nnd)
    cdf_rows.extend(({'dataset': 'Real held-out', 'distance_m': d, 'cdf': value} for d, value in zip(distances, cdf)))
    for label, frame in synthetic.items():
        if cfg.privacy_profile == 'toronto':
            values = ev._nearest_neighbour_distances(deterministic_sample(coordinates[label], n, 102))
            nnd_rows.append({'dataset': label, 'reference_mean_nnd_m': float(real_nnd.mean()), 'synthetic_mean_nnd_m': float(values.mean()), 'rows_per_table': len(values)})
            curve = np.searchsorted(np.sort(values), distances, side='right') / len(values)
            cdf_rows.extend(({'dataset': label, 'distance_m': d, 'cdf': value} for d, value in zip(distances, curve)))
        for repeat_seed in cfg.random_seeds:
            left, right = ev._sample_pair(real_test, frame, repeat_seed, cfg.maximum_equal_sample_rows)
            left_xy, right_xy = (ev._coordinates(left, cfg), ev._coordinates(right, cfg))
            if cfg.privacy_profile == 'ottawa':
                left_nnd, right_nnd = (ev._nearest_neighbour_distances(left_xy), ev._nearest_neighbour_distances(right_xy))
                nnd_rows.append({'dataset': label, 'repeat_seed': repeat_seed, 'reference_mean_nnd_m': float(left_nnd.mean()), 'synthetic_mean_nnd_m': float(right_nnd.mean()), 'rows_per_table': len(left_xy)})
            for size in cfg.grid_sizes_m:
                dimensions = (max(1, int(np.ceil((extent[2] - extent[0]) / size))), max(1, int(np.ceil((extent[3] - extent[1]) / size))))
                left_counts, right_counts = [ev._grid_counts(xy, size, extent[:2], dimensions) for xy in [left_xy, right_xy]]
                left_hotspots = ev._exact_top_q_indices(left_counts, cfg.hotspot_fraction)
                right_hotspots = ev._exact_top_q_indices(right_counts, cfg.hotspot_fraction, count=len(left_hotspots))
                grid_rows.append({'panel': 'held_out_generalization', 'dataset': label, 'repeat_seed': repeat_seed, 'grid_size_m': size, 'count_pearson': ev._safe_pearson(left_counts, right_counts), 'hotspot_jaccard': ev._jaccard(left_hotspots, right_hotspots)})
    write_output_csv(pd.DataFrame(nnd_rows), artifacts.tables / 'nearest_neighbour_fidelity.csv', index=False)
    write_output_csv(pd.DataFrame(grid_rows), artifacts.tables / 'multiscale_grid_fidelity.csv', index=False)
    if cfg.privacy_profile == 'toronto':
        write_output_csv(pd.DataFrame(cdf_rows), artifacts.tables / 'nearest_neighbour_cdf.csv', index=False)
        nnd_figure(pd.DataFrame(cdf_rows), artifacts.figures, primary=True)
        nnd_figure(pd.DataFrame(cdf_rows), artifacts.figures)
    bandwidth = ev._kde_bandwidth(train_xy)
    xx, yy = np.meshgrid(np.linspace(extent[0], extent[2], cfg.kde_grid_size), np.linspace(extent[1], extent[3], cfg.kde_grid_size))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    real_surface = ev._kde_surface(test_xy, points, bandwidth, cfg.maximum_kde_fit_rows, 102)
    surfaces, kde_rows = ({}, [])
    for label, xy in coordinates.items():
        surface = ev._kde_surface(xy, points, bandwidth, cfg.maximum_kde_fit_rows, 103)
        surfaces[label] = surface
        kde_rows.append({'panel': 'held_out_generalization', 'dataset': label, 'bandwidth_m': bandwidth, 'grid_size': cfg.kde_grid_size, 'kde_pearson': ev._safe_pearson(real_surface, surface), 'kde_mae': float(np.mean(np.abs(real_surface - surface)))})
    write_output_csv(pd.DataFrame(kde_rows), artifacts.tables / 'kde_surface_fidelity.csv', index=False)
    means = {}
    mean_rows = []
    for method in ['GeoCTGAN-Full', 'GeoCTGAN-Priv']:
        stack = [surface for label, surface in surfaces.items() if method_seed(label)[0] == method]
        if stack:
            means[method] = np.mean(stack, axis=0)
            mean_rows.append({'dataset': method, 'bandwidth_m': bandwidth, 'grid_size': cfg.kde_grid_size, 'kde_pearson': ev._safe_pearson(real_surface, means[method]), 'kde_mae': float(np.mean(np.abs(real_surface - means[method])))})
    if cfg.privacy_profile == 'toronto':
        write_output_csv(pd.DataFrame(mean_rows), artifacts.tables / 'kde_mean_surface_fidelity.csv', index=False)
        kde_figure(real_surface, means, xx, yy, cfg, artifacts.figures)
    if da is not None:
        da_id = ev._find_da_identifier(da)
        counts = {'Real training': ev._da_counts(real_train, da, da_id, cfg).to_numpy(), 'Real held-out': ev._da_counts(real_test, da, da_id, cfg).to_numpy()}
        counts.update({label: ev._da_counts(frame, da, da_id, cfg).to_numpy() for label, frame in synthetic.items()})
        write_output_csv(pd.DataFrame({da_id: da[da_id].astype(str), **{name: values for name, values in counts.items() if name != 'Real training'}}), artifacts.tables / 'da_counts.csv', index=False)
        fidelity_rows = []
        for label, values in counts.items():
            if label.startswith('Real '):
                continue
            row = {'reference': 'Real held-out', 'dataset': label, 'da_count_pearson': ev._safe_pearson(counts['Real held-out'], values)}
            if cfg.privacy_profile == 'ottawa':
                row['da_count_wape'] = normalized_da_count_wape(counts['Real held-out'], values)
            fidelity_rows.append(row)
        write_output_csv(pd.DataFrame(fidelity_rows), artifacts.tables / 'dissemination_area_fidelity.csv', index=False)
        global_stats, agreement, labels = ev._spatial_autocorrelation(counts, da, da_id, cfg)
        for frame, name in [(global_stats, 'global_moran.csv'), (agreement, 'lisa_agreement.csv'), (labels, 'lisa_cluster_labels.csv')]:
            if 'dataset' in frame:
                frame = frame.loc[frame['dataset'] != 'Real training']
            if 'reference' in frame:
                frame = frame.loc[frame['reference'] == 'Real held-out']
            write_output_csv(frame, artifacts.tables / name, index=False)
        if cfg.privacy_profile == 'toronto':
            area_figure(da, da_id, counts, global_stats, labels, artifacts)
        else:
            ottawa_lisa_figure(da, da_id, global_stats, labels, artifacts)
    return {'nnd_table': 'Tables/nearest_neighbour_fidelity.csv', 'kde_table': 'Tables/kde_surface_fidelity.csv', 'kde_bandwidth_m': bandwidth}

def nnd_figure(table, directory, primary=False):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.6), gridspec_kw={'height_ratios': [1.3, 1]}, sharex=True)
    real = table.loc[table['dataset'] == 'Real held-out'].set_index('distance_m')['cdf']
    axes[0].plot(real.index, real, color=COLORS['Real held-out'], lw=1.5, label='Real held-out')
    chosen = ['GeoCTGAN-Full','GeoCTGAN-Priv'] if primary else ['GeoCTGAN-Full','GeoCTGAN-Priv','GaussianCopula','CTGAN_RawLatLon','KDE_AttributeBootstrap','RoadSupportBootstrap']
    methods = [name for name in chosen if any(value != 'Real held-out' and method_seed(value)[0] == name for value in table['dataset'].unique())]
    inset = axes[0].inset_axes([.60,.14,.37,.46])
    inset.plot(real.index,real,color=COLORS['Real held-out'],lw=1.1)
    for index, method in enumerate(methods):
        names = [name for name in table['dataset'].unique() if name != 'Real held-out' and method_seed(name)[0] == method]
        pivot = table.loc[table['dataset'].isin(names)].pivot(index='distance_m', columns='dataset', values='cdf')
        mean = pivot.mean(axis=1)
        color = COLORS.get(method, plt.cm.tab10(index % 10))
        style = '-' if method in COLORS else ['--', ':', '-.'][index % 3]
        inset.plot(mean.index,mean,color=color,lw=1.1,ls=style)
        for ax, baseline in [(axes[0], 0), (axes[1], real)]:
            ax.plot(mean.index, mean - baseline, color=color, lw=1.5, ls=style, label=method)
            ax.fill_between(mean.index, pivot.min(axis=1) - baseline, pivot.max(axis=1) - baseline, color=color, alpha=0.13)
    axes[0].set_ylabel('Cumulative proportion', fontsize=12)
    axes[1].set_ylabel('Synthetic − real CDF', fontsize=12)
    axes[1].set_xlabel('Distance to nearest neighbouring event (m)', fontsize=12, labelpad=10)
    for index, ax in enumerate(axes):
        ax.text(-0.1, 1.02, 'ab'[index], transform=ax.transAxes, fontweight='bold', fontsize=14)
        ax.set_xlim(0, 50 if primary else 75)
        ax.grid(color='#dddddd', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].axhline(0, color='#666666', lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)
    fig.subplots_adjust(top=0.85, hspace=0.16, bottom=0.12)
    inset.set_xlim(0,10)
    inset.set_ylim(-.025,1.025)
    inset.tick_params(labelsize=8)
    inset.set_facecolor('white')
    for label in inset.get_yticklabels():
        label.set_bbox({'facecolor':'white','edgecolor':'none','pad':1})
    save_figure(fig, directory, 'point_pattern_fidelity' if primary else 'nearest_neighbour_distributions')


def ottawa_lisa_figure(da, da_id, moran, labels, artifacts):
    names = ['Real held-out', 'GeoCTGAN-Full | seed 42', 'GeoCTGAN-UpperQuota | seed 42', 'GeoCTGAN-Priv | seed 42']
    if not set(names).issubset(set(labels['dataset'])):
        raise ValueError('The seed-42 Ottawa LISA results are incomplete.')
    classes = ['NS', 'HH', 'HL', 'LL', 'LH']
    colors = ['#d0d0d0', '#d7191c', '#fdae61', '#2c7bb6', '#abd9e9']
    identifiers = da[da_id].astype(str)
    real = labels.loc[labels['dataset'] == 'Real held-out'].set_index('da_identifier')['lisa_cluster'].reindex(identifiers)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for index, (ax, name) in enumerate(zip(axes.flat, names)):
        local = labels.loc[labels['dataset'] == name].set_index('da_identifier')['lisa_cluster'].reindex(identifiers)
        if local.isna().any():
            raise ValueError('The Ottawa LISA DA identifiers are incomplete.')
        da.assign(_class=local.map({label:i for i,label in enumerate(classes)}).to_numpy()).plot(column='_class',cmap=ListedColormap(colors),vmin=0,vmax=4,linewidth=0,ax=ax)
        statistic = moran.loc[moran['dataset'] == name].iloc[0]
        annotation = f"I = {statistic['moran_i']:.3f}; p = {statistic['permutation_p_value']:.3f}"
        if index:
            annotation += f"\nAgreement = {100*np.mean(local.to_numpy()==real.to_numpy()):.1f}%"
        ax.text(.03,.05,annotation,transform=ax.transAxes,fontsize=11,va='bottom',bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        ax.text(-.04,1.03,'abcd'[index],transform=ax.transAxes,fontsize=15,fontweight='bold')
        ax.set_title(name.replace(' | seed 42',''),fontsize=13,pad=15)
        ax.set_axis_off()
    handles = [Patch(facecolor=color,label=label) for color,label in zip(colors,['Not significant','High-High','High-Low','Low-Low','Low-High'])]
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=10,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1),h_pad=2)
    save_figure(fig,artifacts.figures,'ottawa_da_lisa')

def kde_figure(real, means, xx, yy, cfg, directory):
    from pyproj import Transformer
    from .evaluation import _safe_pearson
    methods = [method for method in ['GeoCTGAN-Full', 'GeoCTGAN-Priv'] if method in means]
    if len(methods) != 2:
        return
    lon, lat = Transformer.from_crs(cfg.metric_crs, cfg.geographic_crs, always_xy=True).transform(xx, yy)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    surfaces = [real, *[means[m] for m in methods]]
    vmax = max((surface.max() for surface in surfaces)) * 1000000.0
    for ax, surface, name in zip(axes[0], surfaces, ['Real held-out', *methods]):
        image = ax.pcolormesh(lon, lat, surface.reshape(xx.shape) * 1000000.0, cmap='inferno', vmin=0, vmax=vmax, shading='auto', rasterized=True)
        ax.set_title(name, fontsize=12, pad=14)
    axes[1, 0].set_visible(False)
    limit = max((np.max(np.abs(real - means[m])) for m in methods)) * 1000000.0
    for ax, method in zip(axes[1, 1:], methods):
        difference = real - means[method]
        difference_image = ax.pcolormesh(lon, lat, difference.reshape(xx.shape) * 1000000.0, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), shading='auto', rasterized=True)
        ax.text(0.02, 0.98, f'r = {_safe_pearson(real, means[method]):.4f}\nMAE = {np.mean(np.abs(difference)):.2e}', transform=ax.transAxes, ha='left', va='top', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
    for row, letter, first in zip(axes, 'ab', [0,1]):
        row[first].text(-0.11, 1.07, letter, transform=row[first].transAxes, fontsize=15, fontweight='bold')
        for ax in row:
            ax.tick_params(labelsize=9, axis='x', rotation=25)
        row[first].set_ylabel('Latitude', fontsize=12, labelpad=12)
        row[1].set_xlabel('←  Longitude  →', fontsize=12, labelpad=15)
    fig.subplots_adjust(hspace=0.55, wspace=0.25, right=0.89)
    cax1 = fig.add_axes([0.92, 0.58, 0.018, 0.27])
    fig.colorbar(image, cax=cax1).set_label('Intensity (×10⁻⁶)', fontsize=10)
    cax2 = fig.add_axes([0.92, 0.13, 0.018, 0.27])
    fig.colorbar(difference_image, cax=cax2).set_label('Real − synthetic (×10⁻⁶)', fontsize=10)
    save_figure(fig, directory, 'kde_surfaces')

def area_figure(da, da_id, counts, moran, lisa, artifacts):
    import mapclassify
    from libpysal.weights import Queen, lag_spatial
    names = ['Real held-out', 'GeoCTGAN-Full | seed 42', 'GeoCTGAN-Priv | seed 42']
    if not all((name in counts for name in names)):
        return
    real = counts[names[0]]
    vectors = [counts[name] * real.sum() / counts[name].sum() for name in names]
    rng_state = np.random.get_state()
    np.random.seed(42)
    try:
        natural = np.asarray(mapclassify.NaturalBreaks(real, k=5).bins)
    finally:
        np.random.set_state(rng_state)
    quantile = np.unique(np.quantile(real, np.linspace(0.2, 1, 5)))
    palettes = [['#fee4c8', '#fdd0a2', '#fdae6b', '#e6550d', '#a63603'], ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']]
    weights = Queen.from_dataframe(da, ids=da[da_id].astype(str).tolist(), silence_warnings=True)
    weights.transform = 'r'
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1, 0.9, 1]})
    agreement_rows, break_rows = ([], [])
    for row, (bins, palette, scheme) in enumerate(zip([natural, quantile], palettes, ['Natural Breaks', 'Quantiles'])):
        real_codes = np.digitize(real, bins[:-1], right=True)
        for index, (name, vector) in enumerate(zip(names, vectors)):
            codes = np.digitize(vector, bins[:-1], right=True)
            da.assign(_class=codes).plot(column='_class', cmap=ListedColormap(palette[:len(bins)]), vmin=0, vmax=max(len(bins) - 1, 1), linewidth=0, ax=axes[row, index])
            axes[row, index].set_axis_off()
            if index:
                left, right = (set(np.flatnonzero(real_codes == len(bins) - 1)), set(np.flatnonzero(codes == len(bins) - 1)))
                agreement_rows.append({'dataset': name, 'classification': scheme, 'class_agreement': float(np.mean(codes == real_codes)), 'highest_class_jaccard': len(left & right) / len(left | right) if left | right else 1.0})
        labels = [f'≤ {bins[0]:g}'] + [f'> {low:g}–{high:g}' for low, high in zip(bins[:-1], bins[1:])]
        axes[row, 1].legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(palette, labels)], loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, fontsize=9)
        break_rows.extend(({'classification': scheme, 'class': i + 1, 'upper_bound': float(bound)} for i, bound in enumerate(bins)))
    classes = ['NS', 'HH', 'HL', 'LL', 'LH']
    colors = ['#d0d0d0', '#d7191c', '#fdae61', '#2c7bb6', '#abd9e9']
    for index, (name, vector) in enumerate(zip(names, vectors)):
        z = (vector - vector.mean()) / vector.std(ddof=0)
        lag = lag_spatial(weights, z)
        ax = axes[2, index]
        color = COLORS.get(name.split(' | ')[0], '#666666')
        ax.scatter(z, lag, s=6, alpha=0.4, color=color, edgecolors='none')
        statistic = moran.loc[moran['dataset'] == name].iloc[0]
        x = np.array([z.min(), z.max()])
        ax.plot(x, float(statistic['moran_i']) * x, color='#222222', lw=1, ls='--')
        ax.axhline(0, color='#888888', lw=0.6)
        ax.axvline(0, color='#888888', lw=0.6)
        ax.text(0.97, 0.96, f"I = {statistic['moran_i']:.3f}\np = {statistic['permutation_p_value']:.3f}", transform=ax.transAxes, ha='right', va='top', fontsize=10)
        if index == 0:
            ax.set_ylabel('Spatial lag (Wz)', fontsize=12, labelpad=10)
        labels = lisa.loc[lisa['dataset'] == name].set_index('da_identifier')['lisa_cluster'].reindex(da[da_id].astype(str))
        da.assign(_class=labels.map({c: i for i, c in enumerate(classes)}).to_numpy()).plot(column='_class', cmap=ListedColormap(colors), vmin=0, vmax=4, linewidth=0, ax=axes[3, index])
        axes[3, index].set_axis_off()
    axes[2, 1].set_xlabel('←  Standardized DA count (z)  →', fontsize=12, labelpad=16)
    for row, letter in enumerate('abcd'):
        axes[row, 0].text(-0.08, 1.04, letter, transform=axes[row, 0].transAxes, fontsize=16, fontweight='bold')
    for ax, name in zip(axes[0], ['Real held-out', 'GeoCTGAN-Full', 'GeoCTGAN-Priv']):
        ax.set_title(name, fontsize=14, pad=16)
    axes[3, 1].legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, ['Not significant', 'High-High', 'High-Low', 'Low-Low', 'Low-High'])], loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=10, frameon=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.16, bottom=0.06, top=0.95)
    write_output_csv(pd.DataFrame(agreement_rows), artifacts.tables / 'da_class_agreement_seed42.csv', index=False)
    write_output_csv(pd.DataFrame(break_rows), artifacts.tables / 'da_class_breaks.csv', index=False)
    save_figure(fig, artifacts.figures, 'area_spatial_structure')


def supplementary_summary_figures(artifacts):
    def read(name):
        return pd.read_csv(artifacts.tables/name)
    def grouped(frame, value):
        selected = frame.loc[frame['panel'] == 'held_out_generalization'].copy() if 'panel' in frame else frame.copy()
        selected['method'] = selected['dataset'].map(lambda name: method_seed(name)[0])
        return selected.groupby(['method','dataset'])[value].mean().reset_index()
    benchmarks = ['GeoCTGAN-Full','GeoCTGAN-Priv','GaussianCopula','CTGAN_RawLatLon','KDE_AttributeBootstrap','RoadSupportBootstrap']
    ablations = ['GeoCTGAN-Full','GeoCTGAN-CapacityOnly','GeoCTGAN-NoCapacity','GeoCTGAN-PermutedEmbedding','GeoCTGAN-UpperQuota']
    quality = read('sdmetrics_quality.csv')
    quality = quality.loc[quality['report'] == 'QualityReport']
    kde = read('kde_surface_fidelity.csv')
    grid = read('multiscale_grid_fidelity.csv')
    learned = read('learned_attribute_fidelity_summary.csv')
    proximity = read('spatial_proximity_risk.csv')
    values = [(grouped(quality,'score'),'score','SDMetrics quality'), (grouped(kde,'kde_pearson'),'kde_pearson','KDE correlation'), (grouped(grid.loc[grid['grid_size_m'] == 100],'count_pearson'),'count_pearson','100 m grid-count correlation'), (grouped(learned,'median'),'median','Median Hellinger distance')]
    for methods,stem in [(benchmarks,'benchmark_fidelity'),(ablations,'ablation_fidelity')]:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        panels=list(values)
        if methods == ablations:
            panels[-1]=(grouped(proximity,'fraction_within_5m'),'fraction_within_5m','Synthetic points within 5 m')
        for i,(ax,(table,value,ylabel)) in enumerate(zip(axes.flat,panels)):
            chosen=[m for m in methods if m in set(table['method'])]
            for x,method in enumerate(chosen):
                sample=table.loc[table['method']==method,value].to_numpy(float)
                mean=float(sample.mean())
                ax.bar(x,mean,width=.65,color=COLORS.get(method,plt.cm.tab10(x)),edgecolor='#333333')
                ax.errorbar(x,mean,yerr=[[max(0,mean-sample.min())],[max(0,sample.max()-mean)]],color='#444444',capsize=3,lw=1)
            ax.set_xticks(range(len(chosen)),[m.replace('GeoCTGAN-','').replace('_','\n') for m in chosen],rotation=25,ha='right',fontsize=9)
            ax.set_ylabel(ylabel,fontsize=11)
            ax.text(-.13,1.03,'abcd'[i],transform=ax.transAxes,fontweight='bold',fontsize=15)
            ax.grid(axis='y',color='#dddddd',lw=.6)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(h_pad=3,w_pad=2)
        save_figure(fig,artifacts.figures,stem)
    for methods,stem in [(benchmarks,'benchmark_grid_fidelity'),(ablations,'ablation_grid_fidelity')]:
        fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True)
        selected=grid.copy()
        selected['method']=selected['dataset'].map(lambda x:method_seed(x)[0])
        for i,(ax,value,ylabel) in enumerate(zip(axes,['count_pearson','hotspot_jaccard'],['Grid-count Pearson r','Hotspot Jaccard overlap'])):
            for j,method in enumerate(methods):
                subset=selected.loc[selected['method']==method]
                if subset.empty:
                    continue
                summary=subset.groupby(['grid_size_m','dataset'])[value].mean().unstack()
                color=COLORS.get(method,plt.cm.tab10(j))
                mean=summary.mean(axis=1)
                ax.plot(mean.index,mean,color=color,marker='o',lw=1.6,label=method)
                ax.fill_between(mean.index,summary.min(axis=1),summary.max(axis=1),color=color,alpha=.14)
            ax.set_ylabel(ylabel,fontsize=12)
            ax.text(-.1,1.04,'ab'[i],transform=ax.transAxes,fontsize=15,fontweight='bold')
            ax.grid(color='#dddddd',lw=.6)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xscale('log')
        axes[1].set_xticks([25,50,100,250,500],[25,50,100,250,500])
        axes[1].set_xlabel('Grid size (m)',fontsize=12,labelpad=10)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9)
        fig.subplots_adjust(top=.84,hspace=.15)
        save_figure(fig,artifacts.figures,stem)
    if (artifacts.tables/'dissemination_area_fidelity.csv').is_file():
        da=read('dissemination_area_fidelity.csv')
        joined=proximity.merge(da,on='dataset',how='inner')
        joined['method']=joined['dataset'].map(lambda x:method_seed(x)[0])
        fig,ax=plt.subplots(figsize=(9,6))
        rows=[]
        for i,(method,group) in enumerate(joined.groupby('method',sort=False)):
            if method not in benchmarks+['GeoCTGAN-UpperQuota']:
                continue
            x=100*group['fraction_within_5m'].to_numpy(float)
            y=group['da_count_pearson'].to_numpy(float)
            mx,my=float(x.mean()),float(y.mean())
            ax.errorbar(mx,my,xerr=[[max(0,mx-x.min())],[max(0,x.max()-mx)]],yerr=[[max(0,my-y.min())],[max(0,y.max()-my)]],fmt='o',color=COLORS.get(method,plt.cm.tab10(i%10)),capsize=3,label=method)
            rows.append({'method':method,'proximity_percent_mean':mx,'proximity_percent_min':float(x.min()),'proximity_percent_max':float(x.max()),'da_pearson_mean':my,'da_pearson_min':float(y.min()),'da_pearson_max':float(y.max())})
        ax.set_xlabel('Synthetic points within 5 m of outer-training points (%)',fontsize=12,labelpad=10)
        ax.set_ylabel('Outer-test DA-count Pearson correlation',fontsize=12,labelpad=10)
        ax.set_xlim(-3,103)
        ax.grid(color='#dddddd',lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=10,frameon=False)
        fig.tight_layout()
        write_output_csv(pd.DataFrame(rows),artifacts.tables/'geographic_proximity_spatial_fidelity.csv',index=False)
        save_figure(fig,artifacts.figures,'geographic_proximity_spatial_fidelity')


def ottawa_summary(artifacts):
    frames = []
    specifications = [
        ('sdmetrics_quality.csv','score','quality',{'report':'QualityReport'}),
        ('learned_attribute_fidelity_summary.csv','median','median_hellinger',{}),
        ('kde_surface_fidelity.csv','kde_pearson','kde_pearson',{}),
        ('multiscale_grid_fidelity.csv','count_pearson','grid_100m_pearson',{'grid_size_m':100}),
        ('dissemination_area_fidelity.csv','da_count_pearson','da_count_pearson',{}),
        ('dissemination_area_fidelity.csv','da_count_wape','da_count_wape',{}),
        ('global_moran.csv','moran_i','moran_i',{}),
        ('lisa_agreement.csv','exact_label_agreement','lisa_agreement',{}),
        ('nearest_neighbour_fidelity.csv','synthetic_mean_nnd_m','mean_nnd_m',{}),
        ('spatial_proximity_risk.csv','fraction_within_5m','fraction_within_5m',{}),
        ('spatial_proximity_risk.csv','distance_median_m','training_distance_median_m',{}),
    ]
    for filename,column,metric,filters in specifications:
        table = pd.read_csv(artifacts.tables/filename)
        if 'panel' in table:
            table = table.loc[table['panel'] == 'held_out_generalization']
        for key,value in filters.items():
            table = table.loc[table[key] == value]
        table = table.loc[~table['dataset'].str.startswith('Real ')]
        grouped = table.groupby('dataset')[column].mean().reset_index(name='value')
        grouped['metric'] = metric
        frames.append(grouped)
    privacy = pd.read_csv(artifacts.tables/'sdmetrics_privacy.csv')
    frames.append(privacy[['dataset','metric','score']].rename(columns={'score':'value'}))
    values = pd.concat(frames,ignore_index=True)
    values['method'] = values['dataset'].map(lambda value:method_seed(value)[0])
    values['seed'] = values['dataset'].map(lambda value:method_seed(value)[1])
    values = values.loc[values['method'].isin(['GeoCTGAN-Full','GeoCTGAN-UpperQuota','GeoCTGAN-Priv'])]
    summary = values.groupby(['method','metric'])['value'].agg(['mean','std','min','max','count']).reset_index()
    write_output_csv(values[['method','seed','metric','value']],artifacts.tables/'ottawa_fidelity_privacy_by_seed.csv',index=False)
    write_output_csv(summary,artifacts.tables/'ottawa_fidelity_privacy_summary.csv',index=False)
