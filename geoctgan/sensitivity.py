from __future__ import annotations
from .utils import write_output_csv
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from .config import Config, scientific_config_payload
from .features import feature_engineering
from .sampling import assign_and_reconstruct, generate_latent_rows
from .temporal import month_boundary_discontinuity
from .training import train_ctgan
from .utils import atomic_json_dump, derived_seed, ensure_dir, load_success_checkpoint, write_success_checkpoint

def _cap_condition_table(input_path: str, output_path: Path, *, maximum_rows_per_condition: int, category_column: str, seed: int, years: Sequence[int] | None=None) -> str:
    frame = pd.read_parquet(input_path)
    if years:
        year_set = {int(value) for value in years}
        frame = frame.loc[frame['OCC_YEAR'].astype(int).isin(year_set)].copy()
        if frame.empty:
            raise RuntimeError(f'Pilot year filter {sorted(year_set)} retained no rows from {input_path}.')
    if maximum_rows_per_condition <= 0:
        raise ValueError('maximum_rows_per_condition must be positive.')
    pieces = []
    for key, group in frame.groupby(['OCC_YEAR', 'OCC_MONTH', category_column], observed=True, sort=True):
        take = min(len(group), int(maximum_rows_per_condition))
        if take == len(group):
            pieces.append(group.copy())
        else:
            condition_seed = derived_seed(int(seed), 'pilot_cap', int(key[0]), int(key[1]), str(key[2]))
            pieces.append(group.sample(n=take, replace=False, random_state=condition_seed))
    reduced = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(output_path, index=False)
    atomic_json_dump({'source': input_path, 'source_rows': int(len(frame)), 'pilot_rows': int(len(reduced)), 'maximum_rows_per_condition': int(maximum_rows_per_condition), 'years': sorted(frame['OCC_YEAR'].astype(int).unique().tolist()), 'conditions': int(reduced.groupby(['OCC_YEAR', 'OCC_MONTH', category_column], observed=True).ngroups)}, output_path.with_suffix('.summary.json'))
    return str(output_path)

def _normalize_distribution(series: pd.Series, support: Sequence[str]) -> np.ndarray:
    counts = series.astype(str).value_counts().reindex(support, fill_value=0).to_numpy(dtype=float)
    return np.zeros(len(support), dtype=float) if counts.sum() == 0 else counts / counts.sum()

def categorical_hellinger(real: pd.Series, synthetic: pd.Series) -> float:
    support = sorted(set(real.dropna().astype(str)).union(set(synthetic.dropna().astype(str))))
    if not support:
        return 0.0
    p, q = (_normalize_distribution(real, support), _normalize_distribution(synthetic, support))
    return float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))

def cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype('string').fillna('<MISSING>'), right.astype('string').fillna('<MISSING>'))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi_square = chi2_contingency(table, correction=False)[0]
    total = float(table.to_numpy().sum())
    if total <= 1:
        return 0.0
    phi_square = chi_square / total
    rows, columns = table.shape
    corrected = max(0.0, phi_square - (columns - 1) * (rows - 1) / (total - 1))
    row_correction = rows - (rows - 1) ** 2 / (total - 1)
    column_correction = columns - (columns - 1) ** 2 / (total - 1)
    denominator = max(min(column_correction - 1, row_correction - 1), 1e-12)
    return float(np.sqrt(corrected / denominator))

def _assert_pilot_condition_support(training_path: str, validation_path: str, *, category_column: str) -> None:
    condition_columns = ['OCC_YEAR', 'OCC_MONTH', category_column]
    training = pd.read_parquet(training_path, columns=condition_columns)
    validation = pd.read_parquet(validation_path, columns=condition_columns)
    training_support = {(int(year), int(month), str(category)) for year, month, category in training[condition_columns].itertuples(index=False, name=None)}
    validation_support = {(int(year), int(month), str(category)) for year, month, category in validation[condition_columns].itertuples(index=False, name=None)}
    unsupported = sorted(validation_support.difference(training_support))
    if unsupported:
        raise RuntimeError(f'The capped inner-validation pilot contains conditions with no inner-training support. Selection cannot score unsupported cells without leakage or relabelling. Examples: {unsupported[:10]}')

def bivariate_association_error(real: pd.DataFrame, synthetic: pd.DataFrame, columns: Sequence[str]) -> float:
    errors = []
    available = [column for column in columns if column in real.columns and column in synthetic.columns]
    for position, left in enumerate(available):
        for right in available[position + 1:]:
            real_value = cramers_v(real[left], real[right])
            synthetic_value = cramers_v(synthetic[left], synthetic[right])
            if np.isfinite(real_value) and np.isfinite(synthetic_value):
                errors.append(abs(real_value - synthetic_value))
    return float(np.mean(errors)) if errors else np.nan

def conditional_segment_js(real: pd.DataFrame, synthetic: pd.DataFrame, condition_column: str, segment_column: str, minimum_group: int=20) -> float:
    values = sorted(set(real[condition_column].dropna().astype(str)).intersection(set(synthetic[condition_column].dropna().astype(str))))
    distances = []
    for value in values:
        real_group = real.loc[real[condition_column].astype(str).eq(value)]
        synthetic_group = synthetic.loc[synthetic[condition_column].astype(str).eq(value)]
        if len(real_group) < minimum_group or len(synthetic_group) < minimum_group:
            continue
        support = sorted(set(real_group[segment_column].astype(str)).union(set(synthetic_group[segment_column].astype(str))))
        p = _normalize_distribution(real_group[segment_column], support)
        q = _normalize_distribution(synthetic_group[segment_column], support)
        distances.append(float(jensenshannon(p, q, base=2.0) ** 2))
    return float(np.mean(distances)) if distances else np.nan

def select_snap_tolerance(snap_results_path: str, relative_retention_tolerance: float=0.001, minimum_assignment_agreement: float=0.99) -> dict:
    frame = pd.read_csv(snap_results_path)
    if 'split' in frame.columns:
        validation = frame.loc[frame['split'].eq('inner_validation')].copy()
        if validation.empty:
            validation = frame.loc[frame['split'].eq('inner_train')].copy()
    else:
        validation = frame.copy()
    maximum_retention = float(validation['retention'].max())
    eligible = validation.loc[validation['retention'] >= maximum_retention - relative_retention_tolerance].copy()
    if eligible.empty:
        eligible = validation.nlargest(1, 'retention')
    if 'assignment_agreement_with_reference' in eligible.columns:
        stable = eligible.loc[pd.to_numeric(eligible['assignment_agreement_with_reference'], errors='coerce').ge(float(minimum_assignment_agreement))].copy()
        if not stable.empty:
            eligible = stable
    order = [column for column in ['snap_tolerance_m', 'distance_p95_m', 'runtime_seconds'] if column in eligible.columns]
    selected = eligible.sort_values(order).iloc[0]
    return selected.to_dict()

def _one_se_choice(summary: pd.DataFrame, *, mean_column: str, se_column: str, complexity_columns: Sequence[str]) -> pd.Series:
    ordered = summary.sort_values(mean_column)
    best = ordered.iloc[0]
    standard_error = pd.to_numeric(pd.Series([best.get(se_column, 0.0)]), errors='coerce').iloc[0]
    if not np.isfinite(standard_error):
        standard_error = 0.0
    threshold = float(best[mean_column]) + float(standard_error)
    eligible = summary.loc[summary[mean_column] <= threshold].copy()
    if eligible.empty:
        raise RuntimeError(f'The one-standard-error rule produced no eligible rows for {mean_column}.')
    return eligible.sort_values([*complexity_columns, mean_column]).iloc[0]

def _score_candidate(real_validation: pd.DataFrame, synthetic: pd.DataFrame, cann_diagnostics: pd.DataFrame, runtime_seconds: float, cfg: Config) -> dict:
    categorical_columns = [cfg.COL_OFFENCE, 'PREMISES_T', 'LOCATION_T']
    hellinger_values = [categorical_hellinger(real_validation[column], synthetic[column]) for column in categorical_columns if column in real_validation.columns and column in synthetic.columns]
    univariate = float(np.mean(hellinger_values)) if hellinger_values else np.nan
    bivariate = bivariate_association_error(real_validation, synthetic, categorical_columns)
    conditional_values = []
    for column in categorical_columns:
        if column in real_validation.columns and column in synthetic.columns:
            value = conditional_segment_js(real_validation, synthetic, column, cfg.COL_SEG)
            if np.isfinite(value):
                conditional_values.append(value)
    conditional = float(np.mean(conditional_values)) if conditional_values else np.nan
    boundary = month_boundary_discontinuity(real_validation, synthetic, date_column=cfg.COL_DATE, category_column=cfg.COL_CAT, window_days=cfg.TEMPORAL_BOUNDARY_WINDOW_DAYS)
    boundary_error = float(boundary['absolute_boundary_error'].mean()) if 'absolute_boundary_error' in boundary.columns and len(boundary) else np.nan
    repair = float(pd.to_numeric(cann_diagnostics.get('selected_rank_median'), errors='coerce').mean())
    return {'univariate_hellinger': univariate, 'bivariate_error': bivariate, 'conditional_segment_js': conditional, 'boundary_error': boundary_error, 'median_selected_rank': repair, 'runtime_seconds': runtime_seconds}

def _add_scaled_score(frame: pd.DataFrame, columns_and_weights: Dict[str, float]) -> pd.DataFrame:
    result = frame.copy()
    score = pd.Series(0.0, index=result.index)
    for column, weight in columns_and_weights.items():
        values = pd.to_numeric(result[column], errors='coerce')
        minimum, maximum = (values.min(), values.max())
        scaled = (values - minimum) / max(maximum - minimum, 1e-09)
        result[f'{column}_scaled'] = scaled
        score += float(weight) * scaled.fillna(1)
    result['selection_score'] = score
    return result

def _summarise_runs(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    aggregate = frame.groupby(list(group_columns), as_index=False).agg(seed_count=('seed', 'nunique'), selection_score_mean=('selection_score', 'mean'), selection_score_sd=('selection_score', 'std'), runtime_seconds_mean=('runtime_seconds', 'mean'), univariate_hellinger_mean=('univariate_hellinger', 'mean'), bivariate_error_mean=('bivariate_error', 'mean'), conditional_segment_js_mean=('conditional_segment_js', 'mean'), boundary_error_mean=('boundary_error', 'mean'), median_selected_rank_mean=('median_selected_rank', 'mean'))
    aggregate['selection_score_se'] = aggregate['selection_score_sd'] / np.sqrt(aggregate['seed_count'].clip(lower=1))
    return aggregate

def temporal_confirmation_schemes(cfg: Config) -> list[str]:
    schemes = list(dict.fromkeys(map(str, cfg.COHORT_CANDIDATES)))
    if set(schemes) != {'global', 'year', 'year_quarter', 'year_month'} or int(cfg.TEMPORAL_FINALIST_COUNT) != 4:
        raise ValueError('The reported temporal comparison requires all four pooling schemes across the confirmation seeds.')
    return schemes


def run_temporal_scheme_pilot(cfg: Config, inner_train_path: str, inner_validation_path: str, embedding_path: str, segments_path: str, out_dir: str) -> str:
    confirmed_schemes = temporal_confirmation_schemes(cfg)
    root = Path(ensure_dir(out_dir))
    pilot_train_path = _cap_condition_table(inner_train_path, root / 'pilot_inner_train.parquet', maximum_rows_per_condition=cfg.TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION, category_column=cfg.COL_CAT, seed=cfg.SEED, years=cfg.TEMPORAL_PILOT_YEARS)
    pilot_validation_path = _cap_condition_table(inner_validation_path, root / 'pilot_inner_validation.parquet', maximum_rows_per_condition=cfg.TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION, category_column=cfg.COL_CAT, seed=cfg.SEED + 1, years=cfg.TEMPORAL_PILOT_YEARS)
    _assert_pilot_condition_support(pilot_train_path, pilot_validation_path, category_column=cfg.COL_CAT)
    real_validation = pd.read_parquet(pilot_validation_path)
    score_weights = {'univariate_hellinger': 0.25, 'bivariate_error': 0.2, 'conditional_segment_js': 0.3, 'boundary_error': 0.15, 'runtime_seconds': 0.1}
    checkpoint_dir = root / 'candidate_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_candidate(scheme: str, seed: int, stage: str) -> dict:
        checkpoint_path = checkpoint_dir / f'{scheme}_seed_{seed}_{stage}.json'
        checkpoint = load_success_checkpoint(checkpoint_path, artifact_root=root)
        if checkpoint is not None:
            return dict(checkpoint['result'])
        candidate_cfg = copy.deepcopy(cfg)
        candidate_cfg.COHORT_KIND = scheme
        candidate_cfg.EPOCHS = cfg.TEMPORAL_PILOT_EPOCHS
        candidate_cfg.REQUIRE_SELECTED_CONFIG = False
        candidate_cfg.SEED = int(seed)
        candidate_cfg.RESUME_TRAINING = True
        candidate_cfg.RESUME_GENERATION = True
        candidate_cfg.RESUME_CANN = True
        candidate_dir = Path(ensure_dir(root / scheme / f'seed_{seed}'))
        start = time.perf_counter()
        raw_features, model_features, crosswalk, pools = feature_engineering(candidate_cfg, pilot_train_path, embedding_path, str(candidate_dir))
        models_dir, _ = train_ctgan(candidate_cfg, model_features, pools, str(candidate_dir), epochs=cfg.TEMPORAL_PILOT_EPOCHS, run_tag='temporal_pilot')
        latent_path, _ = generate_latent_rows(candidate_cfg, raw_features, crosswalk, models_dir, str(candidate_dir), target_table_path=pilot_validation_path)
        synthetic_path, cann_path, _ = assign_and_reconstruct(candidate_cfg, latent_path, raw_features, segments_path, embedding_path, str(candidate_dir), variant='Full', capacity_mode='exact_quota')
        runtime = time.perf_counter() - start
        synthetic = pd.read_parquet(synthetic_path)
        cann = pd.read_csv(cann_path)
        metrics = _score_candidate(real_validation, synthetic, cann, runtime, candidate_cfg)
        result = {'scheme': scheme, 'seed': int(seed), 'stage': stage, 'model_count': len(pools), **metrics}
        write_success_checkpoint(checkpoint_path, result=result, artifacts=[synthetic_path, cann_path, latent_path], artifact_root=root)
        return result
    screen_seed = int(cfg.SENSITIVITY_SCREEN_SEED)
    screen_rows = [run_candidate(str(scheme), screen_seed, 'screen') for scheme in dict.fromkeys(cfg.COHORT_CANDIDATES)]
    screen_runs = _add_scaled_score(pd.DataFrame(screen_rows), score_weights)
    complexity = {'global': 0, 'year': 1, 'year_quarter': 2, 'year_month': 3}
    screen_runs['complexity'] = screen_runs['scheme'].map(complexity)
    screen_runs = screen_runs.sort_values(['selection_score', 'complexity', 'runtime_seconds']).reset_index(drop=True)
    write_output_csv(screen_runs, root / 'temporal_scheme_screening_runs.csv', index=False)
    finalist_schemes = confirmed_schemes
    write_output_csv(screen_runs, root / 'temporal_scheme_finalists.csv', index=False)
    confirmation_rows = [row for row in screen_rows if str(row['scheme']) in set(finalist_schemes)]
    for scheme in finalist_schemes:
        for seed in cfg.SENSITIVITY_SEEDS:
            if int(seed) == screen_seed:
                continue
            confirmation_rows.append(run_candidate(str(scheme), int(seed), 'confirmation'))
    runs = _add_scaled_score(pd.DataFrame(confirmation_rows), score_weights)
    write_output_csv(runs, root / 'temporal_scheme_sensitivity_runs.csv', index=False)
    summary = _summarise_runs(runs, ['scheme', 'model_count'])
    expected_seed_count = len(set(map(int, cfg.SENSITIVITY_SEEDS)))
    if not summary['seed_count'].eq(expected_seed_count).all():
        raise RuntimeError(f"Every temporal finalist must complete all confirmation seeds. Expected {expected_seed_count}; observed {summary[['scheme', 'seed_count']].to_dict('records')}.")
    write_output_csv(summary, root / 'temporal_scheme_sensitivity.csv', index=False)
    return str(root / 'temporal_scheme_sensitivity.csv')

def select_temporal_scheme(results_path: str, cfg: Config) -> dict:
    frame = pd.read_csv(results_path)
    complexity = {'global': 0, 'year': 1, 'year_quarter': 2, 'year_month': 3}
    frame['complexity'] = frame['scheme'].map(complexity)
    selected = _one_se_choice(frame, mean_column='selection_score_mean', se_column='selection_score_se', complexity_columns=['complexity', 'runtime_seconds_mean'])
    return selected.to_dict()


def validate_temporal_confirmation(results_path: str, cfg: Config) -> None:
    expected_schemes = set(temporal_confirmation_schemes(cfg))
    expected_seeds = set(map(int, cfg.SENSITIVITY_SEEDS))
    if len(expected_seeds) != 3:
        raise RuntimeError('Temporal confirmation requires three distinct generator seeds.')
    summary = pd.read_csv(results_path)
    runs = pd.read_csv(Path(results_path).with_name('temporal_scheme_sensitivity_runs.csv'))
    if len(summary) != 4 or set(summary['scheme']) != expected_schemes or set(runs['scheme']) != expected_schemes:
        raise RuntimeError('Temporal confirmation requires results for all four pooling schemes.')
    if not summary['seed_count'].eq(3).all():
        raise RuntimeError('Every temporal-pooling summary must contain three seeds.')
    for scheme in expected_schemes:
        group = runs.loc[runs['scheme'].eq(scheme)]
        if len(group) != 3 or set(group['seed'].astype(int)) != expected_seeds:
            raise RuntimeError('Every temporal-pooling scheme must contain the three configured confirmation seeds exactly once.')

def run_embedding_generation_pilot(cfg: Config, graph_path: str, embedding_screen_results_path: str, inner_train_path: str, inner_validation_path: str, segments_path: str, out_dir: str) -> str:
    from .graph import train_deepwalk
    root = Path(ensure_dir(out_dir))
    pilot_train_path = _cap_condition_table(inner_train_path, root / 'pilot_inner_train.parquet', maximum_rows_per_condition=cfg.TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION, category_column=cfg.COL_CAT, seed=cfg.SEED, years=cfg.TEMPORAL_PILOT_YEARS)
    pilot_validation_path = _cap_condition_table(inner_validation_path, root / 'pilot_inner_validation.parquet', maximum_rows_per_condition=cfg.TEMPORAL_PILOT_MAX_ROWS_PER_CONDITION, category_column=cfg.COL_CAT, seed=cfg.SEED + 1, years=cfg.TEMPORAL_PILOT_YEARS)
    _assert_pilot_condition_support(pilot_train_path, pilot_validation_path, category_column=cfg.COL_CAT)
    screen = pd.read_csv(embedding_screen_results_path).sort_values('screen_score_mean')
    finalists = screen.head(int(cfg.EMBEDDING_FINALIST_COUNT)).copy()
    real_validation = pd.read_parquet(pilot_validation_path)
    checkpoint_dir = root / 'candidate_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for finalist in finalists.itertuples(index=False):
        for seed in cfg.SENSITIVITY_SEEDS:
            candidate_cfg = copy.deepcopy(cfg)
            candidate_cfg.EMB_DIM = int(finalist.dimension)
            candidate_cfg.N_WALKS = int(finalist.walks_per_node)
            candidate_cfg.WALK_LEN = int(finalist.walk_length)
            candidate_cfg.COHORT_KIND = cfg.EMBEDDING_PILOT_SCHEME
            candidate_cfg.EPOCHS = cfg.TEMPORAL_PILOT_EPOCHS
            candidate_cfg.REQUIRE_SELECTED_CONFIG = False
            candidate_cfg.SEED = int(seed)
            candidate_cfg.RESUME_TRAINING = True
            candidate_cfg.RESUME_GENERATION = True
            candidate_cfg.RESUME_CANN = True
            tag = f'd{candidate_cfg.EMB_DIM}_w{candidate_cfg.N_WALKS}_l{candidate_cfg.WALK_LEN}'
            candidate_dir = Path(ensure_dir(root / tag / f'seed_{seed}'))
            candidate_checkpoint_path = checkpoint_dir / f'{tag}_seed_{seed}.json'
            candidate_checkpoint = load_success_checkpoint(candidate_checkpoint_path, artifact_root=root)
            if candidate_checkpoint is not None:
                rows.append(dict(candidate_checkpoint['result']))
                continue
            start = time.perf_counter()
            embedding_checkpoint_path = candidate_dir / 'embedding_complete.json'
            embedding_checkpoint = load_success_checkpoint(embedding_checkpoint_path, artifact_root=candidate_dir, required_artifacts=[candidate_dir / 'segment_embeddings_full_network_pilot.parquet', candidate_dir / 'deepwalk_full_network_pilot.model', candidate_dir / 'deepwalk_full_network_pilot_runtime.json'])
            if embedding_checkpoint is not None:
                embedding_path = str((candidate_dir / 'segment_embeddings_full_network_pilot.parquet').resolve())
                embedding_runtime = dict(embedding_checkpoint['result']['embedding_runtime'])
            else:
                embedding_path, embedding_runtime = train_deepwalk(candidate_cfg, graph_path, str(candidate_dir), dimension=candidate_cfg.EMB_DIM, n_walks=candidate_cfg.N_WALKS, walk_length=candidate_cfg.WALK_LEN, seed=seed, workers=1, tag='full_network_pilot')
                write_success_checkpoint(embedding_checkpoint_path, result={'embedding_path': embedding_path, 'embedding_runtime': embedding_runtime}, artifacts=[embedding_path, candidate_dir / 'deepwalk_full_network_pilot.model', candidate_dir / 'deepwalk_full_network_pilot_runtime.json'], artifact_root=candidate_dir)
            raw_features, model_features, crosswalk, pools = feature_engineering(candidate_cfg, pilot_train_path, embedding_path, str(candidate_dir))
            models_dir, _ = train_ctgan(candidate_cfg, model_features, pools, str(candidate_dir), epochs=cfg.TEMPORAL_PILOT_EPOCHS, run_tag='embedding_pilot')
            latent_path, _ = generate_latent_rows(candidate_cfg, raw_features, crosswalk, models_dir, str(candidate_dir), target_table_path=pilot_validation_path)
            synthetic_path, cann_path, _ = assign_and_reconstruct(candidate_cfg, latent_path, raw_features, segments_path, embedding_path, str(candidate_dir), variant='Full', capacity_mode='exact_quota')
            runtime = time.perf_counter() - start
            synthetic = pd.read_parquet(synthetic_path)
            cann = pd.read_csv(cann_path)
            metrics = _score_candidate(real_validation, synthetic, cann, runtime, candidate_cfg)
            result = {'tag': tag, 'seed': int(seed), 'dimension': candidate_cfg.EMB_DIM, 'walks_per_node': candidate_cfg.N_WALKS, 'walk_length': candidate_cfg.WALK_LEN, 'intrinsic_screen_score': float(finalist.screen_score_mean), 'embedding_runtime_seconds': float(embedding_runtime['runtime_seconds']), **metrics}
            write_success_checkpoint(candidate_checkpoint_path, result=result, artifacts=[synthetic_path, cann_path, latent_path, embedding_path], artifact_root=root)
            rows.append(result)
    runs = _add_scaled_score(pd.DataFrame(rows), {'univariate_hellinger': 0.15, 'bivariate_error': 0.15, 'conditional_segment_js': 0.35, 'median_selected_rank': 0.2, 'runtime_seconds': 0.1})
    runs['selection_score'] += 0.05 * runs['intrinsic_screen_score'].fillna(runs['intrinsic_screen_score'].max())
    write_output_csv(runs, root / 'embedding_generation_pilot_runs.csv', index=False)
    summary = _summarise_runs(runs, ['tag', 'dimension', 'walks_per_node', 'walk_length'])
    write_output_csv(summary, root / 'embedding_generation_pilot.csv', index=False)
    return str(root / 'embedding_generation_pilot.csv')

def select_embedding_from_generation_pilot(results_path: str, cfg: Config) -> dict:
    frame = pd.read_csv(results_path)
    best = frame.sort_values('selection_score_mean').iloc[0]
    best_se = float(best['selection_score_se'])
    if not np.isfinite(best_se):
        raise RuntimeError('Embedding selection requires a finite standard error across the confirmation seeds.')
    cutoff = float(best['selection_score_mean']) + best_se
    retained = frame.loc[frame['dimension'].eq(16) & frame['walks_per_node'].eq(15) & frame['walk_length'].eq(10)]
    if len(retained) != 1 or not float(retained.iloc[0]['selection_score_mean']) <= cutoff:
        raise RuntimeError('The reported 16-dimension, 15-walk, length-10 configuration is absent or outside the one-standard-error cutoff. A configuration decision is required before generation.')
    selected = retained.iloc[0]
    return selected.to_dict()

def write_selected_config(base_config: Config, snap_selection: dict, embedding_selection: dict, temporal_selection: dict, output_path: str, *, outer_split_manifest_path: str) -> str:
    selected = asdict(base_config)
    selected.update({'SNAP_MAX_M': float(snap_selection['snap_tolerance_m']), 'EMB_DIM': int(embedding_selection['dimension']), 'N_WALKS': int(embedding_selection['walks_per_node']), 'WALK_LEN': int(embedding_selection['walk_length']), 'COHORT_KIND': str(temporal_selection['scheme']), 'OUTER_SPLIT_MANIFEST_PATH': str(outer_split_manifest_path), 'REQUIRE_SELECTED_CONFIG': False, 'SELECTED_CONFIG_PATH': None})
    atomic_json_dump(selected, output_path)
    atomic_json_dump({'screening_seed': int(base_config.SENSITIVITY_SCREEN_SEED), 'finalist_confirmation_seeds': list(dict.fromkeys(map(int, base_config.SENSITIVITY_SEEDS))), 'embedding_finalist_count': int(base_config.EMBEDDING_FINALIST_COUNT), 'temporal_finalist_count': int(base_config.TEMPORAL_FINALIST_COUNT), 'inner_validation_size': base_config.INNER_VALIDATION_SIZE, 'outer_split_manifest': outer_split_manifest_path, 'snap': snap_selection, 'embedding': embedding_selection, 'temporal': temporal_selection}, Path(output_path).with_name('selection_evidence.json'))
    return output_path
