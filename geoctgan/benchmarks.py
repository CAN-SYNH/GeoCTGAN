from __future__ import annotations
from .utils import write_output_csv
import calendar
import gc
import json
import logging
import sys
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
try:
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
except ImportError:
    Condition = None
    CTGANSynthesizer = GaussianCopulaSynthesizer = TVAESynthesizer = None
from .config import Config
from .artifacts import ArtifactDirectories
from .training import _adaptive_batch_size, _detect_metadata, _save_metadata, _update_column
from .temporal import condition_values, pool_series, reconstruct_date
from .utils import atomic_json_dump, derived_seed, sdv_gpu_constructor_kwargs, set_seeds
logger = logging.getLogger(__name__)
SDV_METHODS = {'GaussianCopula': GaussianCopulaSynthesizer, 'CTGAN_RawLatLon': CTGANSynthesizer, 'TVAE_RawLatLon': TVAESynthesizer}
_SDV_ZERO_RESULT_MESSAGE = 'unable to sample any rows for the given conditions'
_SDV_ZERO_RESULT_ATTRIBUTE = 'sdv_zero_result_exception'
TVAE_TERMINAL_UNAVAILABLE_POLICY = 'tvae_bounded_strict_conditional_sampling_exhaustion'
_TVAE_METHOD = 'TVAE_RawLatLon'
_TERMINAL_BENCHMARK_STATUSES = frozenset({'success', 'unavailable'})

class BoundedConditionalSamplingError(RuntimeError):
    pass

def is_terminal_tvae_unavailability(method: object, failure: object) -> bool:
    if str(method) != _TVAE_METHOD:
        return False
    if isinstance(failure, BoundedConditionalSamplingError):
        return True
    text = str(failure).casefold()
    return all((marker in text for marker in ('conditional benchmark sampling retained ', 'exact-condition, calendar-valid rows after ', ' bounded rounds for ', 'no padding, relabelling, bootstrap substitution, or model substitution')))

def _prepare_raw_coordinate_table(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    columns = [cfg.COL_CAT, cfg.COL_OFFENCE, 'PREMISES_T', 'LOCATION_T', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW', 'OCC_HOUR', cfg.COL_LON, cfg.COL_LAT]
    return frame[[column for column in columns if column in frame.columns]].copy()

def _prepare_cohort_model_table(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    table = _prepare_raw_coordinate_table(frame, cfg)
    table['OCC_YEAR_CAT'] = table['OCC_YEAR'].astype(int).astype(str)
    table['OCC_MONTH_CAT'] = table['OCC_MONTH'].astype(int).astype(str).str.zfill(2)
    table['__POOL__'] = pool_series(table, cfg.COHORT_KIND)
    table = table.drop(columns=['OCC_DOW'], errors='ignore')
    if cfg.COHORT_KIND == 'year_month':
        table = table.drop(columns=['OCC_YEAR', 'OCC_MONTH', 'OCC_YEAR_CAT', 'OCC_MONTH_CAT'], errors='ignore')
    elif cfg.COHORT_KIND in {'year_quarter', 'year'}:
        table = table.drop(columns=['OCC_YEAR', 'OCC_MONTH', 'OCC_YEAR_CAT'], errors='ignore')
    elif cfg.COHORT_KIND == 'global':
        table = table.drop(columns=['OCC_YEAR', 'OCC_MONTH'], errors='ignore')
    else:
        raise ValueError(f'Unsupported benchmark temporal pooling: {cfg.COHORT_KIND}')
    return table

def _sample_condition_attempt(synthesizer, values: dict, rows: int) -> pd.DataFrame:
    if Condition is None:
        raise ImportError('SDV conditional sampling is unavailable.')
    requested = int(rows)
    if requested <= 0:
        raise ValueError('Conditional benchmark sampling requires rows > 0.')
    condition = Condition(column_values=values, num_rows=requested)
    zero_result_exception = False
    try:
        sampled = synthesizer.sample_from_conditions(conditions=[condition], max_tries_per_batch=10000, batch_size=max(500, min(5000, requested)), output_file_path=None)
    except ValueError as exc:
        if _SDV_ZERO_RESULT_MESSAGE not in str(exc).casefold():
            raise
        zero_result_exception = True
        sampled = pd.DataFrame()
    sampled = sampled.reset_index(drop=True)
    sampled.attrs[_SDV_ZERO_RESULT_ATTRIBUTE] = zero_result_exception
    if len(sampled) > requested:
        raise RuntimeError(f'Conditional benchmark sampling returned {len(sampled)}/{requested} rows.')
    if sampled.empty:
        return sampled
    for column, expected in values.items():
        if column not in sampled.columns or not sampled[column].astype(str).str.strip().eq(str(expected)).all():
            raise RuntimeError(f'Conditional benchmark sampling returned an incorrect value for {column}={expected}.')
    return sampled

def _sample_condition_exact(synthesizer, values: dict, rows: int) -> pd.DataFrame:
    sampled = _sample_condition_attempt(synthesizer, values, rows)
    if len(sampled) != int(rows):
        raise RuntimeError(f'Conditional benchmark sampling returned {len(sampled)}/{rows} rows.')
    return sampled

def _sample_condition_with_valid_calendar(synthesizer, values: dict, rows: int, *, year: int, month: int, maximum_rounds: int) -> tuple[pd.DataFrame, dict]:
    accepted_parts: list[pd.DataFrame] = []
    accepted_rows = 0
    sampled_rows = 0
    requested_rows_across_attempts = 0
    rejected_calendar_rows = 0
    partial_batch_attempts = 0
    zero_progress_attempts = 0
    sdv_zero_result_exceptions = 0
    maximum_day = calendar.monthrange(int(year), int(month))[1]
    rounds_used = 0
    for round_index in range(1, int(maximum_rounds) + 1):
        remaining = int(rows) - accepted_rows
        if remaining <= 0:
            break
        requested_rows_across_attempts += remaining
        sampled = _sample_condition_attempt(synthesizer, values, remaining)
        if bool(sampled.attrs.get(_SDV_ZERO_RESULT_ATTRIBUTE, False)):
            sdv_zero_result_exceptions += 1
        sampled_rows += len(sampled)
        rounds_used = round_index
        if len(sampled) < remaining:
            partial_batch_attempts += 1
        if sampled.empty:
            zero_progress_attempts += 1
            continue
        if 'OCC_DAY' not in sampled.columns:
            accepted_parts.append(sampled)
            accepted_rows += len(sampled)
            continue
        rounded_day = pd.to_numeric(sampled['OCC_DAY'], errors='coerce').round()
        valid = rounded_day.between(1, maximum_day, inclusive='both').fillna(False)
        rejected_calendar_rows += int((~valid).sum())
        accepted = sampled.loc[valid].copy()
        accepted['OCC_DAY'] = rounded_day.loc[valid].astype('Int64')
        if not accepted.empty:
            accepted_parts.append(accepted)
            accepted_rows += len(accepted)
    if accepted_rows != int(rows):
        raise BoundedConditionalSamplingError(f'Conditional benchmark sampling retained {accepted_rows}/{rows} exact-condition, calendar-valid rows after {maximum_rounds} bounded rounds for {year}-{month:02d}. Partial valid batches were retained and invalid rows rejected; no padding, relabelling, bootstrap substitution, or model substitution was used.')
    result = pd.concat(accepted_parts, ignore_index=True).iloc[:int(rows)].copy()
    telemetry = {'year': int(year), 'month': int(month), 'target_rows': int(rows), 'sampled_rows': int(sampled_rows), 'requested_rows_across_attempts': int(requested_rows_across_attempts), 'partial_batch_attempts': int(partial_batch_attempts), 'zero_progress_attempts': int(zero_progress_attempts), 'sdv_zero_result_exceptions': int(sdv_zero_result_exceptions), 'conditional_shortfall_rows': int(max(0, int(rows) - len(result))), 'accepted_rows': int(len(result)), 'rejected_invalid_calendar_rows': int(rejected_calendar_rows), 'invalid_calendar_rate': float(rejected_calendar_rows / sampled_rows) if sampled_rows else 0.0, 'rounds_used': int(rounds_used)}
    return (result, telemetry)

def _cohort_matched_sdv_generation(method: str, frame: pd.DataFrame, cfg: Config, seed: int, method_dir: Path) -> tuple[pd.DataFrame, str]:
    model_table = _prepare_cohort_model_table(frame, cfg)
    target_table = _prepare_raw_coordinate_table(frame, cfg)
    model_dir = method_dir / 'models'
    pool_output_dir = method_dir / 'pool_outputs'
    model_dir.mkdir(parents=True, exist_ok=True)
    pool_output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer_class = SDV_METHODS.get(method)
    if synthesizer_class is None:
        raise ImportError(f'SDV is required for the {method} benchmark.')
    if method in {'CTGAN_RawLatLon', 'TVAE_RawLatLon'}:
        _, gpu_parameter = sdv_gpu_constructor_kwargs(synthesizer_class, cfg.ENABLE_GPU)
    else:
        gpu_parameter = 'not_applicable'
    generated_parts = []
    pool_manifest = []
    calendar_telemetry = []
    targets = target_table.groupby(['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT], observed=True, sort=True).size().rename('target').reset_index()
    targets['__POOL__'] = pool_series(targets, cfg.COHORT_KIND).astype(str)
    training_pools = {str(pool): group.drop(columns=['__POOL__']).reset_index(drop=True) for pool, group in model_table.groupby('__POOL__', observed=True, sort=True)}
    target_pools = set(targets['__POOL__'].astype(str))
    missing_pools = sorted(target_pools.difference(training_pools))
    if missing_pools:
        raise RuntimeError(f'Benchmark target conditions lack fitted temporal-pool support: {missing_pools}')
    for pool in sorted(target_pools):
        training = training_pools[pool]
        pool_targets = targets.loc[targets['__POOL__'].eq(pool)].copy()
        pool_seed = derived_seed(seed, method, pool, 'fit')
        safe_pool = ''.join((character if character.isalnum() or character in {'-', '_'} else '_' for character in pool))
        model_path = model_dir / f'{method}_{safe_pool}.pkl'
        metadata_path = model_dir / f'{method}_{safe_pool}_metadata.json'
        pool_output_path = pool_output_dir / f'synthetic_{safe_pool}.parquet'
        pool_telemetry_path = pool_output_dir / f'calendar_{safe_pool}.csv'
        checkpoint_path = model_dir / f'complete_{safe_pool}.json'
        if cfg.RESUME_TRAINING and checkpoint_path.is_file() and model_path.is_file() and pool_output_path.is_file() and pool_telemetry_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
            pool_generated = pd.read_parquet(pool_output_path)
            expected_rows = int(pool_targets['target'].sum())
            if len(pool_generated) != expected_rows:
                raise RuntimeError(f'Benchmark checkpoint {method}/{pool} has {len(pool_generated)}/{expected_rows} rows.')
            generated_parts.append(pool_generated)
            calendar_telemetry.extend(pd.read_csv(pool_telemetry_path).to_dict(orient='records'))
            manifest_row = dict(checkpoint['manifest_row'])
            manifest_row['resumed'] = True
            pool_manifest.append(manifest_row)
            continue
        synthesizer, metadata = _fit_sdv_method(method, training, cfg, pool_seed)
        _save_metadata(metadata, str(metadata_path))
        synthesizer.save(str(model_path))
        pool_parts = []
        pool_telemetry = []
        for row in pool_targets.itertuples(index=False):
            year = int(row.OCC_YEAR)
            month = int(row.OCC_MONTH)
            category = str(getattr(row, cfg.COL_CAT))
            target = int(row.target)
            values = condition_values(year, month, category, cfg.COHORT_KIND, cfg.COL_CAT)
            set_seeds(derived_seed(seed, method, year, month, category, 'sample'))
            sampled, telemetry = _sample_condition_with_valid_calendar(synthesizer, values, target, year=year, month=month, maximum_rounds=cfg.MAX_WHOLE_ROW_RESAMPLE_ROUNDS)
            telemetry.update({'pool': pool, 'category': category})
            pool_telemetry.append(telemetry)
            sampled['OCC_YEAR'] = year
            sampled['OCC_MONTH'] = month
            if 'OCC_DAY' in sampled.columns:
                sampled['OCC_DATE'] = reconstruct_date(year, month, sampled['OCC_DAY'])
                sampled['OCC_DOW'] = sampled['OCC_DATE'].dt.dayofweek.astype('Int64')
                sampled = sampled.drop(columns=['OCC_DATE'], errors='ignore')
            pool_parts.append(sampled)
        pool_generated = pd.concat(pool_parts, ignore_index=True)
        missing = sorted(set(target_table.columns).difference(pool_generated.columns))
        if missing:
            raise RuntimeError(f'Cohort-matched {method}/{pool} output is missing columns: {missing}')
        pool_generated = pool_generated[target_table.columns].copy()
        pool_generated.to_parquet(pool_output_path, index=False)
        write_output_csv(pd.DataFrame(pool_telemetry), pool_telemetry_path, index=False)
        manifest_row = {'pool': pool, 'training_rows': int(len(training)), 'target_rows': int(len(pool_generated)), 'model_path': str(model_path), 'metadata_path': str(metadata_path) if metadata_path.exists() else None, 'pool_output_path': str(pool_output_path), 'checkpoint_path': str(checkpoint_path), 'sdv_gpu_parameter': gpu_parameter, 'enable_gpu_requested': bool(cfg.ENABLE_GPU), 'resumed': False}
        atomic_json_dump({'manifest_row': manifest_row}, checkpoint_path)
        generated_parts.append(pool_generated)
        calendar_telemetry.extend(pool_telemetry)
        pool_manifest.append(manifest_row)
        del synthesizer
        gc.collect()
    synthetic = pd.concat(generated_parts, ignore_index=True)
    missing = sorted(set(target_table.columns).difference(synthetic.columns))
    if missing:
        raise RuntimeError(f'Cohort-matched {method} output is missing columns: {missing}')
    synthetic = synthetic[target_table.columns].copy()
    pool_manifest_path = method_dir / 'pool_model_manifest.csv'
    write_output_csv(pd.DataFrame(pool_manifest), pool_manifest_path, index=False)
    write_output_csv(pd.DataFrame(calendar_telemetry), method_dir / 'calendar_validity_telemetry.csv', index=False)
    return (synthetic, str(pool_manifest_path))

def _fit_sdv_method(method: str, frame: pd.DataFrame, cfg: Config, seed: int):
    if SDV_METHODS.get(method) is None:
        raise ImportError('SDV is required for this benchmark method.')
    metadata = _detect_metadata(frame)
    for column in frame.columns:
        if frame[column].dtype == 'object' or column in {cfg.COL_CAT, cfg.COL_OFFENCE, 'PREMISES_T', 'LOCATION_T'}:
            _update_column(metadata, column, 'categorical')
    set_seeds(seed)
    cls = SDV_METHODS[method]
    if method == 'GaussianCopula':
        synthesizer = cls(metadata, enforce_rounding=True)
    elif method == 'CTGAN_RawLatLon':
        gpu_kwargs, _ = sdv_gpu_constructor_kwargs(cls, cfg.ENABLE_GPU)
        synthesizer = cls(metadata, epochs=cfg.EPOCHS, batch_size=_adaptive_batch_size(len(frame), cfg.BATCH_TARGET, cfg.PAC), pac=cfg.PAC, generator_lr=cfg.GEN_LR, discriminator_lr=cfg.DISC_LR, enforce_rounding=True, verbose=False, **gpu_kwargs)
    else:
        gpu_kwargs, _ = sdv_gpu_constructor_kwargs(cls, cfg.ENABLE_GPU)
        synthesizer = cls(metadata, epochs=cfg.EPOCHS, batch_size=min(cfg.BATCH_TARGET, len(frame)), enforce_rounding=True, verbose=False, **gpu_kwargs)
    synthesizer.fit(frame)
    return (synthesizer, metadata)

def _kde_attribute_bootstrap(frame: pd.DataFrame, cfg: Config, seed: int) -> pd.DataFrame:
    pieces = []
    for key, group in frame.groupby(['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT], observed=True, sort=True):
        group_seed = derived_seed(seed, 'kde', *key)
        rng = np.random.default_rng(group_seed)
        sampled = group.sample(len(group), replace=True, random_state=group_seed).reset_index(drop=True)
        xy = group[[cfg.COL_LON, cfg.COL_LAT]].dropna().to_numpy(dtype=float).T
        if xy.shape[1] >= 3 and np.linalg.matrix_rank(np.cov(xy)) >= 2:
            kde = gaussian_kde(xy)
            generated = kde.resample(len(sampled), seed=rng).T
            sampled[cfg.COL_LON] = generated[:, 0]
            sampled[cfg.COL_LAT] = generated[:, 1]
        pieces.append(sampled)
    return pd.concat(pieces, ignore_index=True)

def _road_support_bootstrap(frame: pd.DataFrame, cfg: Config, seed: int) -> pd.DataFrame:
    pieces = []
    for key, group in frame.groupby(['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT], observed=True, sort=True):
        pieces.append(group.sample(len(group), replace=True, random_state=derived_seed(seed, *key)))
    return pd.concat(pieces, ignore_index=True)

def _optional_manifest_text(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None

def _load_current_prior_benchmark_rows(manifest_path: Path) -> dict[tuple[str, int], dict]:
    if not manifest_path.is_file():
        return {}
    prior = pd.read_csv(manifest_path)
    required = {'generator_requested', 'seed', 'status'}
    if not required.issubset(prior.columns):
        raise RuntimeError('The benchmark table is missing required fields.')
    rows: dict[tuple[str, int], dict] = {}
    for row in prior.to_dict(orient='records'):
        try:
            key = (str(row['generator_requested']), int(row['seed']))
        except (KeyError, TypeError, ValueError):
            continue
        rows[key] = row
    return rows

def _validate_benchmark_output(method: str, seed: int, output_path: Path, raw_coordinate: pd.DataFrame, output_scope: str) -> None:
    produced = pd.read_parquet(output_path)
    if len(produced) != len(raw_coordinate):
        raise RuntimeError(f'{method} seed={seed} produced {len(produced)} rows; expected {len(raw_coordinate)}.')
    if output_scope == 'complete_table':
        missing = sorted(set(raw_coordinate.columns).difference(produced.columns))
        if missing:
            raise RuntimeError(f'{method} seed={seed} is missing complete-table columns: {missing}')

def _recover_prior_benchmark_row(row: dict, *, method: str, seed: int, effective_seed: int, training_rows: int, output_path: Path, raw_coordinate: pd.DataFrame, output_scope: str) -> Optional[dict]:
    try:
        if int(row.get('effective_seed')) != int(effective_seed):
            return None
        if int(row.get('training_rows')) != int(training_rows):
            return None
    except (TypeError, ValueError):
        return None
    status = str(row.get('status', '')).casefold()
    recovered = dict(row)
    if status == 'success':
        if str(row.get('generator_executed')) != method:
            raise RuntimeError(f'Existing benchmark success for {method} seed={seed} records a different executed generator.')
        if str(row.get('output_scope')) != output_scope:
            raise RuntimeError(f'Existing benchmark success for {method} seed={seed} has an incompatible output scope.')
        _validate_benchmark_output(method, seed, output_path, raw_coordinate, output_scope)
        recovered.update({'output_path': str(output_path), 'reused': True, 'finalized_from_prior_failure': False})
        return recovered
    if status == 'unavailable' and method == 'TVAE_RawLatLon':
        recovered.update({'generator_requested': method, 'status': 'unavailable', 'model_path': None, 'output_path': None, 'reused': True, 'finalized_from_prior_failure': status == 'failed'})
        return recovered
    if status == 'unavailable':
        raise RuntimeError(f'Existing unavailable benchmark row for {method} seed={seed} does not satisfy the terminal TVAE policy.')
    return None

def _write_benchmark_manifest(rows: list[dict], manifest_path: Path) -> None:
    temporary = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    write_output_csv(pd.DataFrame(rows), temporary, index=False)
    temporary.replace(manifest_path)

def run_benchmark_generation(cfg: Config, train_parquet: str, out_dir: str, methods: Optional[List[str]]=None, progress_callback: Optional[Callable[[str, int, int, str, str], None]]=None) -> str:
    frame = pd.read_parquet(train_parquet)
    raw_coordinate = _prepare_raw_coordinate_table(frame, cfg)
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactDirectories.create(output_root)
    benchmark_data = artifacts.data / 'Benchmarks'
    benchmark_models = artifacts.models / 'Benchmarks'
    benchmark_data.mkdir(parents=True, exist_ok=True)
    benchmark_models.mkdir(parents=True, exist_ok=True)
    methods = list(cfg.BENCHMARK_METHODS if methods is None else methods)
    seeds = list(cfg.BENCHMARK_SEEDS or [cfg.SEED])
    requested_pairs = [(str(method), int(seed)) for method in methods for seed in seeds]
    if not requested_pairs:
        raise ValueError('At least one benchmark method-seed pair is required.')
    if len(set(requested_pairs)) != len(requested_pairs):
        raise ValueError('Benchmark method-seed requests must be unique.')
    manifest_path = artifacts.tables / 'benchmark_generation_manifest.csv'
    manifest_by_pair: dict[tuple[str, int], dict] = {}
    progress_total = len(requested_pairs)
    progress_current = 0
    input_path = benchmark_data / 'benchmark_input_raw_coordinates.parquet'
    raw_coordinate.to_parquet(input_path, index=False)
    prior_rows = _load_current_prior_benchmark_rows(manifest_path)
    for method, seed in requested_pairs:
        prior = prior_rows.get((method, seed))
        if prior is None:
            continue
        output_scope = 'complete_table'
        output_path = benchmark_data / method / f'Seed_{seed}' / 'synthetic.parquet'
        recovered = _recover_prior_benchmark_row(prior, method=method, seed=seed, effective_seed=derived_seed(seed, 'benchmark', method), training_rows=len(raw_coordinate), output_path=output_path, raw_coordinate=raw_coordinate, output_scope=output_scope)
        if recovered is not None:
            manifest_by_pair[method, seed] = recovered
    for method in methods:
        for seed in seeds:
            method = str(method)
            seed = int(seed)
            pair = (method, seed)
            if progress_callback is not None:
                progress_callback('benchmark', progress_current, progress_total, f'{method}, seed {int(seed)}', 'running')
            if pair in manifest_by_pair:
                recovered_status = str(manifest_by_pair[pair]['status']).casefold()
                progress_current += 1
                if progress_callback is not None:
                    progress_callback('benchmark', progress_current, progress_total, f'{method}, seed {seed}: {recovered_status} (reused)', 'running')
                continue
            model_dir = benchmark_models / method / f'Seed_{int(seed)}'
            data_dir = benchmark_data / method / f'Seed_{int(seed)}'
            model_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            output_path = data_dir / 'synthetic.parquet'
            model_path = model_dir / 'model.pkl'
            status = 'failed'
            error = None
            executed = None
            runtime = None
            output_scope = 'complete_table'
            terminal_policy = None
            fatal_exception: Optional[Exception] = None
            started = time.perf_counter()
            effective_seed = derived_seed(int(seed), 'benchmark', method)
            try:
                if method in SDV_METHODS:
                    executed = method
                    synthetic, pool_manifest_path = _cohort_matched_sdv_generation(method, frame, cfg, effective_seed, model_dir)
                    synthetic.to_parquet(output_path, index=False)
                    model_path = Path(pool_manifest_path)
                elif method == 'KDE_AttributeBootstrap':
                    _kde_attribute_bootstrap(raw_coordinate, cfg, effective_seed).to_parquet(output_path, index=False)
                    executed = method
                elif method == 'RoadSupportBootstrap':
                    _road_support_bootstrap(frame, cfg, effective_seed).to_parquet(output_path, index=False)
                    executed = method
                else:
                    raise ValueError(f'Unknown benchmark method: {method}')
                _validate_benchmark_output(method, seed, output_path, raw_coordinate, output_scope)
                status = 'success'
            except BoundedConditionalSamplingError as exc:
                error = repr(exc)
                if is_terminal_tvae_unavailability(method, exc):
                    status = 'unavailable'
                    terminal_policy = TVAE_TERMINAL_UNAVAILABLE_POLICY
                else:
                    fatal_exception = exc
            except Exception as exc:
                error = repr(exc)
                fatal_exception = exc
            runtime = time.perf_counter() - started
            succeeded = status == 'success'
            manifest_by_pair[pair] = {'generator_requested': method, 'generator_executed': executed, 'output_scope': output_scope, 'status': status, 'seed': int(seed), 'effective_seed': int(effective_seed), 'training_rows': len(raw_coordinate), 'runtime_seconds': runtime, 'model_path': str(model_path) if succeeded and model_path.exists() else None, 'output_path': str(output_path) if succeeded and output_path.exists() else None, 'reused': False, 'finalized_from_prior_failure': False}
            _write_benchmark_manifest([manifest_by_pair[requested_pair] for requested_pair in requested_pairs if requested_pair in manifest_by_pair], manifest_path)
            progress_current += 1
            if progress_callback is not None:
                progress_callback('benchmark', progress_current, progress_total, f'{method}, seed {int(seed)}: {status}', 'failed' if fatal_exception is not None else 'running')
            if fatal_exception is not None:
                atomic_json_dump({'status': 'FAILED', 'fatal_method_seed_pair': {'generator_requested': method, 'seed': seed}}, artifacts.metadata / 'benchmark_status.json')
                raise fatal_exception.with_traceback(fatal_exception.__traceback__)
    manifest_rows = [manifest_by_pair[pair] for pair in requested_pairs]
    _write_benchmark_manifest(manifest_rows, manifest_path)
    manifest = pd.DataFrame(manifest_rows)
    normalized_status = manifest['status'].astype(str).str.casefold()
    nonterminal = manifest.loc[~normalized_status.isin(_TERMINAL_BENCHMARK_STATUSES)]
    if len(nonterminal):
        raise RuntimeError(f"Benchmark generation reached finalization with nonterminal rows: {nonterminal[['generator_requested', 'seed', 'status']].to_dict('records')}")
    successful_runs = int(normalized_status.eq('success').sum())
    unavailable = manifest.loc[normalized_status.eq('unavailable')]
    unavailable_runs = int(len(unavailable))
    atomic_json_dump({'status': 'COMPLETE', 'requested_runs': int(len(manifest)), 'successful_runs': successful_runs, 'unavailable_runs': unavailable_runs, 'unavailable_method_seed_pairs': unavailable[['generator_requested', 'seed']].to_dict(orient='records')}, artifacts.metadata / 'benchmark_status.json')
    if progress_callback is not None:
        detail = 'Benchmark generation complete'
        if unavailable_runs:
            detail += f': {successful_runs}/{len(manifest)} succeeded; {unavailable_runs} TVAE run(s) terminally unavailable'
        progress_callback('benchmark', progress_total, progress_total, detail, 'complete')
    return str(manifest_path)
