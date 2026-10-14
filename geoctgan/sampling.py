from __future__ import annotations
from .utils import write_output_csv
from .utils import resolve_output_path
import calendar
import json
import logging
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
try:
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer
except ImportError:
    Condition = None
    CTGANSynthesizer = None
from .config import Config, scientific_config_payload
from .features import decode_cyclic, inv_logit
from .flow import assign_capacity_only, assign_flow_cann, assign_no_capacity, build_capacities
from .geometry import point_on_line_with_signed_offset
from .temporal import condition_values, pool_id, reconstruct_date
from .utils import atomic_json_dump, derived_seed, ensure_dir, load_success_checkpoint, set_seeds, write_success_checkpoint
logger = logging.getLogger(__name__)

def _safe_slug(value: str) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('_') or 'UNKNOWN'

def _atomic_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination

def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    write_output_csv(frame, temporary, index=False)
    temporary.replace(destination)
    return destination

def _load_synthesizer(path: str):
    if CTGANSynthesizer is None:
        raise ImportError('SDV is required for conditional generation.')
    return CTGANSynthesizer.load(filepath=str(path))

def _sample_condition(synthesizer: CTGANSynthesizer, values: Dict[str, str], rows: int) -> pd.DataFrame:
    if Condition is None:
        raise ImportError('SDV is required for conditional generation.')
    condition = Condition(column_values=values, num_rows=int(rows))
    return synthesizer.sample_from_conditions(conditions=[condition], max_tries_per_batch=10000, batch_size=max(500, min(5000, int(rows))), output_file_path=None)

def _condition_valid(frame: pd.DataFrame, values: Dict[str, str]) -> pd.Series:
    valid = pd.Series(True, index=frame.index)
    for column, expected in values.items():
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        valid &= frame[column].astype(str).str.strip().eq(str(expected))
    return valid.fillna(False)

def _offence_valid(frame: pd.DataFrame, category: str, crosswalk: Dict[str, List[str]], cfg: Config) -> pd.Series:
    allowed = set(map(str, crosswalk.get(str(category), [])))
    if not allowed or cfg.COL_OFFENCE not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame[cfg.COL_OFFENCE].astype(str).isin(allowed).fillna(False)

def _decode_generated_rows(frame: pd.DataFrame, *, year: int, month: int, category: str, condition: Dict[str, str], crosswalk: Dict[str, List[str]], cfg: Config) -> Tuple[pd.DataFrame, dict]:
    frame = frame.copy().reset_index(drop=True)
    condition_ok = _condition_valid(frame, condition)
    offence_ok = _offence_valid(frame, category, crosswalk, cfg)
    day = pd.to_numeric(frame.get('OCC_DAY'), errors='coerce')
    maximum_day = calendar.monthrange(int(year), int(month))[1]
    day_ok = day.between(1, maximum_day, inclusive='both').fillna(False)
    embedding_columns = [f'gemb_{index}' for index in range(int(cfg.EMB_DIM))]
    missing_embedding_columns = [column for column in embedding_columns if column not in frame.columns]
    if missing_embedding_columns:
        raise RuntimeError(f'Generated table is missing the required spatial-embedding schema: {missing_embedding_columns}. Expected exactly {int(cfg.EMB_DIM)} ordered gemb_* columns.')
    embedding_ok = pd.Series(np.isfinite(frame[embedding_columns].to_numpy(dtype=float)).all(axis=1), index=frame.index)
    if {'OCC_HOUR_sin', 'OCC_HOUR_cos'}.issubset(frame.columns):
        hour_sin = pd.to_numeric(frame['OCC_HOUR_sin'], errors='coerce')
        hour_cos = pd.to_numeric(frame['OCC_HOUR_cos'], errors='coerce')
        hour_norm = np.sqrt(np.square(hour_sin) + np.square(hour_cos))
        hour_missing = frame.get('OCC_HOUR_MISSING', pd.Series(0, index=frame.index)).astype(str).isin({'1', '1.0', 'True', 'true'})
        hour_ok = hour_missing | pd.Series(np.isfinite(hour_sin) & np.isfinite(hour_cos), index=frame.index) & hour_norm.gt(1e-08)
    else:
        hour_ok = pd.Series(True, index=frame.index)
    valid = condition_ok & offence_ok & day_ok & embedding_ok & hour_ok
    output = frame.loc[valid].copy().reset_index(drop=True)
    output['OCC_YEAR'] = int(year)
    output['OCC_MONTH'] = int(month)
    output['OCC_DATE'] = reconstruct_date(year, month, output['OCC_DAY'])
    output['OCC_DOW'] = output['OCC_DATE'].dt.dayofweek.astype('Int64')
    if {'OCC_HOUR_sin', 'OCC_HOUR_cos'}.issubset(output.columns):
        decoded = decode_cyclic(output['OCC_HOUR_sin'], output['OCC_HOUR_cos'], 24, one_based=False)
        if 'OCC_HOUR_MISSING' in output.columns:
            missing = output['OCC_HOUR_MISSING'].astype(str).isin({'1', '1.0', 'True', 'true'})
            decoded.loc[missing] = pd.NA
        output['OCC_HOUR'] = decoded
    if 'seg_pos_logit' in output.columns:
        values = pd.to_numeric(output['seg_pos_logit'], errors='coerce')
        output['seg_pos'] = inv_logit(values.fillna(0))
    else:
        output['seg_pos'] = 0.5
    if 'SEG_MAP_DIST_M_log1p' in output.columns:
        distance = np.expm1(pd.to_numeric(output['SEG_MAP_DIST_M_log1p'], errors='coerce').fillna(0).clip(lower=0))
        output['SEG_MAP_DIST_M'] = np.clip(distance, 0.0, cfg.SNAP_MAX_M)
    else:
        output['SEG_MAP_DIST_M'] = 0.0
    reasons = {'sampled_rows': len(frame), 'valid_rows': len(output), 'condition_mismatch': int((~condition_ok).sum()), 'offence_category_mismatch': int((~offence_ok).sum()), 'invalid_day': int((~day_ok).sum()), 'invalid_hour_vector': int((~hour_ok).sum()), 'nonfinite_embedding': int((~embedding_ok).sum())}
    return (output, reasons)

def _adaptive_request_size(need: int, sampled: int, valid: int, cfg: Config) -> int:
    prior_rate = 0.35
    rate = (valid + 10 * prior_rate) / max(sampled + 10, 1)
    rate = float(np.clip(rate, 0.01, 0.95))
    request = int(math.ceil(need / rate * cfg.CONDITIONAL_REQUEST_SAFETY_FACTOR))
    return max(cfg.MIN_CONDITIONAL_REQUEST, min(cfg.MAX_CONDITIONAL_REQUEST, request))

def _model_from_cache(cache: OrderedDict, pool: str, path: Path, max_size: int):
    if pool in cache:
        cache.move_to_end(pool)
        return cache[pool]
    model = _load_synthesizer(str(path))
    cache[pool] = model
    while len(cache) > int(max_size):
        cache.popitem(last=False)
    return model

def generate_latent_rows(cfg: Config, raw_table_path: str, crosswalk_path: str, models_dir: str, out_dir: str, *, target_table_path: Optional[str]=None) -> Tuple[str, str]:
    raw = pd.read_parquet(raw_table_path)
    target_reference = raw if target_table_path is None else pd.read_parquet(target_table_path)
    with open(crosswalk_path, 'r', encoding='utf-8') as handle:
        crosswalk = json.load(handle)
    targets = target_reference.groupby(['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT], observed=True, sort=True).size().rename('real_count').reset_index()
    targets['target'] = np.rint(targets['real_count'] * cfg.SAMPLE_FACTOR).astype(int)
    targets = targets.loc[targets['target'] > 0].copy()
    generation_dir = Path(ensure_dir(Path(out_dir) / 'generation'))
    condition_dir = Path(ensure_dir(generation_dir / 'condition_tables'))
    diagnostic_dir = Path(ensure_dir(generation_dir / 'condition_diagnostics'))
    checkpoint_dir = Path(ensure_dir(generation_dir / 'condition_checkpoints'))
    partial_diagnostics_path = generation_dir / 'generation_diagnostics_partial.csv'
    generated_parts, diagnostics = ([], [])
    model_cache: OrderedDict[str, CTGANSynthesizer] = OrderedDict()
    for target_row in tqdm(list(targets.itertuples(index=False)), desc='Generating conditions'):
        year = int(target_row.OCC_YEAR)
        month = int(target_row.OCC_MONTH)
        category = str(getattr(target_row, cfg.COL_CAT))
        target = int(target_row.target)
        slug = f"{_safe_slug(category)}_{category.encode('utf-8').hex()}"
        condition_path = condition_dir / f'latent_{year}_{month:02d}_{slug}.parquet'
        diagnostic_path = diagnostic_dir / f'diagnostic_{year}_{month:02d}_{slug}.json'
        checkpoint_path = checkpoint_dir / f'latent_{year}_{month:02d}_{slug}.json'
        pool = pool_id(year, month, cfg.COHORT_KIND)
        model_path = Path(models_dir) / f'ctgan_{pool}.pkl'
        if not model_path.is_file():
            raise FileNotFoundError(f'Missing CTGAN model for temporal pool {pool}: {model_path}')
        checkpoint = None
        if cfg.RESUME_GENERATION:
            checkpoint = load_success_checkpoint(checkpoint_path, artifact_root=generation_dir, required_artifacts=[condition_path, diagnostic_path])
        if checkpoint is not None:
            completed = pd.read_parquet(condition_path)
            try:
                diagnostic_record = json.loads(diagnostic_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                diagnostic_record = {}
            expected_columns = list(checkpoint.get('result', {}).get('columns', []))
            required_columns = {'SYNTHETIC_ID', '__POOL__', 'OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT, *[f'gemb_{index}' for index in range(int(cfg.EMB_DIM))]}
            schema_valid = len(completed) == target and expected_columns == list(completed.columns) and required_columns.issubset(completed.columns) and completed['SYNTHETIC_ID'].astype(str).is_unique and completed['OCC_YEAR'].eq(year).all() and completed['OCC_MONTH'].eq(month).all() and completed[cfg.COL_CAT].astype(str).eq(category).all() and (int(diagnostic_record.get('year', -1)) == year) and (int(diagnostic_record.get('month', -1)) == month) and (str(diagnostic_record.get('category', '')) == category) and (int(diagnostic_record.get('target', -1)) == target) and (int(diagnostic_record.get('accepted', -1)) == target)
            if not not schema_valid:
                generated_parts.append(completed)
                record = dict(diagnostic_record)
                record['resumed'] = True
                record['condition_file'] = str(condition_path)
                diagnostics.append(record)
                _atomic_csv(pd.DataFrame(diagnostics), partial_diagnostics_path)
                continue
        synthesizer = _model_from_cache(model_cache, pool, model_path, cfg.MAX_MODEL_CACHE)
        condition = condition_values(year, month, category, cfg.COHORT_KIND, cfg.COL_CAT)
        accepted_parts, sampled_total, valid_total, aggregate = ([], 0, 0, {})
        total_budget = max(3000, int(target * cfg.MAX_TOTAL_SAMPLE_MULTIPLIER))
        for round_index in range(cfg.MAX_WHOLE_ROW_RESAMPLE_ROUNDS):
            accepted = sum((len(part) for part in accepted_parts))
            need = target - accepted
            if need <= 0:
                break
            if sampled_total >= total_budget:
                break
            request = min(_adaptive_request_size(need, sampled_total, valid_total, cfg), total_budget - sampled_total)
            set_seeds(derived_seed(cfg.SEED, 'generate', year, month, category, round_index))
            sampled = _sample_condition(synthesizer, condition, request)
            sampled_total += len(sampled)
            valid, reasons = _decode_generated_rows(sampled, year=year, month=month, category=category, condition=condition, crosswalk=crosswalk, cfg=cfg)
            valid_total += len(valid)
            for key, value in reasons.items():
                aggregate[key] = aggregate.get(key, 0) + int(value)
            if len(valid):
                accepted_parts.append(valid.iloc[:need].copy())
        accepted = sum((len(part) for part in accepted_parts))
        if accepted < target:
            failure = {'year': year, 'month': month, 'category': category, 'target': target, 'accepted': accepted, 'sampled': sampled_total, 'sample_budget': total_budget, 'diagnostics': aggregate}
            atomic_json_dump(failure, generation_dir / 'generation_failure.json')
            if accepted_parts:
                _atomic_parquet(pd.concat(accepted_parts, ignore_index=True), condition_dir / f'partial_{year}_{month:02d}_{slug}.parquet')
            raise RuntimeError(f'Conditional generation failed for {year}-{month:02d}/{category}: accepted {accepted}/{target} after {sampled_total} sampled rows. Completed condition checkpoints remain reusable.')
        condition_frame = pd.concat(accepted_parts, ignore_index=True).iloc[:target].copy()
        condition_frame['SYNTHETIC_ID'] = [f'SYN_{cfg.SEED}_{year}_{month:02d}_{slug}_{index:07d}' for index in range(target)]
        condition_frame['__POOL__'] = pool
        _atomic_parquet(condition_frame, condition_path)
        diagnostic_record = {'year': year, 'month': month, 'category': category, 'target': target, 'sampled': sampled_total, 'accepted': target, 'acceptance_rate': target / max(sampled_total, 1), 'raw_valid_rate': valid_total / max(sampled_total, 1), 'resumed': False, 'condition_file': str(condition_path), **{f'diagnostic_{key}': value for key, value in aggregate.items()}}
        atomic_json_dump(diagnostic_record, diagnostic_path)
        write_success_checkpoint(checkpoint_path, result={'year': year, 'month': month, 'category': category, 'target': target, 'pool': pool, 'columns': list(condition_frame.columns), 'condition_relative': condition_path.relative_to(generation_dir).as_posix(), 'diagnostic_relative': diagnostic_path.relative_to(generation_dir).as_posix()}, artifacts=[condition_path, diagnostic_path], artifact_root=generation_dir)
        generated_parts.append(condition_frame)
        diagnostics.append(diagnostic_record)
        _atomic_csv(pd.DataFrame(diagnostics), partial_diagnostics_path)
    latent = pd.concat(generated_parts, ignore_index=True)
    latent_path = generation_dir / 'synthetic_latent_rows.parquet'
    diagnostics_path = generation_dir / 'generation_diagnostics.csv'
    _atomic_parquet(latent, latent_path)
    _atomic_csv(pd.DataFrame(diagnostics), diagnostics_path)
    return (str(latent_path), str(diagnostics_path))

def _reconstruct_geometry(cfg: Config, assigned: pd.DataFrame, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    segment_geometry = segments.set_index(cfg.COL_SEG).geometry.to_dict()
    xs, ys = ([], [])
    for row in assigned.itertuples(index=False):
        segment_id = str(getattr(row, cfg.COL_SEG))
        line = segment_geometry.get(segment_id)
        if line is None or line.is_empty:
            raise RuntimeError(f'Assigned segment lacks valid geometry: {segment_id}')
        position = float(np.clip(getattr(row, 'seg_pos'), 0.0, 1.0))
        offset = float(np.clip(getattr(row, 'SEG_MAP_DIST_M'), 0.0, cfg.SNAP_MAX_M))
        side_rng = np.random.default_rng(derived_seed(cfg.SEED, 'geometry', getattr(row, 'SYNTHETIC_ID')))
        signed_offset = offset if side_rng.random() >= 0.5 else -offset
        point = point_on_line_with_signed_offset(line, position, signed_offset)
        xs.append(point.x)
        ys.append(point.y)
    output = gpd.GeoDataFrame(assigned, geometry=gpd.points_from_xy(xs, ys, crs=cfg.CRS_METRIC), crs=cfg.CRS_METRIC).to_crs(cfg.CRS_GEO)
    output[cfg.COL_LON] = output.geometry.x
    output[cfg.COL_LAT] = output.geometry.y
    return output

def run_cann_largest_group_preflight(cfg: Config, raw_table_path: str, embedding_path: str, out_dir: str) -> str:
    raw = pd.read_parquet(raw_table_path)
    embedding = pd.read_parquet(embedding_path)
    embedding_columns = sorted([column for column in embedding.columns if column.startswith('gemb_')], key=lambda value: int(value.split('_')[1]))
    segment_ids = embedding[cfg.COL_SEG].astype(str).tolist()
    segment_vectors = embedding[embedding_columns].to_numpy(dtype=np.float32)
    rows = []
    groups = []
    for key, group in raw.groupby(['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT], observed=True, sort=True):
        support = int(group[cfg.COL_SEG].astype(str).nunique())
        groups.append((int(len(group) * support), key, group.copy()))
    groups.sort(key=lambda item: item[0], reverse=True)
    for full_edges, key, group in groups[:int(cfg.CANN_PREFLIGHT_LARGEST_GROUPS)]:
        year, month, category = (int(key[0]), int(key[1]), str(key[2]))
        observed = group[cfg.COL_SEG].astype(str).value_counts().to_dict()
        capacities = build_capacities(observed, len(group), mode='exact_quota')
        result = assign_flow_cann(group[embedding_columns].to_numpy(dtype=np.float32), segment_ids, segment_vectors, capacities, capacity_mode='exact_quota', candidate_schedule=cfg.K_CANDIDATE_SCHEDULE, cost_scale=cfg.CANN_COST_SCALE, objective_tolerance=cfg.CANN_OBJECTIVE_TOLERANCE, stable_rounds=cfg.CANN_STABLE_ROUNDS, full_support_threshold=cfg.CANN_FULL_SUPPORT_THRESHOLD, time_limit_seconds=cfg.CANN_GROUP_TIME_LIMIT_S, max_edges=cfg.CANN_MAX_EDGES, require_ortools=cfg.CANN_REQUIRE_ORTOOLS)
        rows.append({'year': year, 'month': month, 'category': category, 'events': int(len(group)), 'support_segments': int(len(capacities)), 'full_support_edges': int(full_edges), **result.diagnostics})
    output = Path(out_dir) / 'cann_largest_group_preflight.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_output_csv(pd.DataFrame(rows), output, index=False)
    return str(output)

def _assignment_preflight(cfg: Config, latent: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in latent.groupby(['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT], sort=True, observed=True):
        year, month, category = (int(key[0]), int(key[1]), str(key[2]))
        real_group = raw.loc[raw['OCC_YEAR'].astype(int).eq(year) & raw['OCC_MONTH'].astype(int).eq(month) & raw[cfg.COL_CAT].astype(str).eq(category)]
        support = int(real_group[cfg.COL_SEG].astype(str).nunique())
        events = int(len(group))
        rows.append({'year': year, 'month': month, 'category': category, 'events': events, 'support_segments': support, 'full_support_edges': int(events * support), 'exceeds_edge_budget': bool(events * support > cfg.CANN_MAX_EDGES)})
    return pd.DataFrame(rows)

def assign_and_reconstruct(cfg: Config, latent_path: str, raw_table_path: str, segments_path: str, embedding_path: str, out_dir: str, *, variant: str='Full', capacity_mode: Optional[str]=None) -> Tuple[str, str, str]:
    latent = pd.read_parquet(latent_path)
    raw = pd.read_parquet(raw_table_path)
    segments = gpd.read_parquet(segments_path).to_crs(cfg.CRS_METRIC)
    embedding = pd.read_parquet(embedding_path)
    embedding_columns = sorted([column for column in embedding.columns if column.startswith('gemb_')], key=lambda value: int(value.split('_')[1]))
    segment_ids = embedding[cfg.COL_SEG].astype(str).tolist()
    segment_vectors = embedding[embedding_columns].to_numpy(dtype=np.float32)
    id_to_index = {segment_id: index for index, segment_id in enumerate(segment_ids)}
    mode = cfg.CAPACITY_MODE if capacity_mode is None else str(capacity_mode)
    assignment_dir = Path(ensure_dir(Path(out_dir) / f'cann_{variant}'))
    group_dir = Path(ensure_dir(assignment_dir / 'group_assignments'))
    preflight = _assignment_preflight(cfg, latent, raw)
    write_output_csv(preflight, assignment_dir / 'cann_group_preflight.csv', index=False)
    diagnostics_rows, capacity_rows, assigned_parts = ([], [], [])
    group_columns = ['OCC_YEAR', 'OCC_MONTH', cfg.COL_CAT]
    groups = list(latent.groupby(group_columns, sort=True, observed=True))
    for key, synthetic_group in tqdm(groups, desc=f'CaNN {variant}'):
        year, month, category = (int(key[0]), int(key[1]), str(key[2]))
        slug = _safe_slug(category)
        group_path = group_dir / f'assigned_{year}_{month:02d}_{slug}.parquet'
        diagnostic_path = group_dir / f'diagnostic_{year}_{month:02d}_{slug}.json'
        capacity_path = group_dir / f'capacities_{year}_{month:02d}_{slug}.csv'
        if cfg.RESUME_CANN and group_path.exists() and diagnostic_path.exists() and capacity_path.exists():
            completed = pd.read_parquet(group_path)
            if len(completed) != len(synthetic_group):
                raise RuntimeError(f'CaNN checkpoint {group_path} has the wrong row count.')
            assigned_parts.append(completed)
            diagnostics_rows.append(json.loads(diagnostic_path.read_text(encoding='utf-8')))
            capacity_rows.extend(pd.read_csv(capacity_path).to_dict(orient='records'))
            continue
        real_group = raw.loc[raw['OCC_YEAR'].astype(int).eq(year) & raw['OCC_MONTH'].astype(int).eq(month) & raw[cfg.COL_CAT].astype(str).eq(category)]
        observed = real_group[cfg.COL_SEG].astype(str).value_counts().to_dict()
        capacities = build_capacities(observed, len(synthetic_group), mode=mode, upper_slack=cfg.UPPER_QUOTA_SLACK)
        queries = synthetic_group[embedding_columns].to_numpy(dtype=np.float32)
        support_ids = sorted(capacities)
        support_indices = np.array([id_to_index[s] for s in support_ids], dtype=int)
        support_vectors = segment_vectors[support_indices]
        variant_lower = variant.lower()
        if variant_lower == 'capacityonly':
            result = assign_capacity_only(len(synthetic_group), capacities, seed=derived_seed(cfg.SEED, 'capacity_only', year, month, category))
        elif variant_lower == 'nocapacity':
            result = assign_no_capacity(queries, support_ids, support_vectors)
        else:
            flow_vectors = segment_vectors.copy()
            if variant_lower == 'permutedembedding':
                rng = np.random.default_rng(derived_seed(cfg.SEED, 'permute', year, month, category))
                permutation = rng.permutation(len(flow_vectors))
                flow_vectors = flow_vectors[permutation]
            result = assign_flow_cann(queries, segment_ids, flow_vectors, capacities, capacity_mode=mode, candidate_schedule=cfg.K_CANDIDATE_SCHEDULE, cost_scale=cfg.CANN_COST_SCALE, objective_tolerance=cfg.CANN_OBJECTIVE_TOLERANCE, stable_rounds=cfg.CANN_STABLE_ROUNDS, full_support_threshold=cfg.CANN_FULL_SUPPORT_THRESHOLD, time_limit_seconds=cfg.CANN_GROUP_TIME_LIMIT_S, max_edges=cfg.CANN_MAX_EDGES, require_ortools=cfg.CANN_REQUIRE_ORTOOLS)
        assigned = synthetic_group.copy()
        assigned[cfg.COL_SEG] = result.assigned_segment_ids
        assigned.to_parquet(group_path, index=False)
        capacity_frame = pd.DataFrame([{'year': year, 'month': month, 'category': category, cfg.COL_SEG: segment_id, 'capacity': int(capacity), 'capacity_mode': mode, 'variant': variant} for segment_id, capacity in capacities.items()])
        write_output_csv(capacity_frame, capacity_path, index=False)
        diagnostic = {'year': year, 'month': month, 'category': category, 'variant': variant, **result.diagnostics}
        atomic_json_dump(diagnostic, diagnostic_path)
        assigned_parts.append(assigned)
        diagnostics_rows.append(diagnostic)
        capacity_rows.extend(capacity_frame.to_dict(orient='records'))
    assigned = pd.concat(assigned_parts, ignore_index=True)
    if assigned[cfg.COL_SEG].isna().any():
        raise RuntimeError('CaNN produced missing segment assignments.')
    output = _reconstruct_geometry(cfg, assigned, segments)
    output_path = assignment_dir / f'synthetic_points_{variant}.parquet'
    diagnostics_path = assignment_dir / 'cann_diagnostics.csv'
    capacities_path = assignment_dir / 'capacity_table.csv'
    output.to_parquet(output_path, index=False)
    write_output_csv(pd.DataFrame(diagnostics_rows), diagnostics_path, index=False)
    write_output_csv(pd.DataFrame(capacity_rows), capacities_path, index=False)
    return (str(output_path), str(diagnostics_path), str(capacities_path))

def run_reconstruction_ablations(cfg: Config, latent_path: str, raw_table_path: str, segments_path: str, embedding_path: str, out_dir: str, *, include_full: bool=True) -> str:
    variants = [('CapacityOnly', 'exact_quota'), ('NoCapacity', 'exact_quota'), ('PermutedEmbedding', 'exact_quota'), ('UpperQuota', 'upper_quota')]
    if include_full:
        variants.insert(0, ('Full', 'exact_quota'))
    rows = []
    for variant, mode in variants:
        output, diagnostics, capacities = assign_and_reconstruct(cfg, latent_path, raw_table_path, segments_path, embedding_path, out_dir, variant=variant, capacity_mode=mode)
        rows.append({'variant': variant, 'capacity_mode': mode, 'output_path': output, 'diagnostics_path': diagnostics, 'capacity_path': capacities})
    manifest = Path(out_dir) / 'reconstruction_ablation_manifest.csv'
    write_output_csv(pd.DataFrame(rows), manifest, index=False)
    return str(manifest)

def verified_ablation_output(manifest_path: str, variant: str) -> str:
    manifest = Path(manifest_path).expanduser().resolve()
    frame = pd.read_csv(manifest)
    selected = frame.loc[frame['variant'].astype(str).eq(str(variant))]
    if len(selected) != 1:
        raise RuntimeError(f'Expected exactly one {variant!r} row in {manifest}; found {len(selected)}.')
    row = selected.iloc[0]
    output = resolve_output_path(manifest, row['output_path'])
    if not output.is_file():
        raise FileNotFoundError(f'Recorded {variant} output is missing: {output}')
    return str(output)
