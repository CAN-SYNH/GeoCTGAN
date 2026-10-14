from __future__ import annotations
from .utils import write_output_csv
import hashlib
import json
import logging
import math
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import substring
from sklearn.model_selection import train_test_split
from .config import Config
from .utils import atomic_json_dump, load_success_checkpoint, resolve_in_zip, resolve_source_path, write_success_checkpoint
logger = logging.getLogger(__name__)

def _resolve_spatial_member(base_dir: str, zip_name: str, explicit_member: Optional[str], allowed_extensions: Tuple[str, ...]) -> str:
    if explicit_member:
        return resolve_in_zip(base_dir, zip_name, exact_member=explicit_member)
    zip_path = resolve_source_path(base_dir, zip_name)
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [name for name in archive.namelist() if not name.endswith('/') and name.lower().endswith(tuple((ext.lower() for ext in allowed_extensions)))]
    if not candidates:
        raise FileNotFoundError(f'No spatial source {allowed_extensions} found in {zip_path}.')
    preferred = [name for name in candidates if name.lower().endswith('.gpkg')] or [name for name in candidates if name.lower().endswith('.shp')]
    if len(preferred) != 1:
        raise RuntimeError(f'Ambiguous spatial source in {zip_path}. Configure the archive member explicitly. Candidates: {preferred[:20]}')
    return resolve_in_zip(base_dir, zip_name, exact_member=preferred[0])

def _list_layers(path: str) -> list[str]:
    try:
        import pyogrio
        return [str(row[0]) for row in pyogrio.list_layers(path)]
    except Exception:
        return []

def _read_source(path: str, layer: Optional[str]=None, *, expected_geometry: Optional[str]=None) -> gpd.GeoDataFrame:
    lower = path.lower()
    if lower.endswith('.gpkg') or '.gpkg!' in lower or '.gpkg' in lower:
        layers = _list_layers(path)
        if layer is None and len(layers) > 1:
            raise RuntimeError(f'The GeoPackage contains multiple layers. Configure NETWORK_LAYER/EVENTS layer explicitly instead of relying on the default layer. Layers: {layers}')
        if layer is not None and layers and (layer not in layers):
            raise KeyError(f'Requested layer {layer!r} is not in {path}. Available layers: {layers}')
    frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if 'geometry' not in frame.columns or frame.geometry is None:
        raise ValueError(f'Spatial source {path!r} layer={layer!r} has no active geometry column.')
    if expected_geometry:
        geometry_types = set(frame.geometry.dropna().geom_type.astype(str))
        if expected_geometry == 'line' and (not geometry_types.intersection({'LineString', 'MultiLineString'})):
            raise ValueError(f'Expected a line network layer but found geometry types {sorted(geometry_types)} in {path!r} layer={layer!r}.')
        if expected_geometry == 'point' and (not geometry_types.intersection({'Point', 'MultiPoint'})):
            raise ValueError(f'Expected a point event layer but found geometry types {sorted(geometry_types)} in {path!r} layer={layer!r}.')
    return frame

def _stable_event_id(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    if cfg.COL_ID in frame.columns and frame[cfg.COL_ID].notna().all() and frame[cfg.COL_ID].is_unique:
        return frame[cfg.COL_ID].astype(str)
    payload_columns = [cfg.COL_DATE, cfg.COL_LON, cfg.COL_LAT, cfg.COL_CAT, cfg.COL_OFFENCE]
    payload = frame.apply(lambda row: '|'.join((str(row.get(column, '')) for column in payload_columns)), axis=1)
    base = payload.map(lambda value: 'EVT_' + hashlib.sha1(value.encode('utf-8')).hexdigest()[:18])
    occurrence = base.groupby(base).cumcount()
    return pd.Series(np.where(occurrence.eq(0), base, base + '_' + occurrence.astype(str)), index=frame.index, dtype='object')

def load_events(cfg: Config, out_dir: str) -> str:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = _resolve_spatial_member(cfg.BASE_DIR, cfg.ZIP_EVENTS, cfg.EVENTS_MEMBER, ('.shp', '.gpkg'))
    events = _read_source(events_path, expected_geometry='point').to_crs(cfg.CRS_GEO)
    if cfg.COL_DATE not in events.columns:
        candidates = [column for column in events.columns if 'date' in column.lower()]
        if len(candidates) != 1:
            raise KeyError(f'Expected one date column when {cfg.COL_DATE!r} is absent; found {candidates}.')
        events = events.rename(columns={candidates[0]: cfg.COL_DATE})
    events[cfg.COL_DATE] = pd.to_datetime(events[cfg.COL_DATE], errors='coerce')
    events = events.dropna(subset=[cfg.COL_DATE, 'geometry']).copy()
    events = events.loc[events.geometry.geom_type.eq('Point')].copy()
    events['OCC_YEAR'] = events[cfg.COL_DATE].dt.year.astype(int)
    events['OCC_MONTH'] = events[cfg.COL_DATE].dt.month.astype(int)
    events['OCC_DAY'] = events[cfg.COL_DATE].dt.day.astype(int)
    events['OCC_DOW'] = events[cfg.COL_DATE].dt.dayofweek.astype(int)
    if 'OCC_HOUR' in events.columns:
        hour = pd.to_numeric(events['OCC_HOUR'], errors='coerce')
    else:
        hour = events[cfg.COL_DATE].dt.hour.astype(float)
    valid_hour = hour.between(0, 23)
    events['OCC_HOUR_MISSING'] = (~valid_hour | hour.isna()).astype(int)
    events['OCC_HOUR'] = hour.where(valid_hour, np.nan)
    events = events.loc[events['OCC_YEAR'].between(cfg.START_YEAR, cfg.END_YEAR, inclusive='both')].copy()
    events[cfg.COL_LON] = events.geometry.x.astype(float)
    events[cfg.COL_LAT] = events.geometry.y.astype(float)
    events[cfg.COL_ID] = _stable_event_id(events, cfg)
    if not events[cfg.COL_ID].is_unique:
        raise RuntimeError('Stable event identifiers are not unique after cleaning.')
    output = output_dir / 'events_clean.gpkg'
    events.to_file(output, layer='events', driver='GPKG')
    atomic_json_dump({'rows': len(events), 'source_member': cfg.EVENTS_MEMBER}, output_dir / 'events_clean_summary.json')
    return str(output)

def _resolve_manifest_file(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate

def create_outer_split(cfg: Config, events_gpkg: str, out_dir: str, *, split_manifest_path: Optional[str]=None) -> Tuple[str, str, str]:
    events = gpd.read_file(events_gpkg, layer='events')
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if split_manifest_path:
        manifest_path = Path(split_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        train_ids = pd.read_parquet(_resolve_manifest_file(manifest_path, manifest['train_ids_path']))[cfg.COL_ID].astype(str)
        test_ids = pd.read_parquet(_resolve_manifest_file(manifest_path, manifest['test_ids_path']))[cfg.COL_ID].astype(str)
        train_lookup, test_lookup = (set(train_ids), set(test_ids))
        if train_lookup.intersection(test_lookup):
            raise RuntimeError('Saved outer train/test ID lists overlap.')
        observed = set(events[cfg.COL_ID].astype(str))
        if train_lookup.union(test_lookup) != observed:
            raise RuntimeError('Saved outer split IDs do not exactly partition the current cleaned events.')
        train = events.loc[events[cfg.COL_ID].astype(str).isin(train_lookup)].copy()
        test = events.loc[events[cfg.COL_ID].astype(str).isin(test_lookup)].copy()
        reused = True
    else:
        stratification = events['OCC_YEAR'].astype(str) + '_' + events['OCC_MONTH'].astype(str).str.zfill(2) + '_' + events[cfg.COL_CAT].astype(str)
        counts = stratification.value_counts()
        rare = stratification.isin(counts[counts < 2].index)
        stratification = stratification.where(~rare, events['OCC_YEAR'].astype(str))
        train_index, test_index = train_test_split(np.arange(len(events)), test_size=cfg.TEST_SIZE, random_state=cfg.SEED, stratify=stratification)
        train = events.iloc[train_index].copy()
        test = events.iloc[test_index].copy()
        reused = False
    overlap = set(train[cfg.COL_ID]).intersection(set(test[cfg.COL_ID]))
    if overlap:
        raise RuntimeError(f'Outer train/test event-ID overlap detected: {len(overlap)}')
    train_path = output_dir / 'events_outer_train.gpkg'
    test_path = output_dir / 'events_outer_test.gpkg'
    train.to_file(train_path, layer='events', driver='GPKG')
    test.to_file(test_path, layer='events', driver='GPKG')
    train_ids_path = output_dir / 'outer_train_ids.parquet'
    test_ids_path = output_dir / 'outer_test_ids.parquet'
    train[[cfg.COL_ID]].to_parquet(train_ids_path, index=False)
    test[[cfg.COL_ID]].to_parquet(test_ids_path, index=False)
    manifest_path = output_dir / 'outer_split_manifest.json'
    atomic_json_dump({'seed': cfg.SEED, 'test_size': cfg.TEST_SIZE, 'train_rows': len(train), 'test_rows': len(test), 'event_id_overlap': 0, 'train_ids_path': train_ids_path.name, 'test_ids_path': test_ids_path.name, 'reused_existing_split': reused}, manifest_path)
    return (str(train_path), str(test_path), str(manifest_path))

def create_inner_split(cfg: Config, snapped_train_path: str, out_dir: str, *, seed: int) -> Tuple[str, str, str]:
    frame = pd.read_parquet(snapped_train_path)
    strata = frame['OCC_YEAR'].astype(str) + '_' + frame['OCC_MONTH'].astype(str).str.zfill(2) + '_' + frame[cfg.COL_CAT].astype(str)
    counts = strata.value_counts()
    strata = strata.where(~strata.isin(counts[counts < 2].index), frame['OCC_YEAR'].astype(str))
    train_index, validation_index = train_test_split(np.arange(len(frame)), test_size=cfg.INNER_VALIDATION_SIZE, random_state=int(seed), stratify=strata)
    inner_train = frame.iloc[train_index].copy()
    inner_validation = frame.iloc[validation_index].copy()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / 'inner_train.parquet'
    validation_path = output / 'inner_validation.parquet'
    inner_train.to_parquet(train_path, index=False)
    inner_validation.to_parquet(validation_path, index=False)
    manifest_path = output / 'inner_split_manifest.json'
    atomic_json_dump({'seed': int(seed), 'validation_size': cfg.INNER_VALIDATION_SIZE, 'train_rows': len(inner_train), 'validation_rows': len(inner_validation), 'overlap': int(len(set(inner_train[cfg.COL_ID]).intersection(set(inner_validation[cfg.COL_ID]))))}, manifest_path)
    return (str(train_path), str(validation_path), str(manifest_path))

def _canonical_line_key(line: LineString, precision: int=3) -> Tuple[Tuple[float, float], ...]:
    coordinates = tuple(((round(x, precision), round(y, precision)) for x, y in line.coords))
    reverse = tuple(reversed(coordinates))
    return min(coordinates, reverse)

def _source_value(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return ','.join(map(str, value))
    return str(value)

def _stable_segment_id(source_edge_uid: str, part_index: int, child_index: int, line: LineString) -> str:
    payload = f'{source_edge_uid}|{part_index}|{child_index}|{line.wkb_hex}'
    return 'seg_' + hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]

def segment_network(cfg: Config, out_dir: str) -> str:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    network_path = _resolve_spatial_member(cfg.BASE_DIR, cfg.ZIP_NETWORK, cfg.NETWORK_MEMBER, ('.gpkg', '.shp'))
    roads = _read_source(network_path, layer=cfg.NETWORK_LAYER, expected_geometry='line').to_crs(cfg.CRS_METRIC)
    roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    topology_candidates = {'u': ['u', 'from', 'from_id', 'source'], 'v': ['v', 'to', 'to_id', 'target'], 'key': ['key', 'edge_key'], 'osmid': ['osmid', 'osm_id', 'id'], 'layer': ['layer'], 'bridge': ['bridge'], 'tunnel': ['tunnel'], 'highway': ['highway', 'road_class', 'fclass']}
    lower_lookup = {column.lower(): column for column in roads.columns}
    source_columns: Dict[str, Optional[str]] = {}
    for canonical, candidates in topology_candidates.items():
        source_columns[canonical] = next((lower_lookup[candidate.lower()] for candidate in candidates if candidate.lower() in lower_lookup), None)
    if cfg.REQUIRE_AUTHORITATIVE_TOPOLOGY and (not (source_columns['u'] and source_columns['v'])):
        raise RuntimeError('Authoritative source u/v topology fields were required but not found.')
    records = []
    duplicate_keys = set()
    dropped_reverse_duplicates = 0
    for source_row_position, (_, row) in enumerate(roads.iterrows()):
        geometry = row.geometry
        parts: Iterable[LineString]
        if geometry.geom_type == 'LineString':
            parts = [geometry]
        elif geometry.geom_type == 'MultiLineString':
            parts = list(geometry.geoms)
        else:
            continue
        source_u = _source_value(row[source_columns['u']]) if source_columns['u'] else 'NA'
        source_v = _source_value(row[source_columns['v']]) if source_columns['v'] else 'NA'
        source_key = _source_value(row[source_columns['key']]) if source_columns['key'] else '0'
        source_osmid = _source_value(row[source_columns['osmid']]) if source_columns['osmid'] else 'NA'
        for original_part_index, original_part in enumerate(parts):
            edge_payload = f'row={source_row_position}|u={source_u}|v={source_v}|key={source_key}|part={original_part_index}|geom={original_part.wkb_hex}'
            source_edge_uid = 'edge_' + hashlib.sha1(edge_payload.encode('utf-8')).hexdigest()[:20]
            length = float(original_part.length)
            if length <= 0:
                continue
            number = max(1, int(math.ceil(length / cfg.MAX_SEG_LEN_M)))
            for child_index in range(number):
                start = child_index * length / number
                end = (child_index + 1) * length / number
                child = substring(original_part, start, end)
                if child.is_empty or child.geom_type != 'LineString' or child.length <= 1e-08:
                    continue
                canonical_key = _canonical_line_key(child)
                if cfg.REMOVE_REVERSE_DUPLICATES and canonical_key in duplicate_keys:
                    dropped_reverse_duplicates += 1
                    continue
                duplicate_keys.add(canonical_key)
                record = {cfg.COL_SEG: _stable_segment_id(source_edge_uid, original_part_index, child_index, child), 'source_edge_uid': source_edge_uid, 'source_row_position': source_row_position, 'source_osmid': source_osmid, 'source_part': original_part_index, 'child_index': child_index, 'child_count': number, 'seg_len_m': float(child.length), 'geometry': child}
                for canonical, source_column in source_columns.items():
                    record[f'source_{canonical}'] = row[source_column] if source_column else None
                records.append(record)
    segments = gpd.GeoDataFrame(records, geometry='geometry', crs=cfg.CRS_METRIC)
    if segments.empty:
        raise RuntimeError('Network segmentation produced no road segments.')
    if segments[cfg.COL_SEG].duplicated().any():
        raise RuntimeError('Stable segment identifiers are not unique.')
    if segments['source_edge_uid'].isna().any():
        raise RuntimeError('One or more child segments lack a unique parent-edge identifier.')
    output = output_dir / 'network_segments.parquet'
    segments.to_parquet(output, index=False)
    atomic_json_dump({'source_rows': len(roads), 'segments': len(segments), 'parent_edge_count': int(segments['source_edge_uid'].nunique()), 'max_segment_length_m': float(segments['seg_len_m'].max()), 'reverse_duplicates_removed': dropped_reverse_duplicates, 'source_topology_columns': source_columns, 'source_member': cfg.NETWORK_MEMBER, 'source_layer': cfg.NETWORK_LAYER}, output_dir / 'network_segmentation_summary.json')
    return str(output)

def _deterministic_nearest_join(points: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, max_distance: float, cfg: Config) -> Tuple[gpd.GeoDataFrame, Dict[str, float]]:
    points = points.copy()
    points['__POINT_ORDER__'] = np.arange(len(points), dtype=int)
    joined = gpd.sjoin_nearest(points, segments[[cfg.COL_SEG, 'geometry']], how='left', max_distance=max_distance, distance_col='SEG_MAP_DIST_M')
    joined[cfg.COL_SEG] = joined[cfg.COL_SEG].astype('string')
    joined = joined.sort_values(['__POINT_ORDER__', 'SEG_MAP_DIST_M', cfg.COL_SEG], kind='stable', na_position='last')
    match_counts = joined.groupby('__POINT_ORDER__', dropna=False).size()
    selected = joined.drop_duplicates('__POINT_ORDER__', keep='first').sort_values('__POINT_ORDER__')
    selected = selected.drop(columns=['index_right'], errors='ignore')
    diagnostics = {'input_points': int(len(points)), 'matched_points': int(selected[cfg.COL_SEG].notna().sum()), 'retention': float(selected[cfg.COL_SEG].notna().mean()), 'tied_points': int((match_counts > 1).sum()), 'tie_share': float((match_counts > 1).mean())}
    return (selected, diagnostics)

def snap_events(cfg: Config, events_path: str, segments_path: str, out_path: str, max_distance: Optional[float]=None) -> Tuple[str, Dict[str, float]]:
    max_distance = float(cfg.SNAP_MAX_M if max_distance is None else max_distance)
    events = gpd.read_file(events_path, layer='events').to_crs(cfg.CRS_METRIC)
    segments = gpd.read_parquet(segments_path).to_crs(cfg.CRS_METRIC)
    selected, diagnostics = _deterministic_nearest_join(events, segments, max_distance, cfg)
    geometry_lookup = segments.set_index(cfg.COL_SEG).geometry
    selected['seg_geom'] = selected[cfg.COL_SEG].map(geometry_lookup)

    def position(row) -> float:
        line = row['seg_geom']
        point = row.geometry
        if line is None or pd.isna(line) or line.is_empty or (line.length <= 0):
            return np.nan
        return float(line.project(point) / line.length)
    selected['seg_pos'] = selected.apply(position, axis=1)
    selected = selected.dropna(subset=[cfg.COL_SEG, 'seg_pos']).copy()
    selected[cfg.COL_SEG] = selected[cfg.COL_SEG].astype(str)
    selected = selected.drop(columns=['seg_geom', '__POINT_ORDER__'], errors='ignore')
    selected = selected.to_crs(cfg.CRS_GEO)
    table = pd.DataFrame(selected.drop(columns='geometry'))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    diagnostics['retained_rows'] = len(table)
    diagnostics['distance_p50_m'] = float(table['SEG_MAP_DIST_M'].median()) if len(table) else np.nan
    diagnostics['distance_p95_m'] = float(table['SEG_MAP_DIST_M'].quantile(0.95)) if len(table) else np.nan
    return (out_path, diagnostics)

def run_snap_sensitivity(cfg: Config, inner_train_events_path: str, segments_path: str, out_dir: str, *, inner_validation_events_path: Optional[str]=None) -> str:
    results = []
    scratch = Path(out_dir) / 'snap_sensitivity_tables'
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(out_dir) / 'candidate_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    datasets = [('inner_train', inner_train_events_path)]
    if inner_validation_events_path:
        datasets.append(('inner_validation', inner_validation_events_path))
    tolerances = sorted((float(value) for value in cfg.SNAP_CANDIDATES_M))
    for split_name, event_path in datasets:
        tables = {}
        metrics_by_tolerance = {}
        for tolerance in tolerances:
            temporary = scratch / f'{split_name}_snap_{tolerance:g}m.parquet'
            checkpoint_path = checkpoint_dir / f'{split_name}_snap_{tolerance:g}m.json'
            checkpoint = load_success_checkpoint(checkpoint_path, artifact_root=out_dir, required_artifacts=[temporary])
            if checkpoint is not None:
                metrics = dict(checkpoint['result'])
            else:
                started = time.perf_counter()
                _, metrics = snap_events(cfg, event_path, segments_path, str(temporary), max_distance=float(tolerance))
                metrics['runtime_seconds'] = float(time.perf_counter() - started)
                write_success_checkpoint(checkpoint_path, result=metrics, artifacts=[temporary], artifact_root=out_dir)
            tables[tolerance] = pd.read_parquet(temporary)[[cfg.COL_ID, cfg.COL_SEG]].copy()
            metrics_by_tolerance[tolerance] = metrics
        reference_tolerance = tolerances[-1]
        reference = tables[reference_tolerance].rename(columns={cfg.COL_SEG: 'reference_segment_id'})
        for tolerance in tolerances:
            current = tables[tolerance]
            common = current.merge(reference, on=cfg.COL_ID, how='inner')
            agreement = float((common[cfg.COL_SEG].astype(str) == common['reference_segment_id'].astype(str)).mean()) if len(common) else np.nan
            results.append({'split': split_name, 'snap_tolerance_m': float(tolerance), 'reference_tolerance_m': float(reference_tolerance), 'common_with_reference': int(len(common)), 'assignment_agreement_with_reference': agreement, **metrics_by_tolerance[tolerance]})
    frame = pd.DataFrame(results).sort_values(['snap_tolerance_m', 'split'])
    output = Path(out_dir) / 'snap_sensitivity.csv'
    write_output_csv(frame, output, index=False)
    return str(output)

def create_inner_event_split(cfg: Config, outer_train_gpkg: str, out_dir: str, *, seed: int) -> Tuple[str, str, str]:
    events = gpd.read_file(outer_train_gpkg, layer='events')
    strata = events['OCC_YEAR'].astype(str) + '_' + events['OCC_MONTH'].astype(str).str.zfill(2) + '_' + events[cfg.COL_CAT].astype(str)
    counts = strata.value_counts()
    strata = strata.where(~strata.isin(counts[counts < 2].index), events['OCC_YEAR'].astype(str))
    train_idx, validation_idx = train_test_split(np.arange(len(events)), test_size=cfg.INNER_VALIDATION_SIZE, random_state=int(seed), stratify=strata)
    inner_train = events.iloc[train_idx].copy()
    inner_validation = events.iloc[validation_idx].copy()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / 'inner_train_events.gpkg'
    validation_path = output / 'inner_validation_events.gpkg'
    inner_train.to_file(train_path, layer='events', driver='GPKG')
    inner_validation.to_file(validation_path, layer='events', driver='GPKG')
    manifest_path = output / 'inner_event_split_manifest.json'
    atomic_json_dump({'seed': int(seed), 'validation_size': cfg.INNER_VALIDATION_SIZE, 'train_rows': len(inner_train), 'validation_rows': len(inner_validation), 'overlap': int(len(set(inner_train[cfg.COL_ID]).intersection(set(inner_validation[cfg.COL_ID]))))}, manifest_path)
    return (str(train_path), str(validation_path), str(manifest_path))
