from __future__ import annotations
import inspect
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
from sklearn.neighbors import KDTree, KernelDensity
from tqdm.auto import tqdm
from .artifacts import ArtifactDirectories, create_unique_run_directory
from .utils import atomic_json_dump, clean_user_path, locate_user_input, write_output_csv, resolve_output_path, output_run_root
ProgressCallback = Callable[[str, int, int, str, str], None]
InnerProgressCallback = Callable[[int, int, str], None]
_RUN_ARTIFACT_ANCHORS = {'data', 'figures', 'intermediate', 'logs', 'metadata', 'models', 'tables'}

def _write_table(frame, path, **kwargs):
    drop = {'status', 'error', 'reason', 'procedure', 'controlled_by_design', 'controlled_columns_excluded', 'calibration_constraint', 'fidelity_interpretation', 'interpretation', 'response', 'predictor_scaling', 'missing_data', 'pseudo_p_convention', 'source_file'}
    frame = frame.drop(columns=[name for name in frame if name in drop], errors='ignore')
    if frame.empty or not len(frame.columns):
        raise RuntimeError('The evaluation table contains no results.')
    write_output_csv(frame, path, index=False)

def _annotate_spatial_design(frame):
    return frame.copy()


@dataclass
class EvaluationConfig:
    geographic_crs: str = 'EPSG:4326'
    metric_crs: str = 'EPSG:32617'
    longitude_column: str = 'LONG_WGS84'
    latitude_column: str = 'LAT_WGS84'
    category_column: str = 'MCI_CATEGO'
    segment_column: str = 'segment_id'
    random_seeds: list[int] = field(default_factory=lambda: [101, 202, 303, 404, 505])
    maximum_equal_sample_rows: int = 100000
    maximum_kde_fit_rows: int = 20000
    kde_grid_size: int = 200
    grid_sizes_m: list[int] = field(default_factory=lambda: [25, 50, 100, 250, 500])
    hotspot_fraction: float = 0.1
    near_real_thresholds_m: list[int] = field(default_factory=lambda: [0, 1, 5, 10, 25, 50, 100])
    kmap_grid_m: int = 50
    lisa_permutations: int = 999
    alpha: float = 0.05
    maximum_plotted_synthetic_datasets: int = 8
    privacy_native_subsample_rows: int = 10000
    privacy_native_iterations: int = 3
    dcr_training_to_validation_max_ratio: float = 2.0
    heavy_privacy_datasets: list[str] = field(default_factory=lambda: ['GeoCTGAN-Full | seed 42', 'GeoCTGAN-Full | seed 52', 'GeoCTGAN-Full | seed 62', 'GeoCTGAN-Priv | seed 42', 'GeoCTGAN-Priv | seed 52', 'GeoCTGAN-Priv | seed 62', 'GeoCTGAN-UpperQuota | seed 42', 'GeoCTGAN-UpperQuota | seed 52', 'GeoCTGAN-UpperQuota | seed 62'])
    heavy_privacy_columns: list[str] = field(default_factory=lambda: ['OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW', 'OCC_HOUR', 'MCI_CATEGO', 'OFFENCE', 'PREMISES_T', 'LOCATION_T', 'LONG_WGS84', 'LAT_WGS84'])
    strict_completion: bool = True
    privacy_profile: str = 'toronto'
    spatial_extent_m: Optional[list[float]] = None
    kmap_grid_sizes_m: list[int] = field(default_factory=lambda: [25, 50, 100])
    kmap_grid_origins: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)])

def _balanced_dcr_training_reference(real_training: pd.DataFrame, real_validation: pd.DataFrame, *, maximum_ratio: float, random_state: int) -> pd.DataFrame:
    if not np.isfinite(maximum_ratio) or maximum_ratio <= 0:
        raise ValueError('maximum_ratio must be finite and positive.')
    if real_training.empty or real_validation.empty:
        raise ValueError('DCR training and validation references must be non-empty.')
    maximum_training_rows = max(1, int(np.floor(len(real_validation) * float(maximum_ratio))))
    if len(real_training) <= maximum_training_rows:
        return real_training.reset_index(drop=True).copy()
    return real_training.sample(n=maximum_training_rows, replace=False, random_state=int(random_state)).reset_index(drop=True)

def _notify(callback: Optional[ProgressCallback], stage: str, current: int, total: int, detail: str, state: str='running') -> None:
    if callback is not None:
        callback(stage, int(current), int(total), str(detail), state)

def _notify_inner(callback: Optional[InnerProgressCallback], current: int, total: int, detail: str) -> None:
    if callback is not None:
        callback(int(current), int(total), str(detail))

def _resolve_summary_path(value: str | Path) -> Path:
    path = Path(clean_user_path(value)).expanduser().resolve()
    if path.is_file():
        return path
    candidates = [path / 'Metadata' / 'run_summary.json', path / 'run_summary.json']
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        recursive = sorted(path.rglob('run_summary.json')) if path.is_dir() else []
        if len(recursive) == 1:
            return recursive[0]
    raise FileNotFoundError(f'Could not identify one generation run summary from {path}. Provide Metadata/run_summary.json or its run directory.')

def _manifest_run_root(manifest_path: str | Path) -> Path:
    return output_run_root(manifest_path)

def resolve_recorded_artifact(value: str | Path, *, manifest_path: str | Path, recorded_root: Optional[str | Path]=None, label: str='Recorded artifact', kind: str='file', required: bool=True) -> Optional[Path]:
    if kind not in {'file', 'directory', 'either'}:
        raise ValueError(f'Unsupported artifact kind: {kind!r}')
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if required:
            raise FileNotFoundError(f'{label} was not recorded.')
        return None
    cleaned = clean_user_path(value)
    if not cleaned:
        if required:
            raise FileNotFoundError(f'{label} was not recorded.')
        return None
    recorded = Path(cleaned).expanduser()
    manifest_path = Path(manifest_path).expanduser().resolve()
    current_root = _manifest_run_root(manifest_path)
    candidates: list[Path] = [resolve_output_path(manifest_path, cleaned)]
    if not recorded.is_absolute():
        candidates.extend((manifest_path.parent / recorded, current_root / recorded))
    if recorded_root:
        old_root = Path(clean_user_path(recorded_root)).expanduser()
        try:
            candidates.append(current_root / recorded.relative_to(old_root))
        except ValueError:
            pass
    parts = recorded.parts
    anchor_index = next((index for index, part in enumerate(parts) if str(part).casefold() in _RUN_ARTIFACT_ANCHORS), None)
    if anchor_index is not None:
        candidates.append(current_root.joinpath(*parts[anchor_index:]))

    def matches(candidate: Path) -> bool:
        if kind == 'file':
            return candidate.is_file()
        if kind == 'directory':
            return candidate.is_dir()
        return candidate.exists()
    existing = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if matches(candidate):
            resolved = candidate.resolve()
            if resolved not in existing:
                existing.append(resolved)
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f'{label} is ambiguous after path rebasing. Candidates: {[str(path) for path in existing]}')
    if required:
        raise FileNotFoundError(f'{label} could not be found. Recorded path: {cleaned}; current run root: {current_root}')
    return None

def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.casefold()
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix in {'.gpkg', '.shp', '.geojson'}:
        return gpd.read_file(path)
    if suffix in {'.csv', '.txt'}:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported tabular input: {path}')

def _load_da_boundaries(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if path.suffix.casefold() == '.zip':
        with zipfile.ZipFile(path) as archive:
            shape_members = sorted((name for name in archive.namelist() if name.casefold().endswith('.shp')))
        if len(shape_members) != 1:
            raise RuntimeError(f'The dissemination-area ZIP must contain exactly one shapefile. Found: {shape_members}')
        source = f'/vsizip/{path.resolve().as_posix()}/{shape_members[0]}'
    else:
        source = str(path.resolve())
    frame = gpd.read_file(source)
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        raise RuntimeError('The dissemination-area input contains no valid geometries.')
    return frame

def _find_da_identifier(frame: pd.DataFrame) -> str:
    preferred = ('DAUID', 'DA_ID', 'DA21UID', 'dauid')
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in preferred:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    non_geometry = [column for column in frame.columns if column != 'geometry']
    unique = [column for column in non_geometry if frame[column].notna().all() and frame[column].astype(str).is_unique]
    if not unique:
        raise KeyError('A unique dissemination-area identifier could not be detected.')
    return str(unique[0])

def _coordinates(frame: pd.DataFrame, cfg: EvaluationConfig) -> np.ndarray:
    required = [cfg.longitude_column, cfg.latitude_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f'Coordinate columns are missing: {missing}')
    values = frame[required].apply(pd.to_numeric, errors='coerce')
    valid = values.notna().all(axis=1) & np.isfinite(values).all(axis=1)
    values = values.loc[valid]
    if values.empty:
        raise RuntimeError('No finite coordinates are available for spatial evaluation.')
    points = gpd.GeoDataFrame(values, geometry=gpd.points_from_xy(values[cfg.longitude_column], values[cfg.latitude_column]), crs=cfg.geographic_crs).to_crs(cfg.metric_crs)
    return np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])

def _coordinate_valid_mask(frame: pd.DataFrame, cfg: EvaluationConfig) -> pd.Series:
    values = frame[[cfg.longitude_column, cfg.latitude_column]].apply(pd.to_numeric, errors='coerce')
    return values.notna().all(axis=1) & np.isfinite(values).all(axis=1)

def _validate_dataset(name: str, frame: pd.DataFrame, cfg: EvaluationConfig) -> dict:
    if frame.empty:
        raise RuntimeError(f'Dataset {name!r} is empty.')
    required = {cfg.longitude_column, cfg.latitude_column, 'OCC_YEAR', 'OCC_MONTH', cfg.category_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f'Dataset {name!r} is missing required columns: {missing}')
    xy = _coordinates(frame, cfg)
    return {'dataset': name, 'output_scope': 'complete_table', 'rows': int(len(frame)), 'columns': int(len(frame.columns)), 'finite_coordinate_rows': int(len(xy)), 'finite_coordinate_rate': float(len(xy) / max(len(frame), 1))}

def _validate_spatial_dataset(name: str, frame: pd.DataFrame, cfg: EvaluationConfig) -> dict:
    if frame.empty:
        raise RuntimeError(f'Dataset {name!r} is empty.')
    xy = _coordinates(frame, cfg)
    return {'dataset': name, 'output_scope': 'coordinate_only', 'rows': int(len(frame)), 'columns': int(len(frame.columns)), 'finite_coordinate_rows': int(len(xy)), 'finite_coordinate_rate': float(len(xy) / max(len(frame), 1))}

def _sample_pair(reference: pd.DataFrame, synthetic: pd.DataFrame, seed: int, maximum_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(reference), len(synthetic), int(maximum_rows))
    if n < 2:
        raise RuntimeError('At least two rows are required in each evaluation table.')
    return (reference.sample(n=n, replace=False, random_state=int(seed)).reset_index(drop=True), synthetic.sample(n=n, replace=False, random_state=int(seed) + 1).reset_index(drop=True))

def _hellinger_categorical(reference: pd.Series, synthetic: pd.Series) -> float:
    left = reference.astype('string').fillna('<MISSING>').value_counts(normalize=True)
    right = synthetic.astype('string').fillna('<MISSING>').value_counts(normalize=True)
    support = left.index.union(right.index)
    p = left.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    return float(np.sqrt(np.square(np.sqrt(p) - np.sqrt(q)).sum()) / np.sqrt(2.0))

def _real_derived_numeric_edges(reference: pd.Series, bins: int=32) -> np.ndarray:
    values = pd.to_numeric(reference, errors='coerce').dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.nanmin(values) == np.nanmax(values):
        return np.array([-np.inf, np.inf])
    quantiles = np.linspace(0.0, 1.0, min(int(bins), len(np.unique(values))) + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = (-np.inf, np.inf)
    return edges

def _hellinger_numeric(reference: pd.Series, synthetic: pd.Series, edges: Optional[np.ndarray]=None) -> float:
    edges = _real_derived_numeric_edges(reference) if edges is None else edges
    left = pd.to_numeric(reference, errors='coerce').dropna().to_numpy(dtype=float)
    right = pd.to_numeric(synthetic, errors='coerce').dropna().to_numpy(dtype=float)
    left, right = (left[np.isfinite(left)], right[np.isfinite(right)])
    if not len(left) or not len(right):
        return float('nan')
    p = np.histogram(left, bins=edges)[0].astype(float)
    q = np.histogram(right, bins=edges)[0].astype(float)
    p /= max(p.sum(), 1.0)
    q /= max(q.sum(), 1.0)
    return float(np.sqrt(np.square(np.sqrt(p) - np.sqrt(q)).sum()) / np.sqrt(2.0))

def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype('string').fillna('<MISSING>'), right.astype('string').fillna('<MISSING>'))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / max(n, 1)
    rows, columns = table.shape
    corrected = max(0.0, phi2 - (columns - 1) * (rows - 1) / max(n - 1, 1))
    row_correction = rows - (rows - 1) ** 2 / max(n - 1, 1)
    column_correction = columns - (columns - 1) ** 2 / max(n - 1, 1)
    denominator = max(min(column_correction - 1, row_correction - 1), 1e-12)
    return float(np.sqrt(corrected / denominator))

def _association_difference(reference: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    numeric = [column for column in reference.columns.intersection(synthetic.columns) if pd.api.types.is_numeric_dtype(reference[column]) and reference[column].nunique(dropna=True) > 1]
    numeric = numeric[:20]
    pearson_difference = float('nan')
    if len(numeric) >= 2:
        left = reference[numeric].corr(method='pearson').fillna(0.0)
        right = synthetic[numeric].corr(method='pearson').fillna(0.0)
        pearson_difference = float(np.abs(left.to_numpy() - right.to_numpy()).mean())
    categorical = [column for column in reference.columns.intersection(synthetic.columns) if (not pd.api.types.is_numeric_dtype(reference[column]) or reference[column].nunique(dropna=True) <= 50) and 1 < reference[column].nunique(dropna=True) <= 100][:12]
    differences = []
    for left_index, left_column in enumerate(categorical):
        for right_column in categorical[left_index + 1:]:
            differences.append(abs(_cramers_v(reference[left_column], reference[right_column]) - _cramers_v(synthetic[left_column], synthetic[right_column])))
    return {'pearson_matrix_mean_absolute_difference': pearson_difference, 'cramers_v_matrix_mean_absolute_difference': float(np.mean(differences)) if differences else float('nan')}

def _safe_filename(value: str) -> str:
    return ''.join((character if character.isalnum() else '_' for character in str(value))).strip('_')

def _cramers_v_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(columns), dtype=float), index=list(columns), columns=list(columns))
    for left_index, left_column in enumerate(columns):
        for right_column in columns[left_index + 1:]:
            value = _cramers_v(frame[left_column], frame[right_column])
            matrix.loc[left_column, right_column] = value
            matrix.loc[right_column, left_column] = value
    return matrix

def _association_matrix_artifacts(panels: Mapping[str, pd.DataFrame], synthetic: Mapping[str, pd.DataFrame], columns: Sequence[str], artifacts: ArtifactDirectories, cfg: EvaluationConfig) -> str:
    rows = []
    plotted = 0
    for panel, reference in panels.items():
        for dataset, frame in synthetic.items():
            left, right = _sample_pair(reference, frame, cfg.random_seeds[0], cfg.maximum_equal_sample_rows)
            numeric = [column for column in columns if pd.api.types.is_numeric_dtype(left[column]) and left[column].nunique(dropna=True) > 1][:16]
            categorical = [column for column in columns if (not pd.api.types.is_numeric_dtype(left[column]) or left[column].nunique(dropna=True) <= 50) and 1 < left[column].nunique(dropna=True) <= 100][:12]
            matrices = []
            if len(numeric) >= 2:
                matrices.append(('Pearson', left[numeric].corr(method='pearson').fillna(0.0), right[numeric].corr(method='pearson').fillna(0.0)))
            if len(categorical) >= 2:
                matrices.append(('CramersV', _cramers_v_matrix(left, categorical), _cramers_v_matrix(right, categorical)))
            for matrix_type, real_matrix, synthetic_matrix in matrices:
                difference = (synthetic_matrix - real_matrix).abs()
                for left_column in real_matrix.index:
                    for right_column in real_matrix.columns:
                        rows.append({'panel': panel, 'dataset': dataset, 'matrix_type': matrix_type, 'variable_left': left_column, 'variable_right': right_column, 'real_value': float(real_matrix.loc[left_column, right_column]), 'synthetic_value': float(synthetic_matrix.loc[left_column, right_column]), 'absolute_difference': float(difference.loc[left_column, right_column])})
                if plotted >= cfg.maximum_plotted_synthetic_datasets:
                    continue
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for ax, matrix, title in zip(axes, (real_matrix, synthetic_matrix, difference), ('Real', 'Synthetic', 'Absolute difference')):
                    if title == 'Absolute difference':
                        image = ax.imshow(matrix, cmap='magma', vmin=0)
                    elif matrix_type == 'Pearson':
                        image = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    else:
                        image = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
                    ax.set_xticks(range(len(matrix.columns)))
                    ax.set_yticks(range(len(matrix.index)))
                    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
                    ax.set_yticklabels(matrix.index, fontsize=7)
                    ax.set_title(title)
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.suptitle(f'{matrix_type}: {dataset} — {panel}')
                fig.tight_layout()
                plt.close(fig)
                plotted += 1
    output = artifacts.tables / 'association_matrix_cells.csv'
    _write_table(pd.DataFrame(rows), output, index=False)
    return str(output)

def _controlled_by_design(dataset: str, column: str, cfg: EvaluationConfig, *, dataset_kind: str='') -> bool:
    exact_variant = any((token in dataset for token in ('GeoCTGAN-Full', 'GeoCTGAN-CapacityOnly', 'GeoCTGAN-PermutedEmbedding')))
    if column in {'OCC_YEAR', 'OCC_MONTH', cfg.category_column}:
        return dataset_kind in {'GeoCTGAN', 'Benchmark'} or dataset.startswith('GeoCTGAN-')
    if column == cfg.segment_column:
        return exact_variant
    return False

def evaluate_statistical_fidelity(real_train: pd.DataFrame, real_test: pd.DataFrame, synthetic: Mapping[str, pd.DataFrame], artifacts: ArtifactDirectories, cfg: EvaluationConfig, dataset_kinds: Optional[Mapping[str, str]]=None, progress_callback: Optional[InnerProgressCallback]=None) -> dict:
    excluded = {'geometry', 'EVENT_PK', 'SYNTHETIC_ID', '__POOL__', cfg.longitude_column, cfg.latitude_column}
    common = set(real_train.columns).intersection(real_test.columns)
    for frame in synthetic.values():
        common &= set(frame.columns)
    columns = sorted((column for column in common if column not in excluded))
    if not columns:
        raise RuntimeError('No common non-coordinate columns remain for statistical evaluation.')
    rows = []
    association_rows = []
    panels = {'held_out_generalization': real_test}
    work = len(panels) * len(synthetic) * len(cfg.random_seeds)
    with tqdm(total=work, desc='Statistical fidelity', unit='comparison') as bar:
        for panel, reference in panels.items():
            for dataset, frame in synthetic.items():
                for repeat_seed in cfg.random_seeds:
                    left, right = _sample_pair(reference, frame, repeat_seed, cfg.maximum_equal_sample_rows)
                    for column in columns:
                        if pd.api.types.is_numeric_dtype(left[column]):
                            score = _hellinger_numeric(left[column], right[column])
                            variable_type = 'numerical'
                        else:
                            score = _hellinger_categorical(left[column], right[column])
                            variable_type = 'categorical'
                        rows.append({'panel': panel, 'dataset': dataset, 'repeat_seed': repeat_seed, 'column': column, 'variable_type': variable_type, 'hellinger_distance': score, 'controlled_by_design': _controlled_by_design(dataset, column, cfg, dataset_kind=str(dataset_kinds.get(dataset, '')) if dataset_kinds is not None else ''), 'rows_per_table': len(left)})
                    association_rows.append({'panel': panel, 'dataset': dataset, 'repeat_seed': repeat_seed, **_association_difference(left[columns], right[columns])})
                    bar.update(1)
                    _notify_inner(progress_callback, bar.n, work, f'Hellinger/association: {panel} | {dataset} | seed {repeat_seed}')
    distances = pd.DataFrame(rows)
    associations = pd.DataFrame(association_rows)
    _write_table(distances, artifacts.tables / 'hellinger_distance_by_feature.csv', index=False)
    _write_table(associations, artifacts.tables / 'association_matrix_agreement.csv', index=False)
    learned = distances.loc[~distances['controlled_by_design']].copy()
    learned_summary = learned.groupby(['panel', 'dataset'], observed=True)['hellinger_distance'].agg(['median', 'mean', 'std', 'count']).reset_index()
    _write_table(learned_summary, artifacts.tables / 'learned_attribute_fidelity_summary.csv', index=False)
    sdmetrics_rows = []
    learned_columns = [column for column in columns if column not in {'OCC_YEAR', 'OCC_MONTH', cfg.category_column, cfg.segment_column}]
    association_cells = _association_matrix_artifacts(panels, synthetic, learned_columns, artifacts, cfg)
    try:
        try:
            from sdmetrics.reports import DiagnosticReport, QualityReport
            report_api = 'multi_table'
        except ImportError:
            from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
            report_api = 'single_table'
    except Exception as exc:
        raise
    else:
        sdmetrics_total = len(panels) * len(synthetic) * 2
        sdmetrics_current = 0
        for panel, reference in panels.items():
            for dataset, frame in tqdm(synthetic.items(), desc=f'SDMetrics {panel}', unit='dataset'):
                if not learned_columns:
                    continue
                left, right = _sample_pair(reference, frame, cfg.random_seeds[0], cfg.maximum_equal_sample_rows)
                left, right = (left[learned_columns], right[learned_columns])
                metadata = _infer_sdmetrics_metadata(left)
                for report_name, report_class in (('DiagnosticReport', DiagnosticReport), ('QualityReport', QualityReport)):
                    report = report_class()
                    try:
                        if report_api == 'multi_table':
                            report.generate(real_data={'events': left}, synthetic_data={'events': right}, metadata={'tables': {'events': metadata}}, verbose=False)
                        else:
                            report.generate(real_data=left, synthetic_data=right, metadata=metadata, verbose=False)
                        sdmetrics_rows.append({'panel': panel, 'dataset': dataset, 'report': report_name, 'status': 'success', 'score': float(report.get_score()), 'rows_per_table': int(len(left)), 'columns': int(len(learned_columns)), 'controlled_columns_excluded': True})
                        properties = report.get_properties()
                        if isinstance(properties, pd.DataFrame):
                            safe_dataset = ''.join((character if character.isalnum() else '_' for character in dataset)).strip('_')
                            _write_table(properties, artifacts.tables / f'sdmetrics_{report_name}_{panel}_{safe_dataset}_properties.csv', index=False)
                    except Exception as exc:
                        raise
                    sdmetrics_current += 1
                    _notify_inner(progress_callback, sdmetrics_current, sdmetrics_total, f'SDMetrics: {panel} | {dataset} | {report_name}')
    sdmetrics = pd.DataFrame(sdmetrics_rows)
    _write_table(sdmetrics, artifacts.tables / 'sdmetrics_quality.csv', index=False)
    if not learned.empty:
        plotted = learned.loc[learned['dataset'].isin(list(dict.fromkeys(learned['dataset']))[:cfg.maximum_plotted_synthetic_datasets])]
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = []
        series = []
        for key, group in plotted.groupby(['panel', 'dataset'], sort=False):
            labels.append(f'{key[1]}\n{key[0]}')
            series.append(group['hellinger_distance'].dropna().to_numpy())
        if series:
            ax.boxplot(series, tick_labels=labels, showfliers=False)
            ax.set_ylabel('Hellinger distance')
            ax.set_title('Learned attribute fidelity')
            ax.tick_params(axis='x', labelrotation=45)
            fig.tight_layout()
        plt.close(fig)
    return {'columns': columns, 'hellinger_table': str(artifacts.tables / 'hellinger_distance_by_feature.csv'), 'association_table': str(artifacts.tables / 'association_matrix_agreement.csv'), 'association_cells': association_cells, 'learned_summary': str(artifacts.tables / 'learned_attribute_fidelity_summary.csv'), 'sdmetrics_quality': str(artifacts.tables / 'sdmetrics_quality.csv')}

def evaluate_temporal_fidelity(real_train: pd.DataFrame, real_test: pd.DataFrame, synthetic: Mapping[str, pd.DataFrame], artifacts: ArtifactDirectories) -> dict:
    fields = [field for field in ('OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'OCC_HOUR') if field in real_train]
    rows = []
    for panel, reference in {'held_out_generalization': real_test}.items():
        for dataset, frame in synthetic.items():
            for field in fields:
                rows.append({'panel': panel, 'dataset': dataset, 'field': field, 'hellinger_distance': _hellinger_categorical(reference[field], frame[field]), 'controlled_by_design': field in {'OCC_YEAR', 'OCC_MONTH'}})
    table = pd.DataFrame(rows)
    _write_table(table, artifacts.tables / 'temporal_fidelity.csv', index=False)
    plotted_names = [name for name in synthetic if name.startswith(('GeoCTGAN-Full |','GeoCTGAN-Priv |'))]
    if 'OCC_MONTH' in real_train:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (panel, reference) in zip(axes, (('Training distribution', real_train), ('Held-out generalization', real_test))):
            ref = reference['OCC_MONTH'].value_counts(normalize=True).sort_index()
            ax.plot(ref.index, ref.values, marker='o', color='black', label='Real')
            for name in plotted_names:
                values = synthetic[name]['OCC_MONTH'].value_counts(normalize=True).sort_index()
                ax.plot(values.index, values.values, marker='.', label=name)
            ax.set_title(panel)
            ax.set_xlabel('Month')
            ax.set_ylabel('Proportion')
        axes[-1].legend(fontsize=7, loc='best')
        fig.tight_layout()
        plt.close(fig)
    month_category_rows = []
    category_column = 'MCI_CATEGO'
    temporal_datasets = {'Real held-out': real_test, **{name: synthetic[name] for name in plotted_names}}
    if all(({'OCC_MONTH', category_column}.issubset(frame.columns) for frame in temporal_datasets.values())):
        for dataset, frame in temporal_datasets.items():
            counts = frame.groupby(['OCC_MONTH', category_column], observed=True).size().rename('count').reset_index()
            counts['proportion'] = counts['count'] / max(counts['count'].sum(), 1)
            counts.insert(0, 'dataset', dataset)
            month_category_rows.extend(counts.to_dict(orient='records'))
        month_category = pd.DataFrame(month_category_rows)
        _write_table(month_category, artifacts.tables / 'month_category_distribution.csv', index=False)
        categories = sorted(real_train[category_column].dropna().astype(str).unique())
        rows_count = len(temporal_datasets)
        fig, axes = plt.subplots(rows_count, 1, figsize=(12, max(3.2 * rows_count, 4)), squeeze=False)
        for ax, (dataset, frame) in zip(axes.ravel(), temporal_datasets.items()):
            pivot = frame.assign(**{category_column: frame[category_column].astype(str), 'OCC_MONTH': pd.to_numeric(frame['OCC_MONTH'], errors='coerce')}).groupby([category_column, 'OCC_MONTH'], observed=True).size().unstack(fill_value=0).reindex(index=categories, columns=range(1, 13), fill_value=0)
            values = pivot.to_numpy(dtype=float)
            values /= max(values.sum(), 1.0)
            image = ax.imshow(values, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories, fontsize=8)
            ax.set_xticks(range(12))
            ax.set_xticklabels(range(1, 13))
            ax.set_title(dataset)
            fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
        axes.ravel()[-1].set_xlabel('Month')
        fig.suptitle('Month-by-category event distribution')
        fig.tight_layout()
        plt.close(fig)
    else:
        raise ValueError('The temporal evaluation is missing required fields.')
    hour_rows = []
    if all(('OCC_HOUR' in frame.columns for frame in temporal_datasets.values())):
        fig, ax = plt.subplots(figsize=(11, 5))
        for dataset, frame in temporal_datasets.items():
            hours = pd.to_numeric(frame['OCC_HOUR'], errors='coerce')
            distribution = hours.value_counts(normalize=True).reindex(range(24), fill_value=0.0).sort_index()
            ax.plot(distribution.index, distribution.values, marker='.' if dataset.startswith('Real ') else None, label=dataset)
            hour_rows.extend(({'dataset': dataset, 'hour': int(hour), 'proportion': float(value)} for hour, value in distribution.items()))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Proportion')
        ax.set_title('Hourly event distribution')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        plt.close(fig)
    _write_table(pd.DataFrame(hour_rows), artifacts.tables / 'hour_distribution.csv', index=False)
    return {'temporal_table': str(artifacts.tables / 'temporal_fidelity.csv'), 'month_category_table': str(artifacts.tables / 'month_category_distribution.csv'), 'hour_table': str(artifacts.tables / 'hour_distribution.csv')}

def _nearest_neighbour_distances(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return np.array([], dtype=float)
    distances, _ = KDTree(xy).query(xy, k=2)
    return distances[:, 1]

def _grid_counts(xy: np.ndarray, grid_size: float, origin: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    gx = np.floor((xy[:, 0] - origin[0]) / grid_size).astype(int)
    gy = np.floor((xy[:, 1] - origin[1]) / grid_size).astype(int)
    valid = (gx >= 0) & (gy >= 0) & (gx < shape[0]) & (gy < shape[1])
    flat = gx[valid] * shape[1] + gy[valid]
    return np.bincount(flat, minlength=shape[0] * shape[1]).astype(float)

def _exact_top_q_indices(counts: np.ndarray, q: float, count: Optional[int]=None) -> np.ndarray:
    occupied = np.flatnonzero(counts > 0)
    if not len(occupied):
        return np.array([], dtype=int)
    number = max(1, int(math.ceil(len(occupied) * q))) if count is None else int(count)
    number = min(number, len(occupied))
    order = np.lexsort((occupied, -counts[occupied]))
    return occupied[order[:number]]

def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left, right = (set(map(int, left)), set(map(int, right)))
    union = left.union(right)
    return float(len(left.intersection(right)) / len(union)) if union else 1.0

def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float('nan')
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(pearsonr(left, right).statistic)

def _kde_bandwidth(reference_xy: np.ndarray) -> float:
    n = max(len(reference_xy), 2)
    scale = float(np.mean(np.std(reference_xy, axis=0, ddof=1)))
    return float(np.clip(scale * n ** (-1.0 / 6.0), 25.0, 2000.0))

def _kde_surface(xy: np.ndarray, grid_points: np.ndarray, bandwidth: float, maximum_fit_rows: int, seed: int) -> np.ndarray:
    if len(xy) > maximum_fit_rows:
        rng = np.random.default_rng(seed)
        xy = xy[rng.choice(len(xy), maximum_fit_rows, replace=False)]
    model = KernelDensity(kernel='gaussian', bandwidth=bandwidth, algorithm='ball_tree')
    model.fit(xy)
    density = np.exp(model.score_samples(grid_points))
    return density / max(density.sum(), np.finfo(float).eps)

def _spatial_extent(train_xy: np.ndarray, da_metric: Optional[gpd.GeoDataFrame]) -> tuple[float, float, float, float]:
    if da_metric is not None:
        bounds = da_metric.total_bounds
        return tuple(map(float, bounds))
    lower = np.quantile(train_xy, 0.001, axis=0)
    upper = np.quantile(train_xy, 0.999, axis=0)
    pad = np.maximum((upper - lower) * 0.02, 1.0)
    return (float(lower[0] - pad[0]), float(lower[1] - pad[1]), float(upper[0] + pad[0]), float(upper[1] + pad[1]))

def _da_counts(frame: pd.DataFrame, da_metric: gpd.GeoDataFrame, da_id: str, cfg: EvaluationConfig) -> pd.Series:
    identifiers = da_metric[da_id].astype('string')
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise RuntimeError(f'DA boundary identifier {da_id} must be complete and unique before event counts can be aligned.')
    xy = _coordinates(frame, cfg)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=cfg.metric_crs)
    joined = gpd.sjoin(points, da_metric[[da_id, 'geometry']], how='left', predicate='within')
    counts = joined[da_id].astype('string').value_counts()
    aligned = counts.reindex(identifiers, fill_value=0).astype(float)
    aligned.index = pd.Index(identifiers.astype(str), name=da_id)
    return aligned

def _spatial_autocorrelation(counts_by_dataset: Mapping[str, np.ndarray], da_metric: gpd.GeoDataFrame, da_id: str, cfg: EvaluationConfig, progress: Optional[InnerProgressCallback]=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from esda.moran import Moran, Moran_Local
        from libpysal.weights import Queen
    except Exception:
        raise
    weights = Queen.from_dataframe(da_metric, ids=da_metric[da_id].astype(str).tolist(), silence_warnings=True)
    weights.transform = 'r'
    global_rows = []
    labels_by_dataset = {}
    label_rows = []
    for index, (dataset, values) in enumerate(counts_by_dataset.items()):
        _notify_inner(progress, index, len(counts_by_dataset), f'Moran/LISA: {dataset}')
        values = np.asarray(values, dtype=float)
        moran = Moran(values, weights, permutations=cfg.lisa_permutations)
        local_kwargs = {}
        if 'alternative' in inspect.signature(Moran_Local.__init__).parameters:
            local_kwargs['alternative'] = 'directed'
        local = Moran_Local(values, weights, permutations=cfg.lisa_permutations, seed=cfg.random_seeds[index % len(cfg.random_seeds)], n_jobs=1, **local_kwargs)
        labels = np.full(len(values), 'NS', dtype=object)
        significant = local.p_sim < cfg.alpha
        mapping = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        for quadrant, label in mapping.items():
            labels[significant & (local.q == quadrant)] = label
        labels_by_dataset[dataset] = labels
        label_rows.extend(({'dataset': dataset, 'da_identifier': str(da_identifier), 'lisa_cluster': str(label)} for da_identifier, label in zip(da_metric[da_id].astype(str), labels)))
        global_rows.append({'dataset': dataset, 'moran_i': float(moran.I), 'permutation_p_value': float(moran.p_sim), 'islands': int(len(weights.islands))})
    agreement_rows = []
    for reference_name in ('Real training', 'Real held-out'):
        if reference_name not in labels_by_dataset:
            continue
        reference = labels_by_dataset[reference_name]
        for dataset, labels in labels_by_dataset.items():
            if dataset.startswith('Real '):
                continue
            agreement_rows.append({'reference': reference_name, 'dataset': dataset, 'exact_label_agreement': float(np.mean(reference == labels)), **{f'{cluster.lower()}_jaccard': _jaccard(np.flatnonzero(reference == cluster), np.flatnonzero(labels == cluster)) for cluster in ['HH','HL','LL','LH','NS']}})
    return (pd.DataFrame(global_rows), pd.DataFrame(agreement_rows), pd.DataFrame(label_rows))

def _identify_marginal_columns(marginal: pd.DataFrame, da_id: str) -> tuple[str, list[str], str]:
    lookup = {str(column).casefold(): str(column) for column in marginal.columns}
    marginal_id = next((lookup[name.casefold()] for name in (da_id, 'DAUID', 'DA21UID', 'DA_ID') if name.casefold() in lookup), None)
    if marginal_id is None:
        raise KeyError('The marginalization table has no dissemination-area identifier.')
    population_aliases = ('population', 'population2021', 'population_2021', 'population2016', 'population_2016', 'total_population', 'pop', 'pop2021', 'pop_2021', 'pop2016', 'pop_2016')
    population = next((lookup[name] for name in population_aliases if name in lookup and pd.api.types.is_numeric_dtype(marginal[lookup[name]])), None)
    if population is None:
        population = next((column for column in marginal.columns if 'population' in str(column).strip().casefold() and pd.api.types.is_numeric_dtype(marginal[column])), None)
    requested = ['households_dwellings_DA21', 'material_resources_DA21', 'age_labourforce_DA21', 'racialized_NC_pop_DA21']
    predictors = [lookup[column.casefold()] for column in requested if column.casefold() in lookup]
    if len(predictors) < 2:
        numeric = [column for column in marginal.columns if column != marginal_id and column != population and pd.api.types.is_numeric_dtype(marginal[column]) and (marginal[column].nunique(dropna=True) > 1)]
        predictors = numeric[:4]
    if len(predictors) < 2 or population is None:
        raise KeyError('The marginalization table must provide at least two numerical predictors and one numerical population column. Recognized population names include Pop2021, Population2021, total_population and population.')
    return (marginal_id, predictors, population)

def evaluate_downstream_regression(real_test: pd.DataFrame, synthetic: Mapping[str, pd.DataFrame], da_boundaries: gpd.GeoDataFrame, marginal_path: str | Path, artifacts: ArtifactDirectories, cfg: EvaluationConfig, progress_callback: Optional[InnerProgressCallback]=None) -> dict:
    try:
        import statsmodels.api as sm
    except Exception as exc:
        raise ImportError('statsmodels is required for downstream regression.') from exc
    da_metric = da_boundaries.to_crs(cfg.metric_crs)
    da_id = _find_da_identifier(da_metric)
    marginal = pd.read_csv(marginal_path)
    marginal_id, predictors, population = _identify_marginal_columns(marginal, da_id)
    marginal[marginal_id] = marginal[marginal_id].astype(str)
    covariates = da_metric[[da_id, 'geometry']].copy()
    covariates[da_id] = covariates[da_id].astype(str)
    if covariates[da_id].isna().any() or covariates[da_id].duplicated().any():
        raise RuntimeError(f'DA boundaries must contain one complete, unique row per {da_id}.')
    if marginal[marginal_id].isna().any() or marginal[marginal_id].duplicated().any():
        raise RuntimeError(f'The marginalization table must contain one complete, unique row per {marginal_id}.')
    covariates = covariates.merge(marginal[[marginal_id, population, *predictors]], left_on=da_id, right_on=marginal_id, how='left', validate='one_to_one', sort=False)
    for column in [population, *predictors]:
        covariates[column] = pd.to_numeric(covariates[column], errors='coerce')
    complete = covariates[[population, *predictors]].notna().all(axis=1) & covariates[population].gt(0)
    covariates = covariates.loc[complete].copy()
    means = covariates[predictors].mean()
    standard_deviations = covariates[predictors].std(ddof=0).replace(0, np.nan)
    standardized = (covariates[predictors] - means) / standard_deviations
    if standardized.isna().any().any():
        raise RuntimeError('One or more downstream predictors have zero variance.')
    design = sm.add_constant(standardized, has_constant='add')
    datasets = {'Real held-out': real_test, **synthetic}
    counts_by_dataset = {dataset: _da_counts(frame, da_metric, da_id, cfg) for dataset, frame in datasets.items()}
    comparison_event_total = float(counts_by_dataset['Real held-out'].sum())
    if comparison_event_total <= 0:
        raise RuntimeError('The held-out reference contains no events within the DA support.')
    volume_metadata = {}
    volume_standardized_counts = {}
    selected_da_ids = covariates[da_id].astype(str)
    for dataset, all_counts in counts_by_dataset.items():
        original_event_total = float(all_counts.sum())
        if original_event_total <= 0:
            raise RuntimeError(f'{dataset} contains no events within the DA support.')
        exposure_factor = comparison_event_total / original_event_total
        volume_standardized_counts[dataset] = all_counts.reindex(selected_da_ids, fill_value=0).to_numpy(dtype=float) * exposure_factor
        volume_metadata[dataset] = {'original_event_total_within_da': original_event_total, 'comparison_event_total': comparison_event_total, 'exposure_factor': exposure_factor}
    coefficient_rows = []
    fit_rows = []
    for regression_index, (dataset, frame) in enumerate(tqdm(datasets.items(), desc='Downstream regression', unit='dataset'), start=1):
        counts = volume_standardized_counts[dataset]
        rate = counts / covariates[population].to_numpy(dtype=float) * 1000.0
        model = sm.OLS(rate, design).fit(cov_type='HC3')
        fit_rows.append({'dataset': dataset, 'model': 'OLS-HC3', 'n_das': int(model.nobs), 'r_squared': float(model.rsquared), 'adjusted_r_squared': float(model.rsquared_adj), 'aic': float(model.aic), **volume_metadata[dataset]})
        for term in model.params.index:
            coefficient_rows.append({'dataset': dataset, 'model': 'OLS-HC3', 'term': term, 'coefficient': float(model.params[term]), 'standard_error': float(model.bse[term]), 'p_value': float(model.pvalues[term])})
        _notify_inner(progress_callback, regression_index, len(datasets), f'OLS-HC3: {dataset}')
    try:
        from libpysal.weights import Queen
        from spreg import ML_Error, ML_Lag
    except Exception as exc:
        raise
    else:
        weights = Queen.from_dataframe(covariates, ids=covariates[da_id].astype(str).tolist(), silence_warnings=True)
        weights.transform = 'r'
        x = standardized.to_numpy(dtype=float)
        for spatial_index, (dataset, frame) in enumerate(tqdm(datasets.items(), desc='Spatial regression', unit='dataset'), start=1):
            counts = volume_standardized_counts[dataset]
            rate = (counts / covariates[population].to_numpy(dtype=float) * 1000.0).reshape(-1, 1)
            for model_name, model_class in (('Spatial error', ML_Error), ('Spatial lag', ML_Lag)):
                try:
                    fitted = model_class(rate, x, w=weights, name_y='events_per_1000', name_x=predictors)
                    names = list(getattr(fitted, 'name_x', ['CONSTANT', *predictors]))
                    parameters = np.asarray(fitted.betas).ravel()
                    standard_errors = np.asarray(getattr(fitted, 'std_err', np.full(len(parameters), np.nan))).ravel()
                    z_statistics = list(getattr(fitted, 'z_stat', [(np.nan, np.nan)] * len(parameters)))
                    if len(names) < len(parameters):
                        spatial_term = 'lambda' if model_name == 'Spatial error' else 'rho'
                        names.extend([spatial_term] * (len(parameters) - len(names)))
                    for index, parameter in enumerate(parameters):
                        p_value = float(z_statistics[index][1]) if index < len(z_statistics) else float('nan')
                        coefficient_rows.append({'dataset': dataset, 'model': model_name, 'term': names[index], 'coefficient': float(parameter), 'standard_error': float(standard_errors[index]) if index < len(standard_errors) else float('nan'), 'p_value': p_value})
                    fit_rows.append({'dataset': dataset, 'model': model_name, 'n_das': int(len(rate)), 'r_squared': float(getattr(fitted, 'pr2', np.nan)), 'adjusted_r_squared': float('nan'), 'aic': float(getattr(fitted, 'aic', np.nan)), **volume_metadata[dataset]})
                except Exception as exc:
                    raise
            _notify_inner(progress_callback, spatial_index, len(datasets), f'Spatial regression: {dataset}')
    coefficients = pd.DataFrame(coefficient_rows)
    fits = pd.DataFrame(fit_rows)
    _write_table(coefficients, artifacts.tables / 'downstream_regression_coefficients.csv', index=False)
    _write_table(fits, artifacts.tables / 'downstream_regression_fit.csv', index=False)
    return {'coefficient_table': str(artifacts.tables / 'downstream_regression_coefficients.csv'), 'fit_table': str(artifacts.tables / 'downstream_regression_fit.csv')}

def _infer_sdmetrics_metadata(frame: pd.DataFrame) -> dict:
    metadata = {'columns': {}}
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            sdtype = 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            sdtype = 'datetime'
        elif pd.api.types.is_numeric_dtype(frame[column]):
            sdtype = 'numerical'
        else:
            sdtype = 'categorical'
        metadata['columns'][column] = {'sdtype': sdtype}
    return metadata



def _load_generation_registry(summary_path, benchmark_manifest_path=None):
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    root = _manifest_run_root(summary_path)
    def resolve(value):
        return resolve_recorded_artifact(value, manifest_path=summary_path, recorded_root=summary.get('output_dir'))
    real_train = _read_table(resolve(summary['snapped_train']))
    real_test = _read_table(resolve(summary['snapped_test']))
    synthetic, rows = {}, []
    for seed_summary in summary['seed_summaries']:
        seed = int(seed_summary['seed'])
        candidates = {f'GeoCTGAN-Full | seed {seed}': seed_summary.get('synthetic_internal_exact_quota'), f'GeoCTGAN-Priv | seed {seed}': seed_summary.get('public_profile')}
        if seed_summary.get('minimum_support'):
            candidates[f'GeoCTGAN-MinimumSupport | seed {seed}'] = seed_summary['minimum_support']
        if seed_summary.get('ablation_manifest'):
            for row in pd.read_csv(resolve(seed_summary['ablation_manifest'])).to_dict('records'):
                candidates[f"GeoCTGAN-{row['variant']} | seed {seed}"] = row['output_path']
        for label, value in candidates.items():
            if value:
                synthetic[label] = _read_table(resolve(value))
                rows.append({'dataset': label, 'kind': 'GeoCTGAN', 'seed': seed, 'rows': len(synthetic[label])})
        if 'artifact_directories' in seed_summary:
            seed_summary['artifact_directories'] = {name: str(resolve_recorded_artifact(value, manifest_path=summary_path, recorded_root=summary.get('output_dir'), kind='directory', required=False) or root/name.title()) for name,value in seed_summary['artifact_directories'].items()}
    if benchmark_manifest_path:
        manifest = pd.read_csv(benchmark_manifest_path)
        for row in manifest.to_dict('records'):
            if row.get('status', 'success') != 'success' or not row.get('output_path'):
                continue
            method = row.get('generator_requested',row.get('method'))
            label = f"{method} | seed {int(row['seed'])}"
            path = resolve_recorded_artifact(row['output_path'],manifest_path=benchmark_manifest_path)
            synthetic[label] = _read_table(path)
            rows.append({'dataset': label, 'kind': 'Benchmark', 'seed': int(row['seed']), 'rows': len(synthetic[label])})
    summary['output_dir'] = str(root)
    return summary, real_train, real_test, synthetic, pd.DataFrame(rows)

def evaluate_spatial_fidelity(real_train, real_test, synthetic, artifacts, cfg, da_boundaries, progress_callback=None):
    from .public_figures import spatial_metrics
    return spatial_metrics(real_train, real_test, synthetic, artifacts, cfg, da_boundaries)

def evaluate_privacy(real_train, real_test, synthetic, artifacts, cfg, progress_callback=None):
    from .public_figures import privacy_metrics
    return privacy_metrics(real_train, real_test, synthetic, artifacts, cfg)

def evaluate_training_diagnostics(summary, artifacts, summary_path=None):
    from .public_figures import save_figure
    records=[]
    for seed in summary.get('seed_summaries',[]):
        value=seed.get('artifact_directories',{}).get('logs')
        if not value:
            continue
        logs=resolve_output_path(summary_path,value) if summary_path is not None else Path(value)
        for path in sorted(logs.rglob('losses_*.csv')) if logs.is_dir() else []:
            frame=pd.read_csv(path)
            frame.insert(0,'seed',int(seed['seed']))
            records.append(frame)
    if not records:
        return {}
    losses=pd.concat(records,ignore_index=True)
    _write_table(losses,artifacts.tables/'training_losses.csv')
    epoch=next((c for c in losses if str(c).casefold()=='epoch'),None)
    columns=[next((c for c in losses if kind in str(c).casefold() and 'loss' in str(c).casefold() and pd.api.types.is_numeric_dtype(losses[c])),None) for kind in ('generator','discriminator')]
    if epoch is None or any(c is None for c in columns):
        return {}
    fig,axes=plt.subplots(3,1,figsize=(9,10),gridspec_kw={'height_ratios':[1,1,.8]})
    stats=[]
    for index,column in enumerate(columns):
        smoothed=[]
        for seed,frame in losses.groupby('seed',sort=True):
            frame=frame.sort_values(epoch)
            raw=frame.set_index(epoch)[column]
            smooth=raw.rolling(9,center=True,min_periods=1).mean()
            line=axes[index].plot(smooth.index,smooth,lw=1.3,label=f'Seed {seed}')[0]
            axes[index].plot(raw.index,raw,color=line.get_color(),alpha=.25,lw=.65)
            smoothed.append(smooth.rename(seed))
            tail=raw.iloc[int(np.floor(.8*len(raw))):]
            stats.append({'seed':seed,'loss':column,'standard_deviation':float(tail.std(ddof=1))})
        matrix=pd.concat(smoothed,axis=1)
        mean=matrix.mean(axis=1)
        sd=matrix.std(axis=1,ddof=1)
        axes[index].plot(mean.index,mean,color='#222222',lw=1.3,label='Across-seed mean')
        axes[index].fill_between(mean.index,mean-sd,mean+sd,color='#aab3bf',alpha=.22,label=f'Across-seed ±1 SD (n = {len(smoothed)})')
        axes[index].set_ylabel(column,fontsize=12)
        axes[index].set_xlabel('Epoch',fontsize=12)
        axes[index].grid(color='#dddddd',lw=.6)
    stats=pd.DataFrame(stats)
    for i,(seed,group) in enumerate(stats.groupby('seed',sort=True)):
        values=group.set_index('loss')['standard_deviation'].reindex(columns)
        axes[2].plot(values,[i,i],color='#999999',lw=1.3)
        for value,marker,color in zip(values,['o','s'],['#542788','#e69f00']):
            axes[2].scatter(value,i,marker=marker,color=color,edgecolor='#222222',s=55,zorder=3)
            axes[2].annotate(f'{value:.3f}',(value,i),xytext=(5,7),textcoords='offset points',fontsize=10)
    axes[2].set_yticks(range(stats['seed'].nunique()),[f'Seed {s}' for s in sorted(stats['seed'].unique())])
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Loss standard deviation over final 20% of epochs',fontsize=12)
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([],[],marker='o',color='#542788',linestyle='none',label='Generator loss'),Line2D([],[],marker='s',color='#e69f00',linestyle='none',label='Discriminator loss'),Line2D([],[],color='#999999',label='Paired estimates for each seed')],loc='upper center',bbox_to_anchor=(.5,1.25),ncol=3,frameon=False,fontsize=9)
    for i,ax in enumerate(axes):
        ax.text(-.1,1.04,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=14)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95),h_pad=3)
    save_figure(fig,artifacts.figures,'training_losses')
    _write_table(stats,artifacts.tables/'training_loss_variability.csv')
    return {'loss_table':'Tables/training_losses.csv'}

def run_evaluation(generation_run, *, output_parent, evaluation_dir=None, da_boundaries=None, marginalization_csv=None, benchmark_manifest=None, config=None, progress_callback=None):
    cfg=EvaluationConfig() if config is None else config
    summary_path=_resolve_summary_path(generation_run)
    root=Path(evaluation_dir).expanduser().resolve() if evaluation_dir else create_unique_run_directory(output_parent,'Evaluation')
    root.mkdir(parents=True,exist_ok=True)
    artifacts=ArtifactDirectories.create(root)
    summary,train,test,synthetic,registry=_load_generation_registry(summary_path,benchmark_manifest)
    _validate_dataset('Real training',train,cfg)
    _validate_dataset('Real held-out',test,cfg)
    for name,frame in synthetic.items():
        _validate_dataset(name,frame,cfg)
    da=_load_da_boundaries(da_boundaries) if da_boundaries else None
    comparison={name:frame for name,frame in synthetic.items() if 'MinimumSupport' not in name}
    kinds=registry.set_index('dataset')['kind'].to_dict()
    evaluate_training_diagnostics(summary,artifacts,summary_path)
    evaluate_statistical_fidelity(train,test,comparison,artifacts,cfg,dataset_kinds=kinds)
    evaluate_temporal_fidelity(train,test,comparison,artifacts)
    if cfg.privacy_profile == 'toronto':
        from .reported_figures import generate_reported_figures
        augmented = {name:frame.copy() for name,frame in comparison.items() if name.startswith(('GeoCTGAN-Full |','GeoCTGAN-Priv |'))}
        for seed_summary in summary['seed_summaries']:
            label = f"GeoCTGAN-Priv | seed {int(seed_summary['seed'])}"
            if label not in augmented:
                continue
            value = seed_summary.get('minimum_support') or seed_summary.get('public_assignment_source')
            if not value:
                raise ValueError('The restricted evaluation reference is missing.')
            internal = _read_table(resolve_recorded_artifact(value,manifest_path=summary_path))
            public = augmented[label]
            missing = [column for column in internal if column not in public]
            augmented[label] = public.merge(internal[['SYNTHETIC_ID',*missing]],on='SYNTHETIC_ID',how='left',validate='one_to_one',sort=False)
        generate_reported_figures(test,augmented,artifacts,cfg)
    evaluate_spatial_fidelity(train,test,comparison,artifacts,cfg,da)
    if da is not None and marginalization_csv is not None:
        regression={name:frame for name,frame in comparison.items() if name.startswith(('GeoCTGAN-Full |','GeoCTGAN-Priv |'))}
        evaluate_downstream_regression(test,regression,da,marginalization_csv,artifacts,cfg)
    evaluate_privacy(train,test,synthetic,artifacts,cfg)
    from .public_figures import supplementary_summary_figures
    if cfg.privacy_profile == 'toronto':
        supplementary_summary_figures(artifacts)
    else:
        from .public_figures import ottawa_summary
        ottawa_summary(artifacts)
    saved={'datasets':len(comparison),'figures':'Figures','tables':'Tables'}
    atomic_json_dump(saved,artifacts.metadata/'evaluation_summary.json')
    return {**saved,'evaluation_dir':str(root),'evaluation_summary':str(artifacts.metadata/'evaluation_summary.json')}
