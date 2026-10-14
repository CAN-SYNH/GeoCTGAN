from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from .config import Config
from .utils import atomic_json_dump
logger = logging.getLogger(__name__)

def _project_xy(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, pd.Index]:
    valid = frame.dropna(subset=[cfg.COL_LON, cfg.COL_LAT]).copy()
    if valid.empty:
        return (np.empty((0, 2), dtype=float), valid.index)
    geodata = gpd.GeoDataFrame(valid, geometry=gpd.points_from_xy(valid[cfg.COL_LON], valid[cfg.COL_LAT]), crs=cfg.CRS_GEO).to_crs(cfg.CRS_METRIC)
    return (np.column_stack([geodata.geometry.x, geodata.geometry.y]), geodata.index)

def apply_privacy_gates(cfg: Config, train_parquet: str, synthetic_parquet: str, out_dir: str, *, capacity_mode: str) -> Tuple[str, str]:
    if capacity_mode == 'exact_quota' and (not cfg.ALLOW_EXACT_QUOTA_PUBLIC_RELEASE):
        raise RuntimeError('Exact-quota public release is blocked. It can reveal fine-grained training year-month-category segment margins. Use the upper-quota public profile or set ALLOW_EXACT_QUOTA_PUBLIC_RELEASE=true only after a documented disclosure-risk assessment.')
    real = pd.read_parquet(train_parquet)
    synthetic = pd.read_parquet(synthetic_parquet)
    for frame_name, frame in [('real', real), ('synthetic', synthetic)]:
        required = {cfg.COL_SEG, cfg.COL_LON, cfg.COL_LAT}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f'{frame_name} table is missing privacy columns: {missing}')
    real[cfg.COL_SEG] = real[cfg.COL_SEG].astype(str)
    synthetic[cfg.COL_SEG] = synthetic[cfg.COL_SEG].astype(str)
    segment_counts = real[cfg.COL_SEG].value_counts()
    supported = set(segment_counts[segment_counts >= cfg.MIN_REAL_SEGMENT_SUPPORT].index)
    minimum_support = synthetic.loc[synthetic[cfg.COL_SEG].isin(supported)].copy()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    minimum_path = output_dir / 'synthetic_points_MINIMUM_REAL_SUPPORT_INTERNAL.parquet'
    minimum_support.to_parquet(minimum_path, index=False)
    real_xy, _ = _project_xy(real, cfg)
    synthetic_xy, synthetic_index = _project_xy(minimum_support, cfg)
    if len(real_xy) == 0:
        raise RuntimeError('Privacy gate failed closed: training coordinates are unavailable.')
    if len(synthetic_xy) == 0:
        raise RuntimeError('Privacy gate failed closed: no valid synthetic coordinates remain.')
    tree = KDTree(real_xy)
    distances, _ = tree.query(synthetic_xy, k=1)
    distances = distances.ravel()
    keep = distances > cfg.DCR_THRESH_M
    public_internal = minimum_support.loc[synthetic_index[keep]].copy()
    missing_public = sorted(set(cfg.PUBLIC_COLUMNS).difference(public_internal.columns))
    if missing_public:
        raise KeyError(f'The public release whitelist contains required columns absent from the synthetic output: {missing_public}. Derive them before release or remove them consistently from the frozen schema.')
    public = public_internal[cfg.PUBLIC_COLUMNS].copy()
    public_path = output_dir / 'synthetic_points_PUBLIC_PROFILE.parquet'
    public.to_parquet(public_path, index=False)
    telemetry = {'capacity_mode': capacity_mode, 'minimum_real_support_k': cfg.MIN_REAL_SEGMENT_SUPPORT, 'spatial_proximity_threshold_m': cfg.DCR_THRESH_M, 'real_rows': len(real), 'synthetic_raw_rows': len(synthetic), 'minimum_support_rows': len(minimum_support), 'public_rows': len(public), 'minimum_support_retention': len(minimum_support) / max(len(synthetic), 1), 'public_retention': len(public) / max(len(synthetic), 1), 'pre_gate_distance_quantiles_m': {str(quantile): float(np.quantile(distances, quantile)) for quantile in [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]}, 'post_gate_minimum_distance_m': float(distances[keep].min()) if keep.any() else None, 'public_columns': cfg.PUBLIC_COLUMNS}
    atomic_json_dump(telemetry, output_dir / 'privacy_gate_telemetry.json')
    return (str(minimum_path), str(public_path))
