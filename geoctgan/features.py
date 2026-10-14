from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .config import Config
from .temporal import condition_columns, pool_series
from .utils import atomic_json_dump

def encode_cyclic_zero_based(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    values = pd.to_numeric(series, errors='coerce')
    theta = 2.0 * np.pi * (values / period)
    return pd.DataFrame({f'{name}_sin': np.sin(theta).astype(np.float32), f'{name}_cos': np.cos(theta).astype(np.float32)}, index=series.index)

def encode_cyclic_one_based(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    values = pd.to_numeric(series, errors='coerce')
    theta = 2.0 * np.pi * ((values - 1.0) / period)
    return pd.DataFrame({f'{name}_sin': np.sin(theta).astype(np.float32), f'{name}_cos': np.cos(theta).astype(np.float32)}, index=series.index)

def decode_cyclic(sine: pd.Series, cosine: pd.Series, period: int, *, one_based: bool=False) -> pd.Series:
    sine_values = pd.to_numeric(sine, errors='coerce').to_numpy(dtype=float)
    cosine_values = pd.to_numeric(cosine, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(sine_values) & np.isfinite(cosine_values)
    norm = np.sqrt(np.square(sine_values) + np.square(cosine_values))
    finite &= norm > 1e-08
    result = pd.Series(pd.NA, index=sine.index, dtype='Int64')
    if finite.any():
        theta = np.arctan2(sine_values[finite], cosine_values[finite])
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        values = np.rint(theta / (2 * np.pi) * period).astype(int) % period
        if one_based:
            values += 1
        result.iloc[np.flatnonzero(finite)] = values
    return result

def encode_bounded_logit(series: pd.Series, eps: float=1e-06) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce').fillna(0.5).clip(eps, 1.0 - eps)
    return pd.Series(np.log(values / (1.0 - values)).astype(np.float32), index=series.index)

def inv_logit(values) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -20, 20)
    return 1.0 / (1.0 + np.exp(-values))

def build_offence_crosswalk(frame: pd.DataFrame, cfg: Config) -> Dict[str, List[str]]:
    if cfg.COL_OFFENCE not in frame.columns or cfg.COL_CAT not in frame.columns:
        return {}
    pairs = frame[[cfg.COL_OFFENCE, cfg.COL_CAT]].dropna().astype(str).drop_duplicates()
    mapping: Dict[str, List[str]] = {}
    for category, group in pairs.groupby(cfg.COL_CAT):
        mapping[str(category)] = sorted(group[cfg.COL_OFFENCE].unique().tolist())
    return mapping

def feature_engineering(cfg: Config, train_parquet: str, embedding_parquet: str, out_dir: str) -> Tuple[str, str, str, List[str]]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(train_parquet)
    embedding = pd.read_parquet(embedding_parquet)
    train[cfg.COL_SEG] = train[cfg.COL_SEG].astype(str)
    embedding[cfg.COL_SEG] = embedding[cfg.COL_SEG].astype(str)
    embedding_columns = sorted([column for column in embedding.columns if column.startswith('gemb_')], key=lambda value: int(value.split('_')[1]))
    if len(embedding_columns) != cfg.EMB_DIM:
        raise RuntimeError(f'Expected {cfg.EMB_DIM} DeepWalk dimensions, found {len(embedding_columns)}.')
    merged = train.merge(embedding[[cfg.COL_SEG, *embedding_columns]], on=cfg.COL_SEG, how='left', validate='many_to_one')
    if merged[embedding_columns].isna().any().any():
        raise RuntimeError('Some snapped training events lack segment embeddings.')
    keep = [cfg.COL_SEG, cfg.COL_ID, cfg.COL_DATE, 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW', 'OCC_HOUR', 'OCC_HOUR_MISSING', cfg.COL_CAT, cfg.COL_OFFENCE, 'DIVISION', 'LOCATION_T', 'PREMISES_T', 'seg_pos', 'SEG_MAP_DIST_M', *embedding_columns]
    keep = [column for column in keep if column in merged.columns]
    raw = merged[keep].copy()
    raw['__POOL__'] = pool_series(raw, cfg.COHORT_KIND)
    raw['OCC_YEAR_CAT'] = raw['OCC_YEAR'].astype(int).astype(str)
    raw['OCC_MONTH_CAT'] = raw['OCC_MONTH'].astype(int).astype(str).str.zfill(2)
    model = raw.drop(columns=[cfg.COL_SEG, cfg.COL_ID, cfg.COL_DATE, 'OCC_DOW'], errors='ignore').copy()
    if cfg.COHORT_KIND == 'year_month':
        model = model.drop(columns=['OCC_YEAR_CAT', 'OCC_MONTH_CAT'], errors='ignore')
    elif cfg.COHORT_KIND in {'year_quarter', 'year'}:
        model = model.drop(columns=['OCC_YEAR_CAT'], errors='ignore')
    if 'OCC_HOUR' in model.columns:
        hour_cyclic = encode_cyclic_zero_based(model['OCC_HOUR'], 24, 'OCC_HOUR')
        model = pd.concat([model.drop(columns=['OCC_HOUR']), hour_cyclic], axis=1)
    model = model.drop(columns=['OCC_MONTH', 'OCC_YEAR'], errors='ignore')
    if 'seg_pos' in model.columns:
        model['seg_pos_logit'] = encode_bounded_logit(model['seg_pos'])
        model = model.drop(columns=['seg_pos'])
    if 'SEG_MAP_DIST_M' in model.columns:
        distance = pd.to_numeric(model['SEG_MAP_DIST_M'], errors='coerce').clip(0, cfg.SNAP_MAX_M)
        model['SEG_MAP_DIST_M_log1p'] = np.log1p(distance).astype(np.float32)
        model = model.drop(columns=['SEG_MAP_DIST_M'])
    crosswalk = build_offence_crosswalk(raw, cfg)
    raw_path = output_dir / 'train_features_raw.parquet'
    model_path = output_dir / 'train_features_model.parquet'
    crosswalk_path = output_dir / 'offence_category_crosswalk.json'
    raw.to_parquet(raw_path, index=False)
    model.to_parquet(model_path, index=False)
    atomic_json_dump(crosswalk, crosswalk_path)
    pools = sorted(model['__POOL__'].astype(str).unique().tolist())
    summary = {'cohort_kind': cfg.COHORT_KIND, 'pool_count': len(pools), 'condition_columns': condition_columns(cfg.COHORT_KIND, cfg.COL_CAT), 'model_columns': [column for column in model.columns if column != '__POOL__'], 'derived_after_generation': ['OCC_DOW'], 'embedding_columns': embedding_columns}
    atomic_json_dump(summary, output_dir / 'feature_summary.json')
    return (str(raw_path), str(model_path), str(crosswalk_path), pools)
