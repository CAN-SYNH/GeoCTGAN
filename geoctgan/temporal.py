from __future__ import annotations
import calendar
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
SUPPORTED_SCHEMES = {'year_month', 'year_quarter', 'year', 'global'}

def pool_id(year: int, month: int, scheme: str) -> str:
    if scheme == 'year_month':
        return f'{int(year):04d}_{int(month):02d}'
    if scheme == 'year_quarter':
        quarter = (int(month) - 1) // 3 + 1
        return f'{int(year):04d}_Q{quarter}'
    if scheme == 'year':
        return f'{int(year):04d}'
    if scheme == 'global':
        return 'GLOBAL'
    raise ValueError(f'Unsupported temporal scheme: {scheme!r}')

def pool_series(frame: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f'Unsupported temporal scheme: {scheme!r}')
    return pd.Series([pool_id(year, month, scheme) for year, month in zip(frame['OCC_YEAR'], frame['OCC_MONTH'])], index=frame.index, dtype='object')

def condition_columns(scheme: str, category_column: str) -> List[str]:
    if scheme == 'year_month':
        return [category_column]
    if scheme in {'year_quarter', 'year'}:
        return ['OCC_MONTH_CAT', category_column]
    if scheme == 'global':
        return ['OCC_YEAR_CAT', 'OCC_MONTH_CAT', category_column]
    raise ValueError(f'Unsupported temporal scheme: {scheme!r}')

def condition_values(year: int, month: int, category: str, scheme: str, category_column: str) -> Dict[str, str]:
    values: Dict[str, str] = {category_column: str(category)}
    if scheme in {'year_quarter', 'year', 'global'}:
        values['OCC_MONTH_CAT'] = f'{int(month):02d}'
    if scheme == 'global':
        values['OCC_YEAR_CAT'] = str(int(year))
    return values

def reconstruct_date(year: int, month: int, day: pd.Series) -> pd.Series:
    maximum = calendar.monthrange(int(year), int(month))[1]
    values = pd.to_numeric(day, errors='coerce').round().clip(1, maximum)
    return pd.to_datetime({'year': int(year), 'month': int(month), 'day': values.astype('Int64')}, errors='coerce')

def month_boundary_discontinuity(real: pd.DataFrame, synthetic: pd.DataFrame, *, date_column: str, category_column: str, window_days: int=7) -> pd.DataFrame:
    from scipy.spatial.distance import jensenshannon

    def _distribution(series: pd.Series, support: list[str]) -> np.ndarray:
        counts = series.astype(str).value_counts().reindex(support, fill_value=0).to_numpy(float)
        return counts / counts.sum() if counts.sum() else np.zeros(len(support), float)

    def _js(left: pd.Series, right: pd.Series) -> float:
        support = sorted(set(left.dropna().astype(str)).union(set(right.dropna().astype(str))))
        if not support:
            return 0.0
        return float(jensenshannon(_distribution(left, support), _distribution(right, support), base=2.0) ** 2)

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        table = frame.copy()
        table[date_column] = pd.to_datetime(table[date_column], errors='coerce')
        return table.dropna(subset=[date_column])
    real_table, synthetic_table = (prepare(real), prepare(synthetic))
    all_months = sorted(set(real_table[date_column].dt.to_period('M')).intersection(set(synthetic_table[date_column].dt.to_period('M'))))
    rows = []
    for previous, current in zip(all_months[:-1], all_months[1:]):
        if current - previous != 1:
            continue
        boundary = current.start_time
        for label, table in [('real', real_table), ('synthetic', synthetic_table)]:
            before = table.loc[table[date_column].between(boundary - pd.Timedelta(days=window_days), boundary - pd.Timedelta(days=1))]
            after = table.loc[table[date_column].between(boundary, boundary + pd.Timedelta(days=window_days - 1))]
            before_total, after_total = (float(len(before)), float(len(after)))
            count_change = (after_total - before_total) / max(before_total, 1.0)
            category_shift = _js(before[category_column], after[category_column])
            hour_shift = np.nan
            if 'OCC_HOUR' in before.columns and 'OCC_HOUR' in after.columns:
                hour_shift = _js(before['OCC_HOUR'], after['OCC_HOUR'])
            centroid_shift = np.nan
            if {'LONG_WGS84', 'LAT_WGS84'}.issubset(table.columns) and len(before) and len(after):
                before_xy = before[['LONG_WGS84', 'LAT_WGS84']].apply(pd.to_numeric, errors='coerce').mean()
                after_xy = after[['LONG_WGS84', 'LAT_WGS84']].apply(pd.to_numeric, errors='coerce').mean()
                if np.isfinite(before_xy).all() and np.isfinite(after_xy).all():
                    centroid_shift = float(np.linalg.norm(after_xy.to_numpy() - before_xy.to_numpy()))
            rows.append({'boundary': boundary, 'dataset': label, 'before_count': before_total, 'after_count': after_total, 'count_relative_change': count_change, 'category_js_shift': category_shift, 'hour_js_shift': hour_shift, 'centroid_shift_degrees': centroid_shift})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    wide = result.pivot(index='boundary', columns='dataset')
    output = pd.DataFrame({'boundary': wide.index})
    component_errors = []
    for metric in ['count_relative_change', 'category_js_shift', 'hour_js_shift', 'centroid_shift_degrees']:
        if (metric, 'real') in wide.columns and (metric, 'synthetic') in wide.columns:
            error = (wide[metric, 'synthetic'] - wide[metric, 'real']).abs().to_numpy()
            output[f'{metric}_absolute_error'] = error
            component_errors.append(error)
    if component_errors:
        matrix = np.vstack(component_errors).T
        output['absolute_boundary_error'] = np.nanmean(matrix, axis=1)
    return output
