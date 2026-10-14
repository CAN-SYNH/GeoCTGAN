from __future__ import annotations
from .utils import write_output_csv
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from tqdm.auto import tqdm
try:
    from sdv.single_table import CTGANSynthesizer
except ImportError:
    CTGANSynthesizer = None
try:
    from sdv.metadata import Metadata as SDVMetadata
except ImportError:
    SDVMetadata = None
try:
    from sdv.metadata import SingleTableMetadata
except ImportError:
    SingleTableMetadata = None
from .config import Config
from .temporal import condition_columns
from .utils import atomic_json_dump, derived_seed, ensure_dir, sdv_gpu_constructor_kwargs, set_seeds
logger = logging.getLogger(__name__)

def _detect_metadata(frame: pd.DataFrame):
    if SDVMetadata is not None and hasattr(SDVMetadata, 'detect_from_dataframe'):
        for kwargs in ({'data': frame, 'table_name': 'table', 'infer_keys': None}, {'data': frame, 'table_name': 'table'}, {'data': frame}):
            try:
                return SDVMetadata.detect_from_dataframe(**kwargs)
            except TypeError:
                continue
    if SingleTableMetadata is None:
        raise ImportError('No compatible SDV metadata implementation is available.')
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=frame)
    return metadata

def _update_column(metadata, column: str, sdtype: str) -> None:
    for kwargs in ({'column_name': column, 'sdtype': sdtype}, {'column_name': column, 'table_name': 'table', 'sdtype': sdtype}):
        try:
            metadata.update_column(**kwargs)
            return
        except TypeError:
            continue
    metadata.update_column(column, sdtype=sdtype)

def _save_metadata(metadata, path: str) -> None:
    save = getattr(metadata, 'save_to_json', None)
    if not callable(save):
        return
    for kwargs in ({'filepath': path, 'mode': 'overwrite'}, {'filepath': path}):
        try:
            save(**kwargs)
            return
        except TypeError:
            continue
    save(path)

def _adaptive_batch_size(n_rows: int, target: int, pac: int) -> int:
    if n_rows < pac * 2:
        raise RuntimeError(f'CTGAN pool has only {n_rows} rows; at least {pac * 2} are required for pac={pac}.')
    candidate = min(int(target), int(n_rows))
    candidate -= candidate % pac
    if candidate < pac * 2:
        candidate = pac * 2
    return int(candidate)

def _native_loss_frame(synthesizer) -> pd.DataFrame:
    getter = getattr(synthesizer, 'get_loss_values', None)
    if callable(getter):
        try:
            frame = getter()
            if isinstance(frame, pd.DataFrame):
                return frame.copy()
        except Exception:
            pass
    model = getattr(synthesizer, '_model', None)
    frame = getattr(model, 'loss_values', None)
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

def train_ctgan(cfg: Config, model_table_path: str, pools: List[str], out_dir: str, *, epochs: int | None=None, run_tag: str='definitive', logs_dir: Optional[str]=None) -> Tuple[str, str]:
    cfg.validate()
    if CTGANSynthesizer is None:
        raise ImportError('SDV is required for CTGAN training. Install the exact locked requirements.')
    gpu_kwargs, gpu_parameter = sdv_gpu_constructor_kwargs(CTGANSynthesizer, cfg.ENABLE_GPU)
    table = pd.read_parquet(model_table_path)
    epochs = int(cfg.EPOCHS if epochs is None else epochs)
    models_dir = Path(ensure_dir(Path(out_dir) / f'models_ctgan_{run_tag}'))
    logs_dir = Path(ensure_dir(Path(logs_dir) / f'CTGAN_{run_tag}' if logs_dir is not None else Path(out_dir) / f'logs_ctgan_{run_tag}'))
    categorical_candidates = {cfg.COL_CAT, cfg.COL_OFFENCE, 'DIVISION', 'LOCATION_T', 'PREMISES_T', 'OCC_YEAR_CAT', 'OCC_MONTH_CAT', 'OCC_HOUR_MISSING'}
    manifest_rows = []
    for pool in tqdm(pools, desc=f'Training CTGAN ({run_tag})'):
        subset = table.loc[table['__POOL__'].astype(str).eq(str(pool))].drop(columns=['__POOL__']).copy()
        if len(subset) < cfg.MIN_POOL_ROWS:
            raise RuntimeError(f'Temporal pool {pool!r} has {len(subset)} rows, below MIN_POOL_ROWS={cfg.MIN_POOL_ROWS}. Select a broader temporal scheme rather than silently skipping it.')
        seed = derived_seed(cfg.SEED, 'ctgan', cfg.COHORT_KIND, pool, run_tag)
        batch_size = _adaptive_batch_size(len(subset), cfg.BATCH_TARGET, cfg.PAC)
        model_path = models_dir / f'ctgan_{pool}.pkl'
        metadata_path = models_dir / f'metadata_{pool}.json'
        checkpoint_path = models_dir / f'complete_{pool}.json'
        if cfg.RESUME_TRAINING and model_path.exists() and checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
            manifest_rows.append(checkpoint['manifest_row'])
            continue
        set_seeds(seed)
        metadata = _detect_metadata(subset)
        for column in subset.columns:
            if column in categorical_candidates or subset[column].dtype == 'object':
                _update_column(metadata, column, 'categorical')
            elif column == 'OCC_DAY':
                _update_column(metadata, column, 'numerical')
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, pac=cfg.PAC, generator_lr=cfg.GEN_LR, discriminator_lr=cfg.DISC_LR, enforce_rounding=cfg.ENFORCE_ROUNDING, verbose=False, **gpu_kwargs)
        start = time.perf_counter()
        try:
            synthesizer.fit(subset)
        except Exception as exc:
            raise
        runtime = time.perf_counter() - start
        synthesizer.save(str(model_path))
        _save_metadata(metadata, str(metadata_path))
        losses = _native_loss_frame(synthesizer)
        if not losses.empty:
            losses.insert(0, 'pool', str(pool))
            write_output_csv(losses, logs_dir / f'losses_{pool}.csv', index=False)
        manifest_row = {'pool': str(pool), 'temporal_scheme': cfg.COHORT_KIND, 'rows': len(subset), 'epochs': epochs, 'batch_size': batch_size, 'pac': cfg.PAC, 'enable_gpu_requested': bool(cfg.ENABLE_GPU), 'sdv_gpu_parameter': gpu_parameter, 'seed': seed, 'runtime_seconds': runtime, 'status': 'success', 'model_path': str(model_path), 'metadata_path': str(metadata_path), 'condition_columns': json.dumps(condition_columns(cfg.COHORT_KIND, cfg.COL_CAT))}
        atomic_json_dump({'manifest_row': manifest_row}, checkpoint_path)
        manifest_rows.append(manifest_row)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = models_dir / 'model_manifest.csv'
    write_output_csv(manifest, manifest_path, index=False)
    return (str(models_dir), str(manifest_path))
