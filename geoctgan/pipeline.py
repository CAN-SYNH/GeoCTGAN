from __future__ import annotations
from .utils import write_output_csv
import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence
import pandas as pd

def _bootstrap_gdal_proj_env() -> None:
    try:
        import pyproj
        os.environ.setdefault('PROJ_LIB', pyproj.datadir.get_data_dir())
    except Exception:
        pass
_bootstrap_gdal_proj_env()
from .config import DEFINITIVE_OVERRIDE_FIELDS, PATH_CONFIG_FIELDS, SELECTION_LOCK_FIELDS, Config, scientific_config_payload
from .artifacts import ArtifactDirectories, create_unique_run_directory
from .data import create_inner_event_split, create_inner_split, create_outer_split, load_events, run_snap_sensitivity, segment_network, snap_events
from .features import feature_engineering
from .flow import ortools_available, production_solver_self_test
from .graph import build_graph, run_embedding_sensitivity, train_deepwalk
from .privacy import apply_privacy_gates
from .sampling import assign_and_reconstruct, generate_latent_rows, run_cann_largest_group_preflight, run_reconstruction_ablations, verified_ablation_output
from .sensitivity import run_embedding_generation_pilot, run_temporal_scheme_pilot, select_embedding_from_generation_pilot, select_snap_tolerance, select_temporal_scheme, validate_temporal_confirmation, write_selected_config
from .training import train_ctgan
from .utils import StageLedger, atomic_json_dump, checkpoint_artifact_map as _artifact_map, ensure_dir, load_validated_stage_artifacts as _validated_stage_artifacts, load_success_checkpoint, preserve_prior_file, resolve_source_path, set_seeds, setup_logging, write_success_checkpoint
logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int, str, str], None]

def _notify(callback: Optional[ProgressCallback], stage: str, current: int, total: int, detail: str, state: str='running') -> None:
    if callback is not None:
        callback(stage, int(current), int(total), str(detail), state)

def _load_config(config_path: Optional[str]) -> tuple[Config, dict]:
    if not config_path:
        cfg = Config.from_env()
        cfg.validate()
        return (cfg, {'source': 'environment', 'overrides': {}})
    with open(config_path, 'r', encoding='utf-8') as handle:
        definitive_values = json.load(handle)
    selected_path = definitive_values.get('SELECTED_CONFIG_PATH')
    if not selected_path:
        cfg = Config(**definitive_values)
        cfg.validate()
        return (cfg, {'source': 'definitive_only', 'overrides': definitive_values})
    selected_cfg = Config.from_json(selected_path)
    selected_values = selected_cfg.as_dict()
    overrides = {}
    rejected = {}
    for key, value in definitive_values.items():
        if key == 'SELECTED_CONFIG_PATH':
            overrides[key] = value
            continue
        if key in SELECTION_LOCK_FIELDS and key in selected_values and (value != selected_values[key]):
            rejected[key] = {'selected': selected_values[key], 'requested': value}
            continue
        if key not in DEFINITIVE_OVERRIDE_FIELDS:
            if key in selected_values and value != selected_values[key]:
                rejected[key] = {'selected': selected_values[key], 'requested': value}
            continue
        overrides[key] = value
    if rejected:
        raise RuntimeError(f'The definitive configuration attempted to overwrite sensitivity-selected or non-operational fields: {json.dumps(rejected, indent=2)}')
    merged = dict(selected_values)
    merged.update(overrides)
    merged['SELECTED_CONFIG_PATH'] = selected_path
    cfg = Config(**merged)
    cfg.validate()
    return (cfg, {'source': 'selected_plus_whitelisted_definitive_overrides', 'selected_config': selected_path, 'locked_fields': {field: selected_values[field] for field in SELECTION_LOCK_FIELDS}, 'applied_overrides': overrides})

def load_config(config_path: Optional[str]) -> tuple[Config, dict]:
    return _load_config(config_path)

def _prepare_definitive_run_root(cfg, requested_run_dir):
    if requested_run_dir is None:
        return (create_unique_run_directory(cfg.OUTPUT_DIR or cfg.BASE_DIR, 'GeoCTGAN'), False)
    root = Path(requested_run_dir).expanduser().resolve()
    resumed = root.exists() and any(root.iterdir())
    root.mkdir(parents=True, exist_ok=True)
    return (root, resumed)

def _prepare_selection_run_root(cfg, requested_run_dir):
    if requested_run_dir is None:
        return (create_unique_run_directory(cfg.OUTPUT_DIR or cfg.BASE_DIR, 'GeoCTGAN_Selection'), False)
    root = Path(requested_run_dir).expanduser().resolve()
    resumed = root.exists() and any(root.iterdir())
    root.mkdir(parents=True, exist_ok=True)
    return (root, resumed)

def _validated_recovered_embedding_selection(results_path: str, selection: dict, cfg: Config) -> dict:
    expected = select_embedding_from_generation_pilot(results_path, cfg)
    keys = ['dimension', 'walks_per_node', 'walk_length']
    if tuple(int(selection[key]) for key in keys) != tuple(int(expected[key]) for key in keys):
        raise RuntimeError('The cached embedding configuration differs from the reported eligible pilot selection. The selection stage must be rerun before using its embedding.')
    return expected


def run_sensitivity_pipeline(config_path: str, *, run_dir: Optional[str | Path]=None, progress_callback: Optional[ProgressCallback]=None) -> dict:
    cfg = Config.from_json(config_path)
    cfg.REQUIRE_SELECTED_CONFIG = False
    cfg.validate()
    set_seeds(cfg.SEED)
    root, resumed_run_directory = _prepare_selection_run_root(cfg, run_dir)
    artifacts = ArtifactDirectories.create(root)
    setup_logging(str(artifacts.logs / 'selection.log'))
    cfg.save(str(artifacts.metadata / 'base_config.json'))
    ledger = StageLedger(artifacts.metadata / 'stage_ledger.json')
    progress_total = 5
    _notify(progress_callback, 'inputs', 0, progress_total, 'Loading events and creating splits')
    reference_dir = Path(ensure_dir(artifacts.data / 'Reference'))
    recovered = _validated_stage_artifacts(ledger, 'events', ('events', 'outer_train', 'outer_test', 'outer_split_manifest'), artifact_root=root)
    if recovered is None:
        ledger.start('events')
        events = load_events(cfg, str(reference_dir))
        outer_train, outer_test, outer_split_manifest = create_outer_split(cfg, events, str(reference_dir))
        ledger.complete('events', artifacts=_artifact_map(artifact_root=root, events=events, outer_train=outer_train, outer_test=outer_test, outer_split_manifest=outer_split_manifest))
    else:
        events = recovered['events']
        outer_train = recovered['outer_train']
        outer_test = recovered['outer_test']
        outer_split_manifest = recovered['outer_split_manifest']
    _notify(progress_callback, 'network', 1, progress_total, 'Preparing the road network')
    network_dir = Path(ensure_dir(artifacts.intermediate / 'Network'))
    recovered = _validated_stage_artifacts(ledger, 'network', ('segments', 'graph'), artifact_root=root)
    if recovered is None:
        ledger.start('network')
        segments = segment_network(cfg, str(network_dir))
        graph = build_graph(cfg, segments, str(network_dir))
        ledger.complete('network', artifacts=_artifact_map(artifact_root=root, segments=segments, graph=graph))
    else:
        segments = recovered['segments']
        graph = recovered['graph']
    recovered = _validated_stage_artifacts(ledger, 'snap_selection', ('inner_event_train', 'inner_event_validation', 'inner_event_manifest', 'snap_results', 'snapped_outer_train', 'inner_train', 'inner_validation', 'inner_split_manifest'), artifact_root=root)
    if recovered is None:
        ledger.start('snap_selection')
        inner_event_train, inner_event_validation, inner_event_manifest = create_inner_event_split(cfg, outer_train, str(artifacts.intermediate / 'InnerEventSplit'), seed=cfg.SEED)
        snap_results = run_snap_sensitivity(cfg, inner_event_train, segments, str(artifacts.intermediate / 'SnapSensitivity'), inner_validation_events_path=inner_event_validation)
        snap_selection = select_snap_tolerance(snap_results)
        cfg.SNAP_MAX_M = float(snap_selection['snap_tolerance_m'])
        snapped_outer_train = str(reference_dir / 'events_snapped_outer_train_selected.parquet')
        _, train_snap_summary = snap_events(cfg, outer_train, segments, snapped_outer_train)
        inner_train, inner_validation, inner_split_manifest = create_inner_split(cfg, snapped_outer_train, str(artifacts.intermediate / 'InnerSplit'), seed=cfg.SEED)
        ledger.complete('snap_selection', artifacts=_artifact_map(artifact_root=root, inner_event_train=inner_event_train, inner_event_validation=inner_event_validation, inner_event_manifest=inner_event_manifest, snap_results=snap_results, snapped_outer_train=snapped_outer_train, inner_train=inner_train, inner_validation=inner_validation, inner_split_manifest=inner_split_manifest), snap_selection=snap_selection, train_snap_summary=train_snap_summary)
    else:
        inner_event_train = recovered['inner_event_train']
        inner_event_validation = recovered['inner_event_validation']
        inner_event_manifest = recovered['inner_event_manifest']
        snap_results = recovered['snap_results']
        snapped_outer_train = recovered['snapped_outer_train']
        inner_train = recovered['inner_train']
        inner_validation = recovered['inner_validation']
        inner_split_manifest = recovered['inner_split_manifest']
        snap_selection = dict(recovered['details']['snap_selection'])
        train_snap_summary = dict(recovered['details']['train_snap_summary'])
        cfg.SNAP_MAX_M = float(snap_selection['snap_tolerance_m'])
    _notify(progress_callback, 'embedding', 2, progress_total, 'Selecting the DeepWalk configuration')
    recovered = _validated_stage_artifacts(ledger, 'embedding_selection', ('embedding_results', 'embedding_pilot_results', 'selected_embedding', 'selected_embedding_model', 'selected_embedding_runtime'), artifact_root=root)
    if recovered is None:
        ledger.start('embedding_selection')
        embedding_results = run_embedding_sensitivity(cfg, graph, str(artifacts.intermediate / 'EmbeddingSensitivity'))
        embedding_pilot_results = run_embedding_generation_pilot(cfg, graph, embedding_results, inner_train, inner_validation, segments, str(artifacts.intermediate / 'EmbeddingGenerationPilot'))
        embedding_selection = select_embedding_from_generation_pilot(embedding_pilot_results, cfg)
        cfg.EMB_DIM = int(embedding_selection['dimension'])
        cfg.N_WALKS = int(embedding_selection['walks_per_node'])
        cfg.WALK_LEN = int(embedding_selection['walk_length'])
        selected_fit_dir = artifacts.models / 'Embeddings'
        selected_fit_dir.mkdir(parents=True, exist_ok=True)
        selected_fit_checkpoint = selected_fit_dir / 'selected_full_network_checkpoint.json'
        selected_embedding_model = selected_fit_dir / 'deepwalk_selected_full_network.model'
        selected_embedding_file = selected_fit_dir / 'segment_embeddings_selected_full_network.parquet'
        selected_embedding_runtime = selected_fit_dir / 'deepwalk_selected_full_network_runtime.json'
        selected_fit_commit = load_success_checkpoint(selected_fit_checkpoint, artifact_root=selected_fit_dir, required_artifacts=[selected_embedding_file, selected_embedding_model, selected_embedding_runtime])
        if selected_fit_commit is not None:
            selected_fit_result = selected_fit_commit['result']
            selected_embedding = str(selected_embedding_file.resolve())
            selected_runtime = dict(selected_fit_result['runtime'])
        else:
            selected_embedding, selected_runtime = train_deepwalk(cfg, graph, str(selected_fit_dir), dimension=cfg.EMB_DIM, n_walks=cfg.N_WALKS, walk_length=cfg.WALK_LEN, seed=cfg.SEED, workers=cfg.W2V_WORKERS, tag='selected_full_network')
            write_success_checkpoint(selected_fit_checkpoint, result={'embedding': str(Path(selected_embedding).resolve()), 'model': str(selected_embedding_model.resolve()), 'runtime_file': str(selected_embedding_runtime.resolve()), 'runtime': selected_runtime}, artifacts=[selected_embedding, selected_embedding_model, selected_embedding_runtime], artifact_root=selected_fit_dir)
        ledger.complete('embedding_selection', artifacts=_artifact_map(artifact_root=root, embedding_results=embedding_results, embedding_pilot_results=embedding_pilot_results, selected_embedding=selected_embedding, selected_embedding_model=selected_embedding_model, selected_embedding_runtime=selected_embedding_runtime), embedding_selection=embedding_selection, selected_runtime=selected_runtime)
    else:
        embedding_results = recovered['embedding_results']
        embedding_pilot_results = recovered['embedding_pilot_results']
        selected_embedding = recovered['selected_embedding']
        embedding_selection = _validated_recovered_embedding_selection(embedding_pilot_results, dict(recovered['details']['embedding_selection']), cfg)
        selected_runtime = dict(recovered['details']['selected_runtime'])
        cfg.EMB_DIM = int(embedding_selection['dimension'])
        cfg.N_WALKS = int(embedding_selection['walks_per_node'])
        cfg.WALK_LEN = int(embedding_selection['walk_length'])
    _notify(progress_callback, 'temporal', 3, progress_total, 'Selecting temporal pooling')
    recovered = _validated_stage_artifacts(ledger, 'temporal_selection', ('temporal_results',), artifact_root=root)
    if recovered is None:
        ledger.start('temporal_selection')
        temporal_results = run_temporal_scheme_pilot(cfg, inner_train, inner_validation, selected_embedding, segments, str(artifacts.intermediate / 'TemporalSensitivity'))
        temporal_selection = select_temporal_scheme(temporal_results, cfg)
        ledger.complete('temporal_selection', artifacts=_artifact_map(artifact_root=root, temporal_results=temporal_results), temporal_selection=temporal_selection)
    else:
        temporal_results = recovered['temporal_results']
        temporal_selection = dict(recovered['details']['temporal_selection'])
    validate_temporal_confirmation(temporal_results, cfg)
    expected_temporal_selection = select_temporal_scheme(temporal_results, cfg)
    if str(temporal_selection['scheme']) != str(expected_temporal_selection['scheme']):
        raise RuntimeError('The cached temporal selection differs from the current four-scheme confirmation results. The selection stage must be rerun.')
    temporal_selection = expected_temporal_selection
    atomic_json_dump({'training': train_snap_summary, 'outer_test': {'status': 'UNTOUCHED'}}, artifacts.tables / 'selected_snap_summary.json')
    _notify(progress_callback, 'configuration', 4, progress_total, 'Writing the selected configuration')
    selected_config_path = str(artifacts.metadata / 'selected_config.json')
    write_selected_config(cfg, snap_selection, embedding_selection, temporal_selection, selected_config_path, outer_split_manifest_path=outer_split_manifest)
    ledger.complete('configuration', artifacts=_artifact_map(artifact_root=root, selected_config=selected_config_path))
    summary = {'resumed_run_directory': bool(resumed_run_directory), 'selection_root': str(root), 'outer_train': outer_train, 'outer_test': outer_test, 'outer_split_manifest': outer_split_manifest, 'inner_event_split_manifest': inner_event_manifest, 'inner_split_manifest': inner_split_manifest, 'segments': segments, 'graph': graph, 'snap_results': snap_results, 'embedding_results': embedding_results, 'embedding_generation_pilot_results': embedding_pilot_results, 'temporal_results': temporal_results, 'selected_embedding': selected_embedding, 'selected_config': selected_config_path, 'snap_selection': snap_selection, 'embedding_selection': embedding_selection, 'temporal_selection': temporal_selection, 'replication_design': {'screening_seed': int(cfg.SENSITIVITY_SCREEN_SEED), 'finalist_confirmation_seeds': list(dict.fromkeys(map(int, cfg.SENSITIVITY_SEEDS))), 'embedding_finalists': int(cfg.EMBEDDING_FINALIST_COUNT), 'temporal_finalists': int(cfg.TEMPORAL_FINALIST_COUNT), 'outer_test_used_for_selection': False}}
    summary['artifact_directories'] = {name: str(getattr(artifacts, name)) for name in ('data', 'intermediate', 'models', 'figures', 'tables', 'metadata', 'logs')}
    from .selection_figure import generate_selection_figure
    summary['selection_figure'] = generate_selection_figure(snap_results, embedding_results, embedding_pilot_results, temporal_results, artifacts.figures, snap_selection=snap_selection, embedding_selection=embedding_selection, temporal_selection=temporal_selection)
    summary['selection_summary'] = str(artifacts.metadata / 'selection_summary.json')
    atomic_json_dump(summary, artifacts.metadata / 'selection_summary.json')
    _notify(progress_callback, 'finalize', progress_total, progress_total, 'Configuration selection complete', 'complete')
    return summary

def _run_one_seed(cfg: Config, seed: int, shared: dict, artifacts: ArtifactDirectories) -> dict:
    seed_cfg = copy.deepcopy(cfg)
    seed_cfg.SEED = int(seed)
    set_seeds(seed_cfg.SEED)
    seed_cfg.save(str(artifacts.metadata / 'config_seed.json'))
    ledger = StageLedger(artifacts.metadata / 'stage_ledger.json')
    raw_features = shared['raw_features']
    model_features = shared['model_features']
    crosswalk = shared['crosswalk']
    pools = shared['pools']
    ledger.start('training')
    models_dir, model_manifest = train_ctgan(seed_cfg, model_features, pools, str(artifacts.models), epochs=seed_cfg.EPOCHS, run_tag='definitive', logs_dir=str(artifacts.logs))
    ledger.complete('training', models_dir=models_dir)
    ledger.start('generation')
    latent, generation_diagnostics = generate_latent_rows(seed_cfg, raw_features, crosswalk, models_dir, str(artifacts.intermediate))
    ledger.complete('generation', latent=latent)
    ledger.start('reconstruction')
    synthetic_exact, cann_diagnostics, capacity_table = assign_and_reconstruct(seed_cfg, latent, raw_features, shared['segments'], shared['embedding'], str(artifacts.data), variant='Full', capacity_mode='exact_quota')
    ablation_manifest = run_reconstruction_ablations(seed_cfg, latent, raw_features, shared['segments'], shared['embedding'], str(artifacts.data / 'Ablations'), include_full=False)
    ledger.complete('reconstruction', synthetic_exact=synthetic_exact)
    minimum_support = public_profile = None
    public_assignment_reused_from_ablation = False
    public_source = synthetic_exact
    if seed_cfg.PUBLIC_RELEASE_ENABLED:
        if seed_cfg.PUBLIC_RELEASE_CAPACITY_MODE == 'upper_quota':
            public_source = verified_ablation_output(ablation_manifest, 'UpperQuota')
            public_assignment_reused_from_ablation = True
        minimum_support, public_profile = apply_privacy_gates(seed_cfg, shared['snapped_train'], public_source, str(artifacts.data / 'Privacy'), capacity_mode=seed_cfg.PUBLIC_RELEASE_CAPACITY_MODE)
    summary = {'seed': int(seed), 'raw_features': raw_features, 'model_features': model_features, 'offence_crosswalk': crosswalk, 'models_dir': models_dir, 'model_manifest': model_manifest, 'latent_rows': latent, 'generation_diagnostics': generation_diagnostics, 'synthetic_internal_exact_quota': synthetic_exact, 'cann_diagnostics': cann_diagnostics, 'capacity_table': capacity_table, 'ablation_manifest': ablation_manifest, 'public_assignment_source': public_source, 'public_assignment_reused_from_ablation': public_assignment_reused_from_ablation, 'minimum_support': minimum_support, 'public_profile': public_profile, 'artifact_directories': {name: str(getattr(artifacts, name)) for name in ('data', 'intermediate', 'models', 'figures', 'tables', 'metadata', 'logs')}}
    write_output_csv(pd.DataFrame([{'seed': int(seed), 'latent_rows': latent, 'synthetic_internal_exact_quota': synthetic_exact, 'public_profile': public_profile, 'model_manifest': model_manifest, 'cann_diagnostics': cann_diagnostics, 'capacity_table': capacity_table, 'ablation_manifest': ablation_manifest}]), artifacts.tables / 'seed_artifact_manifest.csv', index=False)
    summary['seed_summary'] = str(artifacts.metadata / 'seed_summary.json')
    atomic_json_dump(summary, artifacts.metadata / 'seed_summary.json')
    return summary

def run_pipeline(config_path: Optional[str]=None, *, seeds: Optional[Sequence[int]]=None, run_dir: Optional[str | Path]=None, progress_callback: Optional[ProgressCallback]=None) -> dict:
    cfg, config_details = _load_config(config_path)
    if cfg.REQUIRE_SELECTED_CONFIG and (not cfg.SELECTED_CONFIG_PATH):
        raise RuntimeError('A sensitivity-selected configuration is required. Run the selection pipeline first.')
    if cfg.CANN_REQUIRE_ORTOOLS and (not ortools_available()):
        raise ImportError('The definitive pipeline requires OR-Tools. The production solver branch must pass its self-test.')
    solver_self_test = production_solver_self_test(min(30, cfg.CANN_GROUP_TIME_LIMIT_S)) if cfg.CANN_REQUIRE_ORTOOLS else None
    if cfg.REQUIRE_REUSED_OUTER_SPLIT and (not cfg.OUTER_SPLIT_MANIFEST_PATH):
        raise RuntimeError('The definitive run must reuse the outer split created by the selection run. OUTER_SPLIT_MANIFEST_PATH is missing.')
    seed_values = list(cfg.DEFINITIVE_SEEDS if seeds is None else seeds)
    if not seed_values:
        raise ValueError('At least one definitive seed is required.')
    root, resumed_run_directory = _prepare_definitive_run_root(cfg, run_dir)
    artifacts = ArtifactDirectories.create(root)
    setup_logging(str(artifacts.logs / 'pipeline.log'))
    cfg.save(str(artifacts.metadata / 'config_used.json'))
    shared_ledger = StageLedger(artifacts.metadata / 'shared_stage_ledger.json')
    progress_total = 4 + len(seed_values)
    _notify(progress_callback, 'inputs', 0, progress_total, 'Loading events and the frozen outer split')
    shared_ledger.start('events_and_split')
    reference_dir = Path(ensure_dir(artifacts.data / 'Reference'))
    events = load_events(cfg, str(reference_dir))
    outer_train, outer_test, outer_split_manifest = create_outer_split(cfg, events, str(reference_dir), split_manifest_path=cfg.OUTER_SPLIT_MANIFEST_PATH)
    shared_ledger.complete('events_and_split', split_manifest=outer_split_manifest)
    _notify(progress_callback, 'network', 1, progress_total, 'Preparing network and DeepWalk embeddings')
    shared_ledger.start('network_and_embedding')
    network_dir = Path(ensure_dir(artifacts.intermediate / 'Network'))
    segments = segment_network(cfg, str(network_dir))
    graph = build_graph(cfg, segments, str(network_dir))
    embedding, embedding_runtime = train_deepwalk(cfg, graph, str(artifacts.models / 'Embeddings'), dimension=cfg.EMB_DIM, n_walks=cfg.N_WALKS, walk_length=cfg.WALK_LEN, seed=cfg.SEED, workers=cfg.W2V_WORKERS, tag='definitive')
    shared_ledger.complete('network_and_embedding', embedding=embedding)
    snapped_train = str(reference_dir / 'events_snapped_train.parquet')
    snapped_test = str(reference_dir / 'events_snapped_test.parquet')
    _, train_snap_summary = snap_events(cfg, outer_train, segments, snapped_train)
    _, test_snap_summary = snap_events(cfg, outer_test, segments, snapped_test)
    atomic_json_dump({'train': train_snap_summary, 'test': test_snap_summary}, artifacts.tables / 'snap_summary.json')
    _notify(progress_callback, 'features', 2, progress_total, 'Engineering features and checking CaNN')
    shared_ledger.start('features_and_cann_preflight')
    raw_features, model_features, crosswalk, pools = feature_engineering(cfg, snapped_train, embedding, str(artifacts.intermediate / 'Features'))
    cann_preflight = run_cann_largest_group_preflight(cfg, raw_features, embedding, str(artifacts.tables / 'CaNN'))
    shared_ledger.complete('features_and_cann_preflight', pools=len(pools), cann_preflight=cann_preflight)
    shared = {'events': events, 'outer_train': outer_train, 'outer_test': outer_test, 'outer_split_manifest': outer_split_manifest, 'segments': segments, 'graph': graph, 'embedding': embedding, 'embedding_runtime': embedding_runtime, 'snapped_train': snapped_train, 'snapped_test': snapped_test, 'raw_features': raw_features, 'model_features': model_features, 'crosswalk': crosswalk, 'pools': pools, 'cann_preflight': cann_preflight}
    seed_summaries = []
    for position, seed in enumerate(seed_values, start=3):
        _notify(progress_callback, f'seed_{int(seed)}', position, progress_total, f'Training, generating and reconstructing seed {int(seed)}')
        seed_summaries.append(_run_one_seed(cfg, int(seed), shared, artifacts.seed(int(seed))))
        _notify(progress_callback, f'seed_{int(seed)}', position + 1, progress_total, f'Seed {int(seed)} complete', 'complete')
    summary = {'resumed_run_directory': bool(resumed_run_directory), 'output_dir': str(root), 'config': str(artifacts.metadata / 'config_used.json'), **shared, 'definitive_seeds': seed_values, 'seed_summaries': seed_summaries, 'solver_self_test': solver_self_test, 'artifact_directories': {name: str(getattr(artifacts, name)) for name in ('data', 'intermediate', 'models', 'figures', 'tables', 'metadata', 'logs')}}
    summary['run_summary'] = str(artifacts.metadata / 'run_summary.json')
    write_output_csv(pd.DataFrame([{'seed': row['seed'], 'synthetic_internal_exact_quota': row['synthetic_internal_exact_quota'], 'public_profile': row['public_profile'], 'model_manifest': row['model_manifest'], 'ablation_manifest': row['ablation_manifest']} for row in seed_summaries]), artifacts.tables / 'definitive_seed_outputs.csv', index=False)
    atomic_json_dump(summary, artifacts.metadata / 'run_summary.json')
    _notify(progress_callback, 'finalize', progress_total, progress_total, 'Definitive generation complete', 'complete')
    return summary
