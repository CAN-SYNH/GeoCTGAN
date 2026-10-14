import json
from pathlib import Path

import pandas as pd
import pytest

from geoctgan.config import Config
from geoctgan.pipeline import _load_config, _validated_recovered_embedding_selection
from geoctgan.graph import embedding_screen_specifications
from geoctgan.sensitivity import select_embedding_from_generation_pilot, select_snap_tolerance, select_temporal_scheme, temporal_confirmation_schemes, validate_temporal_confirmation
from geoctgan import utils


def test_snap_selection_uses_inner_validation_and_prefers_smallest_near_maximum(tmp_path: Path):
    path = tmp_path / "snap.csv"
    pd.DataFrame(
        {
            "split": ["inner_train"] * 4 + ["inner_validation"] * 4,
            "snap_tolerance_m": [10, 20, 30, 50] * 2,
            "retention": [0.98, 0.995, 0.9954, 0.9955, 0.97, 0.9948, 0.9952, 0.9953],
            "distance_p95_m": [8, 15, 20, 30] * 2,
        }
    ).to_csv(path, index=False)
    selected = select_snap_tolerance(str(path), relative_retention_tolerance=0.001)
    assert selected["snap_tolerance_m"] == 20


def test_temporal_selection_uses_actual_one_se_rule(tmp_path: Path):
    path = tmp_path / "temporal.csv"
    pd.DataFrame(
        {
            "scheme": ["year_month", "year", "global"],
            "model_count": [132, 11, 1],
            "selection_score_mean": [0.100, 0.103, 0.104],
            "selection_score_se": [0.005, 0.003, 0.002],
            "runtime_seconds_mean": [100, 30, 10],
        }
    ).to_csv(path, index=False)
    selected = select_temporal_scheme(str(path), Config())
    assert selected["scheme"] == "global"


def test_definitive_operational_override_preserves_selected_fields(tmp_path: Path):
    selected = Config(COHORT_KIND="year", EMB_DIM=16, EPOCHS=100)
    selected.REQUIRE_SELECTED_CONFIG = False
    selected_path = tmp_path / "selected.json"
    selected.save(str(selected_path))
    definitive_path = tmp_path / "definitive.json"
    definitive_path.write_text(
        json.dumps(
            {
                "SELECTED_CONFIG_PATH": str(selected_path),
                "REQUIRE_SELECTED_CONFIG": True,
                "EPOCHS": 300,
                "BASE_DIR": str(tmp_path),
            }
        )
    )
    loaded, provenance = _load_config(str(definitive_path))
    assert loaded.COHORT_KIND == "year"
    assert loaded.EMB_DIM == 16
    assert loaded.EPOCHS == 300
    assert provenance["locked_fields"]["COHORT_KIND"] == "year"


def test_definitive_cannot_override_selected_scientific_field(tmp_path: Path):
    selected = Config(COHORT_KIND="year", EMB_DIM=16)
    selected.REQUIRE_SELECTED_CONFIG = False
    selected_path = tmp_path / "selected.json"
    selected.save(str(selected_path))
    definitive_path = tmp_path / "definitive.json"
    definitive_path.write_text(
        json.dumps(
            {
                "SELECTED_CONFIG_PATH": str(selected_path),
                "REQUIRE_SELECTED_CONFIG": True,
                "COHORT_KIND": "year_month",
            }
        )
    )
    with pytest.raises(RuntimeError, match="overwrite"):
        _load_config(str(definitive_path))


def test_benchmark_operational_overrides_do_not_change_selected_science(tmp_path: Path):
    selected = Config(COHORT_KIND="year", EMB_DIM=16)
    selected.REQUIRE_SELECTED_CONFIG = False
    selected_path = tmp_path / "selected.json"
    selected.save(str(selected_path))
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "SELECTED_CONFIG_PATH": str(selected_path),
                "BENCHMARK_METHODS": ["GaussianCopula"],
                "BENCHMARK_SEEDS": [7, 8],
            }
        )
    )
    loaded, _ = _load_config(str(benchmark_path))
    assert loaded.COHORT_KIND == "year"
    assert loaded.EMB_DIM == 16
    assert loaded.BENCHMARK_METHODS == ["GaussianCopula"]
    assert loaded.BENCHMARK_SEEDS == [7, 8]


def test_intrinsic_screen_uses_seven_reported_configurations():
    cfg = Config()
    candidates = embedding_screen_specifications(cfg)
    assert cfg.WALK_LEN == 10
    assert {row[:3] for row in candidates} == {
        (8, 15, 20), (16, 15, 20), (32, 15, 20),
        (16, 5, 20), (16, 30, 20), (16, 15, 10), (16, 15, 40),
    }
    assert len(candidates) == 7


@pytest.mark.parametrize('retained_mean,expected_success', [(0.30, True), (0.50, False)])
def test_embedding_selection_retains_documented_eligible_configuration(tmp_path, retained_mean, expected_success):
    path = tmp_path / 'embedding.csv'
    pd.DataFrame({
        'dimension': [16, 16, 32], 'walks_per_node': [15, 5, 15],
        'walk_length': [10, 20, 20],
        'selection_score_mean': [retained_mean, .28, .70],
        'selection_score_se': [.03, .06, .04],
        'runtime_seconds_mean': [10, 5, 20],
    }).to_csv(path, index=False)
    if expected_success:
        selected = select_embedding_from_generation_pilot(str(path), Config())
        assert (selected['dimension'], selected['walks_per_node'], selected['walk_length']) == (16, 15, 10)
        assert _validated_recovered_embedding_selection(str(path), selected, Config()) == selected
        with pytest.raises(RuntimeError, match='cached embedding'):
            _validated_recovered_embedding_selection(str(path), {'dimension': 16, 'walks_per_node': 5, 'walk_length': 20}, Config())
    else:
        with pytest.raises(RuntimeError, match='configuration decision'):
            select_embedding_from_generation_pilot(str(path), Config())


def test_temporal_confirmation_includes_all_four_schemes():
    assert set(temporal_confirmation_schemes(Config())) == {'global', 'year', 'year_quarter', 'year_month'}
    with pytest.raises(ValueError, match='all four'):
        temporal_confirmation_schemes(Config(TEMPORAL_FINALIST_COUNT=2))


@pytest.mark.parametrize('incomplete', [None, 'scheme', 'seed'])
def test_temporal_confirmation_rejects_incomplete_resume_evidence(tmp_path, incomplete):
    path = tmp_path / 'temporal_scheme_sensitivity.csv'
    schemes = ['global', 'year', 'year_quarter', 'year_month']
    summary = pd.DataFrame({'scheme': schemes, 'seed_count': [3] * 4})
    runs = pd.DataFrame([{'scheme': scheme, 'seed': seed} for scheme in schemes for seed in [42, 52, 62]])
    if incomplete == 'scheme':
        summary = summary.iloc[:2]
        runs = runs.loc[runs['scheme'].isin(schemes[:2])]
    elif incomplete == 'seed':
        runs = runs.iloc[:-1]
    summary.to_csv(path, index=False)
    runs.to_csv(path.with_name('temporal_scheme_sensitivity_runs.csv'), index=False)
    if incomplete:
        with pytest.raises(RuntimeError, match='confirmation|pooling'):
            validate_temporal_confirmation(str(path), Config())
    else:
        validate_temporal_confirmation(str(path), Config())


def test_csv_path_sanitizer_resolves_root_once(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    (run / 'Data').mkdir(parents=True)
    (run / 'Metadata').mkdir()
    (run / 'Tables').mkdir()
    internal = run / 'Data' / 'events.csv'
    external = tmp_path / 'private' / 'events.csv'
    count = []
    original = utils.output_run_root
    monkeypatch.setattr(utils, 'output_run_root', lambda path: (count.append(path), original(path))[1])
    path = run / 'Tables' / 'values.csv'
    utils.write_output_csv(pd.DataFrame({'label': ['GeoCTGAN-Full'] * 15000, 'value': range(15000)}), path, index=False)
    assert len(count) == 1
    assert pd.read_csv(path)['value'].tolist() == list(range(15000))
    utils.write_output_csv(pd.DataFrame({'path': [str(internal), str(external)], 'label': ['Full', 'Priv']}), path, index=False)
    values = pd.read_csv(path)
    assert values.loc[0, 'path'] == 'Data/events.csv'
    assert pd.isna(values.loc[1, 'path'])
    assert values['label'].tolist() == ['Full', 'Priv']
