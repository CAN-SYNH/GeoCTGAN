import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import geoctgan
from geoctgan import public_figures


def test_da_wape_is_invariant_to_event_volume():
    reference = np.array([8.0, 2.0, 0.0])
    synthetic = np.array([40.0, 30.0, 30.0])
    expected = 0.8
    assert public_figures.normalized_da_count_wape(reference, synthetic) == pytest.approx(expected)
    assert public_figures.normalized_da_count_wape(reference * 7, synthetic) == pytest.approx(expected)
    assert public_figures.normalized_da_count_wape(reference, synthetic * 9) == pytest.approx(expected)


def test_da_wape_retains_zero_count_areas():
    reference = np.array([10.0, 0.0])
    synthetic = np.array([10.0, 10.0])
    assert public_figures.normalized_da_count_wape(reference, synthetic) == pytest.approx(1.0)


def test_da_wape_is_a_ratio_not_percentage():
    reference = np.array([10.0, 0.0])
    synthetic = np.array([0.0, 30.0])
    assert public_figures.normalized_da_count_wape(reference, synthetic) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "reference,synthetic",
    [([0.0, 0.0], [1.0, 2.0]), ([1.0, 2.0], [0.0, 0.0]), ([], [])],
)
def test_da_wape_empty_totals_are_missing(reference, synthetic):
    assert np.isnan(public_figures.normalized_da_count_wape(np.array(reference), np.array(synthetic)))


def test_ottawa_wape_summary_preserves_seeds_and_sample_sd(tmp_path):
    labels = [f"GeoCTGAN-Full | seed {seed}" for seed in [42, 52, 62]]
    tables = tmp_path / "Tables"
    tables.mkdir()
    inputs = {
        "sdmetrics_quality.csv": {"score": [0.9] * 3, "report": ["QualityReport"] * 3},
        "learned_attribute_fidelity_summary.csv": {"median": [0.1] * 3},
        "kde_surface_fidelity.csv": {"kde_pearson": [0.9] * 3},
        "multiscale_grid_fidelity.csv": {"count_pearson": [0.9] * 3, "grid_size_m": [100] * 3},
        "dissemination_area_fidelity.csv": {
            "da_count_pearson": [0.9] * 3,
            "da_count_wape": [0.2, 0.3, 0.7],
        },
        "global_moran.csv": {"moran_i": [0.2] * 3},
        "lisa_agreement.csv": {"exact_label_agreement": [0.9] * 3},
        "nearest_neighbour_fidelity.csv": {"synthetic_mean_nnd_m": [10.0] * 3},
        "spatial_proximity_risk.csv": {
            "fraction_within_5m": [0.2] * 3,
            "distance_median_m": [10.0] * 3,
        },
        "sdmetrics_privacy.csv": {"metric": ["DCRBaselineProtection"] * 3, "score": [0.3] * 3},
    }
    for filename, values in inputs.items():
        pd.DataFrame({"dataset": labels, **values}).to_csv(tables / filename, index=False)

    public_figures.ottawa_summary(SimpleNamespace(tables=tables))

    by_seed = pd.read_csv(tables / "ottawa_fidelity_privacy_by_seed.csv")
    actual = by_seed.loc[by_seed["metric"].eq("da_count_wape")].sort_values("seed")
    assert actual["seed"].tolist() == [42, 52, 62]
    assert actual["method"].tolist() == ["GeoCTGAN-Full"] * 3
    np.testing.assert_allclose(actual["value"], [0.2, 0.3, 0.7])
    summary = pd.read_csv(tables / "ottawa_fidelity_privacy_summary.csv")
    row = summary.loc[summary["metric"].eq("da_count_wape")].iloc[0]
    assert row["mean"] == pytest.approx(0.4)
    assert row["std"] == pytest.approx(np.std([0.2, 0.3, 0.7], ddof=1))
    assert row["min"] == pytest.approx(0.2)
    assert row["max"] == pytest.approx(0.7)
    assert row["count"] == 3


@pytest.mark.parametrize("profile", ["toronto", "ottawa"])
def test_spatial_output_scope_and_da_support(tmp_path, monkeypatch, profile):
    label = "GeoCTGAN-Full | seed 42"
    reference_counts = np.array([8.0, 2.0, 0.0])
    synthetic_counts = np.array([40.0, 30.0, 30.0])
    real_train = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
    real_test = real_train.copy()
    synthetic = real_train.copy()
    count_vectors = {
        id(real_train): reference_counts * 4,
        id(real_test): reference_counts,
        id(synthetic): synthetic_counts,
    }
    ev = ModuleType("geoctgan.evaluation")
    ev._coordinates = lambda frame, cfg: frame[["x", "y"]].to_numpy()
    ev._nearest_neighbour_distances = lambda xy: np.ones(len(xy))
    ev._sample_pair = lambda left, right, seed, maximum: (left, right)
    ev._grid_counts = lambda xy, size, origin, shape: np.array([1.0, 2.0])
    ev._exact_top_q_indices = lambda counts, fraction, count=None: np.array([1])
    ev._jaccard = lambda left, right: 1.0
    ev._safe_pearson = lambda left, right: float(np.corrcoef(left, right)[0, 1])
    ev._kde_bandwidth = lambda xy: 1.0
    ev._kde_surface = lambda xy, points, bandwidth, maximum, seed: np.linspace(0.0, 1.0, len(points))
    ev._find_da_identifier = lambda da: "DAUID"
    ev._da_counts = lambda frame, da, da_id, cfg: pd.Series(count_vectors[id(frame)])
    ev._spatial_autocorrelation = lambda counts, da, da_id, cfg: (
        pd.DataFrame({"dataset": ["Real held-out", label], "moran_i": [0.2, 0.3]}),
        pd.DataFrame({"reference": ["Real held-out"], "dataset": [label], "exact_label_agreement": [1.0]}),
        pd.DataFrame({"dataset": [label], "lisa_cluster": ["NS"]}),
    )
    monkeypatch.setitem(sys.modules, "geoctgan.evaluation", ev)
    monkeypatch.setattr(geoctgan, "evaluation", ev, raising=False)
    for name in ["nnd_figure", "kde_figure", "area_figure", "ottawa_lisa_figure"]:
        monkeypatch.setattr(public_figures, name, lambda *args, **kwargs: None)
    artifacts = SimpleNamespace(tables=tmp_path / "Tables", figures=tmp_path / "Figures")
    artifacts.tables.mkdir()
    artifacts.figures.mkdir()
    cfg = SimpleNamespace(
        metric_crs="EPSG:32618",
        spatial_extent_m=[0.0, 0.0, 2.0, 2.0],
        maximum_equal_sample_rows=3,
        privacy_profile=profile,
        random_seeds=[101],
        grid_sizes_m=[100],
        hotspot_fraction=0.1,
        kde_grid_size=2,
        maximum_kde_fit_rows=3,
    )
    da = pd.DataFrame({"DAUID": ["area_a", "area_b", "area_c"]})
    boundaries = SimpleNamespace(to_crs=lambda crs: da)

    public_figures.spatial_metrics(real_train, real_test, {label: synthetic}, artifacts, cfg, boundaries)

    counts = pd.read_csv(artifacts.tables / "da_counts.csv")
    assert counts["DAUID"].tolist() == ["area_a", "area_b", "area_c"]
    np.testing.assert_allclose(counts["Real held-out"], reference_counts)
    np.testing.assert_allclose(counts[label], synthetic_counts)
    fidelity = pd.read_csv(artifacts.tables / "dissemination_area_fidelity.csv")
    expected_columns = ["reference", "dataset", "da_count_pearson"]
    if profile == "ottawa":
        expected_columns.append("da_count_wape")
        assert fidelity.loc[0, "da_count_wape"] == pytest.approx(0.8)
    assert fidelity.columns.tolist() == expected_columns
    assert fidelity.loc[0, "da_count_pearson"] == pytest.approx(np.corrcoef(reference_counts, synthetic_counts)[0, 1])
    nnd = pd.read_csv(artifacts.tables / "nearest_neighbour_fidelity.csv")
    assert "nnd_ks_statistic" not in nnd.columns
    assert nnd.loc[0, "reference_mean_nnd_m"] == pytest.approx(1.0)
    assert nnd.loc[0, "synthetic_mean_nnd_m"] == pytest.approx(1.0)
    assert nnd.loc[0, "rows_per_table"] == 3
    cdf_path = artifacts.tables / "nearest_neighbour_cdf.csv"
    if profile == "toronto":
        cdf = pd.read_csv(cdf_path)
        assert len(cdf) == 3002
        assert set(cdf["dataset"]) == {"Real held-out", label}
        assert set(cdf["cdf"]) == {0.0, 1.0}
    else:
        assert not cdf_path.exists()
