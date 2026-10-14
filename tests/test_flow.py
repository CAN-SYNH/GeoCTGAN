import sys

import numpy as np
import pytest

import geoctgan.flow as flow_module
from geoctgan.flow import (
    assign_capacity_only,
    assign_exact_flow_cann,
    assign_flow_cann,
    ortools_available,
)


def test_ortools_native_probe_is_cached(monkeypatch):
    class Probe:
        returncode = 0

    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return Probe()

    ortools_available.cache_clear()
    monkeypatch.setattr(flow_module.subprocess, "run", fake_run)
    try:
        assert ortools_available() is True
        assert ortools_available() is True
        assert len(calls) == 1
    finally:
        ortools_available.cache_clear()


def test_exact_flow_fills_quotas_without_overrun_scipy_fallback():
    segment_ids = ["A", "B", "C"]
    segment_embeddings = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    generated = np.array([[0.1, 0.0], [0.2, 0.1], [0.9, 1.0], [1.9, 0.0]], dtype=np.float32)
    result = assign_exact_flow_cann(
        generated,
        segment_ids,
        segment_embeddings,
        {"A": 2, "B": 1, "C": 1},
        candidate_schedule=[1, 2, 3],
        stable_rounds=0,
        full_support_threshold=3,
        require_ortools=False,
    )
    unique, counts = np.unique(result.assigned_segment_ids, return_counts=True)
    assert dict(zip(unique, counts)) == {"A": 2, "B": 1, "C": 1}
    assert result.diagnostics["capacity_violations"] == 0
    assert result.diagnostics["exact_quota_mismatch"] == 0


@pytest.mark.skipif(
    sys.platform == "win32" or not ortools_available(),
    reason="Requires a non-Windows platform with an available OR-Tools backend",
)
def test_production_ortools_branch_executes():
    segment_ids = ["A", "B"]
    segment_embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    generated = np.array([[0.1, 0.0], [0.9, 1.0]], dtype=np.float32)
    result = assign_flow_cann(
        generated,
        segment_ids,
        segment_embeddings,
        {"A": 1, "B": 1},
        capacity_mode="exact_quota",
        candidate_schedule=[1, 2],
        stable_rounds=0,
        full_support_threshold=2,
        time_limit_seconds=20,
        max_edges=100,
        require_ortools=True,
    )
    assert sorted(result.assigned_segment_ids.tolist()) == ["A", "B"]
    assert result.diagnostics["solver_backend"] == "ortools"


def test_hard_edge_budget_blocks_unbounded_graph():
    generated = np.zeros((100, 2), dtype=np.float32)
    embeddings = np.zeros((100, 2), dtype=np.float32)
    capacities = {f"S{i}": 1 for i in range(100)}
    with pytest.raises(RuntimeError, match="hard edge budget|CANN_MAX_EDGES|candidate"):
        assign_flow_cann(
            generated,
            list(capacities),
            embeddings,
            capacities,
            capacity_mode="exact_quota",
            candidate_schedule=[16, 32],
            max_edges=100,
            require_ortools=False,
        )


def test_capacity_only_fills_exact_slots():
    result = assign_capacity_only(4, {"A": 2, "B": 1, "C": 1}, seed=42)
    unique, counts = np.unique(result.assigned_segment_ids, return_counts=True)
    assert dict(zip(unique, counts)) == {"A": 2, "B": 1, "C": 1}
