from pathlib import Path
import importlib.util
import pickle

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from geoctgan.config import Config
from geoctgan.graph import build_graph, load_graph, train_deepwalk


pytestmark = pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow unavailable")


def _base_segments():
    return gpd.GeoDataFrame(
        {
            "segment_id": ["a0", "a1", "b0"],
            "source_edge_uid": ["edge_a", "edge_a", "edge_b"],
            "source_part": [0, 0, 0],
            "child_index": [0, 1, 0],
            "child_count": [2, 2, 1],
            "source_u": [1, 1, 2],
            "source_v": [2, 2, 3],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(2, 0), (3, 0)]),
            ],
        },
        crs="EPSG:32617",
    )


def test_authoritative_topology_connects_only_parent_end_children(tmp_path: Path):
    segments = _base_segments()
    path = tmp_path / "segments.parquet"
    segments.to_parquet(path, index=False)
    cfg = Config(CRS_METRIC="EPSG:32617", MAX_ALLOWED_ISOLATED_SEGMENTS=0)
    graph_path = build_graph(cfg, str(path), str(tmp_path))
    graph = load_graph(graph_path)
    lookup = {segment: index for index, segment in enumerate(graph.segment_ids)}
    assert lookup["a1"] in set(graph.adjacency[lookup["a0"]])
    assert lookup["b0"] in set(graph.adjacency[lookup["a1"]])
    assert lookup["b0"] not in set(graph.adjacency[lookup["a0"]])


def test_repeated_osmid_cannot_chain_distinct_parent_edges(tmp_path: Path):
    segments = _base_segments()
    segments["source_osmid"] = [999, 999, 999]
    path = tmp_path / "segments.parquet"
    segments.to_parquet(path, index=False)
    graph = load_graph(build_graph(Config(CRS_METRIC="EPSG:32617"), str(path), str(tmp_path)))
    lookup = {segment: index for index, segment in enumerate(graph.segment_ids)}
    assert lookup["b0"] not in set(graph.adjacency[lookup["a0"]])


def test_isolated_segments_fail_closed(tmp_path: Path):
    segments = gpd.GeoDataFrame(
        {
            "segment_id": ["a", "b"],
            "source_edge_uid": ["edge_a", "edge_b"],
            "child_index": [0, 0],
            "child_count": [1, 1],
            "source_u": [1, 10],
            "source_v": [2, 11],
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(10, 0), (11, 0)])],
        },
        crs="EPSG:32617",
    )
    path = tmp_path / "isolated.parquet"
    segments.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="isolated"):
        build_graph(Config(CRS_METRIC="EPSG:32617"), str(path), str(tmp_path))


def test_deepwalk_creates_missing_output_directory(tmp_path: Path):
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump(
            {
                "segment_ids": ["a", "b", "c"],
                "adjacency": [
                    np.asarray([1], dtype=np.int32),
                    np.asarray([0, 2], dtype=np.int32),
                    np.asarray([1], dtype=np.int32),
                ],
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    output_dir = tmp_path / "missing" / "Embeddings"
    cfg = Config(
        EMB_DIM=4,
        N_WALKS=1,
        WALK_LEN=3,
        W2V_WINDOW=2,
        W2V_EPOCHS=1,
        W2V_WORKERS=1,
    )
    embedding_path, _ = train_deepwalk(
        cfg,
        str(graph_path),
        str(output_dir),
        tag="directory_test",
    )

    assert output_dir.is_dir()
    assert Path(embedding_path).is_file()
    assert (output_dir / "deepwalk_directory_test.model").is_file()
    assert (output_dir / "deepwalk_directory_test_runtime.json").is_file()
