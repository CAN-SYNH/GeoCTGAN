from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from geoctgan.config import Config
from geoctgan.data import create_outer_split


def test_outer_split_manifest_is_reused_exactly(tmp_path: Path) -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "EVENT_PK": f"event_{index:03d}",
                "OCC_YEAR": 2024,
                "OCC_MONTH": 1 + index % 2,
                "MCI_CATEGO": "Assault" if index % 2 == 0 else "Robbery",
                "geometry": Point(-79.4 + index * 0.0001, 43.7),
            }
        )
    events = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    events_path = tmp_path / "events.gpkg"
    events.to_file(events_path, layer="events", driver="GPKG")

    cfg = Config(
        BASE_DIR=str(tmp_path),
        REQUIRE_SELECTED_CONFIG=False,
        REQUIRE_REUSED_OUTER_SPLIT=False,
    )
    first_dir = tmp_path / "first"
    first_train, first_test, manifest_path = create_outer_split(
        cfg,
        str(events_path),
        str(first_dir),
    )
    second_dir = tmp_path / "second"
    second_train, second_test, reused_manifest_path = create_outer_split(
        cfg,
        str(events_path),
        str(second_dir),
        split_manifest_path=manifest_path,
    )

    first_train_ids = set(
        gpd.read_file(first_train, layer="events")["EVENT_PK"].astype(str)
    )
    first_test_ids = set(
        gpd.read_file(first_test, layer="events")["EVENT_PK"].astype(str)
    )
    second_train_ids = set(
        gpd.read_file(second_train, layer="events")["EVENT_PK"].astype(str)
    )
    second_test_ids = set(
        gpd.read_file(second_test, layer="events")["EVENT_PK"].astype(str)
    )
    reused_manifest = json.loads(
        Path(reused_manifest_path).read_text(encoding="utf-8")
    )

    assert first_train_ids == second_train_ids
    assert first_test_ids == second_test_ids
    assert not first_train_ids.intersection(first_test_ids)
    assert reused_manifest["reused_existing_split"] is True
