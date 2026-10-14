import json
from pathlib import Path

import pandas as pd

from geoctgan.config import Config
from geoctgan.features import (
    decode_cyclic,
    encode_cyclic_one_based,
    feature_engineering,
)


def test_month_round_trip_including_december():
    months = pd.Series(range(1, 13))
    encoded = encode_cyclic_one_based(months, 12, "month")
    decoded = decode_cyclic(encoded["month_sin"], encoded["month_cos"], 12, one_based=True)
    assert decoded.tolist() == list(range(1, 13))


def test_nonfinite_cyclic_values_remain_missing():
    decoded = decode_cyclic(pd.Series([float("nan")]), pd.Series([1.0]), 24)
    assert pd.isna(decoded.iloc[0])


def test_feature_engineering_creates_nonexistent_output_directory(tmp_path):
    cfg = Config(EMB_DIM=2, COHORT_KIND="year_month")
    train = pd.DataFrame(
        {
            cfg.COL_SEG: ["seg_a", "seg_b"],
            cfg.COL_ID: ["event_1", "event_2"],
            cfg.COL_DATE: pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "OCC_YEAR": [2024, 2024],
            "OCC_MONTH": [1, 1],
            "OCC_DAY": [3, 4],
            "OCC_DOW": [2, 3],
            "OCC_HOUR": [10, 11],
            "OCC_HOUR_MISSING": [0, 0],
            cfg.COL_CAT: ["Assault", "Assault"],
            cfg.COL_OFFENCE: ["Assault", "Assault"],
            "seg_pos": [0.25, 0.75],
            "SEG_MAP_DIST_M": [1.0, 2.0],
        }
    )
    embeddings = pd.DataFrame(
        {
            cfg.COL_SEG: ["seg_a", "seg_b"],
            "gemb_0": [0.1, 0.2],
            "gemb_1": [0.3, 0.4],
        }
    )
    train_path = tmp_path / "train.parquet"
    embedding_path = tmp_path / "embeddings.parquet"
    output_dir = tmp_path / "nested" / "Intermediate" / "Features"
    train.to_parquet(train_path, index=False)
    embeddings.to_parquet(embedding_path, index=False)

    raw_path, model_path, crosswalk_path, pools = feature_engineering(
        cfg,
        str(train_path),
        str(embedding_path),
        str(output_dir),
    )

    assert Path(raw_path).is_file()
    assert Path(model_path).is_file()
    assert Path(crosswalk_path).is_file()
    assert (output_dir / "feature_summary.json").is_file()
    assert pools == ["2024_01"]
    assert json.loads(Path(crosswalk_path).read_text(encoding="utf-8")) == {
        "Assault": ["Assault"]
    }
