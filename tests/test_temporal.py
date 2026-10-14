import pandas as pd

from geoctgan.temporal import condition_columns, pool_id, pool_series


def test_pool_ids():
    assert pool_id(2019, 2, "year_month") == "2019_02"
    assert pool_id(2019, 5, "year_quarter") == "2019_Q2"
    assert pool_id(2019, 5, "year") == "2019"
    assert pool_id(2019, 5, "global") == "GLOBAL"


def test_condition_columns():
    assert condition_columns("year_month", "MCI_CATEGO") == ["MCI_CATEGO"]
    assert condition_columns("year", "MCI_CATEGO") == ["OCC_MONTH_CAT", "MCI_CATEGO"]
    assert condition_columns("global", "MCI_CATEGO") == [
        "OCC_YEAR_CAT",
        "OCC_MONTH_CAT",
        "MCI_CATEGO",
    ]
