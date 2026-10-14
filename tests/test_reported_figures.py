import numpy as np
import pandas as pd
import pytest

from geoctgan.reported_figures import (
    association_matrix,
    categorical_hd,
    numeric_hd,
    seasonal_matrix,
)


def test_hellinger_identical_distributions():
    categories = pd.Series(["a", "b", "a", None])
    numbers = pd.Series(np.arange(100))
    assert categorical_hd(categories, categories) == 0
    assert numeric_hd(numbers, numbers) == 0


def test_hellinger_disjoint_categories():
    assert categorical_hd(pd.Series(["a"] * 3), pd.Series(["b"] * 7)) == pytest.approx(1)


def test_seasonal_shares_are_within_category():
    frame = pd.DataFrame({"OFFENCE": ["a", "a", "a", "b"], "OCC_MONTH": [1, 1, 2, 3]})
    matrix = seasonal_matrix(frame, "OFFENCE", ["a", "b", "c"])
    assert matrix.shape == (3, 12)
    np.testing.assert_allclose(matrix.sum(axis=1), [100, 100, 0])
    assert matrix[0, 0] == pytest.approx(200 / 3)
    assert matrix[1, 2] == 100


def test_association_matrices_are_symmetric():
    frame = pd.DataFrame({"a": [0, 1, 0, 1] * 10, "b": [1, 1, 0, 0] * 10})
    for kind in ["Pearson", "CramersV"]:
        matrix = association_matrix(frame, ["a", "b"], kind)
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 1)
        assert matrix[0, 1] == pytest.approx(0, abs=1e-12)
