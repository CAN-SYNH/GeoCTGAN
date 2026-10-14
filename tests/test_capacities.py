from geoctgan.flow import build_capacities, largest_remainder_counts


def test_largest_remainder_exact_and_nonnegative():
    result = largest_remainder_counts({"A": 5, "B": 3, "C": 2}, 7)
    assert result == {"A": 4, "B": 2, "C": 1}
    assert sum(result.values()) == 7
    assert min(result.values()) >= 0


def test_exact_quota_uses_exact_total():
    result = build_capacities({"A": 5, "B": 3, "C": 2}, 10, mode="exact_quota")
    assert result == {"A": 5, "B": 3, "C": 2}


def test_upper_quota_has_slack():
    result = build_capacities({"A": 5, "B": 3, "C": 2}, 10, mode="upper_quota", upper_slack=0.2)
    assert sum(result.values()) == 12
