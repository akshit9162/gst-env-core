from gst_env.stats import bootstrap_mean, tasks_for_half_width, wilson


def test_wilson_is_wide_at_three_tasks():
    lo, hi = wilson(0, 3)
    assert lo == 0 and 0.5 < hi < 0.6  # 0/3 is still consistent with a 56% pass rate
    lo, hi = wilson(3, 3)
    assert 0.4 < lo < 0.5 and hi == 1


def test_wilson_matches_known_value():
    lo, hi = wilson(50, 100)
    assert round(lo, 3) == 0.404 and round(hi, 3) == 0.596


def test_bootstrap_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.5, 1.0, 1.0, 0.25]
    lo, hi = bootstrap_mean(values)
    assert lo <= sum(values) / len(values) <= hi
    assert (lo, hi) == bootstrap_mean(values)


def test_sample_size_for_half_width():
    assert tasks_for_half_width(0.10) == 97
    assert tasks_for_half_width(0.15) == 43


def test_pass_at_k():
    from gst_env.stats import pass_at_k
    assert pass_at_k(16, 0, 16) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert pass_at_k(16, 1, 1) == 1 / 16
    assert pass_at_k(16, 1, 16) == 1.0
    assert abs(pass_at_k(4, 1, 2) - 0.5) < 1e-12  # 1 - C(3,2)/C(4,2)
    import pytest
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)
