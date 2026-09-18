import numpy as np
import pytest

from scripts.report_tirole_content_uncertainty import block_weights, composition_interval, mean_interval


def test_identical_values_have_exact_interval_and_one_block_is_insufficient():
    codes, w = block_weights([0, 10, 100, 200], 60, 3)
    result = mean_interval(np.ones(4) * 0.4, codes, w)
    assert result["ci025"] == pytest.approx(0.4)
    assert result["ci975"] == pytest.approx(0.4)
    c, w = block_weights([0, 10], 60, 3)
    result = mean_interval([1, 2], c, w)
    assert np.isnan(result["ci025"])


def test_missing_content_does_not_become_a_zero_effect():
    codes, w = block_weights([0, 100, 200], 60, 3)
    result = mean_interval([np.nan] * 3, codes, w)
    assert np.isnan(result["point"]) and result["finite_bootstrap_fraction"] == 0


def test_split_duplication_cannot_tighten_composition_interval():
    codes, w = block_weights([0, 100, 200, 300], 60, 3)
    q = np.array([[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.9, 0.8]])
    full = np.ones_like(q, bool)
    half = np.array([[1, 0], [1, 1], [0, 1], [1, 1]], bool)
    a = composition_interval(q, full, half, codes, w, True)
    b = composition_interval(np.tile(q, (1, 5)), np.tile(full, (1, 5)), np.tile(half, (1, 5)), codes, w, True)
    for k in ["point", "ci025", "ci975"]:
        assert a[k] == pytest.approx(b[k])


def test_total_change_and_selective_loss_are_different():
    codes, w = block_weights([0, 100, 200], 60, 3)
    q = np.array([[0.1], [0.3], [0.9]])
    full = np.array([[1], [1], [0]], bool)
    thin = np.array([[0], [1], [1]], bool)
    loss = composition_interval(q, full, thin, codes, w, True)
    total = composition_interval(q, full, thin, codes, w, False)
    assert loss["point"] == pytest.approx(0.1)
    assert total["point"] == pytest.approx(0.4)
    assert loss["finite_bootstrap_fraction"] < 1
