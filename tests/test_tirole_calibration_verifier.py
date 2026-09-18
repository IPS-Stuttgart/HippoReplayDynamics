import numpy as np
import pytest

from scripts.verify_tirole_measurement_calibration import check_close, direct_content, draw, path


def test_direct_context_is_sequenceless_and_signed():
    rates = np.array([[[10.0, 10.0], [1.0, 1.0]], [[1.0, 1.0], [10.0, 10.0]]])
    counts = np.array([[3, 0], [1, 0], [0, 0]])
    valid = np.ones((2, 2), bool)
    swaps = np.array([[False, True], [True, False], [True, True]] * 20)
    for conditional in [False, True]:
        a = direct_content(counts, rates, valid, swaps, conditional)
        assert a[0] > 0
        np.testing.assert_allclose(a, direct_content(counts[::-1], rates, valid, swaps, conditional))
        assert np.isnan(direct_content(np.zeros_like(counts), rates, valid, swaps, conditional)[0])


def test_independent_draw_preserves_counts_and_path_support():
    rng = np.random.default_rng(4)
    x = draw(np.array([0, 3, 7]), np.ones((3, 5)), rng)
    np.testing.assert_equal(x.sum(axis=1), [0, 3, 7])
    valid = np.ones((2, 20), bool)
    valid[:, :4] = False
    p = path(valid, 10, rng)
    assert valid[:, p].all()
    assert (np.diff(p) >= 0).all() or (np.diff(p) <= 0).all()


def test_independent_comparison_rejects_corruption():
    with pytest.raises(ValueError, match="mismatch"):
        check_close(1.0, 1.1)
