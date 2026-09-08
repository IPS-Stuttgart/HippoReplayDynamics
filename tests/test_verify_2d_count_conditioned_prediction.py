import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext
from scripts.verify_2d_count_conditioned_prediction import parse_ids, read_fold_table, reference_posterior


@pytest.mark.parametrize("n_time", [1, 3, 17])
@pytest.mark.parametrize("model", ["diffusion", "first_order_imm"])
def test_independent_dynamic_solver(n_time, model):
    rng = np.random.default_rng(235)
    centers = np.array([[0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [16, 8.0]])
    ll = rng.normal(-4, 2, (n_time, len(centers)))
    times = np.arange(n_time) * 0.02
    if n_time > 1:
        times[-1] -= 0.004
    expected = SpatialPredictionContext(centers).infer(ll, times)[0][model]
    actual = reference_posterior(ll, centers, times, model == "first_order_imm")
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(logsumexp(actual, axis=1), 0, atol=1e-10)


def test_independent_reference_depends_on_training_input():
    centers = np.array([[0, 0], [8, 0], [0, 8], [8, 8.0]])
    times = np.arange(3) * 0.02
    flat = reference_posterior(np.zeros((3, 4)), centers, times, True)
    strong = np.zeros((3, 4))
    strong[1, 0] = 10
    changed = reference_posterior(strong, centers, times, True)
    assert not np.allclose(flat, changed)


def test_single_excluded_id_and_empty_csv_cells(tmp_path):
    path = tmp_path / "folds.csv"
    path.write_text("fold,test_ids,calibration_ids,excluded_ids\n0,1,2,116\n1,2,1,\n")
    frame = read_fold_table(path)
    assert parse_ids(frame.iloc[0].excluded_ids) == [116]
    assert parse_ids(frame.iloc[1].excluded_ids) == []
    assert parse_ids(frame.iloc[0].test_ids) == [1]


def test_fractional_identifier_is_not_silently_rounded():
    with pytest.raises(ValueError):
        parse_ids("116.3")
