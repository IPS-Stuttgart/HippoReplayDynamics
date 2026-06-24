import numpy as np
import pytest

from hipporeplayimm.kd_reference import best_grid_params, empirical_grid_prior


def test_kd_empirical_prior_ignores_nonfinite_2d_event_rows() -> None:
    sd_meters = np.array([0.1, 0.5, 1.0], dtype=float)
    grid = np.array(
        [
            [1.0, 2.0, -np.inf],
            [np.nan, -np.inf, np.nan],
        ],
        dtype=float,
    )

    prior, diagnostics = empirical_grid_prior({"sd_meters": sd_meters}, grid)
    rows = best_grid_params("diffusion", [10, 11], {"sd_meters": sd_meters}, grid)

    assert prior.shape == sd_meters.shape
    assert np.all(np.isfinite(prior))
    assert diagnostics["sd_prior_mass"] == pytest.approx(float(prior.sum()))
    assert diagnostics["finite_event_rows"] == 1
    assert rows[0]["best_sd_meters"] == 0.5
    assert rows[0]["best_log_evidence"] == 2.0
    assert np.isnan(rows[1]["best_sd_meters"])
    assert np.isnan(rows[1]["best_log_evidence"])


def test_kd_empirical_prior_ignores_nonfinite_3d_event_rows() -> None:
    sd_meters = np.array([40.0, 120.0], dtype=float)
    decay = np.array([1.0, 100.0], dtype=float)
    grid = np.array(
        [
            [[0.0, 3.0], [1.0, 2.0]],
            [[-np.inf, np.nan], [np.nan, -np.inf]],
        ],
        dtype=float,
    )

    prior, diagnostics = empirical_grid_prior({"sd_meters": sd_meters, "decay": decay}, grid)
    rows = best_grid_params("momentum", [20, 21], {"sd_meters": sd_meters, "decay": decay}, grid)

    assert prior.shape == (sd_meters.shape[0], decay.shape[0])
    assert np.all(np.isfinite(prior))
    assert diagnostics["joint_prior_mass"] == pytest.approx(float(prior.sum()))
    assert diagnostics["finite_event_rows"] == 1
    assert rows[0]["best_sd_meters"] == 40.0
    assert rows[0]["best_decay"] == 100.0
    assert rows[0]["best_log_evidence"] == 3.0
    assert np.isnan(rows[1]["best_sd_meters"])
    assert np.isnan(rows[1]["best_decay"])
    assert np.isnan(rows[1]["best_log_evidence"])


def test_kd_empirical_prior_rejects_all_nonfinite_grid_rows() -> None:
    sd_meters = np.array([0.1, 0.5], dtype=float)
    grid = np.full((2, 2), -np.inf, dtype=float)

    with pytest.raises(ValueError, match="at least one finite"):
        empirical_grid_prior({"sd_meters": sd_meters}, grid)
