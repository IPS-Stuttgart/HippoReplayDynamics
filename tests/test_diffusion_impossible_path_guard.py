import warnings

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import DiffusionModel


_POSTERIOR_DIAGNOSTIC_KEYS = {
    "decoded_endpoint_x",
    "decoded_endpoint_y",
    "decoded_map_x",
    "decoded_map_y",
    "decoded_map_bin",
    "terminal_posterior_entropy",
}


def _emissions(log_likelihood: np.ndarray) -> LogEmissionTensor:
    values = np.asarray(log_likelihood, dtype=float)
    return LogEmissionTensor(
        log_likelihood=values,
        spike_counts=np.zeros((values.shape[0], 1), dtype=int),
        times=np.arange(values.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def _disconnected_bin_centers() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
        ]
    )


def test_diffusion_model_clears_undefined_disconnected_path_posterior() -> None:
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
            ]
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        score = DiffusionModel(sigma_cm=1.0, max_step_sigma=1.0).score(
            emissions,
            _disconnected_bin_centers(),
        )

    assert np.isneginf(score.log_likelihood)
    assert score.terminal_log_posterior is None
    assert score.trajectory_log_posterior is None
    assert score.diagnostics["diffusion_path_support"] == "no_finite_path"
    assert _POSTERIOR_DIAGNOSTIC_KEYS.isdisjoint(score.diagnostics)


def test_diffusion_model_keeps_connected_emission_posterior() -> None:
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
            ]
        )
    )

    score = DiffusionModel(sigma_cm=1.0, max_step_sigma=1.0).score(
        emissions,
        _disconnected_bin_centers(),
    )

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert score.trajectory_log_posterior is not None
    assert not np.any(np.isnan(score.terminal_log_posterior))
    assert np.isclose(np.exp(score.terminal_log_posterior).sum(), 1.0)
    assert np.isneginf(score.terminal_log_posterior[1])
    assert score.diagnostics["decoded_map_bin"] == 0
