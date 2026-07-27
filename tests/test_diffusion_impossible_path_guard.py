import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import DiffusionModel


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


def test_diffusion_model_rejects_disconnected_emission_support() -> None:
    emissions = _emissions(
        np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
            ]
        )
    )

    with pytest.raises(ValueError, match="diffusion model has no finite path mass"):
        DiffusionModel(sigma_cm=1.0, max_step_sigma=1.0).score(
            emissions,
            _disconnected_bin_centers(),
        )


def test_diffusion_model_keeps_connected_emission_support() -> None:
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
    assert not np.any(np.isnan(score.terminal_log_posterior))
    assert np.isclose(np.exp(score.terminal_log_posterior).sum(), 1.0)
    assert np.isneginf(score.terminal_log_posterior[1])
    assert score.diagnostics["decoded_map_bin"] == 0


def test_diffusion_model_does_not_promote_unreachable_finite_sentinel() -> None:
    emissions = _emissions(
        np.array(
            [
                [-6.0e299, -np.inf],
                [-6.0e299, 0.0],
            ]
        )
    )

    score = DiffusionModel(sigma_cm=1.0, max_step_sigma=1.0).score(
        emissions,
        _disconnected_bin_centers(),
    )

    assert np.isfinite(score.log_likelihood)
    assert score.log_likelihood < -1.1e300
    assert score.terminal_log_posterior is not None
    assert score.terminal_log_posterior[0] == pytest.approx(0.0)
    assert np.isneginf(score.terminal_log_posterior[1])
    assert score.diagnostics["decoded_map_bin"] == 0
