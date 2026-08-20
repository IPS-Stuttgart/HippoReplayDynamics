from __future__ import annotations

import importlib

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.kd_reference as kd


def test_kd_poisson_rejects_overflowing_expected_counts() -> None:
    with pytest.raises(ValueError, match="scaled expected spike counts must be finite"):
        kd.poisson_log_emissions(
            np.array([[0]], dtype=int),
            np.array([[np.finfo(float).max]], dtype=float),
            2.0,
        )


def test_kd_poisson_preserves_exact_zero_rate_support() -> None:
    actual = kd.poisson_log_emissions(
        np.array([[0], [1]], dtype=int),
        np.array([[0.0]], dtype=float),
        1.0,
    )

    assert actual[0, 0] == 0.0
    assert np.isneginf(actual[1, 0])


@pytest.mark.parametrize(
    ("spike_counts", "rates_hz", "dt", "spike_rate_scale", "match"),
    [
        (np.array([[True]]), np.array([[1.0]]), 1.0, 1.0, "spike_counts.*boolean"),
        (np.array([[0]]), np.array([[True]]), 1.0, 1.0, "rates_hz.*boolean"),
        (np.array([[0]]), np.array([[1.0]]), True, 1.0, "dt.*boolean"),
        (np.array([[0]]), np.array([[1.0]]), 1.0, True, "spike_rate_scale.*boolean"),
    ],
)
def test_kd_poisson_rejects_boolean_numeric_inputs(
    spike_counts,
    rates_hz,
    dt,
    spike_rate_scale,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        kd.poisson_log_emissions(
            spike_counts,
            rates_hz,
            dt,
            spike_rate_scale=spike_rate_scale,
        )


def test_kd_validation_is_restored_after_kd_reference_reload() -> None:
    reloaded = importlib.reload(kd)

    # Python reload preserves dynamically-added module sentinels even though it
    # replaces the module-defined functions with fresh, unwrapped callables.
    assert getattr(reloaded, "_kd_encoding_config_validation_patch_applied", False)
    assert getattr(reloaded, "_kd_transition_parameter_validation_patch_applied", False)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="n_bins must be a positive integer"):
        reloaded.diffusion_transition_1d(True, 0.1, 4.0, 0.02)

    invalid_config = reloaded.KDEncodingConfig(n_bins_x=0)
    with pytest.raises(ValueError, match="n_bins_x must be positive"):
        reloaded.fit_kd_place_field_encoding(object(), invalid_config)

    with pytest.raises(ValueError, match="scaled expected spike counts must be finite"):
        reloaded.poisson_log_emissions(
            np.array([[0]], dtype=int),
            np.array([[np.finfo(float).max]], dtype=float),
            2.0,
        )
