from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipporeplayimm.result_improvement_extensions import build_sorted_emissions_with_replay_calibration


def _config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "time_bin_s": 0.005,
        "spike_rate_scale": 1.0,
        "likelihood_temperature": 1.0,
        "negative_binomial_overdispersion": 0.0,
        "cell_weights": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _calibration(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "gain_prior_count": 10.0,
        "max_gain": 20.0,
        "negative_binomial_dispersion": 50.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("parameter", "config", "calibration"),
    [
        ("time_bin_s", _config(time_bin_s=10**400), None),
        ("spike_rate_scale", _config(spike_rate_scale=10**400), None),
        ("likelihood_temperature", _config(likelihood_temperature=10**400), None),
        (
            "negative_binomial_overdispersion",
            _config(negative_binomial_overdispersion=10**400),
            None,
        ),
        ("gain_prior_count", None, _calibration(gain_prior_count=10**400)),
        ("max_gain", None, _calibration(max_gain=10**400)),
        (
            "negative_binomial_dispersion",
            None,
            _calibration(negative_binomial_dispersion=10**400),
        ),
    ],
)
def test_replay_calibrated_emissions_normalize_numeric_overflow(
    parameter: str,
    config: SimpleNamespace | None,
    calibration: SimpleNamespace | None,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        build_sorted_emissions_with_replay_calibration(
            object(),
            object(),
            0,
            config,
            calibration,
        )
