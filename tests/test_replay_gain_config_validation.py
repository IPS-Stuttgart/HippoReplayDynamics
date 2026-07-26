from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm.accuracy_upgrades import ReplayGainConfig


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prior_observed_spikes", -1.0, "prior_observed_spikes must be nonnegative"),
        ("prior_observed_spikes", np.nan, "prior_observed_spikes must be finite"),
        ("prior_expected_spikes", 0.0, "prior_expected_spikes must be positive"),
        ("prior_expected_spikes", np.inf, "prior_expected_spikes must be finite"),
        ("min_gain", 0.0, "min_gain must be positive"),
        ("max_gain", -1.0, "max_gain must be positive"),
    ],
)
def test_replay_gain_config_rejects_invalid_numeric_parameters(field, value, message):
    with pytest.raises(ValueError, match=message):
        ReplayGainConfig(**{field: value})


def test_replay_gain_config_rejects_boolean_and_text_parameters():
    with pytest.raises(TypeError, match="min_gain must be numeric, not boolean"):
        ReplayGainConfig(min_gain=True)
    with pytest.raises(ValueError, match="max_gain must be numeric, not text"):
        ReplayGainConfig(max_gain="2.0")


def test_replay_gain_config_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="min_gain must be less than or equal to max_gain"):
        ReplayGainConfig(min_gain=2.0, max_gain=1.0)


def test_replay_gain_config_accepts_and_canonicalizes_scalar_wrappers():
    config = ReplayGainConfig(
        prior_observed_spikes=np.array(0.0),
        prior_expected_spikes=np.float32(0.5),
        min_gain=np.float32(0.2),
        max_gain=np.array(0.3),
    )

    assert config.prior_observed_spikes == 0.0
    assert np.isclose(config.prior_expected_spikes, 0.5)
    assert np.isclose(config.min_gain, 0.2)
    assert config.max_gain == 0.3
    assert all(
        isinstance(getattr(config, field), float)
        for field in ("prior_observed_spikes", "prior_expected_spikes", "min_gain", "max_gain")
    )


def test_replay_gain_config_validation_remains_idempotent_after_runtime_refresh():
    apply_runtime_patches()
    apply_runtime_patches()

    with pytest.raises(ValueError, match="min_gain must be positive"):
        ReplayGainConfig(min_gain=0.0)
