import numpy as np
import pytest

from hipporeplayimm.duration_dynamics import DurationFloat
from hipporeplayimm.kd_reference import (
    diffusion_transition_1d,
    momentum_transition_1d,
    stationary_gaussian_transition_1d,
)


@pytest.mark.parametrize(
    ("builder", "args", "error", "match"),
    [
        (diffusion_transition_1d, (3, 1.0, 4.0, 0.0), ValueError, "dt"),
        (diffusion_transition_1d, (3, 1.0, 4.0, -0.1), ValueError, "dt"),
        (diffusion_transition_1d, (3, 1.0, 0.0, 0.02), ValueError, "bin_size_cm"),
        (diffusion_transition_1d, (3, -1.0, 4.0, 0.02), ValueError, "sd_meters"),
        (diffusion_transition_1d, (0, 1.0, 4.0, 0.02), ValueError, "n_bins"),
        (diffusion_transition_1d, (True, 1.0, 4.0, 0.02), TypeError, "n_bins"),
        (momentum_transition_1d, (3, 1.0, np.nan, 4.0, 0.02), ValueError, "decay"),
        (momentum_transition_1d, (3, 1.0, 1.0, 4.0, np.inf), ValueError, "dt"),
        (stationary_gaussian_transition_1d, (3, 1.0, 0.0), ValueError, "bin_size_cm"),
    ],
)
def test_kd_transition_builders_reject_invalid_physical_parameters(builder, args, error, match):
    with pytest.raises(error, match=match):
        builder(*args)


def test_kd_diffusion_transition_remains_column_stochastic_for_valid_parameters():
    transition = diffusion_transition_1d(4, 0.25, 5.0, 0.02)

    assert transition.shape == (4, 4)
    assert np.all(np.isfinite(transition))
    assert np.allclose(transition.sum(axis=0), 1.0)


def test_kd_diffusion_transition_preserves_duration_metadata():
    transitions = diffusion_transition_1d(
        3,
        0.25,
        5.0,
        DurationFloat(0.02, transition_durations=(0.01, 0.03)),
    )

    assert isinstance(transitions, list)
    assert len(transitions) == 2
    assert all(np.allclose(transition.sum(axis=0), 1.0) for transition in transitions)
