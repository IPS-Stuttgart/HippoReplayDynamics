import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import _posterior_diagnostics
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel
from hipporeplayimm.state_space_displacement_momentum import _shifted_gaussian_transition_matrix
from hipporeplayimm.state_space_utils import _as_log_probs, _mean_entropy, _scaled_emissions


def test_scaled_emissions_uses_valid_mask_before_row_offset():
    log_likelihood = np.array(
        [
            [1000.0, 0.0, -1.0],
            [999.0, -0.25, -0.75],
        ]
    )
    valid_mask = np.array([False, True, True])

    scaled, offsets = _scaled_emissions(log_likelihood, valid_bin_mask=valid_mask)

    np.testing.assert_allclose(offsets, [0.0, -0.25])
    np.testing.assert_allclose(scaled[:, 0], 0.0)
    np.testing.assert_allclose(scaled[0, 1:], [1.0, np.exp(-1.0)])
    np.testing.assert_allclose(scaled[1, 1:], [1.0, np.exp(-0.5)])


def test_displacement_transition_rejects_nonbinary_valid_masks():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="boolean or 0/1"):
        _shifted_gaussian_transition_matrix(
            centers,
            displacement=np.zeros(2, dtype=float),
            sigma_cm=1.0,
            max_step_sigma=2.0,
            valid_bin_mask=np.array([2, 0], dtype=int),
        )


def test_as_log_probs_rejects_zero_mass_rows():
    with pytest.raises(ValueError, match="positive finite mass"):
        _as_log_probs(np.array([[0.0, 0.0]], dtype=float))


def test_entropy_helpers_ignore_impossible_states():
    log_probs = _as_log_probs(np.array([[1.0, 0.0]], dtype=float))

    assert _mean_entropy(log_probs) == pytest.approx(0.0)
    diagnostics = _posterior_diagnostics(
        log_probs[0],
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    assert diagnostics["terminal_posterior_entropy"] == pytest.approx(0.0)


def test_fragmented_state_space_occupancy_mask_ignores_invalid_dominant_bins():
    log_likelihood = np.array(
        [
            [1000.0, 0.0, -1.0],
            [999.0, -0.25, -0.75],
        ]
    )
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((2, 0), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    occupancy_s = np.array([0.0, 1.0, 1.0])
    model = StateSpaceReplayModel(
        mode="fragmented",
        config=StateSpaceDecoderConfig(
            mode="fragmented",
            valid_occupancy_threshold_s=0.5,
        ),
    )

    score = model.score(emissions, bin_centers, occupancy_s=occupancy_s)

    expected_logp = (
        np.log(np.exp(0.0) + np.exp(-1.0))
        + np.log(np.exp(-0.25) + np.exp(-0.75))
        - 2.0 * np.log(2.0)
    )
    assert np.isclose(score.log_likelihood, expected_logp)
    assert score.trajectory_log_posterior is not None
    np.testing.assert_allclose(
        np.exp(score.trajectory_log_posterior[:, 0]),
        0.0,
        atol=0.0,
    )
