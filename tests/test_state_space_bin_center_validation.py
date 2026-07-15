import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceReplayModel


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.6, 0.4],
                    [0.3, 0.7],
                ]
            )
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 1.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_state_space_score_accepts_sequence_bin_centers():
    score = StateSpaceReplayModel(mode="fragmented").score(
        _emissions(),
        [[0.0, 0.0], [1.0, 0.0]],
    )

    assert np.isfinite(score.log_likelihood)
    assert score.terminal_log_posterior is not None
    assert np.all(np.isfinite(score.terminal_log_posterior))


def test_state_space_score_accepts_1d_bin_centers():
    score = StateSpaceReplayModel(mode="fragmented").score(
        _emissions(),
        np.array([0.0, 1.0]),
    )

    assert np.isfinite(score.diagnostics["decoded_endpoint_x"])
    assert score.diagnostics["decoded_endpoint_y"] == 0.0


def test_state_space_score_rejects_nonfinite_bin_centers():
    with pytest.raises(ValueError, match="bin_centers must be finite"):
        StateSpaceReplayModel(mode="fragmented").score(
            _emissions(),
            np.array([[0.0, 0.0], [np.nan, 0.0]]),
        )


def test_state_space_score_normalizes_bin_center_overflow():
    with pytest.raises(ValueError, match="bin_centers must contain numeric real coordinates"):
        StateSpaceReplayModel(mode="fragmented").score(
            _emissions(),
            [[0, 0], [10**1000, 0]],
        )
