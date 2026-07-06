import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _mass_retaining_candidate_indices,
    _mode_transition_matrix,
    _top_candidate_indices,
)


def _emissions() -> LogEmissionTensor:
    log_likelihood = np.log(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.80, 0.10],
                [0.15, 0.25, 0.60],
            ],
            dtype=float,
        )
    )
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((log_likelihood.shape[0], 1), dtype=int),
        times=np.arange(log_likelihood.shape[0], dtype=float),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mass_threshold": "0.75"}, "mass_threshold"),
        ({"mass_threshold": np.asarray("0.75")}, "mass_threshold"),
        ({"mass_threshold": 0.75, "top_k": "2"}, "top_k"),
        ({"mass_threshold": 0.75, "min_k": np.asarray("1")}, "min_k"),
        ({"mass_threshold": 0.75, "max_k": "0"}, "max_k"),
    ],
)
def test_mass_retaining_candidate_indices_reject_string_scalars(kwargs, message):
    with pytest.raises(TypeError, match=rf"{message}.*string"):
        _mass_retaining_candidate_indices(np.log(np.array([0.6, 0.3, 0.1])), **kwargs)


def test_top_candidate_indices_rejects_string_top_k():
    with pytest.raises(TypeError, match="top_k.*string"):
        _top_candidate_indices(np.log(np.array([0.6, 0.3, 0.1])), "2")


def test_state_space_model_rejects_string_candidate_config_before_coercion():
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold="0.75",
            momentum_candidate_top_k=2,
            momentum_predicted_candidate_top_k=0,
        ),
    )

    with pytest.raises(TypeError, match="momentum_candidate_mass_threshold.*string"):
        model.candidate_indices(_emissions())


def test_state_space_model_rejects_string_candidate_counts_before_coercion():
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_mass_threshold=0.75,
            momentum_candidate_top_k="2",
            momentum_predicted_candidate_top_k=0,
        ),
    )

    with pytest.raises(TypeError, match="momentum_candidate_top_k.*string"):
        model.candidate_indices(_emissions())


def test_mode_transition_matrix_rejects_string_stickiness():
    with pytest.raises(TypeError, match="mode_stickiness.*string"):
        _mode_transition_matrix(3, "0.95")
