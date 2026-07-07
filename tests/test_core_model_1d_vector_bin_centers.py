import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel, RandomModel, StationaryModel


def _simple_1d_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.10, 0.20, 0.70],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_core_models_accept_vector_bin_centers_for_one_dimensional_tracks():
    emissions = _simple_1d_emissions()
    vector_centers = np.array([0.0, 1.0, 2.0])
    column_centers = vector_centers[:, None]
    replay_models = [
        RandomModel(),
        StationaryModel(),
        DiffusionModel(sigma_cm=1.0, max_step_sigma=10.0),
        CandidateKinematicModel(mode="diffusion", top_k=3, diffusion_sigma_cm=1.0),
    ]

    for model in replay_models:
        vector_score = model.score(emissions, vector_centers)
        column_score = model.score(emissions, column_centers)

        assert np.isfinite(vector_score.log_likelihood)
        assert np.allclose(vector_score.log_likelihood, column_score.log_likelihood)
        np.testing.assert_allclose(
            vector_score.terminal_log_posterior,
            column_score.terminal_log_posterior,
        )
        np.testing.assert_allclose(
            vector_score.trajectory_log_posterior,
            column_score.trajectory_log_posterior,
        )
        assert vector_score.diagnostics["decoded_endpoint_y"] == 0.0
        assert vector_score.diagnostics["decoded_map_y"] == 0.0
