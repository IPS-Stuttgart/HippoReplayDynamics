import numpy as np
import pandas as pd

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    add_evidence_columns,
    build_scoring_models,
    confusion_matrix,
    emissions_from_counts,
    expected_scoring_model,
    parse_model_list,
    recovery_summary,
    simulate_latent_path,
    simulate_replay_event,
)
from hipporeplayimm.state_space import StateSpaceDecoderConfig


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array(
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ]
        ),
        rates_hz=np.array(
            [
                [20.0, 0.1, 0.1, 0.1],
                [0.1, 20.0, 0.1, 0.1],
                [0.1, 0.1, 20.0, 0.1],
            ]
        ),
        occupancy_s=np.ones(4),
        cell_ids=np.array([1, 2, 3]),
        config=EncodingConfig(bin_size_cm=1.0),
    )


def test_simulated_emissions_have_expected_shape_and_finite_likelihoods():
    rng = np.random.default_rng(1)
    emissions, path = simulate_replay_event(
        _encoding(),
        true_model="diffusion",
        n_time=5,
        dt=0.02,
        rng=rng,
        state_space=StateSpaceDecoderConfig(diffusion_sigma_cm_sqrt_s=4.0),
    )

    assert path.shape == (5,)
    assert emissions.log_likelihood.shape == (5, 4)
    assert emissions.spike_counts.shape == (5, 3)
    assert emissions.n_spikes == int(emissions.spike_counts.sum())
    assert np.isfinite(emissions.log_likelihood).all()


def test_stationary_path_repeats_one_bin_and_fragmented_can_jump():
    encoding = _encoding()
    stationary = simulate_latent_path(
        encoding,
        true_model="stationary",
        n_time=6,
        dt=0.02,
        rng=np.random.default_rng(2),
    )
    fragmented = simulate_latent_path(
        encoding,
        true_model="fragmented",
        n_time=20,
        dt=0.02,
        rng=np.random.default_rng(2),
    )

    assert np.unique(stationary).size == 1
    assert np.unique(fragmented).size > 1


def test_emissions_from_counts_scores_true_preferred_bin():
    encoding = _encoding()
    counts = np.array([[3, 0, 0], [4, 0, 0]])
    emissions = emissions_from_counts(encoding, counts, dt=0.05)

    assert emissions.log_likelihood[0, 0] == emissions.log_likelihood[0].max()
    assert emissions.log_likelihood[1, 0] == emissions.log_likelihood[1].max()


def test_recovery_summary_and_confusion_matrix_use_expected_models():
    rows = pd.DataFrame(
        [
            _row(0, "stationary", "sorted-spike-state-space-stationary", -1.0),
            _row(0, "stationary", "sorted-spike-state-space-diffusion", -3.0),
            _row(1, "diffusion", "sorted-spike-state-space-stationary", -4.0),
            _row(1, "diffusion", "sorted-spike-state-space-diffusion", -2.0),
            _row(2, "momentum", "sorted-spike-state-space-diffusion", -2.0),
            _row(2, "momentum", "sorted-spike-state-space-momentum", -1.0),
        ]
    )
    scored = add_evidence_columns(rows)

    summary = recovery_summary(scored)
    confusion = confusion_matrix(
        scored,
        ("sorted-spike-state-space-stationary", "sorted-spike-state-space-diffusion", "sorted-spike-state-space-momentum"),
    )

    overall = summary[summary["true_model"] == "overall"].iloc[0]
    assert overall["recovered_events"] == 3
    assert overall["simulated_events"] == 3
    assert expected_scoring_model("momentum") == "sorted-spike-state-space-momentum"
    assert confusion.loc[confusion["true_model"] == "momentum", "sorted-spike-state-space-momentum"].iloc[0] == 1


def test_build_scoring_models_and_model_parser_accept_space_or_comma_lists():
    assert parse_model_list("stationary,diffusion momentum") == ("stationary", "diffusion", "momentum")
    models = build_scoring_models(
        SimulationRecoveryConfig(
            scoring_models=("sorted-spike-state-space-stationary", "sorted-spike-state-space-momentum"),
            state_space=StateSpaceDecoderConfig(momentum_candidate_top_k=4),
        )
    )

    assert list(models) == ["sorted-spike-state-space-stationary", "sorted-spike-state-space-momentum"]


def _row(event_index: int, true_model: str, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "true_model": true_model,
        "expected_model": expected_scoring_model(true_model),
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }
