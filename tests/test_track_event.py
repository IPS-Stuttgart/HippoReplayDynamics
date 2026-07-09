import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import CandidateKinematicModel, EventScore
from scripts.track_event import _TRACK_MODEL_CHOICES, _mode_probability_row, _trajectory_from_prefix_scores, _trajectory_rows_from_log_posteriors


def test_trajectory_rows_entropy_handles_zero_probability_bins():
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0, -np.inf], [0.0, 0.0]]),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    log_posteriors = np.array(
        [
            [0.0, -np.inf],
            [np.log(0.25), np.log(0.75)],
        ]
    )
    score = EventScore("synthetic", 0.0, emissions.n_time, emissions.n_spikes)

    rows = _trajectory_rows_from_log_posteriors(
        log_posteriors=log_posteriors,
        emissions=emissions,
        bin_centers=np.array([[0.0, 0.0], [1.0, 0.0]]),
        score=score,
        likelihood_column="event_log_likelihood",
    )

    assert np.isfinite(rows.loc[0, "posterior_entropy"])
    assert rows.loc[0, "posterior_entropy"] == pytest.approx(0.0)
    expected_entropy = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert rows.loc[1, "posterior_entropy"] == pytest.approx(expected_entropy)


def test_trajectory_rows_accept_one_dimensional_bin_centers():
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0, 0.0]]),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    score = EventScore("synthetic", 0.0, emissions.n_time, emissions.n_spikes)

    rows = _trajectory_rows_from_log_posteriors(
        log_posteriors=np.array([[np.log(0.25), np.log(0.75)]]),
        emissions=emissions,
        bin_centers=np.array([[0.0], [2.0]]),
        score=score,
        likelihood_column="event_log_likelihood",
    )

    assert rows.loc[0, "posterior_mean_x"] == pytest.approx(1.5)
    assert rows.loc[0, "posterior_mean_y"] == pytest.approx(0.0)
    assert rows.loc[0, "map_x"] == pytest.approx(2.0)
    assert rows.loc[0, "map_y"] == pytest.approx(0.0)


def test_mode_probability_row_normalizes_valid_probabilities():
    row = _mode_probability_row(("stationary", "diffusion"), np.array([2.0, 6.0]))

    assert row["mode_stationary_probability"] == pytest.approx(0.25)
    assert row["mode_diffusion_probability"] == pytest.approx(0.75)
    assert row["most_likely_mode"] == "diffusion"


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([1.0, -0.1]),
        np.array([0.0, 0.0]),
        np.ones((2, 1)),
        np.ones(3),
    ],
)
def test_mode_probability_row_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="mode probabilities"):
        _mode_probability_row(("stationary", "diffusion"), probabilities)


def test_full_trajectory_imm_export_preserves_mode_probability_columns():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    model = CandidateKinematicModel(mode="imm", top_k=0)

    rows, log_posteriors = _trajectory_from_prefix_scores(model, emissions, bin_centers)

    assert log_posteriors.shape == (emissions.n_time, emissions.n_bins)
    probability_columns = [f"mode_{mode}_probability" for mode in ("stationary", "diffusion", "momentum", "jump")]
    assert set(probability_columns).issubset(rows.columns)
    assert "most_likely_mode" in rows.columns
    np.testing.assert_allclose(rows[probability_columns].sum(axis=1), np.ones(emissions.n_time))
    assert rows["most_likely_mode"].notna().all()


def test_track_model_choices_include_benchmark_state_space_variants():
    required = {
        "sorted-spike-state-space-momentum-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse",
        "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse",
        "sorted-spike-state-space-displacement-momentum",
        "sorted-spike-state-space-velocity-momentum",
        "sorted-spike-state-space-displacement-imm",
        "sorted-spike-state-space-first-order-imm",
        "sorted-spike-state-space-goal",
        "state-space-momentum-exact-sparse",
        "state-space-trajectory-imm-exact-sparse",
        "state-space-displacement-momentum",
        "state-space-velocity-momentum",
        "state-space-displacement-imm",
        "state-space-first-order-imm",
        "state-space-goal",
    }

    assert required.issubset(set(_TRACK_MODEL_CHOICES))
