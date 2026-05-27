from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    add_evidence_columns,
    certified_vs_exact_event_recovery,
    certified_vs_exact_recovery_summary,
    _candidate_indices_with_path,
    _candidate_path_support_diagnostics,
    _recovery_state_space_config,
    _write_simulation_recovery_checkpoint,
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


def test_recovery_summary_counts_exact_sparse_momentum_surrogate():
    rows = pd.DataFrame(
        [
            _row(0, "momentum", "sorted-spike-state-space-diffusion", -2.0),
            _row(0, "momentum", "sorted-spike-state-space-momentum-exact-sparse", -1.0),
        ]
    )

    scored = add_evidence_columns(rows)

    exact_sparse = scored[
        scored["model"] == "sorted-spike-state-space-momentum-exact-sparse"
    ].iloc[0]
    assert bool(exact_sparse["is_best_model"])
    assert bool(exact_sparse["recovered_expected_model"])
    assert bool(exact_sparse["exact_surrogate_recovered_expected_model"])

    summary = recovery_summary(scored)
    momentum = summary[summary["true_model"] == "momentum"].iloc[0]
    assert momentum["recovered_events"] == 1
    assert momentum["recovery_accuracy"] == 1.0
    assert momentum["exact_surrogate_recovered_events"] == 1


def test_build_scoring_models_and_model_parser_accept_space_or_comma_lists():
    assert parse_model_list("stationary,diffusion momentum") == ("stationary", "diffusion", "momentum")
    models = build_scoring_models(
        SimulationRecoveryConfig(
            scoring_models=(
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
                "sorted-spike-state-space-momentum",
            ),
            state_space=StateSpaceDecoderConfig(momentum_candidate_top_k=4),
        )
    )

    assert list(models) == [
        "sorted-spike-state-space-stationary",
        "sorted-spike-state-space-first-order-imm",
        "sorted-spike-state-space-momentum",
    ]


def test_candidate_path_diagnostics_measure_oracle_support_coverage():
    path = np.asarray([0, 1, 2], dtype=int)
    candidates = [
        np.asarray([0], dtype=int),
        np.asarray([1], dtype=int),
        np.asarray([3], dtype=int),
    ]

    diagnostics = _candidate_path_support_diagnostics(candidates, path)
    augmented = _candidate_indices_with_path(candidates, path)
    augmented_diagnostics = _candidate_path_support_diagnostics(augmented, path)

    assert diagnostics["candidate_true_bin_coverage"] == 2.0 / 3.0
    assert diagnostics["candidate_true_pair_coverage"] == 0.5
    assert diagnostics["candidate_true_triplet_coverage"] == 0.0
    assert diagnostics["candidate_true_path_fully_supported"] == 0
    assert augmented_diagnostics["candidate_true_path_fully_supported"] == 1
    assert augmented_diagnostics["candidate_true_triplet_coverage"] == 1.0


def test_recovery_state_space_config_enables_positive_occupancy_mask_by_default():
    enabled = _recovery_state_space_config(
        SimulationRecoveryConfig(
            state_space=StateSpaceDecoderConfig(valid_occupancy_threshold_s=0.0),
            score_with_occupancy=True,
        )
    )
    disabled = _recovery_state_space_config(
        SimulationRecoveryConfig(
            state_space=StateSpaceDecoderConfig(valid_occupancy_threshold_s=0.0),
            score_with_occupancy=False,
        )
    )

    assert enabled.valid_occupancy_threshold_s > 0.0
    assert disabled.valid_occupancy_threshold_s == 0.0


def test_recovery_summary_does_not_mix_truncated_lower_bounds_with_exact_evidence():
    rows = pd.DataFrame(
        [
            _row(0, "momentum", "sorted-spike-state-space-diffusion", -1.0),
            _row(
                0,
                "momentum",
                "sorted-spike-state-space-momentum",
                100.0,
                diagnostic_state_space_momentum_evidence_support="truncated_full_grid",
            ),
        ]
    )
    scored = add_evidence_columns(rows)

    momentum = scored[scored["model"] == "sorted-spike-state-space-momentum"].iloc[0]
    diffusion = scored[scored["model"] == "sorted-spike-state-space-diffusion"].iloc[0]

    assert bool(diffusion["is_best_model"])
    assert diffusion["best_model"] == "sorted-spike-state-space-diffusion"
    assert not bool(momentum["evidence_comparable"])
    assert not bool(momentum["is_best_model"])
    assert pd.isna(momentum["model_probability"])
    assert momentum["best_truncated_lower_bound_model"] == "sorted-spike-state-space-momentum"

    summary = recovery_summary(scored)
    overall = summary[summary["true_model"] == "overall"].iloc[0]
    assert overall["recovered_events"] == 0


def test_certified_vs_exact_summary_counts_lower_bound_wins_conservatively():
    rows = pd.DataFrame(
        [
            _row(0, "momentum", "sorted-spike-state-space-diffusion", -1.0),
            _row(
                0,
                "momentum",
                "sorted-spike-state-space-momentum",
                2.0,
                diagnostic_state_space_momentum_evidence_support="truncated_full_grid",
            ),
        ]
    )
    scored = add_evidence_columns(rows)

    events = certified_vs_exact_event_recovery(scored)
    summary = certified_vs_exact_recovery_summary(scored)
    momentum = summary[summary["true_model"] == "momentum"].iloc[0]

    assert bool(events.iloc[0]["certified_vs_exact_recovered_expected_model"])
    assert events.iloc[0]["certified_vs_exact_reason"] == "expected_lower_bound_beats_best_comparable"
    assert events.iloc[0]["expected_minus_best_comparable_log_evidence"] == 3.0
    assert momentum["certified_vs_exact_recovered_events"] == 1
    assert momentum["certified_vs_exact_recovery_accuracy"] == 1.0


def test_simulation_recovery_checkpoint_writes_partial_outputs(tmp_path):
    class FakeSession:
        session_id = "RatX/OpenY"

    rows = [
        _row(0, "momentum", "sorted-spike-state-space-diffusion", -2.0),
        _row(0, "momentum", "sorted-spike-state-space-momentum-exact-sparse", -1.0),
    ]
    config = SimulationRecoveryConfig(
        true_models=("momentum",),
        scoring_models=(
            "sorted-spike-state-space-diffusion",
            "sorted-spike-state-space-momentum-exact-sparse",
        ),
        checkpoint_output=tmp_path,
        progress_log=True,
    )

    _write_simulation_recovery_checkpoint(
        rows,
        tmp_path,
        FakeSession(),  # type: ignore[arg-type]
        config,
        [7],
        _encoding(),
        StateSpaceDecoderConfig(),
        run_started_at=0.0,
        planned_synthetic_events=2,
        completed_synthetic_events=1,
        stop_reason="completed",
        checkpoint_status="running",
    )

    scores = pd.read_csv(tmp_path / "simulation_recovery_event_scores.csv")
    summary = pd.read_csv(tmp_path / "simulation_recovery_summary.csv")
    settings = Path(tmp_path / "simulation_recovery_settings.yml").read_text(encoding="utf-8")

    assert len(scores) == 2
    assert summary.loc[summary["true_model"] == "momentum", "recovered_events"].iloc[0] == 1
    assert "checkpoint_status: running" in settings
    assert "checkpoint_completed_synthetic_events: 1" in settings


def _row(event_index: int, true_model: str, model: str, log_evidence: float, **extra: object) -> dict[str, object]:
    row = {
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
    row.update(extra)
    return row
