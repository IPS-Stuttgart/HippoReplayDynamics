import numpy as np
import pandas as pd

import hipporeplayimm.simulation_recovery as simulation_recovery
from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_EXACT,
    EVIDENCE_COMPARISON_LOWER_BOUND,
    EVIDENCE_COMPARISON_DEGENERATE,
    EVIDENCE_COMPARISON_PARTICLE_APPROXIMATION,
    EVIDENCE_COMPARISON_UNKNOWN,
    DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
    EXACT_EVIDENCE_SUPPORT,
    PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
    simulation_add_evidence_columns,
)


def test_state_space_imm_support_diagnostic_is_truncated_not_exact():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "expected_model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-imm",
                "expected_model": "sorted-spike-state-space-diffusion",
                "log_evidence": 100.0,
                "n_time": 3,
                "n_spikes": 5,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    diffusion = scored[scored["model"] == "sorted-spike-state-space-diffusion"].iloc[0]
    imm = scored[scored["model"] == "sorted-spike-state-space-imm"].iloc[0]

    assert diffusion["evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert bool(diffusion["evidence_comparable"])
    assert diffusion["evidence_comparison"] == EVIDENCE_COMPARISON_EXACT
    assert bool(diffusion["is_best_model"])
    assert diffusion["best_model"] == "sorted-spike-state-space-diffusion"

    assert imm["evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert not bool(imm["evidence_comparable"])
    assert imm["evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND
    assert not bool(imm["is_best_model"])
    assert pd.isna(imm["model_probability"])
    assert imm["truncated_relative_log_evidence"] == 0.0
    assert bool(imm["is_best_truncated_lower_bound"])
    assert imm["best_truncated_lower_bound_model"] == "sorted-spike-state-space-imm"


def test_simulation_reporting_counts_exact_sparse_momentum_as_recovery():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -2.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-momentum-exact-sparse",
                "log_evidence": -1.0,
                "n_time": 3,
                "n_spikes": 5,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    exact_sparse = scored[
        scored["model"] == "sorted-spike-state-space-momentum-exact-sparse"
    ].iloc[0]

    assert bool(exact_sparse["is_best_model"])
    assert bool(exact_sparse["recovered_expected_model"])


def test_simulation_reporting_counts_clusterless_exact_momentum_surrogate_as_recovery():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -2.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "clusterless-state-space-momentum-exact-sparse",
                "log_evidence": -1.0,
                "n_time": 3,
                "n_spikes": 5,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    clusterless = scored[
        scored["model"] == "clusterless-state-space-momentum-exact-sparse"
    ].iloc[0]

    assert bool(clusterless["is_best_model"])
    assert bool(clusterless["recovered_expected_model"])
    assert clusterless["exact_surrogate_best_model"] == "clusterless-state-space-momentum-exact-sparse"


def test_simulation_reporting_counts_velocity_momentum_alias_as_recovery():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -2.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-velocity-momentum",
                "log_evidence": -1.0,
                "n_time": 3,
                "n_spikes": 5,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    velocity_momentum = scored[
        scored["model"] == "sorted-spike-state-space-velocity-momentum"
    ].iloc[0]

    assert bool(velocity_momentum["is_best_model"])
    assert bool(velocity_momentum["recovered_expected_model"])
    assert velocity_momentum["exact_surrogate_best_model"] == "sorted-spike-state-space-velocity-momentum"


def test_simulation_reporting_ignores_nonfinite_log_evidence_rows():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -2.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-momentum-exact-sparse",
                "log_evidence": np.nan,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": np.inf,
                "n_time": 3,
                "n_spikes": 5,
                "diagnostic_state_space_momentum_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-imm",
                "log_evidence": -1.0,
                "n_time": 3,
                "n_spikes": 5,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    diffusion = scored[scored["model"] == "sorted-spike-state-space-diffusion"].iloc[0]
    exact_sparse = scored[
        scored["model"] == "sorted-spike-state-space-momentum-exact-sparse"
    ].iloc[0]
    nonfinite_lower_bound = scored[
        scored["model"] == "sorted-spike-state-space-momentum"
    ].iloc[0]
    finite_lower_bound = scored[scored["model"] == "sorted-spike-state-space-imm"].iloc[0]

    assert bool(diffusion["is_best_model"])
    assert diffusion["model_probability"] == 1.0
    assert not bool(exact_sparse["evidence_comparable"])
    assert not bool(exact_sparse["is_best_model"])
    assert pd.isna(exact_sparse["model_probability"])
    assert not bool(nonfinite_lower_bound["is_best_truncated_lower_bound"])
    assert bool(finite_lower_bound["is_best_truncated_lower_bound"])
    assert finite_lower_bound["best_truncated_lower_bound_model"] == "sorted-spike-state-space-imm"


def test_simulation_recovery_patch_exposes_velocity_momentum_surrogate_aliases():
    surrogates = simulation_recovery.exact_surrogate_scoring_models("momentum")

    assert "sorted-spike-state-space-velocity-momentum" in surrogates
    assert "state-space-velocity-momentum" in surrogates
    assert "clusterless-state-space-momentum-exact-sparse" in surrogates
    assert "clusterless-state-space-displacement-momentum" in surrogates
    assert "clusterless-state-space-velocity-momentum" in surrogates


def test_state_space_imm_support_column_is_used_by_generic_inference():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "sorted-spike-state-space-imm",
                "log_evidence": 10.0,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND


def test_explicit_noncomparable_without_support_stays_unknown():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "legacy-candidate-pruned-row",
                "log_evidence": 100.0,
                "evidence_comparable": "False",
            },
            {
                "status": "success",
                "model": "exact-row",
                "log_evidence": 1.0,
            },
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    legacy = scored[scored["model"] == "legacy-candidate-pruned-row"].iloc[0]
    exact = scored[scored["model"] == "exact-row"].iloc[0]
    assert legacy["evidence_support"] == EVIDENCE_COMPARISON_UNKNOWN
    assert not bool(legacy["evidence_comparable"])
    assert legacy["evidence_comparison"] == EVIDENCE_COMPARISON_UNKNOWN
    assert exact["evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert bool(exact["evidence_comparable"])


def test_state_space_displacement_imm_support_column_is_used_by_generic_inference():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "sorted-spike-state-space-displacement-imm",
                "log_evidence": 10.0,
                "diagnostic_state_space_displacement_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND


def test_pyrecest_particle_support_is_classified_as_particle_approximation():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "pyrecest-goal-particle",
                "log_evidence": 10.0,
                "diagnostic_pyrecest_evidence_support": PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == PYRECEST_PARTICLE_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_PARTICLE_APPROXIMATION


def test_specific_state_space_support_overrides_generic_component_support():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "sorted-spike-state-space-momentum-exact-sparse",
                "log_evidence": 0.0,
                "diagnostic_state_space_momentum_evidence_support": EXACT_EVIDENCE_SUPPORT,
                "diagnostic_state_space_sparse_momentum_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_DEGENERATE
