from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from aggregate_all_session_model_evidence import (
    exact_core_model_claim_decisions,
    exact_core_model_claim_summary,
    exact_trajectory_dynamics_gate_summary,
    exact_trajectory_dynamics_threshold_sensitivity,
    _load_combined,
    exact_sparse_momentum_core_margin_summary,
    exact_sparse_momentum_core_margins,
    exact_sparse_momentum_core_threshold_sensitivity,
    leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity,
    leave_one_rat_out_exact_sparse_momentum_core_margin_summary,
    leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity,
    leave_one_rat_out_paired_momentum_diffusion_margin_summary,
    leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity,
    paper_readiness_gate_summary,
    paired_momentum_diffusion_margin_decisions,
    paired_momentum_diffusion_margin_summary,
    paired_momentum_diffusion_threshold_sensitivity,
    rat_exact_sparse_momentum_core_margin_summary,
    rat_bootstrap_exact_sparse_momentum_core_margin_summary,
    rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity,
    rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity,
    rat_bootstrap_paired_momentum_diffusion_margin_summary,
    rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity,
    rat_exact_trajectory_dynamics_threshold_sensitivity,
    rat_paired_momentum_diffusion_margin_summary,
    random_effects_model_probabilities,
    required_full_core_model_coverage_table,
    session_exact_core_model_claim_summary,
    session_exact_trajectory_dynamics_threshold_sensitivity,
    session_exact_sparse_momentum_core_margin_summary,
    session_exact_sparse_momentum_core_threshold_sensitivity,
    session_best_model_counts,
    session_model_evidence_summary,
    session_paired_momentum_diffusion_margin_summary,
    session_paired_momentum_diffusion_threshold_sensitivity,
)


def test_all_session_model_evidence_workflow_exports_expected_outputs():
    workflow = Path(".github/workflows/model-evidence-all-sessions.yml").read_text(encoding="utf-8")

    assert "name: Benchmark replay model evidence all sessions" in workflow
    assert "Rat1/Open1 Rat1/Open2 Rat2/Open1 Rat2/Open2 Rat3/Open1 Rat3/Open2 Rat4/Open1 Rat4/Open2" in workflow
    assert "scripts/plan_model_evidence_event_shards.py" in workflow
    assert "scripts/aggregate_all_session_model_evidence.py" in workflow
    assert "spike_rate_scale:" in workflow
    assert "--spike-rate-scale" in workflow
    assert "state_space_momentum_initial_sigma_cm_sqrt_s:" in workflow
    assert (
        "STATE_SPACE_MOMENTUM_INITIAL_SIGMA_CM_SQRT_S: "
        "${{ inputs.state_space_momentum_initial_sigma_cm_sqrt_s }}"
    ) in workflow
    assert "--state-space-momentum-initial-sigma-cm-sqrt-s" in workflow
    assert 'CLUSTERLESS_MARK_SMOOTHING_SIGMA_BINS: "1.0"' in workflow
    assert "--clusterless-mark-smoothing-sigma-bins" in workflow
    assert "all_sessions_model_evidence_summary.csv" in workflow
    assert "session_model_evidence_summary.csv" in workflow
    assert "random_effects_model_probabilities.csv" in workflow
    assert "paper_readiness_gate_summary.csv" in workflow
    assert "exact_trajectory_dynamics_gate_summary.csv" in workflow
    assert "exact_trajectory_dynamics_threshold_sensitivity.csv" in workflow
    assert "session_exact_trajectory_dynamics_threshold_sensitivity.csv" in workflow
    assert "rat_exact_trajectory_dynamics_threshold_sensitivity.csv" in workflow
    assert "leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity.csv" in workflow
    assert "rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity.csv" in workflow
    assert "required_full_core_model_coverage.csv" in workflow
    assert "exact_core_model_claim_decisions.csv" in workflow
    assert "exact_core_model_claim_summary.csv" in workflow
    assert "session_exact_core_model_claim_summary.csv" in workflow
    assert "paired_momentum_diffusion_margin_summary.csv" in workflow
    assert "paired_momentum_diffusion_threshold_sensitivity.csv" in workflow
    assert "session_paired_momentum_diffusion_margin_summary.csv" in workflow
    assert "session_paired_momentum_diffusion_threshold_sensitivity.csv" in workflow
    assert "rat_paired_momentum_diffusion_margin_summary.csv" in workflow
    assert "leave_one_rat_out_paired_momentum_diffusion_margin_summary.csv" in workflow
    assert "leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity.csv" in workflow
    assert "rat_bootstrap_paired_momentum_diffusion_margin_summary.csv" in workflow
    assert "rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity.csv" in workflow
    assert "exact_sparse_momentum_core_margin_summary.csv" in workflow
    assert "exact_sparse_momentum_core_threshold_sensitivity.csv" in workflow
    assert "session_exact_sparse_momentum_core_margin_summary.csv" in workflow
    assert "session_exact_sparse_momentum_core_threshold_sensitivity.csv" in workflow
    assert "rat_exact_sparse_momentum_core_margin_summary.csv" in workflow
    assert "leave_one_rat_out_exact_sparse_momentum_core_margin_summary.csv" in workflow
    assert "leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity.csv" in workflow
    assert "rat_bootstrap_exact_sparse_momentum_core_margin_summary.csv" in workflow
    assert "rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity.csv" in workflow


def test_all_session_summary_helpers_group_by_session():
    rows = []
    for session, winner in (("Rat1/Open1", "diffusion"), ("Rat1/Open2", "momentum")):
        for event_index in (0, 1):
            for model in ("diffusion", "momentum"):
                rows.append(
                    {
                        "session": session,
                        "event_index": event_index,
                        "model": model,
                        "model_family": "trajectory",
                        "status": "success",
                        "log_evidence": 2.0 if model == winner else 1.0,
                        "relative_log_evidence": 0.0 if model == winner else -1.0,
                        "model_probability": 0.75 if model == winner else 0.25,
                        "is_best_model": model == winner,
                        "best_model": winner,
                        "best_trajectory_model": winner,
                        "best_nontrajectory_model": "",
                        "runtime_s": 0.1,
                    }
                )
    frame = pd.DataFrame(rows)

    session_summary = session_model_evidence_summary(frame)
    session_counts = session_best_model_counts(frame)
    random_effects = random_effects_model_probabilities(frame)

    assert set(session_summary["session"]) == {"Rat1/Open1", "Rat1/Open2"}
    assert set(session_counts["comparison"]) >= {"best_model", "best_trajectory_model"}
    assert set(random_effects["model"]) == {"diffusion", "momentum"}
    assert random_effects["random_effects_probability"].sum() == 1.0


def test_all_session_aggregation_rejects_mixed_run_settings(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([_score_row(event_index=0, spike_rate_scale=1.0)]).to_csv(first, index=False)
    pd.DataFrame([_score_row(event_index=1, spike_rate_scale=2.0)]).to_csv(second, index=False)

    with pytest.raises(ValueError, match="spike_rate_scale"):
        _load_combined(str(tmp_path / "*.csv"))


def test_all_session_paired_momentum_margin_summaries_separate_claim_states():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat2/Open1", 2, "sorted-spike-state-space-diffusion", 7.0),
            _paired_score_row(
                "Rat2/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
        ]
    )

    decisions = paired_momentum_diffusion_margin_decisions(frame)
    summary = paired_momentum_diffusion_margin_summary(decisions).iloc[0]
    session_summary = session_paired_momentum_diffusion_margin_summary(decisions)

    assert summary["events"] == 3
    assert summary["positive_raw_wins"] == 2
    assert summary["reference_raw_wins"] == 1
    assert summary["positive_model_claims"] == 1
    assert summary["reference_model_claims"] == 1
    assert summary["ambiguous_events"] == 1
    assert summary["margin_threshold"] == 5.5
    assert set(session_summary["session"]) == {"Rat1/Open1", "Rat2/Open1"}
    rat1 = session_summary[session_summary["session"] == "Rat1/Open1"].iloc[0]
    assert rat1["positive_model_claims"] == 1
    assert rat1["ambiguous_events"] == 1


def test_all_session_threshold_sensitivity_sweeps_margin_gate():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat2/Open1", 2, "sorted-spike-state-space-diffusion", 7.0),
            _paired_score_row(
                "Rat2/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
        ]
    )

    paired = paired_momentum_diffusion_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5, 10.0),
    )
    core = exact_sparse_momentum_core_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5, 10.0),
    )

    assert paired["margin_threshold"].tolist() == [0.0, 5.5, 10.0]
    assert paired["positive_model_claims"].tolist() == [2, 1, 0]
    assert paired["reference_model_claims"].tolist() == [1, 1, 0]
    assert paired["ambiguous_events"].tolist() == [0, 1, 3]
    assert core["margin_threshold"].tolist() == [0.0, 5.5, 10.0]
    assert core["positive_confident_core_claims"].tolist() == [2, 1, 0]


def test_all_session_threshold_sensitivity_groups_by_session():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat2/Open1", 2, "sorted-spike-state-space-diffusion", 7.0),
            _paired_score_row(
                "Rat2/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
        ]
    )

    paired = session_paired_momentum_diffusion_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5),
    )
    core = session_exact_sparse_momentum_core_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5),
    )

    assert paired["margin_threshold"].tolist() == [0.0, 0.0, 5.5, 5.5]
    assert paired["session"].tolist() == ["Rat1/Open1", "Rat2/Open1", "Rat1/Open1", "Rat2/Open1"]
    rat1_t0 = paired[(paired["session"] == "Rat1/Open1") & (paired["margin_threshold"] == 0.0)].iloc[0]
    assert rat1_t0["positive_model_claims"] == 2
    assert rat1_t0["ambiguous_events"] == 0
    rat1_t55 = paired[(paired["session"] == "Rat1/Open1") & (paired["margin_threshold"] == 5.5)].iloc[0]
    assert rat1_t55["positive_model_claims"] == 1
    assert rat1_t55["ambiguous_events"] == 1
    rat2_t55 = paired[(paired["session"] == "Rat2/Open1") & (paired["margin_threshold"] == 5.5)].iloc[0]
    assert rat2_t55["reference_model_claims"] == 1
    assert core["positive_confident_core_claims"].tolist() == paired["positive_model_claims"].tolist()


def test_all_session_exact_sparse_momentum_core_margins_compare_best_other_exact_model():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 1.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 4.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                10.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 2.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-first-order-imm", 9.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat2/Open1", 2, "sorted-spike-state-space-diffusion", 2.0),
            _paired_score_row("Rat2/Open1", 2, "sorted-spike-state-space-fragmented", 3.0),
            _paired_score_row(
                "Rat2/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                7.0,
            ),
        ]
    )

    margins = exact_sparse_momentum_core_margins(frame)
    summary = exact_sparse_momentum_core_margin_summary(margins).iloc[0]
    session_summary = session_exact_sparse_momentum_core_margin_summary(margins)

    assert summary["events"] == 3
    assert summary["positive_exact_best_events"] == 2
    assert summary["non_positive_exact_best_events"] == 1
    assert summary["positive_confident_core_claims"] == 1
    assert summary["ambiguous_or_other_best_events"] == 2
    assert summary["mean_positive_minus_best_other_exact_log_evidence"] == pytest.approx(3.0)
    assert summary["median_positive_minus_best_other_exact_log_evidence"] == pytest.approx(4.0)
    assert set(margins["best_other_exact_model"]) == {
        "sorted-spike-state-space-fragmented",
        "sorted-spike-state-space-first-order-imm",
    }
    rat1 = session_summary[session_summary["session"] == "Rat1/Open1"].iloc[0]
    assert rat1["positive_exact_best_events"] == 1
    assert rat1["positive_confident_core_claims"] == 1


def test_all_session_exact_core_model_claims_track_full_core_winners():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 15.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-fragmented", -1.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-first-order-imm", -2.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat1/Open2", 2, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open2",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                20.0,
            ),
        ]
    )

    decisions = exact_core_model_claim_decisions(frame)
    summary = exact_core_model_claim_summary(decisions)
    session_summary = session_exact_core_model_claim_summary(decisions)

    assert decisions["claim_model"].tolist() == [
        "sorted-spike-state-space-first-order-imm",
        "sorted-spike-state-space-momentum-exact-sparse",
        "incomplete_core",
    ]
    assert decisions["required_models_complete"].tolist() == [True, True, False]
    assert decisions.loc[2, "missing_required_models"] == (
        "sorted-spike-state-space-stationary "
        "sorted-spike-state-space-fragmented "
        "sorted-spike-state-space-first-order-imm"
    )

    imm = summary[summary["model"] == "sorted-spike-state-space-first-order-imm"].iloc[0]
    momentum = summary[summary["model"] == "sorted-spike-state-space-momentum-exact-sparse"].iloc[0]
    assert imm["raw_best_events"] == 1
    assert imm["confident_claims"] == 1
    assert momentum["raw_best_events"] == 2
    assert momentum["confident_claims"] == 1
    assert momentum["incomplete_core_events"] == 1

    rat1_open1 = session_summary[
        (session_summary["session"] == "Rat1/Open1")
        & (session_summary["model"] == "sorted-spike-state-space-momentum-exact-sparse")
    ].iloc[0]
    assert rat1_open1["events"] == 2
    assert rat1_open1["confident_claims"] == 1


def test_all_session_exact_trajectory_dynamics_gate_summary_accepts_imm_dominance():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 15.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-fragmented", -1.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-first-order-imm", -2.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-fragmented", 4.0),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-first-order-imm", 5.0),
            _paired_score_row(
                "Rat1/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                6.0,
            ),
        ]
    )

    summary = exact_trajectory_dynamics_gate_summary(frame).set_index("gate")

    assert bool(summary.loc["required_exact_core_models_present", "passed"])
    assert bool(summary.loc["exact_trajectory_raw_best_majority", "passed"])
    assert summary.loc["exact_trajectory_raw_best_majority", "observed"] == 1.0
    assert bool(summary.loc["exact_trajectory_confident_claim_majority", "passed"])
    assert summary.loc["exact_trajectory_confident_claim_majority", "observed"] == pytest.approx(2 / 3)
    assert bool(summary.loc["no_confident_static_or_other_core_claims", "passed"])
    assert bool(summary.loc["overall", "passed"])


def test_all_session_exact_trajectory_dynamics_threshold_sensitivity_sweeps_claims():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 15.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-fragmented", -1.0),
            _paired_score_row("Rat1/Open1", 1, "sorted-spike-state-space-first-order-imm", -2.0),
            _paired_score_row(
                "Rat1/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-fragmented", 4.0),
            _paired_score_row("Rat1/Open1", 2, "sorted-spike-state-space-first-order-imm", 5.0),
            _paired_score_row(
                "Rat1/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                6.0,
            ),
        ]
    )

    summary = exact_trajectory_dynamics_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5, 10.0),
    )
    session_summary = session_exact_trajectory_dynamics_threshold_sensitivity(
        frame,
        thresholds=(5.5,),
    )

    assert summary["margin_threshold"].tolist() == [0.0, 5.5, 10.0]
    assert summary["trajectory_raw_best_events"].tolist() == [3, 3, 3]
    assert summary["trajectory_confident_claims"].tolist() == [3, 2, 0]
    assert summary["ambiguous_events"].tolist() == [0, 1, 3]
    assert summary["nontrajectory_confident_claims"].tolist() == [0, 0, 0]
    assert session_summary["session"].tolist() == ["Rat1/Open1"]
    assert session_summary["trajectory_confident_claims"].tolist() == [2]


def test_all_session_exact_trajectory_dynamics_rat_and_leave_one_out_summaries():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 10.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-fragmented", 1.0),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-first-order-imm", 2.0),
            _paired_score_row(
                "Rat1/Open2",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-first-order-imm", 4.0),
            _paired_score_row(
                "Rat2/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                5.0,
            ),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-stationary", 12.0),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-first-order-imm", 3.0),
            _paired_score_row(
                "Rat2/Open2",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                4.0,
            ),
        ]
    )

    rat_summary = rat_exact_trajectory_dynamics_threshold_sensitivity(
        frame,
        thresholds=(5.5,),
    )
    leave_one_out = leave_one_rat_out_exact_trajectory_dynamics_threshold_sensitivity(
        frame,
        thresholds=(5.5,),
    )

    rat1 = rat_summary[rat_summary["rat"] == "Rat1"].iloc[0]
    rat2 = rat_summary[rat_summary["rat"] == "Rat2"].iloc[0]
    assert rat1["trajectory_confident_claims"] == 2
    assert rat1["nontrajectory_confident_claims"] == 0
    assert rat2["trajectory_confident_claims"] == 0
    assert rat2["nontrajectory_confident_claims"] == 1
    assert rat2["ambiguous_events"] == 1

    held_out_rat1 = leave_one_out[leave_one_out["held_out_rat"] == "Rat1"].iloc[0]
    held_out_rat2 = leave_one_out[leave_one_out["held_out_rat"] == "Rat2"].iloc[0]
    assert held_out_rat1["included_rats"] == "Rat2"
    assert held_out_rat1["trajectory_confident_claims"] == 0
    assert held_out_rat1["nontrajectory_confident_claims"] == 1
    assert held_out_rat2["included_rats"] == "Rat1"
    assert held_out_rat2["trajectory_confident_claims"] == 2
    assert held_out_rat2["nontrajectory_confident_claims"] == 0


def test_all_session_exact_trajectory_dynamics_rat_bootstrap_reports_uncertainty():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 10.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-fragmented", 1.0),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-first-order-imm", 2.0),
            _paired_score_row(
                "Rat1/Open2",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-stationary", -3.0),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat2/Open1", 0, "sorted-spike-state-space-first-order-imm", 4.0),
            _paired_score_row(
                "Rat2/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                5.0,
            ),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-stationary", 12.0),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-fragmented", 2.0),
            _paired_score_row("Rat2/Open2", 1, "sorted-spike-state-space-first-order-imm", 3.0),
            _paired_score_row(
                "Rat2/Open2",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                4.0,
            ),
        ]
    )

    bootstrap = rat_bootstrap_exact_trajectory_dynamics_threshold_sensitivity(
        frame,
        thresholds=(5.5,),
        n_bootstrap=50,
        random_seed=7,
    ).iloc[0]

    assert bootstrap["margin_threshold"] == 5.5
    assert bootstrap["observed_events"] == 4
    assert bootstrap["observed_rats"] == 2
    assert bootstrap["observed_required_complete_fraction"] == 1.0
    assert bootstrap["observed_trajectory_raw_best_fraction"] == pytest.approx(0.75)
    assert bootstrap["observed_trajectory_confident_claim_fraction"] == pytest.approx(0.5)
    assert bootstrap["observed_nontrajectory_confident_claim_fraction"] == pytest.approx(0.25)
    assert bootstrap["observed_ambiguous_fraction"] == pytest.approx(0.25)
    assert 0.0 <= bootstrap["trajectory_confident_claim_fraction_ci95_low"] <= 1.0
    assert 0.0 <= bootstrap["trajectory_confident_claim_fraction_ci95_high"] <= 1.0


def test_all_session_exact_trajectory_dynamics_gate_summary_rejects_stationary_claims():
    rows: list[dict[str, object]] = []
    for event_index in range(3):
        rows.extend(
            [
                _paired_score_row("Rat1/Open1", event_index, "sorted-spike-state-space-stationary", 20.0),
                _paired_score_row("Rat1/Open1", event_index, "sorted-spike-state-space-diffusion", 0.0),
                _paired_score_row("Rat1/Open1", event_index, "sorted-spike-state-space-fragmented", 2.0),
                _paired_score_row("Rat1/Open1", event_index, "sorted-spike-state-space-first-order-imm", 3.0),
                _paired_score_row(
                    "Rat1/Open1",
                    event_index,
                    "sorted-spike-state-space-momentum-exact-sparse",
                    4.0,
                ),
            ]
        )

    summary = exact_trajectory_dynamics_gate_summary(pd.DataFrame(rows)).set_index("gate")

    assert not bool(summary.loc["exact_trajectory_raw_best_majority", "passed"])
    assert not bool(summary.loc["exact_trajectory_confident_claim_majority", "passed"])
    assert not bool(summary.loc["no_confident_static_or_other_core_claims", "passed"])
    assert not bool(summary.loc["overall", "passed"])


def test_all_session_rat_robustness_summaries_group_and_hold_out_rats():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat1/Open2", 1, "sorted-spike-state-space-diffusion", 1.0),
            _paired_score_row(
                "Rat1/Open2",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
            _paired_score_row("Rat2/Open1", 2, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat2/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
            _paired_score_row("Rat3/Open1", 3, "sorted-spike-state-space-diffusion", 6.0),
            _paired_score_row(
                "Rat3/Open1",
                3,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
        ]
    )
    decisions = paired_momentum_diffusion_margin_decisions(frame)
    core_margins = exact_sparse_momentum_core_margins(frame)

    rat_paired = rat_paired_momentum_diffusion_margin_summary(decisions)
    rat_core = rat_exact_sparse_momentum_core_margin_summary(core_margins)
    paired_leave_one_out = leave_one_rat_out_paired_momentum_diffusion_margin_summary(decisions)
    core_leave_one_out = leave_one_rat_out_exact_sparse_momentum_core_margin_summary(core_margins)

    assert rat_paired["rat"].tolist() == ["Rat1", "Rat2", "Rat3"]
    assert rat_core["rat"].tolist() == ["Rat1", "Rat2", "Rat3"]
    assert paired_leave_one_out["held_out_rat"].tolist() == ["Rat1", "Rat2", "Rat3"]
    assert core_leave_one_out["held_out_rat"].tolist() == ["Rat1", "Rat2", "Rat3"]
    rat1 = rat_paired[rat_paired["rat"] == "Rat1"].iloc[0]
    assert rat1["events"] == 2
    assert rat1["positive_model_claims"] == 1
    held_out_rat1 = paired_leave_one_out[paired_leave_one_out["held_out_rat"] == "Rat1"].iloc[0]
    assert held_out_rat1["included_rats"] == "Rat2 Rat3"
    assert held_out_rat1["events"] == 2
    assert held_out_rat1["positive_model_claims"] == 1


def test_all_session_leave_one_rat_out_threshold_sensitivity_sweeps_margin_gate():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat2/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat2/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat3/Open1", 2, "sorted-spike-state-space-diffusion", 7.0),
            _paired_score_row(
                "Rat3/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
        ]
    )

    paired = leave_one_rat_out_paired_momentum_diffusion_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5),
    )
    core = leave_one_rat_out_exact_sparse_momentum_core_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5),
    )

    assert paired["margin_threshold"].tolist() == [0.0, 0.0, 0.0, 5.5, 5.5, 5.5]
    assert paired["held_out_rat"].tolist() == ["Rat1", "Rat2", "Rat3", "Rat1", "Rat2", "Rat3"]
    t0 = paired[paired["margin_threshold"] == 0.0]
    assert t0["positive_model_claims"].tolist() == [1, 1, 2]
    assert t0["reference_model_claims"].tolist() == [1, 1, 0]
    t55 = paired[paired["margin_threshold"] == 5.5]
    assert t55["positive_model_claims"].tolist() == [0, 1, 1]
    assert t55["reference_model_claims"].tolist() == [1, 1, 0]
    assert t55["ambiguous_events"].tolist() == [1, 0, 1]
    assert core["margin_threshold"].tolist() == paired["margin_threshold"].tolist()
    assert core["held_out_rat"].tolist() == paired["held_out_rat"].tolist()
    assert core["positive_confident_core_claims"].tolist() == paired["positive_model_claims"].tolist()


def test_all_session_rat_bootstrap_summaries_report_uncertainty():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat2/Open1", 1, "sorted-spike-state-space-diffusion", 2.0),
            _paired_score_row(
                "Rat2/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
            _paired_score_row("Rat3/Open1", 2, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat3/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                8.0,
            ),
        ]
    )
    decisions = paired_momentum_diffusion_margin_decisions(frame)
    core_margins = exact_sparse_momentum_core_margins(frame)

    paired = rat_bootstrap_paired_momentum_diffusion_margin_summary(
        decisions,
        n_bootstrap=50,
        random_seed=7,
    ).iloc[0]
    core = rat_bootstrap_exact_sparse_momentum_core_margin_summary(
        core_margins,
        n_bootstrap=50,
        random_seed=7,
    ).iloc[0]

    assert paired["bootstrap_unit"] == "rat"
    assert paired["bootstrap_replicates"] == 50
    assert paired["observed_events"] == 3
    assert paired["observed_rats"] == 3
    assert paired["observed_positive_raw_win_fraction"] == pytest.approx(2 / 3)
    assert paired["observed_positive_claim_fraction"] == pytest.approx(2 / 3)
    assert paired["observed_mean_delta"] == pytest.approx(5.0)
    assert 0.0 <= paired["probability_mean_delta_gt_0"] <= 1.0
    assert core["observed_mean_delta"] == pytest.approx(paired["observed_mean_delta"])


def test_all_session_rat_bootstrap_threshold_sensitivity_reports_claim_uncertainty():
    frame = pd.DataFrame(
        [
            _paired_score_row("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat1/Open1",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                9.0,
            ),
            _paired_score_row("Rat2/Open1", 1, "sorted-spike-state-space-diffusion", 0.0),
            _paired_score_row(
                "Rat2/Open1",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                3.0,
            ),
            _paired_score_row("Rat3/Open1", 2, "sorted-spike-state-space-diffusion", 7.0),
            _paired_score_row(
                "Rat3/Open1",
                2,
                "sorted-spike-state-space-momentum-exact-sparse",
                0.0,
            ),
        ]
    )

    paired = rat_bootstrap_paired_momentum_diffusion_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5, 10.0),
        n_bootstrap=50,
        random_seed=7,
    )
    core = rat_bootstrap_exact_sparse_momentum_core_threshold_sensitivity(
        frame,
        thresholds=(0.0, 5.5, 10.0),
        n_bootstrap=50,
        random_seed=7,
    )

    assert paired["margin_threshold"].tolist() == [0.0, 5.5, 10.0]
    assert paired["observed_positive_claim_fraction"].tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert paired["observed_positive_raw_win_fraction"].tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3])
    assert paired["observed_mean_delta"].tolist() == pytest.approx([5 / 3, 5 / 3, 5 / 3])
    assert core["margin_threshold"].tolist() == [0.0, 5.5, 10.0]
    assert core["observed_positive_claim_fraction"].tolist() == pytest.approx(
        paired["observed_positive_claim_fraction"].tolist()
    )


def test_all_session_paper_readiness_gate_summary_reports_pass_fail():
    frame = _paper_readiness_frame()

    summary = paper_readiness_gate_summary(
        frame,
        n_bootstrap=50,
        random_seed=7,
    ).set_index("gate")

    assert bool(summary.loc["no_scoring_failures", "passed"])
    assert bool(summary.loc["minimum_rats_present", "passed"])
    assert bool(summary.loc["minimum_sessions_present", "passed"])
    assert bool(summary.loc["minimum_paired_events_per_session", "passed"])
    assert bool(summary.loc["paired_no_confident_diffusion_claims", "passed"])
    assert bool(summary.loc["paired_raw_momentum_win_majority", "passed"])
    assert bool(summary.loc["paired_confident_momentum_claim_majority", "passed"])
    assert bool(summary.loc["all_sessions_have_confident_momentum_claims", "passed"])
    assert bool(summary.loc["leave_one_rat_out_median_delta_positive", "passed"])
    assert bool(summary.loc["rat_bootstrap_median_delta_ci_positive", "passed"])
    assert bool(summary.loc["full_core_required_exact_models_present", "passed"])
    assert bool(summary.loc["full_core_min_exact_models_compared", "passed"])
    assert bool(summary.loc["full_core_exact_sparse_claims_present", "passed"])
    assert bool(summary.loc["full_core_exact_sparse_best_majority", "passed"])
    assert bool(summary.loc["full_core_confident_exact_sparse_claim_majority", "passed"])
    assert bool(summary.loc["overall", "passed"])


def test_all_session_paper_readiness_gate_summary_rejects_canary_coverage():
    rows: list[dict[str, object]] = []
    for event_index in range(5):
        rows.extend(
            [
                _paired_score_row("Rat1/Open1", event_index, "sorted-spike-state-space-diffusion", 0.0),
                _paired_score_row(
                    "Rat1/Open1",
                    event_index,
                    "sorted-spike-state-space-momentum-exact-sparse",
                    8.0,
                ),
            ]
        )
    summary = paper_readiness_gate_summary(pd.DataFrame(rows), n_bootstrap=50, random_seed=7).set_index("gate")

    assert not bool(summary.loc["minimum_rats_present", "passed"])
    assert not bool(summary.loc["minimum_sessions_present", "passed"])
    assert bool(summary.loc["minimum_paired_events_per_session", "passed"])
    assert not bool(summary.loc["overall", "passed"])


def test_all_session_paper_readiness_gate_summary_rejects_sparse_confident_claims():
    rows: list[dict[str, object]] = []
    for rat in range(1, 5):
        for session_idx in range(1, 3):
            session = f"Rat{rat}/Open{session_idx}"
            for event_index in range(5):
                momentum_delta = 8.0 if event_index == 0 else 1.0
                rows.extend(
                    [
                        _paired_score_row(session, event_index, "sorted-spike-state-space-diffusion", 0.0),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-momentum-exact-sparse",
                            momentum_delta,
                        ),
                    ]
                )

    summary = paper_readiness_gate_summary(pd.DataFrame(rows), n_bootstrap=50, random_seed=7).set_index("gate")

    assert bool(summary.loc["paired_raw_momentum_win_majority", "passed"])
    assert not bool(summary.loc["paired_confident_momentum_claim_majority", "passed"])
    assert bool(summary.loc["all_sessions_have_confident_momentum_claims", "passed"])
    assert not bool(summary.loc["overall", "passed"])


def test_all_session_paper_readiness_gate_summary_rejects_two_model_full_core_coverage():
    rows: list[dict[str, object]] = []
    for rat in range(1, 5):
        for session_idx in range(1, 3):
            session = f"Rat{rat}/Open{session_idx}"
            for event_index in range(5):
                rows.extend(
                    [
                        _paired_score_row(session, event_index, "sorted-spike-state-space-diffusion", 0.0),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-momentum-exact-sparse",
                            8.0,
                        ),
                    ]
                )

    frame = pd.DataFrame(rows)
    coverage = required_full_core_model_coverage_table(frame)
    summary = paper_readiness_gate_summary(frame, n_bootstrap=50, random_seed=7).set_index("gate")

    assert len(coverage) == 40
    assert set(coverage["required_models_present"]) == {2}
    assert set(coverage["required_models_complete"]) == {False}
    assert set(coverage["missing_required_models"]) == {
        (
            "sorted-spike-state-space-stationary "
            "sorted-spike-state-space-fragmented "
            "sorted-spike-state-space-first-order-imm"
        )
    }

    assert bool(summary.loc["paired_raw_momentum_win_majority", "passed"])
    assert bool(summary.loc["paired_confident_momentum_claim_majority", "passed"])
    assert not bool(summary.loc["full_core_required_exact_models_present", "passed"])
    assert summary.loc["full_core_required_exact_models_present", "observed"] == "2/5"
    assert not bool(summary.loc["full_core_min_exact_models_compared", "passed"])
    assert summary.loc["full_core_min_exact_models_compared", "observed"] == 2
    assert not bool(summary.loc["overall", "passed"])


def test_all_session_paper_readiness_gate_summary_rejects_wrong_full_core_models():
    rows: list[dict[str, object]] = []
    for rat in range(1, 5):
        for session_idx in range(1, 3):
            session = f"Rat{rat}/Open{session_idx}"
            for event_index in range(5):
                rows.extend(
                    [
                        _paired_score_row(session, event_index, "sorted-spike-state-space-stationary", -3.0),
                        _paired_score_row(session, event_index, "sorted-spike-state-space-diffusion", 0.0),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-first-order-imm",
                            -1.0,
                        ),
                        _paired_score_row(session, event_index, "sorted-spike-state-space-goal", -2.0),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-momentum-exact-sparse",
                            8.0,
                        ),
                    ]
                )

    summary = paper_readiness_gate_summary(pd.DataFrame(rows), n_bootstrap=50, random_seed=7).set_index("gate")

    assert bool(summary.loc["full_core_min_exact_models_compared", "passed"])
    assert not bool(summary.loc["full_core_required_exact_models_present", "passed"])
    assert summary.loc["full_core_required_exact_models_present", "observed"] == "4/5"
    assert not bool(summary.loc["overall", "passed"])


def test_all_session_paper_readiness_gate_summary_rejects_non_momentum_core_majority():
    rows: list[dict[str, object]] = []
    for rat in range(1, 5):
        for session_idx in range(1, 3):
            session = f"Rat{rat}/Open{session_idx}"
            for event_index in range(5):
                fragmented_log_evidence = 9.0 if event_index < 4 else 1.0
                rows.extend(
                    [
                        _paired_score_row(session, event_index, "sorted-spike-state-space-diffusion", 0.0),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-momentum-exact-sparse",
                            8.0,
                        ),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-fragmented",
                            fragmented_log_evidence,
                        ),
                    ]
                )

    summary = paper_readiness_gate_summary(pd.DataFrame(rows), n_bootstrap=50, random_seed=7).set_index("gate")

    assert bool(summary.loc["paired_raw_momentum_win_majority", "passed"])
    assert bool(summary.loc["paired_confident_momentum_claim_majority", "passed"])
    assert not bool(summary.loc["full_core_exact_sparse_best_majority", "passed"])
    assert not bool(summary.loc["full_core_confident_exact_sparse_claim_majority", "passed"])
    assert not bool(summary.loc["overall", "passed"])


def _paper_readiness_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rat in range(1, 5):
        for session_idx in range(1, 3):
            session = f"Rat{rat}/Open{session_idx}"
            for event_index in range(5):
                rows.extend(
                    [
                        _paired_score_row(session, event_index, "sorted-spike-state-space-stationary", -3.0),
                        _paired_score_row(session, event_index, "sorted-spike-state-space-diffusion", 0.0),
                        _paired_score_row(session, event_index, "sorted-spike-state-space-fragmented", -2.0),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-first-order-imm",
                            -1.0,
                        ),
                        _paired_score_row(
                            session,
                            event_index,
                            "sorted-spike-state-space-momentum-exact-sparse",
                            8.0 + float(event_index),
                        ),
                    ]
                )
    return pd.DataFrame(rows)


def _score_row(*, event_index: int, spike_rate_scale: float = 1.0) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "model": "diffusion",
        "requested_model": "diffusion",
        "model_family": "trajectory",
        "log_evidence": -1.0,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
        "spike_rate_scale": spike_rate_scale,
        "clusterless_mark_smoothing_sigma_bins": 1.0,
        "clusterless_mark_prior_count": 1.0,
        "clusterless_mark_variance_floor": 1.0,
        "clusterless_rate_floor_hz": 1e-4,
    }


def _paired_score_row(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "model_family": "trajectory",
        "evidence_comparable": True,
        "evidence_support": "exact_full_grid",
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
        "spike_rate_scale": 1.0,
        "clusterless_mark_smoothing_sigma_bins": 1.0,
        "clusterless_mark_prior_count": 1.0,
        "clusterless_mark_variance_floor": 1.0,
        "clusterless_rate_floor_hz": 1e-4,
    }
