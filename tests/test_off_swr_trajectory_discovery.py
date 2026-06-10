import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from off_swr_trajectory_discovery import (  # noqa: E402
    AMBIGUOUS_CLASS,
    EXCLUDED_SWR_OVERLAP_CLASS,
    FULL_CORE_REQUIRED_MODELS,
    INTERESTING_CANDIDATE_LABEL,
    MOVEMENT_SPIKING_LIKE_LABEL,
    STATIC_NONTRAJECTORY_CLASS,
    TRAJECTORY_CANDIDATE_CLASS,
    _off_swr_run_state_window_table,
    write_off_swr_trajectory_discovery_outputs,
)


def test_off_swr_discovery_classifies_clusters_and_reports_covariates(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows(
                "Rat1/Open1",
                0,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=8.0,
                start=10.0,
                ripple_power=1.2,
                animal_speed_mean=1.0,
            ),
            *_event_rows(
                "Rat1/Open1",
                1,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=9.0,
                start=10.25,
                ripple_power=1.4,
                animal_speed_mean=18.0,
            ),
            *_event_rows("Rat1/Open1", 2, "matched_null", 0, stationary=12.0, trajectory=0.0, start=11.0, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 3, "matched_null", 0, stationary=0.0, trajectory=2.0, start=12.0, ripple_power=0.6),
            *_event_rows(
                "Rat1/Open1",
                4,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=9.0,
                start=13.0,
                ripple_power=1.5,
                off_swr=False,
            ),
            *_event_rows("Rat1/Open1", 5, "real", -1, stationary=0.0, trajectory=20.0, start=1.0, ripple_power=2.0, off_swr=False),
        ]
    )

    outputs = write_off_swr_trajectory_discovery_outputs(scores, tmp_path, cluster_gap_s=0.5)

    decisions = outputs["off_swr_trajectory_discovery_decisions.csv"]
    classes_by_event = dict(zip(decisions["event_index"], decisions["candidate_class"], strict=True))
    assert classes_by_event[0] == TRAJECTORY_CANDIDATE_CLASS
    assert classes_by_event[1] == TRAJECTORY_CANDIDATE_CLASS
    assert classes_by_event[2] == STATIC_NONTRAJECTORY_CLASS
    assert classes_by_event[3] == AMBIGUOUS_CLASS
    assert classes_by_event[4] == EXCLUDED_SWR_OVERLAP_CLASS
    assert 5 not in classes_by_event

    candidates = outputs["off_swr_trajectory_candidate_events.csv"]
    assert len(candidates) == 2
    assert set(candidates["event_index"]) == {0, 1}
    assert candidates["passes_known_swr_exclusion"].all()

    triage = outputs["off_swr_candidate_table.csv"]
    requested_columns = {
        "session",
        "rat",
        "window_start_s",
        "window_end_s",
        "duration_s",
        "n_spikes",
        "active_cell_count",
        "trajectory_family_margin",
        "candidate_tier",
        "best_trajectory_model",
        "trajectory_confidence",
        "trajectory_posterior_entropy",
        "distance_to_nearest_swr_s",
        "overlaps_known_swr",
        "animal_speed_mean",
        "animal_speed_median",
        "animal_speed_max",
        "position_sample_count",
        "run_or_immobility_state",
        "decoded_path_length",
        "decoded_speed",
        "decoded_endpoint_distance",
        "decoded_start_to_end_distance",
        "candidate_cluster_id",
    }
    assert requested_columns.issubset(triage.columns)
    assert len(triage) == 2
    assert triage["candidate_rank"].tolist() == [1, 2]
    assert triage["candidate_cluster_id"].nunique() == 1
    assert triage["trajectory_confidence"].notna().all()
    assert triage["trajectory_posterior_entropy"].notna().all()
    assert triage["distance_to_nearest_swr_s"].notna().all()
    assert not triage["overlaps_known_swr"].any()
    assert set(triage["candidate_specificity_label"]) == {INTERESTING_CANDIDATE_LABEL, MOVEMENT_SPIKING_LIKE_LABEL}
    assert set(triage["candidate_tier"]) == {"weak"}

    triage_clusters = outputs["off_swr_candidate_cluster_table.csv"]
    assert len(triage_clusters) == 1
    assert int(triage_clusters.iloc[0]["window_count"]) == 2
    assert int(triage_clusters.iloc[0]["interesting_candidate_windows"]) == 1
    assert int(triage_clusters.iloc[0]["movement_spiking_like_windows"]) == 1

    session_summary = outputs["off_swr_candidate_session_summary.csv"].iloc[0]
    assert int(session_summary["candidate_windows"]) == 2
    assert int(session_summary["candidate_clusters"]) == 1

    rat_summary = outputs["off_swr_candidate_rat_summary.csv"].iloc[0]
    assert int(rat_summary["candidate_windows"]) == 2

    candidate_vs_swr = outputs["off_swr_candidate_vs_swr_summary.csv"].iloc[0]
    assert int(candidate_vs_swr["off_swr_candidate_windows"]) == 2
    assert int(candidate_vs_swr["swr_reference_windows"]) == 1
    assert candidate_vs_swr["off_swr_vs_swr_interpretation"] == "B_weaker_but_directionally_similar_tail"
    assert not bool(candidate_vs_swr["claim_should_narrow"])
    assert pd.notna(candidate_vs_swr["candidate_median_decoded_speed"])
    assert pd.notna(candidate_vs_swr["swr_median_decoded_speed"])
    assert "sorted-spike-state-space-first-order-imm" in candidate_vs_swr["off_swr_best_trajectory_model_distribution"]

    contrast = outputs["off_swr_candidate_vs_swr_window_table.csv"]
    assert {"off_swr_candidate", "swr_replay"} == set(contrast["window_set"])
    contrast_columns = {
        "trajectory_family_margin",
        "best_trajectory_model",
        "n_spikes",
        "active_cell_count",
        "trajectory_posterior_entropy",
        "decoded_path_length",
        "decoded_speed",
        "duration_s",
        "distance_to_nearest_swr_s",
    }
    assert contrast_columns.issubset(contrast.columns)

    model_distribution = outputs["off_swr_candidate_vs_swr_model_distribution.csv"]
    assert {"off_swr_candidate", "swr_replay"} == set(model_distribution["window_set"])
    assert model_distribution["fraction"].between(0.0, 1.0).all()

    run_state = outputs["off_swr_run_state_stratified_summary.csv"]
    assert {
        "off_swr_immobile_windows",
        "off_swr_running_windows",
        "off_swr_unknown_speed_windows",
        "swr_replay_windows",
    } == set(run_state["stratum"])
    immobile = run_state[run_state["stratum"].eq("off_swr_immobile_windows")].iloc[0]
    running = run_state[run_state["stratum"].eq("off_swr_running_windows")].iloc[0]
    swr = run_state[run_state["stratum"].eq("swr_replay_windows")].iloc[0]
    assert int(immobile["windows"]) == 3
    assert int(immobile["trajectory_family_candidates"]) == 1
    assert int(running["windows"]) == 1
    assert int(running["trajectory_family_candidates"]) == 1
    assert int(swr["windows"]) == 1

    run_state_specificity = outputs["off_swr_run_state_specificity_summary.csv"].iloc[0]
    assert run_state_specificity["run_state_specificity_interpretation"] == "immobile_off_swr_candidates_present"
    assert bool(run_state_specificity["immobile_candidate_signal_present"])
    assert not bool(run_state_specificity["claim_should_narrow_for_run_state"])

    nearest_swr = outputs["off_swr_nearest_swr_exclusion_summary.csv"]
    assert nearest_swr["exclusion_radius_s"].tolist() == [0.1, 0.25, 0.5, 1.0]
    one_second = nearest_swr[nearest_swr["exclusion_radius_s"].eq(1.0)].iloc[0]
    assert int(one_second["windows_after_exclusion"]) == 4
    assert int(one_second["candidate_windows_after_exclusion"]) == 2
    assert float(one_second["candidate_fraction_after_exclusion"]) == 0.5
    assert one_second["nearest_swr_exclusion_interpretation"] == "candidate_signal_persists_after_nearest_swr_exclusion"

    nearest_specificity = outputs["off_swr_nearest_swr_specificity_summary.csv"].iloc[0]
    assert nearest_specificity["nearest_swr_specificity_interpretation"] == "candidate_signal_persists_beyond_500ms_and_1s"
    assert not bool(nearest_specificity["claim_should_narrow_for_nearest_swr"])

    tier_summary = outputs["off_swr_candidate_tier_threshold_summary.csv"]
    assert tier_summary["candidate_tier"].tolist() == ["weak", "moderate", "strong", "extreme"]
    weak_tier = tier_summary[tier_summary["candidate_tier"].eq("weak")].iloc[0]
    moderate_tier = tier_summary[tier_summary["candidate_tier"].eq("moderate")].iloc[0]
    assert int(weak_tier["candidate_windows"]) == 2
    assert int(weak_tier["immobile_candidate_windows"]) == 1
    assert int(weak_tier["running_candidate_windows"]) == 1
    assert int(weak_tier["candidate_windows_after_1s_swr_exclusion"]) == 2
    assert int(moderate_tier["candidate_windows"]) == 0

    tier_distance = outputs["off_swr_candidate_tier_nearest_swr_exclusion_summary.csv"]
    assert len(tier_distance) == 16
    assert set(tier_distance["candidate_tier"]) == {"weak", "moderate", "strong", "extreme"}

    promotion = outputs["off_swr_promotion_readiness_summary.csv"].iloc[0]
    assert promotion["promotion_status"] == "exploratory_no_strong_candidates"
    assert not bool(promotion["promotion_ready"])

    speed_coverage = outputs["off_swr_speed_coverage_summary.csv"].iloc[0]
    assert speed_coverage["speed_coverage_status"] == "speed_available_for_promotion_gate"
    assert bool(speed_coverage["speed_coverage_ready"])
    assert int(speed_coverage["candidate_windows_with_speed"]) == 2

    specificity_gates = outputs["off_swr_candidate_specificity_gate_summary.csv"]
    assert bool(specificity_gates[specificity_gates["gate"].eq("overall")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("movement_spiking_like_candidates_flagged")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("candidate_vs_swr_window_table_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("swr_reference_windows_available_for_contrast")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("run_state_stratified_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("run_state_specificity_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("nearest_swr_exclusion_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("nearest_swr_specificity_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("candidate_tier_threshold_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("candidate_tier_rat_session_summaries_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("candidate_tier_nearest_swr_exclusion_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("high_specificity_candidate_table_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("promotion_readiness_summary_written")].iloc[0]["passed"])
    assert bool(specificity_gates[specificity_gates["gate"].eq("speed_coverage_summary_written")].iloc[0]["passed"])

    summary = outputs["off_swr_trajectory_candidate_summary.csv"].iloc[0]
    assert int(summary["windows"]) == 5
    assert int(summary["off_swr_windows"]) == 4
    assert int(summary["trajectory_family_candidates"]) == 2
    assert int(summary["static_nontrajectory_windows"]) == 1
    assert int(summary["ambiguous_windows"]) == 1
    assert int(summary["excluded_known_swr_overlap_windows"]) == 1

    clusters = outputs["off_swr_candidate_clusters.csv"]
    assert len(clusters) == 1
    assert int(clusters.iloc[0]["window_count"]) == 2
    assert clusters.iloc[0]["template_event_indices"] == "0 1"

    behavior = outputs["off_swr_candidate_behavior_lfp_summary.csv"]
    ripple_rows = behavior[behavior["feature"].eq("ripple_power")]
    assert not ripple_rows.empty
    assert ripple_rows["feature_available"].all()
    assert set(ripple_rows["candidate_class"]) >= {TRAJECTORY_CANDIDATE_CLASS, STATIC_NONTRAJECTORY_CLASS, AMBIGUOUS_CLASS}

    gates = outputs["off_swr_candidate_gate_summary.csv"]
    overall = gates[gates["gate"].eq("overall")].iloc[0]
    assert bool(overall["passed"])
    candidate_gate = gates[gates["gate"].eq("trajectory_candidates_detected")].iloc[0]
    assert bool(candidate_gate["passed"])
    assert not bool(candidate_gate["required_for_overall"])

    for filename in outputs:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_off_swr_candidate_table_parses_string_false_claim_flags():
    scores = pd.DataFrame(
        _event_rows(
            "Rat1/Open1",
            0,
            "matched_null",
            0,
            stationary=0.0,
            trajectory=8.0,
            start=10.0,
            ripple_power=1.2,
            animal_speed_mean=1.0,
        )
    )
    decisions = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 0,
                "window_role": "matched_null",
                "null_index": 0,
                "candidate_class": TRAJECTORY_CANDIDATE_CLASS,
                "is_trajectory_family_candidate": "True",
                "trajectory_confident_claim": "True",
                "nontrajectory_confident_claim": "False",
                "passes_known_swr_exclusion": "True",
                "window_start_s": 10.0,
                "window_end_s": 10.1,
                "window_duration_s": 0.1,
                "trajectory_minus_nontrajectory_log_evidence": 8.0,
                "best_trajectory_model": "sorted-spike-state-space-first-order-imm",
                "n_spikes": 10,
                "active_cell_count": 5,
            }
        ]
    )

    table = _off_swr_run_state_window_table(
        decisions,
        scores,
        required_models=FULL_CORE_REQUIRED_MODELS,
        trajectory_models=FULL_CORE_REQUIRED_MODELS[1:],
    )

    row = table.iloc[0]
    assert bool(row["is_trajectory_family_candidate"]) is True
    assert bool(row["trajectory_confident_claim"]) is True
    assert bool(row["nontrajectory_confident_claim"]) is False


def test_off_swr_triage_preserves_lightweight_comparison_scope(tmp_path):
    lightweight_models = {
        "sorted-spike-state-space-stationary",
        "sorted-spike-state-space-first-order-imm",
    }
    rows = [
        *_event_rows("Rat1/Open1", 0, "matched_null", 0, stationary=0.0, trajectory=8.0, start=10.0, ripple_power=1.2),
        *_event_rows("Rat1/Open1", 1, "real", -1, stationary=0.0, trajectory=12.0, start=1.0, ripple_power=2.0, off_swr=False),
    ]
    scores = pd.DataFrame([row for row in rows if row["model"] in lightweight_models])

    outputs = write_off_swr_trajectory_discovery_outputs(
        scores,
        tmp_path,
        comparison_scope="lightweight-first-order-imm-vs-stationary",
    )

    triage = outputs["off_swr_candidate_table.csv"]
    assert len(triage) == 1
    assert triage.iloc[0]["best_trajectory_model"] == "sorted-spike-state-space-first-order-imm"
    assert triage.iloc[0]["trajectory_confidence"] > 0.99
    assert outputs["off_swr_candidate_specificity_gate_summary.csv"].query("gate == 'overall'").iloc[0]["passed"]


def test_off_swr_candidate_tier_summaries_report_selective_thresholds(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows("Rat1/Open1", 0, "matched_null", 0, stationary=0.0, trajectory=8.0, start=10.0, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 1, "matched_null", 0, stationary=0.0, trajectory=25.0, start=10.25, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 2, "matched_null", 0, stationary=0.0, trajectory=60.0, start=10.5, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 3, "matched_null", 0, stationary=0.0, trajectory=120.0, start=10.75, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 4, "matched_null", 0, stationary=80.0, trajectory=0.0, start=11.0, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 5, "matched_null", 0, stationary=80.0, trajectory=0.0, start=11.25, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 6, "matched_null", 0, stationary=80.0, trajectory=0.0, start=11.5, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 7, "matched_null", 0, stationary=80.0, trajectory=0.0, start=11.75, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 8, "matched_null", 0, stationary=80.0, trajectory=0.0, start=12.0, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 9, "matched_null", 0, stationary=80.0, trajectory=0.0, start=12.25, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 10, "real", -1, stationary=0.0, trajectory=20.0, start=1.0, ripple_power=2.0, off_swr=False),
        ]
    )

    outputs = write_off_swr_trajectory_discovery_outputs(scores, tmp_path, cluster_gap_s=0.5)

    triage = outputs["off_swr_candidate_table.csv"]
    assert set(triage["candidate_tier"]) == {"weak", "moderate", "strong", "extreme"}

    tier_summary = outputs["off_swr_candidate_tier_threshold_summary.csv"].set_index("candidate_tier")
    assert int(tier_summary.loc["weak", "candidate_windows"]) == 4
    assert int(tier_summary.loc["moderate", "candidate_windows"]) == 3
    assert int(tier_summary.loc["strong", "candidate_windows"]) == 2
    assert int(tier_summary.loc["extreme", "candidate_windows"]) == 1
    assert int(tier_summary.loc["extreme", "immobile_candidate_windows"]) == 1
    assert int(tier_summary.loc["extreme", "candidate_windows_after_1s_swr_exclusion"]) == 1

    session_summary = outputs["off_swr_candidate_tier_session_summary.csv"]
    rat_summary = outputs["off_swr_candidate_tier_rat_summary.csv"]
    assert int(session_summary[session_summary["candidate_tier"].eq("strong")].iloc[0]["candidate_windows"]) == 2
    assert int(rat_summary[rat_summary["candidate_tier"].eq("extreme")].iloc[0]["candidate_windows"]) == 1

    tier_distance = outputs["off_swr_candidate_tier_nearest_swr_exclusion_summary.csv"]
    extreme_one_second = tier_distance[
        tier_distance["candidate_tier"].eq("extreme") & tier_distance["exclusion_radius_s"].eq(1.0)
    ].iloc[0]
    assert int(extreme_one_second["candidate_windows_after_exclusion"]) == 1
    assert float(extreme_one_second["candidate_retention_after_exclusion"]) == 1.0

    high_specificity = outputs["off_swr_high_specificity_candidate_table.csv"]
    assert len(high_specificity) == 2
    assert high_specificity["passes_high_specificity_promotion_filter"].map(bool).all()
    assert high_specificity["passes_specificity_label_filter"].map(bool).all()
    assert set(high_specificity["candidate_specificity_label"]) == {INTERESTING_CANDIDATE_LABEL}

    promotion = outputs["off_swr_promotion_readiness_summary.csv"].iloc[0]
    assert promotion["promotion_status"] == "ready_for_off_swr_replay_candidate_claim"
    assert bool(promotion["promotion_ready"])
    assert int(promotion["strong_candidate_windows"]) == 2
    assert int(promotion["high_specificity_candidate_windows"]) == 2

    speed_coverage = outputs["off_swr_speed_coverage_summary.csv"].iloc[0]
    assert speed_coverage["speed_coverage_status"] == "speed_available_for_promotion_gate"
    assert bool(speed_coverage["speed_coverage_ready"])
    assert int(speed_coverage["strong_candidate_windows_with_speed"]) == 2


def test_off_swr_high_specificity_excludes_movement_spiking_like_rows(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows("Rat1/Open1", 0, "matched_null", 0, stationary=0.0, trajectory=60.0, start=10.0, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 1, "matched_null", 0, stationary=0.0, trajectory=120.0, start=10.25, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 2, "real", -1, stationary=0.0, trajectory=20.0, start=1.0, ripple_power=2.0, off_swr=False),
        ]
    )

    outputs = write_off_swr_trajectory_discovery_outputs(scores, tmp_path, cluster_gap_s=0.5)

    high_specificity = outputs["off_swr_high_specificity_candidate_table.csv"]
    assert len(high_specificity) == 2
    assert not high_specificity["passes_high_specificity_promotion_filter"].map(bool).any()
    assert not high_specificity["passes_specificity_label_filter"].map(bool).any()
    assert set(high_specificity["candidate_specificity_label"]) == {MOVEMENT_SPIKING_LIKE_LABEL}
    assert set(high_specificity["high_specificity_label"]) == {
        "tier_distance_candidate_movement_spiking_or_low_information",
    }

    promotion = outputs["off_swr_promotion_readiness_summary.csv"].iloc[0]
    assert promotion["promotion_status"] == "exploratory_high_specificity_filter_failed"
    assert not bool(promotion["promotion_ready"])
    assert int(promotion["high_specificity_candidate_windows"]) == 0


def test_off_swr_vs_swr_contrast_flags_movement_like_candidate_tail(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows(
                "Rat1/Open1",
                0,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=8.0,
                start=10.0,
                ripple_power=1.2,
                animal_speed_mean=20.0,
            ),
            *_event_rows(
                "Rat1/Open1",
                1,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=9.0,
                start=10.25,
                ripple_power=1.4,
                animal_speed_mean=22.0,
            ),
            *_event_rows("Rat1/Open1", 2, "real", -1, stationary=0.0, trajectory=20.0, start=1.0, ripple_power=2.0, off_swr=False),
        ]
    )

    outputs = write_off_swr_trajectory_discovery_outputs(scores, tmp_path, cluster_gap_s=0.5)

    summary = outputs["off_swr_candidate_vs_swr_summary.csv"].iloc[0]
    assert summary["off_swr_vs_swr_interpretation"] == "C_mostly_movement_behavioral_decoding_windows"
    assert bool(summary["claim_should_narrow"])
    assert float(summary["candidate_fraction_run"]) == 1.0

    run_state_specificity = outputs["off_swr_run_state_specificity_summary.csv"].iloc[0]
    assert run_state_specificity["run_state_specificity_interpretation"] == "candidate_signal_concentrated_in_running_windows"
    assert not bool(run_state_specificity["immobile_candidate_signal_present"])
    assert bool(run_state_specificity["claim_should_narrow_for_run_state"])

    gate = outputs["off_swr_candidate_specificity_gate_summary.csv"]
    narrowing = gate[gate["gate"].eq("movement_like_claim_narrowing_flagged")].iloc[0]
    assert bool(narrowing["passed"])
    assert "claim_should_narrow=True" in str(narrowing["observed"])
    run_state_narrowing = gate[gate["gate"].eq("immobile_off_swr_candidate_signal_reported")].iloc[0]
    assert "claim_should_narrow=True" in str(run_state_narrowing["observed"])


def test_nearest_swr_exclusion_flags_candidates_that_vanish_near_swr(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows(
                "Rat1/Open1",
                0,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=8.0,
                start=1.15,
                ripple_power=1.2,
                animal_speed_mean=1.0,
            ),
            *_event_rows(
                "Rat1/Open1",
                1,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=9.0,
                start=1.2,
                ripple_power=1.4,
                animal_speed_mean=1.0,
            ),
            *_event_rows("Rat1/Open1", 2, "real", -1, stationary=0.0, trajectory=20.0, start=1.0, ripple_power=2.0, off_swr=False),
        ]
    )

    outputs = write_off_swr_trajectory_discovery_outputs(scores, tmp_path, cluster_gap_s=0.5)

    nearest_swr = outputs["off_swr_nearest_swr_exclusion_summary.csv"]
    half_second = nearest_swr[nearest_swr["exclusion_radius_s"].eq(0.5)].iloc[0]
    assert int(half_second["candidate_windows_after_exclusion"]) == 0
    assert half_second["nearest_swr_exclusion_interpretation"] == "no_evaluable_windows_after_exclusion"

    nearest_specificity = outputs["off_swr_nearest_swr_specificity_summary.csv"].iloc[0]
    assert nearest_specificity["nearest_swr_specificity_interpretation"] == "candidate_signal_vanishes_by_500ms_nearest_swr_exclusion"
    assert bool(nearest_specificity["claim_should_narrow_for_nearest_swr"])

    gate = outputs["off_swr_candidate_specificity_gate_summary.csv"]
    nearest_gate = gate[gate["gate"].eq("nearest_swr_distance_specificity_reported")].iloc[0]
    assert "claim_should_narrow=True" in str(nearest_gate["observed"])


def _event_rows(
    session: str,
    event_index: int,
    window_role: str,
    null_index: int,
    *,
    stationary: float,
    trajectory: float,
    start: float,
    ripple_power: float,
    off_swr: bool = True,
    animal_speed_mean: float = 0.5,
) -> list[dict[str, object]]:
    models = [
        ("sorted-spike-state-space-stationary", stationary, "nontrajectory"),
        ("sorted-spike-state-space-diffusion", trajectory - 3.0, "trajectory"),
        ("sorted-spike-state-space-fragmented", trajectory - 2.0, "trajectory"),
        ("sorted-spike-state-space-first-order-imm", trajectory, "trajectory"),
        ("sorted-spike-state-space-momentum-exact-sparse", trajectory - 1.0, "trajectory"),
    ]
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "window_role": window_role,
            "event_window_variant": "core" if window_role == "real" else "matched_null",
            "null_index": null_index,
            "matched_null_rank": max(null_index, 0),
            "template_event_index": event_index,
            "window_start_s": start,
            "window_end_s": start + 0.1,
            "window_duration_s": 0.1,
            "real_event_start_s": 1.0 + event_index,
            "real_event_end_s": 1.1 + event_index,
            "real_event_duration_s": 0.1,
            "model": model,
            "requested_model": model,
            "model_family": family,
            "log_evidence": log_evidence,
            "n_time": 25,
            "n_spikes": 10 + event_index,
            "null_active_cell_count": 5,
            "real_n_spikes": 10,
            "n_spikes_delta": event_index,
            "n_spikes_relative_delta": event_index / 10.0,
            "ripple_power": ripple_power,
            "animal_speed_mean": animal_speed_mean,
            "animal_speed_median": animal_speed_mean,
            "animal_speed_max": animal_speed_mean + 1.0,
            "position_sample_count": 3,
            "diagnostic_mean_trajectory_posterior_entropy": 0.5 + 0.1 * event_index,
            "diagnostic_decoded_start_x": 1.0,
            "diagnostic_decoded_start_y": 2.0,
            "diagnostic_decoded_endpoint_x": 4.0 + event_index,
            "diagnostic_decoded_endpoint_y": 6.0,
            "diagnostic_posterior_mean_path_length_cm": 10.0 + event_index,
            "animal_x": 2.0,
            "animal_y": 3.0,
            "off_swr": off_swr,
            "evidence_comparable": True,
        }
        for model, log_evidence, family in models
    ]
