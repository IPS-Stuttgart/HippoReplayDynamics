from pathlib import Path

import pandas as pd

from scripts.report_hc11_paper_grade_robustness import (
    build_event_table,
    read_event_model_evidence,
    write_outputs,
)


def test_hc11_robustness_outputs_mark_missing_shuffle_as_not_paper_grade(tmp_path: Path) -> None:
    evidence = pd.DataFrame(
        [
            *_event("AnimalA/Session1", 0, stationary=0.0, diffusion=15.0, fragmented=20.0, first_order=50.0, momentum=30.0),
            *_event("AnimalA/Session2", 1, stationary=0.0, diffusion=10.0, fragmented=15.0, first_order=40.0, momentum=25.0),
            *_event("AnimalB/Session1", 2, stationary=0.0, diffusion=12.0, fragmented=8.0, first_order=35.0, momentum=20.0),
            *_event("AnimalB/Session2", 3, stationary=0.0, diffusion=14.0, fragmented=10.0, first_order=38.0, momentum=22.0),
        ]
    )

    outputs = write_outputs(
        evidence,
        tmp_path,
        margin_threshold=5.5,
        n_bootstrap=50,
        seed=3,
    )

    expected = {
        "hc11_event_claim_table.csv",
        "hc11_by_animal_summary.csv",
        "hc11_by_session_summary.csv",
        "hc11_leave_one_animal_out_summary.csv",
        "hc11_animal_cluster_bootstrap.csv",
        "hc11_imm_vs_fragmented_audit.csv",
        "hc11_time_order_shuffle_clean_imm.csv",
        "hc11_posterior_content_audit.csv",
        "hc11_posterior_content_summary.csv",
        "hc11_gate_summary.csv",
    }
    assert set(outputs) == expected
    for name in expected:
        assert (tmp_path / name).is_file()
    assert (tmp_path / "hc11_rat" / "animal_cluster_bootstrap.csv").is_file()

    gates = outputs["hc11_gate_summary.csv"].set_index("gate")
    assert bool(gates.loc["technical_overall", "passed"])
    assert bool(gates.loc["robustness_overall", "passed"])
    assert not bool(gates.loc["time_order_shuffle_artifact_present", "passed"])
    assert not bool(gates.loc["posterior_content_artifact_present", "passed"])
    assert not bool(gates.loc["paper_grade_overall", "passed"])

    by_animal = outputs["hc11_by_animal_summary.csv"].set_index("animal")
    assert set(by_animal.index) == {"AnimalA", "AnimalB"}
    assert int(by_animal.loc["AnimalA", "events"]) == 2
    assert float(by_animal.loc["AnimalB", "trajectory_confident_fraction"]) == 1.0

    imm_audit = outputs["hc11_imm_vs_fragmented_audit.csv"]
    assert imm_audit["within_family_classification"].eq("clean_imm_candidate").all()

    time_order = outputs["hc11_time_order_shuffle_clean_imm.csv"].iloc[0]
    assert time_order["status"] == "not_run"
    assert not bool(time_order["time_order_gate_passed"])
    posterior = outputs["hc11_posterior_content_summary.csv"].iloc[0]
    assert posterior["status"] == "not_run"
    assert not bool(posterior["posterior_content_gate_passed"])


def test_hc11_robustness_requires_posterior_content_for_paper_grade(tmp_path: Path) -> None:
    evidence = pd.DataFrame(
        [
            *_event("AnimalA/Session1", 0, stationary=0.0, diffusion=15.0, fragmented=20.0, first_order=50.0, momentum=30.0),
            *_event("AnimalA/Session2", 1, stationary=0.0, diffusion=10.0, fragmented=15.0, first_order=40.0, momentum=25.0),
            *_event("AnimalB/Session1", 2, stationary=0.0, diffusion=12.0, fragmented=8.0, first_order=35.0, momentum=20.0),
            *_event("AnimalB/Session2", 3, stationary=0.0, diffusion=14.0, fragmented=10.0, first_order=38.0, momentum=22.0),
        ]
    )
    shuffle = tmp_path / "clean_imm_time_order_shuffle_decisions.csv"
    pd.DataFrame(
        [
            _shuffle("AnimalA/Session1", 0, advantage=20.0, above_p95=True),
            _shuffle("AnimalA/Session2", 1, advantage=16.0, above_p95=False),
            _shuffle("AnimalB/Session1", 2, advantage=12.0, above_p95=True),
            _shuffle("AnimalB/Session2", 3, advantage=14.0, above_p95=False),
        ]
    ).to_csv(shuffle, index=False)

    outputs = write_outputs(
        evidence,
        tmp_path / "out",
        margin_threshold=5.5,
        n_bootstrap=50,
        seed=3,
        time_order_shuffle_decisions=shuffle,
    )

    gates = outputs["hc11_gate_summary.csv"].set_index("gate")
    assert bool(gates.loc["time_order_shuffle_artifact_present", "passed"])
    assert bool(gates.loc["time_order_shuffle_clean_imm_gate_passed", "passed"])
    assert not bool(gates.loc["posterior_content_artifact_present", "passed"])
    assert not bool(gates.loc["paper_grade_overall", "passed"])

    time_order = outputs["hc11_time_order_shuffle_clean_imm.csv"].iloc[0]
    assert time_order["status"] == "provided"
    assert int(time_order["clean_imm_events"]) == 4
    assert bool(time_order["time_order_gate_passed"])


def test_hc11_robustness_accepts_time_order_and_posterior_content(tmp_path: Path) -> None:
    evidence = pd.DataFrame(
        [
            *_event("AnimalA/Session1", 0, stationary=0.0, diffusion=15.0, fragmented=20.0, first_order=50.0, momentum=30.0),
            *_event("AnimalA/Session2", 1, stationary=0.0, diffusion=10.0, fragmented=15.0, first_order=40.0, momentum=25.0),
            *_event("AnimalB/Session1", 2, stationary=0.0, diffusion=12.0, fragmented=8.0, first_order=35.0, momentum=20.0),
            *_event("AnimalB/Session2", 3, stationary=0.0, diffusion=14.0, fragmented=10.0, first_order=38.0, momentum=22.0),
        ]
    )
    shuffle = tmp_path / "clean_imm_time_order_shuffle_decisions.csv"
    pd.DataFrame(
        [
            _shuffle("AnimalA/Session1", 0, advantage=20.0, above_p95=True),
            _shuffle("AnimalA/Session2", 1, advantage=16.0, above_p95=False),
            _shuffle("AnimalB/Session1", 2, advantage=12.0, above_p95=True),
            _shuffle("AnimalB/Session2", 3, advantage=14.0, above_p95=False),
        ]
    ).to_csv(shuffle, index=False)
    posterior = tmp_path / "first_order_imm_mode_usage_event_summary.csv"
    pd.DataFrame(
        [
            _posterior("AnimalA/Session1", 0, mean_nonstationary=0.8, map_nonstationary=0.7, path=24.0),
            _posterior("AnimalA/Session2", 1, mean_nonstationary=0.75, map_nonstationary=0.6, path=18.0),
            _posterior("AnimalB/Session1", 2, mean_nonstationary=0.7, map_nonstationary=0.65, path=16.0),
            _posterior("AnimalB/Session2", 3, mean_nonstationary=0.65, map_nonstationary=0.55, path=14.0),
        ]
    ).to_csv(posterior, index=False)

    outputs = write_outputs(
        evidence,
        tmp_path / "out",
        margin_threshold=5.5,
        n_bootstrap=50,
        seed=3,
        time_order_shuffle_decisions=shuffle,
        posterior_content_summary=posterior,
    )

    gates = outputs["hc11_gate_summary.csv"].set_index("gate")
    assert bool(gates.loc["time_order_shuffle_clean_imm_gate_passed", "passed"])
    assert bool(gates.loc["posterior_content_gate_passed", "passed"])
    assert bool(gates.loc["paper_grade_overall", "passed"])

    posterior_summary = outputs["hc11_posterior_content_summary.csv"].iloc[0]
    assert posterior_summary["status"] == "provided"
    assert int(posterior_summary["first_order_imm_best_events"]) == 4
    assert bool(posterior_summary["posterior_content_gate_passed"])


def test_hc11_reader_canonicalizes_short_and_long_model_names(tmp_path: Path) -> None:
    path = tmp_path / "event_model_evidence.csv"
    pd.DataFrame(
        [
            _score("Achilles/day1", 0, "sorted-spike-state-space-stationary", 0.0),
            _score("Achilles/day1", 0, "diffusion", 10.0),
            _score("Achilles/day1", 0, "fragmented", 15.0),
            _score("Achilles/day1", 0, "first_order_imm", 40.0),
            _score("Achilles/day1", 0, "momentum_cv", 22.0),
        ]
    ).to_csv(path, index=False)

    evidence = read_event_model_evidence(path)
    events = build_event_table(evidence, margin_threshold=5.5)

    row = events.iloc[0]
    assert row["animal"] == "Achilles"
    assert row["minimal_core_complete"]
    assert row["best_model"] == "first_order_imm"
    assert row["trajectory_confident_claim"]
    assert row["momentum_raw_win_vs_diffusion"]


def _event(
    session: str,
    event_index: int,
    *,
    stationary: float,
    diffusion: float,
    fragmented: float,
    first_order: float,
    momentum: float,
) -> list[dict[str, object]]:
    return [
        _score(session, event_index, "stationary", stationary),
        _score(session, event_index, "diffusion", diffusion),
        _score(session, event_index, "fragmented", fragmented),
        _score(session, event_index, "first_order_imm", first_order),
        _score(session, event_index, "momentum_cv", momentum),
    ]


def _score(session: str, event_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "duration_ms": 40.0,
        "n_spikes": 12,
        "n_active_units": 5,
    }


def _shuffle(session: str, event_index: int, *, advantage: float, above_p95: bool) -> dict[str, object]:
    return {
        "session": session,
        "rat": session.split("/", 1)[0],
        "event_index": event_index,
        "event_group": "clean_imm",
        "original_delta_imm_minus_fragmented": 30.0,
        "median_shuffle_delta_imm_minus_fragmented": 30.0 - advantage,
        "mean_shuffle_delta_imm_minus_fragmented": 30.0 - advantage,
        "p95_shuffle_delta_imm_minus_fragmented": 29.0 if above_p95 else 35.0,
        "time_order_advantage": advantage,
        "empirical_p_value": 0.05,
        "original_above_shuffle_median": True,
        "original_above_shuffle_p95": above_p95,
        "n_shuffles": 20,
        "duration_ms": 40.0,
        "n_spikes": 12,
        "n_active_units": 5,
    }


def _posterior(
    session: str,
    event_index: int,
    *,
    mean_nonstationary: float,
    map_nonstationary: float,
    path: float,
) -> dict[str, object]:
    return {
        "session": session,
        "rat": session.split("/", 1)[0],
        "event_index": event_index,
        "first_order_imm_is_best_exact_core": True,
        "mean_nonstationary_mode_probability": mean_nonstationary,
        "fraction_time_map_nonstationary": map_nonstationary,
        "nonstationary_bout_count": 1,
        "longest_nonstationary_bout_s": 0.04,
        "posterior_expected_path_length_cm": path,
        "posterior_net_displacement_cm": path / 2.0,
        "posterior_path_speed_cm_s": path * 10.0,
        "trajectory_content_gate_passed": True,
        "strong_trajectory_content_gate_passed": True,
        "content_diagnostic_status": "moderate_posterior_content_gate_passed",
    }
