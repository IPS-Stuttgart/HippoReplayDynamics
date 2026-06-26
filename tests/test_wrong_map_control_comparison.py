from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from compare_wrong_map_evidence_controls import (
    rat_bootstrap_wrong_map_family_evidence_attenuation,
    wrong_map_control_gate_summary,
    wrong_map_family_evidence_attenuation,
    wrong_map_family_evidence_attenuation_summary,
    wrong_map_model_evidence_attenuation,
    write_wrong_map_comparison_outputs,
)


def test_wrong_map_absolute_attenuation_can_pass_when_margin_did_is_negative(tmp_path: Path):
    real = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, "sorted-spike-state-space-stationary", 0.0),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 20.0),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 25.0),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 60.0),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-momentum-exact-sparse", 40.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-stationary", 0.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-diffusion", 18.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-fragmented", 30.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-first-order-imm", 50.0),
            _score("Rat2/Open1", 1, "sorted-spike-state-space-momentum-exact-sparse", 35.0),
        ]
    )
    wrong = pd.DataFrame(
        [
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-stationary", -300.0),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-diffusion", -170.0),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-fragmented", -150.0),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-first-order-imm", -120.0),
            _wrong_score(
                "Rat1/Open1",
                "Rat1/Open2",
                0,
                "sorted-spike-state-space-momentum-exact-sparse",
                -140.0,
            ),
            _wrong_score("Rat2/Open1", "Rat2/Open2", 1, "sorted-spike-state-space-stationary", -500.0),
            _wrong_score("Rat2/Open1", "Rat2/Open2", 1, "sorted-spike-state-space-diffusion", -230.0),
            _wrong_score("Rat2/Open1", "Rat2/Open2", 1, "sorted-spike-state-space-fragmented", -220.0),
            _wrong_score("Rat2/Open1", "Rat2/Open2", 1, "sorted-spike-state-space-first-order-imm", -200.0),
            _wrong_score(
                "Rat2/Open1",
                "Rat2/Open2",
                1,
                "sorted-spike-state-space-momentum-exact-sparse",
                -210.0,
            ),
        ]
    )

    attenuation = wrong_map_model_evidence_attenuation(real, wrong)
    family = wrong_map_family_evidence_attenuation(real, wrong)
    summary = wrong_map_family_evidence_attenuation_summary(family)
    gate = wrong_map_control_gate_summary(family, n_bootstrap=100, random_seed=1)

    assert len(attenuation) == 10
    assert (family["best_trajectory_delta_real_minus_wrong"] > 0).all()
    assert (family["family_margin_difference_in_differences"] < 0).all()
    assert float(summary.iloc[0]["mean_best_trajectory_delta_real_minus_wrong"]) > 0.0
    assert float(summary.iloc[0]["mean_family_margin_difference_in_differences"]) < 0.0
    assert gate.loc[gate["gate"] == "overall", "passed"].iloc[0]
    assert gate.loc[
        gate["gate"] == "family_margin_difference_in_differences_reported",
        "criterion",
    ].iloc[0] == "family-margin DID is diagnostic and is not required to be positive"

    write_wrong_map_comparison_outputs(real, wrong, tmp_path, n_bootstrap=100, random_seed=1)
    for name in (
        "wrong_map_model_evidence_attenuation.csv",
        "wrong_map_family_evidence_attenuation.csv",
        "wrong_map_family_evidence_attenuation_summary.csv",
        "rat_wrong_map_family_evidence_attenuation.csv",
        "leave_one_rat_out_wrong_map_family_evidence_attenuation.csv",
        "rat_bootstrap_wrong_map_family_evidence_attenuation.csv",
        "wrong_map_margin_difference_in_differences.csv",
        "wrong_map_control_gate_summary.csv",
    ):
        assert (tmp_path / name).is_file()


def test_wrong_map_comparison_keeps_legacy_missing_status_rows():
    real = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, "sorted-spike-state-space-stationary", 0.0, status=pd.NA),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-diffusion", 5.0, status=""),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-fragmented", 6.0, status=float("nan")),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-first-order-imm", 9.0, status="success"),
            _score("Rat1/Open1", 0, "sorted-spike-state-space-momentum-exact-sparse", 7.0, status=None),
            _score("Rat1/Open1", 0, "failed-model", 999.0, status="failed"),
        ]
    )
    wrong = pd.DataFrame(
        [
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-stationary", -20.0, status=pd.NA),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-diffusion", -10.0, status=""),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-fragmented", -9.0, status=float("nan")),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-first-order-imm", -8.0, status="success"),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "sorted-spike-state-space-momentum-exact-sparse", -7.0, status=None),
            _wrong_score("Rat1/Open1", "Rat1/Open2", 0, "failed-model", -999.0, status="failed"),
        ]
    )

    attenuation = wrong_map_model_evidence_attenuation(real, wrong)
    family = wrong_map_family_evidence_attenuation(real, wrong)
    summary = wrong_map_family_evidence_attenuation_summary(family).iloc[0]

    assert len(attenuation) == 5
    assert "failed-model" not in set(attenuation["model"])
    assert len(family) == 1
    assert bool(family.loc[0, "required_models_complete_both_maps"])
    assert summary["complete_family_events"] == 1
    assert summary["mean_best_trajectory_delta_real_minus_wrong"] > 0.0


def test_wrong_map_summary_treats_string_false_complete_flag_as_false():
    family = pd.DataFrame(
        [
            _family_row(
                rat="Rat1",
                session="Rat1/Open1",
                event_index=0,
                complete=True,
                best_delta=10.0,
                core_delta=8.0,
                stationary_delta=4.0,
                margin_real=12.0,
                margin_wrong=7.0,
            ),
            _family_row(
                rat="Rat2",
                session="Rat2/Open1",
                event_index=1,
                complete="False",
                best_delta=1000.0,
                core_delta=900.0,
                stationary_delta=800.0,
                margin_real=700.0,
                margin_wrong=-300.0,
            ),
        ]
    )

    summary = wrong_map_family_evidence_attenuation_summary(family).iloc[0]
    bootstrap = rat_bootstrap_wrong_map_family_evidence_attenuation(
        family,
        n_bootstrap=20,
        random_seed=3,
    ).iloc[0]

    assert summary["events"] == 2
    assert summary["complete_family_events"] == 1
    assert summary["mean_best_trajectory_delta_real_minus_wrong"] == 10.0
    assert summary["mean_family_margin_difference_in_differences"] == 5.0
    assert bootstrap["observed_mean_best_trajectory_delta_real_minus_wrong"] == 10.0


def _score(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    status: object = "success",
) -> dict[str, object]:
    return {
        "status": status,
        "session": session,
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "log_evidence": log_evidence,
    }


def _wrong_score(
    session: str,
    map_session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    status: object = "success",
) -> dict[str, object]:
    row = _score(session, event_index, model, log_evidence, status=status)
    row["map_session"] = map_session
    row["wrong_map_control"] = True
    return row


def _family_row(
    *,
    rat: str,
    session: str,
    event_index: int,
    complete: object,
    best_delta: float,
    core_delta: float,
    stationary_delta: float,
    margin_real: float,
    margin_wrong: float,
) -> dict[str, object]:
    return {
        "rat": rat,
        "session": session,
        "event_index": event_index,
        "map_session": f"{session}-wrong",
        "required_models_complete_real_map": complete,
        "required_models_complete_wrong_map": complete,
        "required_models_complete_both_maps": complete,
        "missing_required_models_real_map": "",
        "missing_required_models_wrong_map": "",
        "best_trajectory_model_real_map": "sorted-spike-state-space-first-order-imm",
        "best_trajectory_log_evidence_real_map": 0.0,
        "same_trajectory_model_log_evidence_wrong_map": -best_delta,
        "best_trajectory_delta_real_minus_wrong": best_delta,
        "best_trajectory_model_wrong_map": "sorted-spike-state-space-first-order-imm",
        "best_trajectory_log_evidence_wrong_map": -best_delta,
        "best_core_model_real_map": "sorted-spike-state-space-first-order-imm",
        "best_core_log_evidence_real_map": 0.0,
        "same_core_model_log_evidence_wrong_map": -core_delta,
        "best_core_delta_real_minus_wrong": core_delta,
        "best_nontrajectory_model_real_map": "sorted-spike-state-space-stationary",
        "best_nontrajectory_log_evidence_real_map": -margin_real,
        "best_nontrajectory_model_wrong_map": "sorted-spike-state-space-stationary",
        "best_nontrajectory_log_evidence_wrong_map": -margin_wrong,
        "stationary_log_evidence_real_map": 0.0,
        "stationary_log_evidence_wrong_map": -stationary_delta,
        "stationary_delta_real_minus_wrong": stationary_delta,
        "family_margin_real_map": margin_real,
        "family_margin_wrong_map": margin_wrong,
        "family_margin_difference_in_differences": margin_real - margin_wrong,
    }
