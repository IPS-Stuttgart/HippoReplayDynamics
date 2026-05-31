from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from compare_wrong_map_evidence_controls import (
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


def _score(session: str, event_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
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
) -> dict[str, object]:
    row = _score(session, event_index, model, log_evidence)
    row["map_session"] = map_session
    row["wrong_map_control"] = True
    return row
