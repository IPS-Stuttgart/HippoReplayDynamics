from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_replay_dynamics_axis import (
    DIFFUSION,
    DYNAMICS_INDICES,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    build_dynamics_axis_covariate_model,
    build_dynamics_axis_gate_summary,
    build_dynamics_axis_leave_one_rat_out,
    build_event_dynamics_axis,
    build_dynamics_axis_bootstrap_summary,
    write_dynamics_axis_pack,
)


def test_replay_dynamics_axis_computes_indices_and_margins(tmp_path: Path):
    evidence = pd.DataFrame(
        [
            *_event_scores("Rat1/Open1", 1, [0.0, 1.0, 2.0, 4.0, 3.0], 0.10, 30),
            *_event_scores("Rat1/Open1", 2, [0.0, 3.0, 1.0, 2.0, 4.0], 0.20, 40),
            *_event_scores("Rat2/Open1", 3, [4.0, 1.0, 0.0, 2.0, 3.0], 0.30, 50),
            *_event_scores("Rat2/Open1", 4, [0.0, 2.0, 1.0, 5.0, 3.0], 0.40, 60),
        ]
    )

    event_axis = build_event_dynamics_axis(
        evidence,
        covariates=("pre_event_rate", "event_duration", "spike_count"),
    )

    assert len(event_axis) == 4
    assert event_axis["exact_core_complete"].all()
    first = event_axis[event_axis["event_index"].eq(1)].iloc[0]
    probabilities = _softmax([0.0, 1.0, 2.0, 4.0, 3.0])

    assert first["P_stationary"] == pytest.approx(probabilities[0])
    assert first["P_diffusion"] == pytest.approx(probabilities[1])
    assert first["P_fragmented"] == pytest.approx(probabilities[2])
    assert first["P_first_order_imm"] == pytest.approx(probabilities[3])
    assert first["P_momentum_exact_sparse"] == pytest.approx(probabilities[4])
    assert first["diffusivity_index"] == pytest.approx(
        probabilities[1] + probabilities[2] - probabilities[0]
    )
    assert first["momentum_index"] == pytest.approx(probabilities[4] - probabilities[1])
    assert first["trajectory_family_index"] == pytest.approx(1.0 - 2 * probabilities[0])
    assert first["switching_index"] == pytest.approx(
        probabilities[3] - max(probabilities[1], probabilities[4])
    )
    assert first["trajectory_family_margin"] == pytest.approx(4.0)
    assert first["momentum_minus_diffusion_margin"] == pytest.approx(2.0)
    assert first["first_order_imm_minus_momentum_margin"] == pytest.approx(1.0)

    covariate_model = build_dynamics_axis_covariate_model(
        event_axis,
        ("pre_event_rate", "missing_metric", "event_duration"),
    )
    assert set(covariate_model["outcome"]) == {
        "diffusivity_index",
        "momentum_index",
        "trajectory_family_index",
        "switching_index",
    }
    assert "missing_metric" in covariate_model["covariates_missing"].iloc[0]

    leave_one_rat_out = build_dynamics_axis_leave_one_rat_out(
        event_axis,
        ("pre_event_rate", "event_duration"),
    )
    assert set(leave_one_rat_out["held_out_rat"]) == {"Rat1", "Rat2"}

    bootstrap = build_dynamics_axis_bootstrap_summary(
        event_axis,
        replicates=25,
        random_seed=1,
    )
    assert len(bootstrap) == 8
    assert set(bootstrap["statistic"]) == {"mean", "median"}

    gate = build_dynamics_axis_gate_summary(
        event_axis,
        covariate_model,
        leave_one_rat_out,
        bootstrap,
        ("pre_event_rate", "missing_metric", "event_duration"),
    )
    assert gate[gate["gate"].eq("overall")]["passed"].iloc[0]
    assert not gate[gate["gate"].eq("requested_external_covariates_present")][
        "passed"
    ].iloc[0]

    outputs = write_dynamics_axis_pack(
        evidence,
        tmp_path,
        covariates=("pre_event_rate", "event_duration"),
        bootstrap_replicates=25,
    )
    assert set(outputs) == {
        "event_dynamics_axis.csv",
        "rat_dynamics_axis_summary.csv",
        "dynamics_axis_covariate_model.csv",
        "dynamics_axis_leave_one_rat_out.csv",
        "dynamics_axis_bootstrap_summary.csv",
        "dynamics_axis_gate_summary.csv",
    }
    for filename in outputs:
        assert (tmp_path / filename).is_file()


def test_replay_dynamics_axis_pack_handles_no_successful_rows(tmp_path: Path):
    evidence = pd.DataFrame(
        _event_scores("Rat1/Open1", 1, [0.0, 1.0, 2.0, 4.0, 3.0], 0.10, 30)
    )
    evidence["status"] = "failed"

    outputs = write_dynamics_axis_pack(
        evidence,
        tmp_path,
        covariates=("pre_event_rate",),
        bootstrap_replicates=0,
    )

    event_axis = outputs["event_dynamics_axis.csv"]
    assert event_axis.empty
    assert set(DYNAMICS_INDICES).issubset(event_axis.columns)
    assert "rat" in event_axis.columns
    assert "pre_event_rate" in event_axis.columns

    gate = outputs["dynamics_axis_gate_summary.csv"].set_index("gate")
    assert not bool(gate.loc["event_rows_present", "passed"])
    assert gate.loc["event_rows_present", "observed"] == "0"
    assert not bool(gate.loc["overall", "passed"])
    for filename in outputs:
        assert (tmp_path / filename).is_file()


def test_dynamics_axis_gate_parses_string_false_exact_core_flags():
    event_axis = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 1,
                "exact_core_complete": "True",
                "pre_event_rate": 0.1,
                **dict.fromkeys(DYNAMICS_INDICES, 0.1),
            },
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 2,
                "exact_core_complete": "False",
                "pre_event_rate": 0.2,
                **dict.fromkeys(DYNAMICS_INDICES, 0.2),
            },
            {
                "session": "Rat2/Open1",
                "rat": "Rat2",
                "event_index": 3,
                "exact_core_complete": "True",
                "pre_event_rate": 0.3,
                **dict.fromkeys(DYNAMICS_INDICES, 0.3),
            },
        ]
    )
    covariate_model = pd.DataFrame(
        [{"outcome": outcome, "term": "intercept"} for outcome in DYNAMICS_INDICES]
    )

    gate = build_dynamics_axis_gate_summary(
        event_axis,
        covariate_model,
        pd.DataFrame([{"held_out_rat": "Rat1"}]),
        pd.DataFrame([{"outcome": "diffusivity_index"}]),
        ("pre_event_rate",),
    ).set_index("gate")

    assert not bool(gate.loc["exact_core_complete", "passed"])
    assert gate.loc["exact_core_complete", "observed"] == "2/3"
    assert not bool(gate.loc["overall", "passed"])


def _event_scores(
    session: str,
    event_index: int,
    log_evidences: list[float],
    pre_event_rate: float,
    n_spikes: int,
) -> list[dict[str, object]]:
    models = [
        STATIONARY,
        DIFFUSION,
        FRAGMENTED,
        FIRST_ORDER_IMM,
        MOMENTUM_EXACT,
    ]
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "model": model,
            "log_evidence": log_evidence,
            "evidence_comparable": True,
            "n_time": 50 + event_index,
            "time_bin_s": 0.004,
            "n_spikes": n_spikes,
            "pre_event_rate": pre_event_rate,
        }
        for model, log_evidence in zip(models, log_evidences, strict=True)
    ]


def _softmax(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    exp_values = np.exp(arr - np.max(arr))
    return exp_values / exp_values.sum()
