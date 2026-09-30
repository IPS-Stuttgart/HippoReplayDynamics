import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
)
from hipporeplayimm.simulation_recovery import (
    add_evidence_columns,
    certified_vs_exact_event_recovery,
)


def _row(
    model: str,
    log_evidence: float,
    *,
    true_model: str,
    expected_model: str,
    evidence_support: str,
    evidence_comparable: bool,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": 0,
        "true_model": true_model,
        "expected_model": expected_model,
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
        "evidence_support": evidence_support,
        "evidence_comparable": evidence_comparable,
    }


def test_negative_infinite_exact_evidence_remains_comparable_zero_mass():
    expected_model = "sorted-spike-state-space-diffusion"
    rows = pd.DataFrame(
        [
            _row(
                expected_model,
                -5.0,
                true_model="diffusion",
                expected_model=expected_model,
                evidence_support=EXACT_EVIDENCE_SUPPORT,
                evidence_comparable=True,
            ),
            _row(
                "sorted-spike-state-space-stationary",
                -np.inf,
                true_model="diffusion",
                expected_model=expected_model,
                evidence_support=EXACT_EVIDENCE_SUPPORT,
                evidence_comparable=True,
            ),
        ]
    )

    scored = add_evidence_columns(rows)
    diffusion = scored[scored["model"] == expected_model].iloc[0]
    stationary = scored[
        scored["model"] == "sorted-spike-state-space-stationary"
    ].iloc[0]

    assert bool(stationary["evidence_comparable"])
    assert np.isneginf(stationary["relative_log_evidence"])
    assert stationary["model_probability"] == 0.0
    assert diffusion["model_probability"] == 1.0
    assert bool(diffusion["is_best_model"])
    assert scored["best_model"].iloc[0] == expected_model


def test_finite_truncated_lower_bound_beats_negative_infinite_exact_reference():
    expected_model = "sorted-spike-state-space-momentum"
    exact_reference = "sorted-spike-state-space-diffusion"
    rows = pd.DataFrame(
        [
            _row(
                expected_model,
                -10.0,
                true_model="momentum",
                expected_model=expected_model,
                evidence_support=TRUNCATED_EVIDENCE_SUPPORT,
                evidence_comparable=False,
            ),
            _row(
                exact_reference,
                -np.inf,
                true_model="momentum",
                expected_model=expected_model,
                evidence_support=EXACT_EVIDENCE_SUPPORT,
                evidence_comparable=True,
            ),
        ]
    )

    scored = add_evidence_columns(rows)
    exact = scored[scored["model"] == exact_reference].iloc[0]
    assert bool(exact["evidence_comparable"])

    event = certified_vs_exact_event_recovery(scored).iloc[0]

    assert bool(event["certified_vs_exact_recovered_expected_model"])
    assert event["certified_vs_exact_reason"] == (
        "expected_lower_bound_beats_best_comparable"
    )
    assert event["best_comparable_model"] == ""
    assert np.isneginf(event["best_comparable_log_evidence"])
    assert np.isposinf(event["expected_minus_best_comparable_log_evidence"])

def test_all_negative_infinite_exact_evidence_remains_unresolved_independent_of_row_order():
    expected_model = "sorted-spike-state-space-diffusion"
    competitor = "sorted-spike-state-space-stationary"
    base_rows = [
        _row(
            expected_model,
            -np.inf,
            true_model="diffusion",
            expected_model=expected_model,
            evidence_support=EXACT_EVIDENCE_SUPPORT,
            evidence_comparable=True,
        ),
        _row(
            competitor,
            -np.inf,
            true_model="diffusion",
            expected_model=expected_model,
            evidence_support=EXACT_EVIDENCE_SUPPORT,
            evidence_comparable=True,
        ),
    ]

    for event_rows in (base_rows, list(reversed(base_rows))):
        scored = add_evidence_columns(pd.DataFrame(event_rows))

        assert scored["best_model"].eq("").all()
        assert not scored["is_best_model"].to_numpy(dtype=bool).any()
        assert scored["evidence_comparable"].to_numpy(dtype=bool).all()

        event = certified_vs_exact_event_recovery(scored).iloc[0]

        assert not bool(event["certified_vs_exact_recovered_expected_model"])
        assert event["certified_vs_exact_reason"] == "all_comparable_impossible"
        assert event["best_comparable_model"] == ""
        assert np.isneginf(event["best_comparable_log_evidence"])
        assert np.isnan(event["expected_minus_best_comparable_log_evidence"])


def test_negative_infinite_truncated_lower_bound_gets_relative_negative_infinity():
    expected_model = "sorted-spike-state-space-momentum"
    impossible_lower_bound = "sorted-spike-state-space-momentum-alt"
    exact_reference = "sorted-spike-state-space-diffusion"
    rows = pd.DataFrame(
        [
            _row(
                expected_model,
                -10.0,
                true_model="momentum",
                expected_model=expected_model,
                evidence_support=TRUNCATED_EVIDENCE_SUPPORT,
                evidence_comparable=False,
            ),
            _row(
                impossible_lower_bound,
                -np.inf,
                true_model="momentum",
                expected_model=expected_model,
                evidence_support=TRUNCATED_EVIDENCE_SUPPORT,
                evidence_comparable=False,
            ),
            _row(
                exact_reference,
                -20.0,
                true_model="momentum",
                expected_model=expected_model,
                evidence_support=EXACT_EVIDENCE_SUPPORT,
                evidence_comparable=True,
            ),
        ]
    )

    scored = add_evidence_columns(rows)
    impossible = scored[scored["model"] == impossible_lower_bound].iloc[0]

    assert scored["best_truncated_lower_bound_model"].eq(expected_model).all()
    assert np.isneginf(impossible["truncated_relative_log_evidence"])

