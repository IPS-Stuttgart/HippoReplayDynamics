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
    assert event["best_comparable_model"] == exact_reference
    assert np.isneginf(event["best_comparable_log_evidence"])
    assert np.isposinf(event["expected_minus_best_comparable_log_evidence"])
