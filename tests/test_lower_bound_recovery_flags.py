from __future__ import annotations

import pandas as pd

import hipporeplayimm
import hipporeplayimm.evidence_reporting as evidence_reporting


_MOMENTUM_EXACT_SURROGATE = "sorted-spike-state-space-momentum-exact-sparse"


def test_lower_bound_recovery_flags_accept_exact_surrogates_event_wide() -> None:
    hipporeplayimm.apply_runtime_patches()
    frame = pd.DataFrame(
        [
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "momentum",
                "model": _MOMENTUM_EXACT_SURROGATE,
                "log_evidence": 2.0,
                "diagnostic_state_space_momentum_evidence_support": evidence_reporting.TRUNCATED_EVIDENCE_SUPPORT,
                "status": "success",
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "stationary",
                "model": "stationary",
                "log_evidence": 0.0,
                "diagnostic_state_space_momentum_evidence_support": evidence_reporting.EXACT_EVIDENCE_SUPPORT,
                "status": "success",
            },
        ]
    )

    scored = evidence_reporting.simulation_add_evidence_columns(frame)

    assert scored["best_truncated_lower_bound_model"].eq(_MOMENTUM_EXACT_SURROGATE).all()
    assert scored["lower_bound_recovered_expected_model"].to_numpy(dtype=bool).all()
