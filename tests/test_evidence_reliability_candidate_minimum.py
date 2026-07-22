from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def _score_row(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "status": "success",
        "n_spikes": 4,
        "n_time": 3,
        "evidence_support": "truncated_full_grid",
        "mean_candidate_log_mass": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_event_reliability_prefers_minimum_candidate_mass_over_mean():
    flagged = add_event_reliability_flags(
        _score_row(diagnostic_min_candidate_log_mass=-1.0),
    )

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_low_candidate_mass"])
    assert flagged.loc[0, "event_reliability_reasons"] == "low_candidate_mass"


def test_event_reliability_uses_worst_model_specific_minimum():
    flagged = add_event_reliability_flags(
        _score_row(
            min_candidate_log_mass=-0.005,
            diagnostic_state_space_momentum_min_candidate_log_mass=-1.0,
        ),
    )

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_low_candidate_mass"])
    assert flagged.loc[0, "event_reliability_reasons"] == "low_candidate_mass"


def test_event_reliability_keeps_legacy_mean_candidate_mass_fallback():
    flagged = add_event_reliability_flags(
        _score_row(mean_candidate_log_mass=-1.0),
    )

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_low_candidate_mass"])
    assert flagged.loc[0, "event_reliability_reasons"] == "low_candidate_mass"
