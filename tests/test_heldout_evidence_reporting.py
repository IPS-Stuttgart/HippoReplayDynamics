import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import _add_relative_metrics
from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_LOWER_BOUND,
    EVIDENCE_COMPARISON_RESTRICTED_DIFFERENCE,
    EVIDENCE_COMPARISON_UNKNOWN,
    EXACT_EVIDENCE_SUPPORT,
    RESTRICTED_HELDOUT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)


def test_heldout_rows_with_nonfinite_likelihood_are_noncomparable():
    rows = pd.DataFrame(
        {
            "model": ["random", "stationary", "imm"],
            "heldout_log_likelihood": [np.inf, -5.0, -4.0],
            "evidence_support": [EXACT_EVIDENCE_SUPPORT] * 3,
        }
    )

    scored = ensure_evidence_support_columns(rows)

    assert not bool(scored.loc[0, "evidence_comparable"])
    assert bool(scored.loc[1, "evidence_comparable"])
    assert bool(scored.loc[2, "evidence_comparable"])


def test_relative_metrics_ignore_nonfinite_static_heldout_baseline():
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s1"],
            "event_index": [0, 0, 0],
            "model": ["random", "stationary", "imm"],
            "heldout_log_likelihood": [np.inf, -5.0, -4.0],
            "test_spikes": [2, 2, 2],
            "evidence_support": [EXACT_EVIDENCE_SUPPORT] * 3,
        }
    )

    result = _add_relative_metrics(rows).set_index("model")

    assert not bool(result.loc["random", "evidence_comparable"])
    assert result.loc["stationary", "best_static_heldout_log_likelihood"] == -5.0
    assert result.loc["imm", "best_static_heldout_log_likelihood"] == -5.0
    assert result.loc["imm", "delta_vs_best_static"] == 1.0
    assert np.isnan(result.loc["random", "delta_vs_best_static"])


def test_missing_support_metadata_fails_closed_for_unknown_model():
    rows = pd.DataFrame(
        {
            "model": ["new-unclassified-model"],
            "heldout_log_likelihood": [-3.0],
        }
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == EVIDENCE_COMPARISON_UNKNOWN
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_UNKNOWN
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_known_exact_legacy_baselines_remain_explicitly_recognized():
    rows = pd.DataFrame(
        {
            "model": ["random", "stationary"],
            "heldout_log_likelihood": [-3.0, -2.0],
        }
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored["evidence_support"].tolist() == [
        EXACT_EVIDENCE_SUPPORT,
        EXACT_EVIDENCE_SUPPORT,
    ]
    assert scored["evidence_comparable"].tolist() == [True, True]


def test_truncated_heldout_score_is_not_reported_as_lower_bound():
    rows = pd.DataFrame(
        {
            "model": ["imm"],
            "heldout_log_likelihood": [-4.0],
            "diagnostic_candidate_evidence_support": [TRUNCATED_EVIDENCE_SUPPORT],
        }
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == RESTRICTED_HELDOUT_EVIDENCE_SUPPORT
    assert (
        scored.loc[0, "evidence_comparison"]
        == EVIDENCE_COMPARISON_RESTRICTED_DIFFERENCE
    )
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_legacy_truncated_heldout_support_is_reclassified():
    rows = pd.DataFrame(
        {
            "model": ["imm"],
            "heldout_log_likelihood": [-4.0],
            "evidence_support": [TRUNCATED_EVIDENCE_SUPPORT],
        }
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == RESTRICTED_HELDOUT_EVIDENCE_SUPPORT
    assert (
        scored.loc[0, "evidence_comparison"]
        == EVIDENCE_COMPARISON_RESTRICTED_DIFFERENCE
    )


def test_raw_truncated_log_evidence_remains_a_lower_bound():
    rows = pd.DataFrame(
        {
            "model": ["imm"],
            "log_evidence": [-4.0],
            "evidence_support": [TRUNCATED_EVIDENCE_SUPPORT],
        }
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND
    assert not bool(scored.loc[0, "evidence_comparable"])
