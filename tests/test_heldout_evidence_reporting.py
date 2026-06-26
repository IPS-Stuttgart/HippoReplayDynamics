import numpy as np
import pandas as pd

from hipporeplayimm.benchmarks import _add_relative_metrics
from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns


def test_heldout_rows_with_nonfinite_likelihood_are_noncomparable():
    rows = pd.DataFrame(
        {
            "model": ["random", "stationary", "imm"],
            "heldout_log_likelihood": [np.inf, -5.0, -4.0],
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
        }
    )

    result = _add_relative_metrics(rows).set_index("model")

    assert not bool(result.loc["random", "evidence_comparable"])
    assert result.loc["stationary", "best_static_heldout_log_likelihood"] == -5.0
    assert result.loc["imm", "best_static_heldout_log_likelihood"] == -5.0
    assert result.loc["imm", "delta_vs_best_static"] == 1.0
    assert np.isnan(result.loc["random", "delta_vs_best_static"])
