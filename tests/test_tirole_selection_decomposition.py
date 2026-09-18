import numpy as np
import pandas as pd
import pytest

from scripts.report_tirole_selection_decomposition import analyze, decompose


def test_gained_content_is_not_attributed_to_selective_loss():
    q = [0.1, 0.3, 0.9]
    out = decompose(q, [1, 1, 0], [0, 1, 1], 3)
    assert out["loss_shift"] == pytest.approx(0.1)
    assert out["gain_shift"] == pytest.approx(0.3)
    assert out["total_shift"] == pytest.approx(0.4)
    assert out["n_gained"] == 1
    assert out["loss_deletion_null_p_two_sided"] == 1


def test_no_retained_content_does_not_pass_a_deletion_test():
    out = decompose([0.1, 0.9], [1, 0], [0, 1], 3)
    assert out["total_shift"] == pytest.approx(0.8)
    assert np.isnan(out["loss_shift"])
    assert np.isnan(out["loss_deletion_null_p_two_sided"])


def test_pure_deletion_closes_and_no_change_has_null_p_one():
    q = [0.1, 0.3, 0.8]
    out = decompose(q, [1, 1, 1], [1, 0, 0], 3)
    assert out["gain_shift"] == 0
    assert out["loss_shift"] == out["total_shift"]
    unchanged = decompose(q, [1, 1, 1], [1, 1, 1], 3)
    assert unchanged["loss_deletion_null_p_two_sided"] == 1


def test_empty_evaluation_is_explicitly_missing():
    out = decompose([np.nan, np.nan], [1, 1], [1, 0], 3)
    assert out["n_full"] == 2
    assert out["n_full_with_content"] == 0
    assert np.isnan(out["total_shift"])


def fixture():
    records = []
    for fraction in [1, 0.5]:
        for event in range(3):
            records.append(
                {
                    "animal": "A",
                    "session": "S",
                    "cohort_stratum": "test",
                    "candidate_stratum": "ripple",
                    "epoch": "POST",
                    "split": 0,
                    "repeat": 0,
                    "arm": "real",
                    "fraction": fraction,
                    "event_id": event,
                    "sequence_accepted": event < 2 if fraction == 1 else event > 0,
                    "evaluation_track2_probability": [0.1, 0.3, 0.9][event],
                    "conditional_evaluation_track2_probability": [0.2, 0.4, 0.8][event],
                }
            )
    return pd.DataFrame(records)


def test_table_is_paired_and_readout_does_not_change():
    rows, summary = analyze(fixture())
    assert len(rows) == len(summary) == 2
    x = fixture()
    x.loc[(x.fraction == 0.5) & (x.event_id == 1), "evaluation_track2_probability"] = 0.7
    with pytest.raises(ValueError, match="readout changed"):
        analyze(x)
    with pytest.raises(ValueError, match="identities differ"):
        analyze(fixture().iloc[:-1])
