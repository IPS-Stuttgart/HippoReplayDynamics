import numpy as np
import pandas as pd
import pytest

from scripts.report_tirole_selection_transport import anchor_statistics, ratios


def fixture(n_draws=2):
    scores, content = [], []
    for anchor in range(2):
        for draw in range(n_draws):
            for truth in [1, 2]:
                base = {"session": "fixture", "animal": "RAT", "anchor_event": anchor, "epoch": "POST", "ripple_positive": True, "split": 0, "draw": draw, "truth_track": truth}
                content.append({**base, "poisson_true_signed_z": 2.0, "conditional_count_true_signed_z": 1.0})
                for generator in ["ordered", "whole_bin_shuffled"]:
                    for repeat in [-1, 0, 1]:
                        scores.append(
                            {
                                **base,
                                "generator": generator,
                                "repeat": repeat,
                                "fraction": 1.0 if repeat == -1 else 0.5,
                                "inferred_track": truth,
                                "sequence_accepted": generator == "ordered" and (repeat == -1 or truth == 2),
                            }
                        )
    return pd.DataFrame(scores), pd.DataFrame(content)


def test_known_truth_selection_change_not_conflated_with_classification():
    s, c = fixture()
    a = anchor_statistics(s, c)
    r = ratios(a[(a.generator == "ordered") & (a.draw_partition == "all")])
    assert r["full_true_track2_fraction"] == 0.5
    assert r["half_true_track2_fraction"] == 1.0
    assert r["retained_minus_full_true_track2_fraction"] == 0.5
    assert r["lost_true_track2_fraction"] == 0.0
    assert r["loss_given_full_track1"] == 1.0
    assert r["loss_given_full_track2"] == 0.0
    assert r["lost_conditional_count_true_signed_z"] == 1.0


def test_empty_selected_group_remains_missing():
    s, c = fixture()
    a = anchor_statistics(s, c)
    r = ratios(a[(a.generator == "whole_bin_shuffled") & (a.draw_partition == "all")])
    assert np.isnan(r["full_true_track2_fraction"])
    assert np.isnan(r["retained_minus_full_true_track2_fraction"])
    assert r["full_acceptance_track1"] == 0.0


def test_extra_mc_copies_do_not_change_original_anchor_weight():
    result = []
    for draws in [2, 8]:
        s, c = fixture(draws)
        a = anchor_statistics(s, c)
        result.append(ratios(a[(a.generator == "ordered") & (a.draw_partition == "all")]))
    np.testing.assert_allclose(list(result[0].values()), list(result[1].values()), equal_nan=True)


def test_missing_paired_full_score_fails():
    s, c = fixture()
    s = s.drop(s.index[0])
    with pytest.raises(ValueError, match="missing paired full"):
        anchor_statistics(s, c)


def test_missing_evaluation_and_duplicate_keys_fail():
    s, c = fixture()
    with pytest.raises(ValueError, match="missing independent evaluation"):
        anchor_statistics(s, c.iloc[1:])
    with pytest.raises(ValueError, match="duplicate"):
        anchor_statistics(pd.concat([s, s.iloc[:1]]), c)
