import numpy as np
import pandas as pd

from scripts.report_tirole_measurement_calibration import temporal_summary, track_summary


def test_unclassified_track_is_not_counted_as_wrong_track():
    x = pd.DataFrame(
        [
            {
                "session": "s",
                "animal": "r",
                "cohort_stratum": "primary",
                "epoch": "POST",
                "anchor_ripple_positive": True,
                "generator": "ordered",
                "fraction": 1.0,
                "truth_track": 1,
                "anchor_event": i,
                "sequence_eligible": (i == 0),
                "sequence_accepted": (i == 0),
                "correct_inferred_track": (i == 0),
                "evaluation_true_signed_log_odds": 1.0,
                "conditional_true_signed_log_odds": 1.0,
                "conditional_true_signed_z": 1.0,
                "n_inference_spikes": 5,
            }
            for i in range(10)
        ]
    )
    r = track_summary(x).iloc[0]
    assert r.acceptance_fraction == 0.1
    assert r.correct_track_given_opportunity == 1.0
    x.sequence_eligible = False
    x.sequence_accepted = False
    assert np.isnan(track_summary(x).iloc[0].correct_track_given_acceptance)


def test_zero_target_spikes_are_missing_not_zero_gain():
    x = pd.DataFrame(
        [
            {
                "session": "s",
                "animal": "r",
                "cohort_stratum": "primary",
                "epoch": "POST",
                "anchor_ripple_positive": True,
                "generator": g,
                "horizon_ms": 40,
                "anchor_event": i,
                "n_heldout_target_spikes": int(i != 0),
                "score_dynamic": 2.0,
                "score_matched_own": 1.0,
            }
            for i in range(2)
            for g in ["dynamic", "matched_null"]
        ]
    )
    e, s, p = temporal_summary(x)
    assert e[e.anchor_event == 0].event_median_gain.isna().all()
    assert (s.n_anchors_with_target_spikes == 1).all()
    assert p.n_paired_anchors.iloc[0] == 1


def test_repeated_splits_do_not_reweight_anchors():
    records = [
        {
            "session": "s",
            "animal": "r",
            "cohort_stratum": "primary",
            "epoch": "POST",
            "anchor_ripple_positive": True,
            "generator": g,
            "horizon_ms": 40,
            "anchor_event": i,
            "n_heldout_target_spikes": 1,
            "score_dynamic": float(i),
            "score_matched_own": 0.0,
        }
        for i in range(2)
        for g in ["dynamic", "matched_null"]
    ]
    x = pd.DataFrame(records)
    _, a, _ = temporal_summary(x)
    _, b, _ = temporal_summary(pd.concat([x, x[x.anchor_event == 1]]))
    np.testing.assert_equal(a.mean_event_median_gain.to_numpy(), b.mean_event_median_gain.to_numpy())
