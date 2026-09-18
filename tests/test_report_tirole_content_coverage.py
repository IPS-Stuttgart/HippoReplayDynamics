import numpy as np
import pandas as pd
import pytest

from scripts.report_tirole_content_coverage import event_summaries, selected_content_bias, summarize_content


def fixture():
    rows = []
    for repeat in range(5):
        for event, q in enumerate([0.1, 0.2, 0.8, 0.9]):
            for fraction in (1.0, 0.5):
                rows.append(
                    {
                        "animal": "a",
                        "session": "s",
                        "cohort_stratum": "strict_RUN_pass",
                        "candidate_stratum": "ripple",
                        "epoch": "POST",
                        "event_id": event,
                        "split": 0,
                        "repeat": repeat,
                        "fraction": fraction,
                        "arm": "real",
                        "sequence_accepted": fraction == 1 or event >= 2,
                        "sequence_eligible": True,
                        "inferred_track": 1 if q < 0.5 else 2,
                        "evaluation_z_log_odds": 2 if q < 0.5 else -2,
                        "conditional_evaluation_z_log_odds": 1 if q < 0.5 else -1,
                        "evaluation_n_eval_spikes": 10,
                        "n_evaluation_active_units": 4,
                        "duration_s": 0.2,
                        "ripple_peak_z": 5,
                        "evaluation_track2_probability": q,
                    }
                )
    return pd.DataFrame(rows)


def test_lost_content_counts_events_not_repeated_splits():
    events = event_summaries(fixture())
    lost = events[(events.fraction == 0.5) & (events.group == "lost")]
    assert len(lost) == 2 and (lost.n_contributing_split_repeats == 5).all()
    summary = summarize_content(events)
    row = summary[(summary.fraction == 0.5) & (summary.group == "lost")].iloc[0]
    assert row.n_events == 2 and row.mean_signed_evaluation_z == 2


def test_fixed_readout_exposes_selection_bias():
    bias = selected_content_bias(fixture())
    np.testing.assert_allclose(bias.track2_selection_shift, 0.35)
    assert (bias.n_full_accepted_content == 4).all() and (bias.n_thinned_accepted_content == 2).all()
    assert bias.status.eq("estimable").all()


def test_readout_cannot_change_when_inference_cells_change():
    data = fixture()
    data.loc[data.fraction == 0.5, "evaluation_track2_probability"] += 0.01
    with pytest.raises(ValueError, match="readout changed"):
        selected_content_bias(data)


def test_no_accepted_events_does_not_mean_no_bias():
    data = fixture()
    data["sequence_accepted"] = False
    bias = selected_content_bias(data)
    assert bias.track2_selection_shift.isna().all()
    assert bias.deletion_null_p_two_sided.isna().all()
