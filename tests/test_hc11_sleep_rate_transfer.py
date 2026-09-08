import numpy as np
import pandas as pd
import pytest

from scripts import audit_hc11_count_conditioned_prediction as frozen
from scripts.audit_hc11_sleep_rate_transfer import REGIMES, calibrate, check_scores, folds, paired_contrasts, predict, summarize


def events():
    return pd.DataFrame({"event_id": np.arange(20), "start_time_s": np.arange(20) * 3.0, "end_time_s": np.arange(20) * 3.0 + 0.2})


def test_fold_identity_and_guard():
    data = events()
    for _, test, cal, ex in folds(data):
        assert len(test) == 10 and len(cal) == 10 and ex.empty
        assert not set(test.event_id) & set(cal.event_id)
    data.loc[10, ["start_time_s", "end_time_s"]] = [27.4, 27.6]
    result = folds(data)
    assert list(result[0][3].event_id) == [10]
    assert list(result[1][3].event_id) == [9]


def test_empty_calibration_rejected():
    data = events()
    data["start_time_s"] = np.arange(20) * 0.01
    data["end_time_s"] = data.start_time_s + 0.1
    with pytest.raises(ValueError, match="empty guarded"):
        folds(data)


def test_gain_is_spatially_constant_and_normalized():
    rates = np.array([[1.0, 2.0, 3.0], [6.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
    p, gain, p0 = calibrate(rates, np.array([100, 10, 0]), 100)
    assert p.sum() == pytest.approx(1)
    assert (p > 0).all()
    assert gain[0] > gain[1]
    adapted = rates * gain[:, None]
    assert np.allclose(adapted.mean(axis=1) / adapted.mean(axis=1).sum(), p)
    assert np.allclose(adapted / adapted.sum(axis=1, keepdims=True), rates / rates.sum(axis=1, keepdims=True))
    pp, gg, _ = calibrate(rates, np.array([100, 10, 0]), None)
    assert np.allclose(pp, p0) and np.all(gg == 1)


@pytest.mark.parametrize("counts,alpha", [(np.zeros(3, int), 100), (np.array([1.0, 2.0, 3.0]), 100), (np.array([-1, 2, 3]), 100), (np.ones(3, int), 0), (np.ones(3, int), np.nan)])
def test_bad_calibration(counts, alpha):
    with pytest.raises(ValueError):
        calibrate(np.ones((3, 4)), counts, alpha)


def test_heldout_changes_do_not_change_posterior():
    rates = np.array([[5.0, 1.0, 0.1], [0.1, 1.0, 5.0], [3.0, 1.0, 0.2], [0.2, 1.0, 3.0]])
    edges = np.arange(5) * 0.02
    kernels = frozen.transitions(np.arange(3) * 4.0, edges, "linear", 12.0)
    counts = np.array([[2, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 2, 0, 1]])
    args = (edges, [rates], np.array([0, 1]), np.array([2, 3]), kernels, np.full(4, 0.25))
    original, hashes = predict(counts, *args)
    changed = counts.copy()
    changed[:, 2:] = changed[::-1, 2:]
    alternative, other = predict(changed, *args)
    assert hashes == other
    assert original != alternative
    assert all(v <= 0 for v in original.values())
    changed[:, 2:] = 0
    zeros, other = predict(changed, *args)
    assert all(v == pytest.approx(0, abs=1e-12) for v in zeros.values())
    assert hashes == other


def test_event_counts_cannot_change_calibration():
    data = events()
    _, test, cal, _ = folds(data)[0]
    counts = {i: np.ones((2, 4), int) for i in data.event_id}
    before = sum(counts[i].sum(axis=0) for i in cal.event_id)
    counts[int(test.event_id.iloc[0])] *= 1000
    after = sum(counts[i].sum(axis=0) for i in cal.event_id)
    assert np.array_equal(before, after)


def test_no_vacuous_pass():
    assert not check_scores(pd.DataFrame(), events())


def score_fixture():
    rows = []
    for regime in REGIMES:
        for variant in frozen.VARIANTS:
            for spatial_map in frozen.MAPS:
                for split in range(5):
                    rows.append(
                        {
                            "session": "s",
                            "rat": "r",
                            "phase": "POST",
                            "event_id": 1,
                            "split": split,
                            "regime": regime,
                            "encoding_variant": variant,
                            "map": spatial_map,
                            **{f"score_{m}": -2.0 for m in (*frozen.MODELS, "global")},
                            "n_spikes": 3,
                            "n_train_spikes": 2,
                            "n_heldout_spikes": 1,
                            "status": "success",
                            "posterior_unchanged": True,
                            "heldout_used_for_inference": False,
                            "train_cell_ids": "1,2",
                            "heldout_cell_ids": "3",
                        }
                    )
    return pd.DataFrame(rows)


def test_complete_and_missing_factors():
    table = score_fixture()
    selected = table[["session", "phase", "event_id"]].drop_duplicates()
    assert check_scores(table, selected)
    assert not check_scores(table.iloc[1:], selected)
    assert not check_scores(pd.concat([table, table.iloc[:1]]), selected)
    table.loc[0, "heldout_cell_ids"] = "2"
    assert not check_scores(table, selected)


def test_paired_contrasts():
    table = score_fixture()
    table.loc[table.regime.eq("sleep_alpha100"), "score_first_order_imm"] = -1.0
    paired, event = paired_contrasts(table)
    improved = event[event.regime.eq("sleep_alpha100") & event.contrast.eq("adaptation_temporal_interaction")]
    assert len(improved) == 2
    assert improved.delta.eq(1).all()
    assert paired.groupby(["session", "phase", "event_id", "regime", "encoding_variant", "contrast"]).size().eq(5).all()


def test_equal_animal_weight_not_event_pooling():
    rows = []
    for animal, n, value in (("A", 20, 2), ("B", 3, -2), ("C", 8, 0), ("D", 1, 0)):
        rows.extend(
            {
                "phase": "POST",
                "regime": "run_original",
                "encoding_variant": "pooled",
                "contrast": "imm_minus_iid",
                "rat": animal,
                "session": animal,
                "event_id": i,
                "delta": value,
                "delta_per_heldout_spike": value / 4,
            }
            for i in range(n)
        )
    summary, animals, _ = summarize(pd.DataFrame(rows))
    assert summary.iloc[0]["mean"] == pytest.approx(0)
    assert len(animals) == 4
