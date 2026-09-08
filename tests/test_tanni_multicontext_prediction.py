import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.multicontext_prediction import context_priors, infer_contexts, predict_contexts, score_frozen_contexts
from scripts.audit_tanni_multicontext_prediction import balanced_means, bootstrap_mua, build_bank, count_running_windows, neural_splits, running_windows


def bank_fixture():
    # Training and held-out pairs share context and position selectivity.
    maps = [np.array([[9.0, 2], [1, 3], [8, 1], [1, 4]]), np.array([[1.0, 2], [9, 3], [1, 4], [8, 1]])]
    comp = [m.mean(axis=1) for m in maps]
    return maps, comp


def test_context_priors_not_proportional_to_templates():
    assert np.allclose(context_priors(list("ABCDA")), [0.125, 0.25, 0.25, 0.25, 0.125])
    with pytest.raises(ValueError):
        context_priors([])


def test_predictive_target_is_normalized_multinomial():
    maps, comp = bank_fixture()
    base = np.array([[3, 1, 0, 0]])
    inferred = infer_contexts(base, maps, comp, [0, 1], ["A", "B"])
    scores = []
    for first in range(3):
        x = base.copy()
        x[0, 2:] = [first, 2 - first]
        scores.append(score_frozen_contexts(x, maps, comp, [2, 3], inferred, "A", np.ones(4)))
    for model in ("mix_iid", "current_iid", "mix_global", "current_global", "blind_iid", "blind_global", "event_global"):
        assert sum(np.exp(r[model]) for r in scores) == pytest.approx(1)


def test_heldout_counts_do_not_change_context_or_path():
    maps, comp = bank_fixture()
    a = predict_contexts(np.array([[3, 0, 0, 5], [2, 1, 0, 3]]), maps, comp, [0, 1], [2, 3], ["A", "B"], "A")
    b = predict_contexts(np.array([[3, 0, 5, 0], [2, 1, 3, 0]]), maps, comp, [0, 1], [2, 3], ["A", "B"], "A")
    assert a["training_hash"] == b["training_hash"]
    assert a["iid_p_A"] == b["iid_p_A"]
    assert a["global_p_A"] == b["global_p_A"]
    assert a["mix_iid"] != b["mix_iid"]
    assert not a["heldout_used_for_inference"]


def test_remote_context_can_improve_correctly_matched_prediction():
    maps, comp = bank_fixture()
    x = np.array([[0, 9, 0, 9]])
    r = predict_contexts(x, maps, comp, [0, 1], [2, 3], ["A", "B"], "A")
    assert r["iid_p_B"] > 0.8
    assert r["mix_iid"] > r["current_iid"]


def test_neural_partition_rejects_overlap():
    maps, comp = bank_fixture()
    with pytest.raises(ValueError):
        predict_contexts(np.ones((2, 4), int), maps, comp, [0, 1], [1, 2, 3], ["A", "B"], "A")


def test_zero_heldout_spikes_have_zero_logscore():
    maps, comp = bank_fixture()
    r = predict_contexts(np.array([[3, 0, 0, 0]]), maps, comp, [0, 1], [2, 3], ["A", "B"], "A")
    assert abs(r["mix_iid"]) < 1e-12
    assert abs(r["mix_global"]) < 1e-12


def test_bank_selection_ignores_test_half_and_full_QC():
    sources = []
    for _ in range(5):
        sources.append(
            {
                "cell_ids": np.arange(12),
                "rates_first_half_hz": np.tile([0.5, 3.0], (12, 1)),
                "occupancy_first_half_s": np.array([100.0, 100.0]),
                "rates_second_half_hz": np.full((12, 2), 99),
                "unit_qc_mask": np.zeros(12, bool),
            }
        )
    units, maps, _, _, _ = build_bank(sources)
    for s in sources:
        s["rates_second_half_hz"] *= 100
        s["unit_qc_mask"][:] = True
    units2, maps2, _, _, _ = build_bank(sources)
    assert np.array_equal(units, units2)
    assert all(np.array_equal(a, b) for a, b in zip(maps, maps2, strict=True))


def test_RUN_windows_exclude_training_and_tracking_gaps():
    t = np.r_[np.arange(0, 2, 0.02), np.arange(3, 6, 0.02)]
    a = {"position": np.column_stack([t, 20 * t, np.zeros(len(t))]), "supported_run_intervals": np.array([[0, 2], [3, 6]])}
    w = running_windows(a, 1, maximum=100)
    assert len(w)
    assert (w[:, 0] >= 3).all()
    assert np.allclose(w[:, 1] - w[:, 0], 0.2)
    assert (w[1:, 0] >= w[:-1, 1] - 1e-10).all()


def test_RUN_spikes_half_open_interval():
    spikes = np.array([[0.0, 1], [0.02, 1], [0.2, 1], [0.4, 1]])
    counts = count_running_windows(spikes, np.array([1]), np.array([[0.0, 0.2], [0.2, 0.4]]))
    assert counts[0].sum() == 2
    assert counts[1].sum() == 1


def test_partitions_deterministic_and_disjoint():
    a, b = neural_splits("R1", 20), neural_splits("R1", 20)
    for (tr, he), (tr2, he2) in zip(a, b, strict=True):
        assert sorted([*tr, *he]) == list(range(20))
        assert np.array_equal(tr, tr2) and np.array_equal(he, he2)


def balanced_fixture():
    rows = []
    for animal in ("r1", "r2", "r3", "r4", "r5"):
        for j, context in enumerate("ABCDA"):
            for eid in range(2 + j):
                rows.append({"animal": animal, "context": context, "session": str(j), "phase": "MUA", "event_id": eid, "effect": 1.0 if context == "A" else 0.0})
    return pd.DataFrame(rows)


def test_physical_context_balancing_not_event_or_template_weighting():
    _, contexts, animals = balanced_means(balanced_fixture(), ["effect"])
    assert len(contexts) == 20
    assert np.allclose(animals.effect, 0.25)
    result = bootstrap_mua(balanced_fixture(), ["effect"], draws=30)
    assert result.iloc[0]["mean"] == pytest.approx(0.25)
    assert result.iloc[0].ci_low == pytest.approx(0.25)
    assert result.iloc[0].ci_high == pytest.approx(0.25)
