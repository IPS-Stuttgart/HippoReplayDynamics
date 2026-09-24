import numpy as np
import pandas as pd
import pytest

from scripts import calibrate_kleinman_joint_temporal_specificity as joint


def synthetic_bank(animal="A"):
    pos = np.arange(8)
    rates = np.array([2 + 12 * np.exp(-0.5 * ((pos - i) / 1.3) ** 2) for i in range(4)])
    return {
        "identity": animal + "/s/p",
        "animal": animal,
        "session": "s",
        "drug": 0,
        "novel": 0,
        "direction": 0,
        "generating_rates": rates,
        "unit_keys": np.column_stack([np.ones(4, int), np.arange(1, 5)]),
        "reference_exposure": np.full(8, 3.0),
        "readout_exposure": np.full((3, 8), 0.7),
        "predictor": np.array([-1.5, -0.5, 0.5, 1.5]),
    }


def test_selection_is_balanced_and_ignores_outcomes():
    rows = [
        {"animal": f"A{a}", "session": f"s{i}", "packet_id": f"p{i}", "drug": d, "novel": n, "selected": True, "availability_descriptor": True}
        for a in range(6)
        for d in (0, 1)
        for n in (0, 1)
        for i in range(3)
    ]
    # Packet IDs are session-unique, including the condition.
    for r in rows:
        r["session"] += f"_{r['drug']}_{r['novel']}"
    f = pd.DataFrame(rows)
    selected = joint.select_packets(f)
    f["native_change"] = np.arange(len(f))[::-1]
    other = joint.select_packets(f.sample(frac=1, random_state=123))
    assert len(selected) == 48
    pd.testing.assert_frame_equal(selected, other.drop(columns="native_change"))
    assert selected.groupby(["animal", "drug", "novel"]).size().eq(2).all()
    f.loc[(f.animal == "A0") & (f.drug == 0) & (f.novel == 0), "availability_descriptor"] = False
    with pytest.raises(ValueError, match="two_packets"):
        joint.select_packets(f)


def test_no_shape_change_from_scalar_gain_and_total_normalization():
    b = synthetic_bank()
    exposure = b["readout_exposure"].copy()
    exposure[2] *= np.linspace(0.5, 2, 8)
    r = b["generating_rates"][0]
    gains = np.array([0.7, 1.1, 2.0])
    scalar = joint.expected_rates(r, exposure, 1.0, "scalar_gain", np.zeros(8), gains)
    assert np.allclose(np.ptp(scalar / r, axis=1), 0)
    shape = joint.expected_rates(r, exposure, 1.0, "future_only", np.zeros(8), gains)
    np.testing.assert_allclose((shape * exposure).sum(axis=1), (r * exposure).sum(axis=1) * gains)


def test_continuous_drift_has_equal_log_shape_steps():
    b = synthetic_bank()
    rates = joint.expected_rates(b["generating_rates"][0], b["readout_exposure"], 1.2, "continuous_positive_drift", np.zeros(8), np.ones(3))
    steps = np.diff(np.log(rates), axis=0)
    steps -= steps.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(steps[0], steps[1], atol=1e-12)


def test_past_only_and_future_only_have_correct_change_locations():
    b = synthetic_bank()
    for case, unchanging in (("future_only", (0, 1)), ("past_only", (1, 2))):
        rates = joint.expected_rates(b["generating_rates"][0], b["readout_exposure"], 1.2, case, np.zeros(8), np.ones(3))
        np.testing.assert_allclose(rates[unchanging[0]], rates[unchanging[1]])
        assert not np.allclose(rates[0], rates[2])


def test_shared_baseline_is_not_resampled(monkeypatch):
    b = synthetic_bank()
    counts = np.arange(4 * 3 * 8).reshape(4, 3, 8)
    calls = []
    original = joint.spatial_score

    def capture(before, after, *rest):
        calls.append((before.copy(), after.copy()))
        return original(before, after, *rest)

    monkeypatch.setattr(joint, "spatial_score", capture)
    joint.paired_scores(counts, b["readout_exposure"], b["generating_rates"], b["predictor"])
    for i in range(4):
        np.testing.assert_array_equal(calls[i][1], calls[i + 4][0])


def test_both_lags_use_same_support_and_exclusions_are_recorded():
    b = synthetic_bank()
    b["readout_exposure"][0, 0] = 0
    b["readout_exposure"][2, 7] = 0
    counts = np.ones((4, 3, 8), int)
    counts[:, 0, 0] = 0
    counts[:, 2, 7] = 0
    r = joint.paired_scores(counts, b["readout_exposure"], b["generating_rates"], b["predictor"])
    assert r["common_bins"] == 6
    assert r["preceding_excluded_spikes"] == 12
    assert r["prospective_excluded_spikes"] == 12


def test_equal_animal_weight_and_missing_animal_failure():
    rows = [{"animal": f"A{i}", "preceding_score": 0.0, "preceding_information": 10.0, "prospective_score": float(i + 1), "prospective_information": 10.0} for i in range(6)]
    _, base = joint.infer(rows)
    _, repeated = joint.infer(rows + [rows[0]] * 100)
    assert base["mean_contrast"] == pytest.approx(0.35)
    assert base["mean_contrast"] == repeated["mean_contrast"]
    _, missing = joint.infer(rows[:-1])
    assert missing["status"] == "missing_information" and not missing["flag"]


def test_deterministic_joint_replica_and_complete_case_ledger():
    banks = [synthetic_bank(f"A{i}") for i in range(6)]
    joint.init_worker(banks)
    a, b, c = joint.simulate_replica(0)
    x, _y, z = joint.simulate_replica(0)
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(x))
    pd.testing.assert_frame_equal(pd.DataFrame(c), pd.DataFrame(z))
    assert len(a) == 6 * 8 and len(b) == 6 * 8 and len(c) == 8
    assert all(e["n_animals"] == 6 for e in c)


def test_failed_null_or_missing_animals_cannot_pass():
    rows = []
    for i in range(128):
        rows.append(
            {
                "case": "unchanged",
                "replicate": i,
                "n_animals": 6,
                "status": "scored",
                "positive_flag": i < 20,
                "negative_flag": False,
                "flag": i < 20,
                "mean_contrast": 0.0,
                "mean_preceding": 0.0,
                "mean_prospective": 0.0,
            }
        )
    summary = joint.summarize(pd.DataFrame(rows))
    assert summary.iloc[0].n_flags == 20 and not summary.iloc[0].engineering_screen_passed
    for r in rows:
        r.update(n_animals=5, status="missing_information", positive_flag=False, flag=False)
    assert not joint.summarize(pd.DataFrame(rows)).iloc[0].engineering_screen_passed


def test_independent_conditional_moments_match_joint_pair(monkeypatch):
    from scripts.verify_kleinman_conditional_coupling import independent_anchor, independent_score

    b = synthetic_bank()
    counts = np.random.default_rng(12).poisson(b["readout_exposure"][None, :, :] * b["generating_rates"][:, None, :])
    original = joint.paired_scores(counts, b["readout_exposure"], b["generating_rates"], b["predictor"])
    monkeypatch.setattr(joint, "spatial_score", independent_score)
    monkeypatch.setattr(joint, "anchor_score", independent_anchor)
    other = joint.paired_scores(counts, b["readout_exposure"], b["generating_rates"], b["predictor"])
    for lag in ("preceding", "prospective"):
        for what in ("score", "information"):
            assert original[lag + "_" + what] == pytest.approx(other[lag + "_" + what], abs=2e-4)
