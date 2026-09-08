from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multinomial

from scripts import audit_hc11_native_maze_transfer as audit


def synthetic_session():
    times = np.arange(0, 100, 0.02)
    position = 40 * (1 + np.sin(2 * np.pi * times / 12))
    velocity = np.gradient(position, 0.02)
    track = audit.native.TrackSamples(
        times,
        position,
        np.abs(velocity),
        np.sign(velocity).astype(int),
        np.full(len(times), 0.02),
        np.ones(len(times), bool),
        80.0,
        "linear",
        "synthetic linear",
        np.array([[0.0, 100.0]]),
        np.array([[100.0, 200.0]]),
    )
    rng = np.random.default_rng(97)
    spikes = {}
    for uid, center in enumerate(np.linspace(5, 75, 8)):
        rates = 0.1 + 25 * np.exp(-0.5 * ((position - center) / 8) ** 2)
        spikes[uid] = np.repeat(times + 0.001, rng.poisson(rates * 0.02))
    return track, audit.native.SpikeData(tuple(spikes), spikes)


def test_guarded_folds():
    track, _ = synthetic_session()
    assert audit.fold_intervals(track, 0) == ((5.0, 45.0), (55.0, 95.0))
    assert audit.fold_intervals(track, 1) == ((55.0, 95.0), (5.0, 45.0))
    with pytest.raises(ValueError):
        audit.fold_intervals(replace(track, maze_epoch=np.array([[0.0, 5.0]])), 0)


def test_behavior_only_deterministic_windows():
    track, _ = synthetic_session()
    a, qa = audit.select_windows(track, (55, 95), "synthetic", 0)
    b, qb = audit.select_windows(track, (55, 95), "synthetic", 0)
    pd.testing.assert_frame_equal(a, b)
    assert qa == qb
    assert len(a) > 0
    assert np.all(a.end_time_s.to_numpy()[:-1] <= a.start_time_s.to_numpy()[1:] + 1e-10)
    for event in a.itertuples(index=False):
        indices = audit.native.nearest_frame_indices(track.times_s, event.start_time_s + (np.arange(10) + 0.5) * 0.02)
        assert (track.speed_cm_s[indices] >= 5).all()
        assert len(np.unique(track.direction[indices])) == 1


@pytest.mark.parametrize("regime", audit.UNIT_REGIMES)
def test_test_half_spikes_do_not_change_maps_or_primary_unit_selection(regime):
    track, spikes = synthetic_session()
    changed = audit.native.SpikeData(spikes.unit_ids, {uid: np.concatenate([times[times < 50], np.arange(50.0, 100.0, 0.001)]) for uid, times in spikes.times_by_unit.items()})
    a, qa, mask = audit.fit_maps(track, spikes, (5, 45), regime, spikes.unit_ids)
    b, qb, other = audit.fit_maps(track, changed, (5, 45), regime, spikes.unit_ids)
    assert_array_equal(mask, other)
    pd.testing.assert_frame_equal(qa, qb)
    for variant in a:
        for left, right in zip(a[variant], b[variant], strict=True):
            assert left.unit_ids == right.unit_ids
            assert_array_equal(left.rates_hz, right.rates_hz)
            assert_array_equal(left.occupancy_s, right.occupancy_s)
    _, restricted, _ = audit.training_inputs(track, spikes, (5, 45))
    for times in restricted.times_by_unit.values():
        assert ((times >= 5) & (times < 45)).all()


@pytest.mark.parametrize("target", [0, 3, 9, 10, 50])
def test_thinning_preserves_observed_spike_locations(target):
    counts = np.array([[2, 0, 3], [1, 4, 0]])
    a, attained = audit.capped_counts(counts, target, 4)
    b, _ = audit.capped_counts(counts, target, 4)
    assert_array_equal(a, b)
    assert (a <= counts).all()
    assert a.sum() == min(counts.sum(), target)
    assert attained == (target <= counts.sum())


def test_noninteger_cap_rejected():
    with pytest.raises(ValueError):
        audit.capped_counts(np.ones((3, 2), int), 2.5, 3)


def test_heldout_cannot_change_neural_inference():
    rng = np.random.default_rng(23)
    counts = rng.poisson(0.3, (10, 8))
    edges = np.arange(11) * 0.02
    rates = [rng.uniform(0.1, 8, (8, 9)) for _ in range(2)]
    kernels = audit.frozen.transitions(np.arange(9) * 4 + 2, edges, "linear", 36)
    train, held = np.arange(5), np.arange(5, 8)
    a, pa, ha = audit.infer_and_predict(counts, edges, rates, train, held, kernels)
    changed = counts.copy()
    changed[:, held] += 5
    b, pb, hb = audit.infer_and_predict(changed, edges, rates, train, held, kernels)
    assert ha == hb
    for model in pa:
        assert_array_equal(pa[model], pb[model])
    assert a != b
    assert all(value <= 1e-8 for value in b.values())


def test_known_behavior_probability_is_separate_and_normalized():
    rates = np.array([[1.0, 3.0], [4.0, 2.0], [2.0, 1.0]])
    counts = np.array([[1, 2, 1], [0, 1, 1]])
    held = np.array([1, 2])
    result = audit.behavior_scores(
        counts, np.array([0.0, 0.02, 0.04]), [rates], rates, np.ones(2), held, np.array([1.0, 3.0]), np.ones(2), np.array([0.0, 2.0, 4.0]), np.array([1, 0])
    )
    expected = sum(multinomial.logpmf(counts[t, held], counts[t, held].sum(), rates[held, b] / rates[held, b].sum()) for t, b in enumerate([0, 1]))
    assert_allclose(result["score_behavior"], expected, atol=1e-12)


def test_circle_mean_handles_wrap():
    q = np.log(np.array([[0.5, 0.5]]))
    result = audit.position_metrics(q, [None], np.array([1.0, 99.0]), np.array([0.0, 50.0, 100.0]), np.array([0.0]), np.array([[1, 1]]), "circular", 100.0)
    assert result["median_mean_error_cm_all"] < 1e-8
    assert result["median_map_error_cm_all"] == 1


def test_summary_weights_halves_sessions_animals_equally():
    rows = []
    for rat, session, half, n, value in [
        ("a", "a1", 0, 1, 100.0),
        ("a", "a1", 1, 100, 0.0),
        ("b", "b1", 0, 30, 0.0),
        ("b", "b1", 1, 5, 0.0),
        ("b", "b2", 0, 30, 10.0),
        ("b", "b2", 1, 3, 10.0),
    ]:
        rows.extend(
            {
                "rat": rat,
                "session": session,
                "fold": half,
                "window_id": i,
                "unit_regime": "train_only_qc",
                "count_regime": "native",
                "encoding_variant": "pooled",
                "contrast": "imm_minus_iid",
                "delta": value,
                "delta_per_heldout_spike": value,
            }
            for i in range(n)
        )
    summary, _, _, _ = audit.summaries(pd.DataFrame(rows))
    assert summary.equal_animal_mean.iloc[0] == 27.5


def test_empty_completeness_fails():
    assert not audit.technical_gates(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()).passed.any()


def test_fold_scoring_integration_and_nonvacuous_factor_gates(tmp_path, monkeypatch):
    track, spikes = synthetic_session()
    source, dataset, output = tmp_path / "source", tmp_path / "data", tmp_path / "result"
    source.mkdir()
    (dataset / "Syn" / "Syn_session").mkdir(parents=True)
    output.mkdir()
    np.savez(source / "Syn_session_cache.npz", unit_ids=np.asarray(spikes.unit_ids))
    pd.DataFrame({"session": ["Syn_session"] * 20, "phase": ["POST"] * 20, "event_id": range(20), "n_spikes": [20] * 20}).to_csv(source / "frozen_selection.csv", index=False)
    monkeypatch.setattr(audit.native, "load_track_samples", lambda path: track)
    monkeypatch.setattr(audit.native, "load_spikes", lambda path: spikes)
    monkeypatch.setattr(audit, "MAX_WINDOWS", 2)
    result = audit.score_fold(("Syn_session", "Syn", 0, source, dataset, output))
    assert result["rows"] == 80
    scores = pd.read_csv(output / "Syn_session_fold0_scores.csv")
    windows = pd.read_csv(output / "Syn_session_fold0_selection.csv")
    qc = pd.read_csv(output / "Syn_session_fold0_qc.csv")
    all_scores, all_windows, all_qc = [], [], []
    for i in range(8):
        for fold in (0, 1):
            all_scores.append(scores.assign(session=f"s{i}", rat=f"r{i // 2}", fold=fold))
            all_windows.append(windows.assign(session=f"s{i}", rat=f"r{i // 2}", fold=fold))
            all_qc.append(qc.assign(session=f"s{i}", rat=f"r{i // 2}", fold=fold))
    scores, windows, qc = (pd.concat(parts, ignore_index=True) for parts in (all_scores, all_windows, all_qc))
    assert audit.technical_gates(scores, windows, qc).passed.all()
    assert not audit.technical_gates(scores.iloc[1:], windows, qc).passed.all()
    paired, events = audit.event_contrasts(scores)
    assert np.isfinite(paired.delta).all()
    assert len(events) * 5 == len(paired)
