from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat
from scipy.special import logsumexp

from hipporeplayimm.tirole_two_track import (
    TrackSession,
    blocked_training_mask,
    decode_counts,
    fit_maps,
    load_session,
    nearest_samples,
    run_crossvalidation,
)
from scripts.audit_tirole_two_track_inputs import duplicate_sessions, session_gate


def synthetic_session():
    rng = np.random.default_rng(47)
    t = np.arange(0, 400, 0.04)
    x = (t * 20) % 200
    positions = np.stack([np.where((t >= 20) & (t < 180), x, np.nan), np.where((t >= 220) & (t < 380), x, np.nan)])
    peaks = np.linspace(5, 195, 30)
    track_peaks = [peaks, np.random.default_rng(91).permutation(peaks)]
    all_t, all_u = [], []
    for k in range(2):
        valid = np.flatnonzero(np.isfinite(positions[k]))
        lam = 0.1 + 12 * np.exp(-(((positions[k, valid, None] - track_peaks[k][None, :]) / 12) ** 2) / 2)
        counts = rng.poisson(0.04 * lam)
        ii, uu = np.nonzero(counts)
        all_t.extend(np.repeat(t[valid[ii]], counts[ii, uu]))
        all_u.extend(np.repeat(uu, counts[ii, uu]))
    st, su = np.array(all_t), np.array(all_u)
    order = np.argsort(st)
    st, su = st[order], su[order]
    return TrackSession(
        "synthetic",
        t,
        np.full(len(t), 20.0),
        ~np.isfinite(positions).any(axis=0),
        positions,
        np.array([200.0, 200.0]),
        np.arange(1, 31),
        np.arange(1001, 1031),
        st,
        su,
        nearest_samples(t, st),
    )


def write_session(tmp_path, session, mutate=None):
    clusters = {"spike_id": session.unit_ids[session.spike_units], "spike_times": session.spike_times, "id_conversion": np.c_[session.unit_ids, session.original_ids]}
    position = {
        "t": session.times,
        "v_cm": session.speed,
        "sleepbox": np.where(session.sleepbox, session.times, np.nan),
        "linear": np.array([{"linear": x, "length": length / 100} for x, length in zip(session.positions, session.lengths_cm)], dtype=object),
    }
    if mutate:
        mutate(clusters, position)
    savemat(tmp_path / "synthetic_extracted_clusters.mat", {"clusters": clusters})
    savemat(tmp_path / "synthetic_extracted_position.mat", {"position": position})


def test_reader_cm_units_and_ids(tmp_path):
    data = synthetic_session()
    data = replace(data, unit_ids=data.unit_ids * 7)
    write_session(tmp_path, data)
    actual = load_session(tmp_path, "synthetic")
    np.testing.assert_equal(actual.positions, data.positions)
    np.testing.assert_equal(actual.unit_ids, data.unit_ids)
    np.testing.assert_equal(actual.spike_units, data.spike_units)
    np.testing.assert_equal(actual.lengths_cm, [200, 200])


@pytest.mark.parametrize("corruption", ["times", "identity", "shape", "position"])
def test_reader_rejects_bad_inputs(tmp_path, corruption):
    def corrupt(c, p):
        if corruption == "times":
            p["t"][3] = p["t"][2]
        elif corruption == "identity":
            c["spike_id"][0] = 900
        elif corruption == "shape":
            p["v_cm"] = p["v_cm"][:-1]
        else:
            p["linear"][0]["linear"][600] = 2000

    write_session(tmp_path, synthetic_session(), corrupt)
    with pytest.raises(ValueError):
        load_session(tmp_path, "synthetic")


def test_nearest_no_out_of_range_clipping():
    np.testing.assert_equal(nearest_samples(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.4, 1.5, 2.8, 4.0])), [-1, 0, 0, 1, 2, -1])


def test_training_maps_and_unit_selection_ignore_heldout_and_rest_spikes():
    s = synthetic_session()
    training = blocked_training_mask(s.times, 0, 0)
    before = fit_maps(s, training)
    outside = s.times[~training | ~s.epoch_mask()][::5]
    times = np.r_[s.spike_times, np.repeat(outside, 20)]
    modified = replace(s, spike_times=times, spike_units=np.r_[s.spike_units, np.zeros(len(outside) * 20, int)], spike_samples=nearest_samples(s.times, times))
    after = fit_maps(modified, training)
    for k in before:
        np.testing.assert_equal(before[k], after[k], err_msg=k)


def test_fold_mask_guards_neighbours():
    times = np.array([8.5, 9.5, 10, 11, 19, 20, 20.5, 21.5, 59.5, 60])
    np.testing.assert_equal(blocked_training_mask(times, 0, 1), [True, False, False, False, False, False, False, True, False, False])


def test_source_aligned_maps_and_occupancy():
    s = synthetic_session()
    maps = fit_maps(s)
    assert maps["rates"].shape == (2, 30, 20)
    assert len(maps["common_units"]) >= 25
    np.testing.assert_allclose(maps["occupancy_s"].sum(axis=1), [160, 160])
    raw = maps["raw_rates"]
    np.testing.assert_allclose(maps["rates"][:, :, 1:-1], 0.25 * raw[:, :, :-2] + 0.5 * raw[:, :, 1:-1] + 0.25 * raw[:, :, 2:])


def test_poisson_decoder_against_direct_enumeration():
    rates = np.array([[[1.0, 2.0, 8.0], [7.0, 3.0, 1.0]], [[3.0, 6.0, 1.0], [1.0, 2.0, 9.0]]])
    counts = np.array([[2, 0], [0, 1]])
    valid = np.array([[True, True, True], [True, False, True]])
    actual = decode_counts(counts, rates, 0.02, valid)
    expect = np.zeros_like(actual)
    for t in range(2):
        weights = np.full((2, 3), -np.inf)
        for k in range(2):
            for x in range(3):
                if valid[k, x]:
                    weights[k, x] = sum(counts[t] * np.log(rates[k, :, x])) - 0.02 * sum(rates[k, :, x]) - np.log(valid[k].sum())
        expect[t] = np.exp(weights - logsumexp(weights))
    np.testing.assert_allclose(actual, expect)
    np.testing.assert_allclose(decode_counts(counts[::-1], rates, 0.02, valid), actual[::-1])


def test_identical_context_maps_cannot_discriminate_tracks():
    rates = np.random.default_rng(19).uniform(0.1, 10, (1, 20, 20))
    rates = np.repeat(rates, 2, axis=0)
    posterior = decode_counts(np.ones((3, 20), int), rates, 0.25, np.ones((2, 20), bool))
    np.testing.assert_allclose(posterior.sum(axis=2), 0.5)


@pytest.mark.parametrize("kind", ["fractional", "negative", "no_units", "no_bins", "bad_rates"])
def test_decoder_fail_closed(kind):
    counts = np.array([[1.0, 2.0]])
    rates = np.ones((2, 2, 20))
    valid = np.ones((2, 20), bool)
    if kind == "fractional":
        counts[0, 0] = 0.5
    if kind == "negative":
        counts[0, 0] = -1
    if kind == "no_units":
        counts = counts[:, :0]
        rates = rates[:, :0]
    if kind == "no_bins":
        valid[0] = False
    if kind == "bad_rates":
        rates[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        decode_counts(counts, rates, 0.02, valid)


def test_crossvalidation_synthetic_recovery():
    rows = run_crossvalidation(synthetic_session())
    assert len(rows) == 10
    assert all(r["status"] == "complete" for r in rows)
    assert min(r["context_accuracy"] for r in rows) > 0.8
    assert max(r["median_position_error_cm"] for r in rows) < 20


def test_duplicate_identity_both_quarantined():
    bad, groups = duplicate_sessions(
        [
            {"session": "a", "kind": "clusters", "sha256": "same"},
            {"session": "b", "kind": "clusters", "sha256": "same"},
            {"session": "c", "kind": "clusters", "sha256": "different"},
        ]
    )
    assert bad == {"a", "b"}
    assert groups == [["a", "b"]]


def test_gate_does_not_pass_empty_cv_or_unknown_identity():
    row = {"duplicate_identity": True, "n_tracks": 4, "n_common_units": 30, "minimum_track_occupancy_fraction": 1, "pre_rest_s": 500, "post_rest_s": 500}
    reasons = session_gate(row, [])
    assert "duplicate_raw_identity_quarantined" in reasons
    assert "track_identity_requires_external_resolution" in reasons
    assert "incomplete_RUN_crossvalidation" in reasons


def test_gate_rejects_nan_metrics():
    row = {"duplicate_identity": False, "n_tracks": 2, "n_common_units": 30, "minimum_track_occupancy_fraction": 1, "pre_rest_s": 500, "post_rest_s": 500}
    cv = [{"status": "complete", "n_scored_windows": 100, "context_accuracy": 0.9, "median_position_error_cm": 10, "valid_bin_fraction": 1} for _ in range(10)]
    assert session_gate(row, cv) == []
    cv[3]["context_accuracy"] = np.nan
    assert "RUN_decoder_not_calibrated" in session_gate(row, cv)
