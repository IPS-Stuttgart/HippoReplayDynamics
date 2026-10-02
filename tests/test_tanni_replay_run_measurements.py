"""Actual LFP phase and native-time measurement checks; no biology calibration."""
import json
from pathlib import Path

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.signal import butter, convolve, filtfilt, hilbert
from scipy.signal.windows import gaussian

from scripts.audit_replay_order_run_coordination import epoch_links, order_asymmetry, whole_bin_shuffles
from scripts import measure_tanni_replay_run_coordination as measure


def protocols():
    root = Path(__file__).parents[1] / "docs"
    return (json.loads((root / "tanni_replay_run_measurement_protocol.json").read_text()),
            json.loads((root / "replay_order_run_coordination_protocol.json").read_text()))


def test_source_theta_matches_pinned_filter_hilbert_and_smoothing():
    p, _ = protocols()
    fs = 200.0
    time = np.arange(12 * int(fs)) / fs
    raw = np.cos(2 * np.pi * 8 * time) + .15 * np.sin(2 * np.pi * 20 * time)
    phase, valid = measure.source_phase(raw, fs, p)
    b, a = butter(2, [7 / (fs / 2), 11 / (fs / 2)], btype="band")
    filtered = filtfilt(b, a, raw)
    analytic = hilbert(filtered, next_fast_len(len(filtered)))[:len(filtered)]
    sigma = .005 * fs
    width = int(round(10 * sigma))
    width -= int(width % 2 == 0)
    kernel = gaussian(width, sigma)
    expected = convolve(np.unwrap(np.angle(analytic)), kernel / kernel.sum(), mode="same")
    expected = (expected + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(phase[valid], expected[valid], atol=1e-10)
    assert not valid[:int(fs)].any() and not valid[-int(fs):].any()


def test_filtered_noise_is_not_automatically_valid_theta():
    p, _ = protocols()
    fs = 200.0
    time = np.arange(12 * int(fs)) / fs
    raw = np.cos(2 * np.pi * 13 * time) + .01 * np.sin(2 * np.pi * 8 * time)
    _, valid = measure.source_phase(raw, fs, p)
    support, rows = measure.theta_bouts(raw, time, fs, np.ones(len(time), bool), valid, p)
    assert rows and not any(row["theta_spectral_supported"] for row in rows)
    assert not support.any()
    raw = np.cos(2 * np.pi * 8 * time) + .1 * np.sin(2 * np.pi * 13 * time)
    _, valid = measure.source_phase(raw, fs, p)
    support, rows = measure.theta_bouts(raw, time, fs, np.ones(len(time), bool), valid, p)
    assert rows and all(row["theta_spectral_supported"] for row in rows)
    assert support.sum() == valid.sum()


def test_nonfinite_lfp_and_integer_clipping_break_filter_segments():
    p, _ = protocols()
    fs = 200.0
    raw = np.sin(2 * np.pi * 8 * np.arange(12 * int(fs)) / fs)
    raw[6 * int(fs)] = np.nan
    phase, valid = measure.source_phase(raw, fs, p)
    assert not valid[5 * int(fs):7 * int(fs) + 1].any()
    assert np.isnan(phase[~valid]).all()
    integer = np.round(np.nan_to_num(raw) * 100).astype(np.int16)
    integer[6 * int(fs)] = np.iinfo(np.int16).max
    _, valid = measure.source_phase(integer, fs, p)
    assert not valid[5 * int(fs):7 * int(fs) + 1].any()
    _, valid = measure.source_phase(np.zeros(1200), fs, p)
    assert not valid.any()


def test_circular_phase_interpolation_and_outside_source_bounds():
    phase = measure.sample_phase(np.array([0., 1., 2.]), np.array([3., -3., -2.]),
                                 np.ones(3, bool), np.array([-.1, .5, 1.5, 2.1]))
    assert np.isnan(phase[[0, 3]]).all()
    assert abs(abs(phase[1]) - np.pi) < 1e-8
    assert abs(phase[2] + 2.5) < 1e-8


def stationary_fixture():
    p, matching = protocols()
    time = np.arange(0, 5.001, 1 / 30)
    position = np.column_stack((time, np.zeros((len(time), 2))))
    links = epoch_links(position, np.array([[time[0], time[-1]]]), matching)
    spikes = np.array([(t, cell) for cell in range(10) for t in np.arange(2., 2.101, .01)])
    return p, matching, links, spikes


def test_mua_detection_crosses_regular_tracking_knots_not_gaps():
    p, _, links, spikes = stationary_fixture()
    rows, diagnostics = measure.detect_mua(spikes, 10, links, p)
    assert diagnostics["immobile_baseline_bins"] > 4000
    assert len(rows) == 1
    assert rows[0]["start_s"] < 2. and rows[0]["end_s"] > 2.1
    assert rows[0]["detector_active_units"] == 10
    assert measure.detect_mua(spikes, 10, links, p) == (rows, diagnostics)
    links["good"][(links["t"][:-1] > 1.9) & (links["t"][:-1] < 2.2)] = False
    rows, _ = measure.detect_mua(spikes, 10, links, p)
    assert not rows


def test_mua_movement_cannot_qualify_as_immobile_candidate():
    p, _, links, spikes = stationary_fixture()
    links["speed"][:] = 20.
    rows, diagnostics = measure.detect_mua(spikes, 10, links, p)
    assert rows == [] and diagnostics["immobile_baseline_bins"] == 0


def test_run_bank_retains_zero_spikes_native_time_and_controls():
    p, _, links, spikes = stationary_fixture()
    mask = np.ones(len(links["good"]), bool)
    mask[40:50] = False
    clock = np.arange(0, 5.01, .0005)
    phase = (2 * np.pi * 8 * clock + np.pi) % (2 * np.pi) - np.pi
    bank = measure.run_bank(spikes, np.arange(10), links, mask, clock, [phase],
                            [np.ones(len(clock), bool)], p)
    assert bank["counts"].shape[1] == 10
    assert (bank["counts"].sum(axis=1) == 0).any()
    assert np.all(mask[bank["native_link"]])
    assert np.all(bank["time_s"] - .0005 >= links["t"][bank["native_link"]] - 1e-10)
    assert np.all(bank["time_s"] + .0005 <= links["t"][bank["native_link"] + 1] + 1e-10)
    assert not np.isin(bank["native_link"], np.arange(40, 50)).any()
    assert bank["theta_phase_rad"].shape == (len(bank["counts"]), 1)
    assert np.isfinite(bank["theta_phase_rad"]).all()
    altered = np.vstack((spikes, [[8., 4.], [9., 2.]]))
    changed = measure.run_bank(altered, np.arange(10), links, mask, clock, [phase],
                               [np.ones(len(clock), bool)], p)
    for key in bank:
        np.testing.assert_array_equal(bank[key], changed[key])


def test_order_shuffles_keep_full_vectors_not_individual_cell_times():
    counts = np.array([[4, 0], [3, 0], [0, 2], [0, 5]])
    original = order_asymmetry(counts, .02, .005, .06)
    assert original[0, 1] > 0
    for shuffled in whole_bin_shuffles(counts, 20, 20261002):
        np.testing.assert_array_equal(shuffled.sum(axis=0), counts.sum(axis=0))
        assert sorted(map(tuple, shuffled)) == sorted(map(tuple, counts))


def test_protocol_cannot_claim_replay_or_completed_association():
    p, _ = protocols()
    assert not p["replay_sequence_validated"] and not p["association_fit_enabled"]
    assert "not_validated_replay" in p["candidate_label"]
