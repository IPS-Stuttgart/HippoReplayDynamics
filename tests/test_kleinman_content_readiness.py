import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts.audit_kleinman_content_readiness import (
    counts_in_windows,
    event_windows,
    immobile_windows,
    inspect_session,
    numeric_matrix,
    parse_session,
    position_alignment_diagnostic,
    snapshot_inputs,
    split_units,
    summarize_support,
    unit_spike_trains,
    verify_inputs,
)


def info():
    t = np.arange(0, 2, 0.025)
    return {"position": np.arange(len(t) + 1, dtype=float), "velocity": np.c_[t, np.zeros(len(t))], "drug": 0, "novel": 0}


def test_compound_identity_does_not_merge_clusters():
    spikes = np.array([[0.02, 1, 1], [0.01, 1, 2], [0.03, 1, 1]])
    keys, trains = unit_spike_trains(spikes)
    np.testing.assert_array_equal(keys, [[1, 1], [2, 1]])
    np.testing.assert_array_equal(trains[0], [0.02, 0.03])
    np.testing.assert_array_equal(counts_in_windows(trains, [0, 0.02], [0.02, 0.04]), [[0, 1], [2, 0]])


def test_position_extra_sample_remains_unresolved():
    position, velocity, status, flags = parse_session(info())
    assert status == "one_extra_position_sample_unresolved"
    assert flags == {"drug": 0, "novel": 0}
    results = position_alignment_diagnostic(position, velocity, np.array([[0.5, 0.6, 0.55, 20.0]]))
    assert {r["convention"] for r in results} == {"drop_first", "drop_last"}
    assert len(position) == len(velocity) + 1


def test_unavailable_novelty_preserved_not_fabricated(tmp_path):
    metadata = info()
    del metadata["novel"]
    assert parse_session(metadata)[3]["novel"] is None
    folder = tmp_path / "Experiment_2/Con_1/session"
    folder.mkdir(parents=True)
    savemat(folder / "session_info.mat", {"session_info": metadata})
    savemat(folder / "ripple_events.mat", {"ripple_events": np.array([[0.5, 0.7, 0.6, 20.0]])})
    row, events, _, _, _ = inspect_session(folder / "session_info.mat", tmp_path, 12)
    assert row["novel"] is None and not row["prospective_control_stratum"]
    summary = summarize_support(pd.DataFrame(events))
    assert len(summary) == 6 and summary.novel.isna().all()


def test_invalid_clock_and_nonfinite_arrays_rejected():
    bad = info()
    bad["velocity"][4, 0] = bad["velocity"][3, 0]
    with pytest.raises(ValueError, match="timestamps"):
        parse_session(bad)
    with pytest.raises(ValueError):
        numeric_matrix([[1.0, np.nan, 3]], 3, "spikes")


def test_native_endpoint_never_shifted_to_peak():
    events = np.array([[0.1, 0.2, 0.11, 10.0], [0.3, 0.31, 0.305, 20.0], [0.5, 0.7, 0.69, 30.0]])
    starts, ends, valid = event_windows(events, "native_endpoint")
    np.testing.assert_allclose(ends, events[:, 1])
    np.testing.assert_allclose(starts, ends - 0.02)
    np.testing.assert_array_equal(valid, [True, False, True])
    _, _, peak_valid = event_windows(events, "native_peak")
    assert not peak_valid[1]


def test_gap_and_motion_not_marked_immobile():
    velocity = np.array([[0.0, 0.0], [0.025, 0.0], [0.05, 6.0], [0.25, 0.0], [0.275, 0.0]])
    np.testing.assert_array_equal(immobile_windows(velocity, [0.001, 0.03, 0.1, 0.255], [0.02, 0.04, 0.12, 0.27]), [True, False, False, True])


def test_partition_disjoint_equal_and_seeded():
    a, b = split_units(11, "Experiment_1/Con_1/session", 0, 19)
    assert len(a) == len(b) == 5 and not set(a).intersection(b)
    a2, b2 = split_units(11, "Experiment_1/Con_1/session", 0, 19)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)


def test_read_native_mini_fixture_with_zero_support_and_missing_spikes(tmp_path):
    folder = tmp_path / "Experiment_1/Con_1/session"
    folder.mkdir(parents=True)
    savemat(folder / "session_info.mat", {"session_info": info()})
    savemat(folder / "ripple_events.mat", {"ripple_events": np.array([[0.5, 0.7, 0.6, 20.0]])})
    row, events, _, units, _ = inspect_session(folder / "session_info.mat", tmp_path, 12)
    assert row["has_spikes"] is False and row["decoder_ready"] is False
    assert not units and len(events) == 6
    summary = summarize_support(pd.DataFrame(events))
    assert summary.both_halves_supported.eq(0).all()
    assert summary.support_fraction.eq(0).all()
    assert row["prospective_control_stratum"]


def test_short_candidate_stays_in_denominator_inventory(tmp_path):
    folder = tmp_path / "Experiment_1/Exp_1/session"
    folder.mkdir(parents=True)
    savemat(folder / "session_info.mat", {"session_info": info()})
    savemat(folder / "spike_data.mat", {"spike_data": np.array([[0.605, 1, 1], [0.606, 1, 2]])})
    savemat(folder / "sdes.mat", {"sdes": np.array([[0.6, 0.61, 0.605, 20.0]])})
    row, events, _, _, _ = inspect_session(folder / "session_info.mat", tmp_path, 12)
    assert row["n_units_raw"] == 2 and not row["prospective_control_stratum"]
    assert not any(x["valid_20ms_window"] for x in events)
    summary = summarize_support(pd.DataFrame(events))
    assert summary.native_events.eq(1).all()
    assert summary.valid_immobile_windows.eq(0).all()
    assert summary.support_fraction.isna().all()


def test_source_modification_cannot_finish_as_verified(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("before")
    inputs = {"source": path}
    frozen = snapshot_inputs(inputs)
    assert verify_inputs(inputs, frozen) == frozen
    path.write_text("after")
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs(inputs, frozen)


def test_rate_denominator_uses_only_behavior_clock_spikes(tmp_path):
    folder = tmp_path / "Experiment_1/Con_1/session"
    folder.mkdir(parents=True)
    savemat(folder / "session_info.mat", {"session_info": info()})
    savemat(folder / "spike_data.mat", {"spike_data": np.array([[-0.2, 1, 1], [0.5, 1, 1], [3.0, 1, 1]])})
    _, _, _, units, _ = inspect_session(folder / "session_info.mat", tmp_path, 12)
    assert units[0]["n_spikes"] == 3
    assert units[0]["n_spikes_in_velocity_span"] == 1
    assert units[0]["mean_rate_over_velocity_span_hz"] == pytest.approx(1 / 1.975)
