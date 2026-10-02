"""Source identity and matching checks, not biological association calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_tanni_replay_run_inputs as audit
from scripts import verify_tanni_replay_run_inputs as verifier
from scripts.audit_replay_order_run_coordination import epoch_links, immobile_pauses, matched_run, link_spikes


def test_curated_labels_index_retained_not_raw_spikes():
    times = np.asarray([1., 2., 3., 4.])
    spikes, units = audit.curated_units(times, np.asarray([False, True, True, True]),
                                       np.asarray([2, 0, 3]), 4)
    np.testing.assert_array_equal(spikes[:, 0], [2, 4])
    np.testing.assert_array_equal(spikes[:, 1], [4 * 65536 + 2, 4 * 65536 + 3])
    assert len(units) == 2
    with pytest.raises(ValueError, match="artifact-retained"):
        audit.curated_units(times, np.asarray([False, True, True, True]), np.asarray([1, 2, 0, 3]), 4)
    with pytest.raises(ValueError, match="Boolean"):
        audit.curated_units(times, np.asarray([0, 1, 1, 1]), np.asarray([2, 0, 3]), 4)


def test_empty_and_zero_clusters_do_not_create_units():
    spikes, units = audit.curated_units(np.asarray([]), np.asarray([], dtype=bool),
                                       np.asarray([], dtype=int), 1)
    assert spikes.shape == (0, 2) and not units
    spikes, units = audit.curated_units(np.asarray([1., 2.]), np.asarray([True, True]),
                                       np.asarray([0, 0]), 1)
    assert spikes.shape == (0, 2) and not units


def test_release_int16_labels_do_not_overflow_stable_ids():
    spikes, units = audit.curated_units(np.asarray([1., 2.]), np.asarray([True, True]),
                                       np.asarray([2, 300], dtype=np.int16), 31)
    assert [row["unit_id"] for row in units] == [31 * 65536 + 2, 31 * 65536 + 300]
    np.testing.assert_array_equal(spikes[:, 1], [31 * 65536 + 2, 31 * 65536 + 300])


@pytest.mark.parametrize("times", [[1, 0], [1, np.nan]])
def test_bad_spike_chronology_not_repaired(times):
    with pytest.raises(ValueError):
        audit.curated_units(np.asarray(times), np.asarray([True, True]), np.asarray([1, 1]), 0)


def test_channel_lists_not_human_readable_one_based_strings():
    mapping = audit.ca1_mapping({"CA1_LH": np.arange(64, 72), "CA1_RH": np.arange(8)})
    assert mapping == {0: "CA1_RH", 1: "CA1_RH", 16: "CA1_LH", 17: "CA1_LH"}
    assert audit.bad_channels(b"1,68,75-76") == {0, 67, 74, 75}
    with pytest.raises(ValueError, match="Duplicate"):
        audit.bad_channels("1,1")
    with pytest.raises(ValueError, match="mixed-area"):
        audit.ca1_mapping({"CA1_LH": np.asarray([0, 1]), "CA1_RH": np.asarray([2, 3])})
    with pytest.raises(ValueError, match="Conflicting"):
        audit.ca1_mapping({"CA1_LH": np.arange(4), "CA1_RH": np.arange(4)})


def test_source_pulse_conversion_and_tie_rule():
    oe = np.asarray([100., 100.1, 100.2])
    camera = np.asarray([1000000., 1100000., 1200000.])
    converted = audit.camera_to_acquisition(oe, camera, np.asarray([1010000., 1180000.]), 1e6)
    np.testing.assert_allclose(converted, [100.01, 100.18])
    np.testing.assert_array_equal(audit.closest_indices(np.asarray([1., 3.]), np.asarray([2.])), [0])


@pytest.mark.parametrize("oe,camera", [([0, .1], [0, 100000, 200000]),
                                    ([0, .1, .1], [0, 100000, 200000]),
                                    ([0, .1, .2], [0, 200000, 100000])])
def test_clock_disagreement_not_offset_fitted_or_truncated(oe, camera):
    with pytest.raises(ValueError):
        audit.camera_to_acquisition(np.asarray(oe), np.asarray(camera), np.asarray([10000., 20000.]), 1e6)


def test_source_reconstruction_allows_removed_rows_but_not_changed_coordinates():
    reference = np.asarray([[1., 1., 2.], [2., 3., 4.], [3., 5., 6.]])
    processed = reference[[0, 2]]
    result = audit.reconcile_processed(processed, reference, .005, .05)
    assert result["source_clock_reconciliation_passed"]
    assert not result["hardware_latency_measured"]
    changed = processed.copy()
    changed[1, 1] += 1
    assert not audit.reconcile_processed(changed, reference, .005, .05)["source_clock_reconciliation_passed"]


def test_resampled_grid_alone_cannot_prove_clock_alignment():
    times = np.arange(100) / 30
    reference = np.column_stack((times, times * 20, times * 10))
    processed = reference[10:80].copy()
    processed[:, 0] += .1  # Three grid steps: modulo-grid checks would pass.
    result = audit.reconcile_processed(processed, reference, .005, .05)
    assert result["maximum_time_error_s"] < 1e-8
    assert not result["source_clock_reconciliation_passed"]


def test_missing_tracking_and_nan_patterns_not_reconciled():
    reference = np.asarray([[1., 1., 2.], [2., 3., 4.]])
    processed = reference.copy()
    processed[0, 1] = np.nan
    assert not audit.reconcile_processed(processed, reference, .005, .05)["source_clock_reconciliation_passed"]
    with pytest.raises(ValueError, match="duplicate"):
        audit.reconcile_processed(reference[[0, 0]], reference, .005, .05)


def test_array_digest_contains_dtype_and_shape():
    a = np.arange(8, dtype=np.int64)
    assert audit.array_digest(a) != audit.array_digest(a.reshape(2, 4))
    assert audit.array_digest(a) != audit.array_digest(a.astype(float))
    with pytest.raises(ValueError):
        audit.array_digest(np.asarray([{}], dtype=object))


def test_native_json_accounting_for_numpy_threshold_flags():
    total = 0
    for flag in [np.bool_(True), np.bool_(False), np.bool_(True)]:
        total += int(flag)
    assert json.dumps({"verified_supported_nonoverlap_pauses": total}) == '{"verified_supported_nonoverlap_pauses": 2}'


def test_future_spikes_cannot_qualify_preceding_unit_support():
    matching = json.loads((Path(__file__).parents[1] / "docs/replay_order_run_coordination_protocol.json").read_text())
    times = np.arange(0, 41.001, .025)
    x = np.zeros(len(times))
    for mask, offset in [(times < 20, 0), (times >= 21, 21)]:
        phase = np.mod(times[mask] - offset, 4)
        x[mask] = np.where(phase < 2, phase, 4 - phase) * 15
    position = np.column_stack((times, x, np.zeros(len(times))))
    spikes = np.asarray([(t, 1) for t in np.arange(.1, 19.9, .1)])
    baseline = audit.pause_support(position, spikes, matching)
    future = np.asarray([(t, c) for t in np.arange(22., 40., .01) for c in range(2, 20)])
    assert baseline == audit.pause_support(position, np.vstack((spikes, future)), matching)


def test_source_combiner_executes_only_pinned_functions(tmp_path, monkeypatch):
    folder = tmp_path / "openEPhys_DACQ"
    folder.mkdir()
    path = folder / "TrackingDataProcessing.py"
    path.write_text("raise RuntimeError('module code must not execute')\n"
                    "def combineCamerasData(*args):\n    return 7\n"
                    "def remove_tracking_data_outside_boundaries(*args, **kwargs):\n    return 9\n")
    monkeypatch.setattr(audit.subprocess, "check_output", lambda cmd, **kwargs: "abc" if "rev-parse" in cmd else "")
    combine, crop, digest = audit.source_combiner(tmp_path, "abc")
    assert combine() == 7 and crop() == 9 and len(digest) == 64
    with pytest.raises(ValueError, match="differs"):
        audit.source_combiner(tmp_path, "other")


def test_zero_or_header_only_support_never_completes_goal():
    for sessions in ([], [{"animal": "A", "source_inputs_verified": False}],
                     [{"animal": "A", "source_inputs_verified": True, "earliest_nonoverlap_supported_pauses": 10}]):
        gates = {x["gate"]: x["passed"] for x in audit.decision_gates(sessions)}
        assert not gates["overall_goal_complete"]
        assert not gates["theta_control_implemented_and_validated"]
        assert not gates["replay_order_and_run_change_tested"]
    assert not {x["gate"]: x["passed"] for x in audit.decision_gates([])}["all_inventory_sessions_resolved"]


def test_independent_pause_and_run_matching_with_rates_and_gaps():
    p = json.loads((Path(__file__).parents[1] / "docs/replay_order_run_coordination_protocol.json").read_text())
    rng = np.random.default_rng(27)
    t = np.arange(0, 80.01, .025)
    x = (np.mod(t, 8) < 4) * np.mod(t, 4) * 15
    x[(t >= 30) & (t <= 31)] = 0
    x[(t >= 40) & (t <= 41)] = np.nan
    position = np.column_stack((t, x, np.zeros(len(t))))
    links = epoch_links(position, np.asarray([[t[0], t[-1]]]), p)
    actual = immobile_pauses(links, p)
    assert verifier.reference_pauses(position, p) == [(r["tracking_start_index"], r["start_s"], r["end_s"]) for r in actual]
    spikes = np.column_stack((np.sort(rng.uniform(0, 80, 2000)), rng.integers(1, 12, 2000)))
    for pause in actual:
        observed, before, _ = matched_run(links, pause, p)
        pre = spikes[link_spikes(spikes[:, 0], links, before)]
        _, counts = np.unique(pre[:, 1], return_counts=True)
        observed["eligible_preceding_units"] = int(np.sum(counts >= p["minimum_preceding_run_spikes_per_unit"]))
        independent = verifier.reference_match(position, spikes, (pause["start_s"], pause["end_s"]), p)
        for key in observed:
            assert observed[key] == pytest.approx(independent[key])
