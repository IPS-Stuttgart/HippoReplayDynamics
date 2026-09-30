import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_igata_obsolete_route_relapse import block_inventory, duplicate_inventory, scan_file
from scripts.igata_route_audit import adjacent_transitions, classify_route, field_string, file_identity, grid_labels, inspect_log, modified_levenshtein, stimulation_alignment
from scripts.verify_igata_route_audit import reference_distance


def good_info():
    return {"log_status": "ok", "tracking_qc_passed": True, "active_checkpoint_phase": "new_checkpoint_active", "task_success_proxy": True}


def test_source_geometry_and_boundary_precedence():
    assert grid_labels([[100, 900], [900, 100], [700, 700], [300, 300]]).tolist() == ["U", "E", "S", "G"]
    assert grid_labels([[200, 200], [-1, 300], [np.nan, 300]]).tolist() == ["A", "-", "-"]
    assert field_string("lUVWGBCDEa") == "UVWGBCDE"
    assert field_string("lUV-WGBCDEa") == "UV-WGBCDE"


def test_new_route_uses_published_length_not_arbitrary_distance():
    result = classify_route("lUVQ L G B C D Ea".replace(" ", ""), good_info())
    assert result["route_label"] == "new"
    assert result["optimized_new_success"]
    assert classify_route("UVWRMHGBCDE", good_info())["route_label"] == "new"
    assert classify_route("UVWRRMHGBCDE", good_info())["route_label"] == "new"  # adjacent dwell duplicate collapsed
    assert classify_route("UVWRMNHGBCDE", good_info())["route_label"] == "other"  # length == 12


def test_obsolete_attempt_requires_old_goal_before_correction():
    r = classify_route("lUVWRSNIDE DCBGBCDEa".replace(" ", ""), good_info())
    assert r["route_label"] == "obsolete"
    assert r["old_before_new"] and not r["optimized_new_success"]
    mixed = classify_route("UVWRSMHGBCDE", good_info())
    assert mixed["route_label"] == "other"
    assert mixed["route_reason"] == "mixed_old_new_without_early_goal"


@pytest.mark.parametrize("string,change", [("UV-GBCDE", {}), ("VGBCDE", {}), ("UVWGBCD", {}), ("UVWGBCDE", {"tracking_qc_passed": False}), ("UVWGBCDE", {"task_success_proxy": False}), ("UVWGBCDE", {"active_checkpoint_phase": "old_checkpoint_active"})])
def test_unknown_tracking_or_task_is_not_other(string, change):
    r = classify_route(string, dict(good_info(), **change))
    assert r["route_label"] == "unclassifiable"
    assert not r["optimized_new_success"]


def make_log():
    x = np.zeros((30, 16))
    x[:, 0] = 20_000 + np.arange(30) * 40
    x[:, 1:3] = [300, 300]
    x[:, 3] = 1
    x[10:, 8] = 1
    x[20:, 10] = 1
    return x


def test_checkpoint_arrival_flag_uses_rising_edge_and_sensor_order():
    info = inspect_log(make_log())
    assert info["n_checkpoint_rising_edges"] == 1
    assert info["active_checkpoint_phase"] == "new_checkpoint_active"
    assert info["task_success_proxy"]
    x = make_log()
    x[:, 10] = 0
    x[2:5, 10] = 1
    assert not inspect_log(x)["task_success_proxy"]
    x[15:17, 8] = 0
    assert inspect_log(x)["active_checkpoint_phase"] == "unverified_checkpoint"


def test_timestamp_failure_and_tracking_gaps():
    x = make_log()
    x[8:18, 3] = 0
    assert not inspect_log(x)["tracking_qc_passed"]
    x[9, 0] = x[8, 0]
    assert inspect_log(x)["log_status"] == "nonmonotone_or_nonfinite_timestamps"


def test_stimulation_clock_is_relative_not_absolute_and_not_trigger_latency():
    raster = np.zeros(1000)
    raster[[10, 200, 600]] = 1
    r = stimulation_alignment([10.2, 200.8, 600.0], raster, 1000)
    assert r["stimulation_raster_aligned"] and r["stimulation_within_trial_support"]
    assert not r["online_trigger_latency_verified"]
    assert not stimulation_alignment([20010.2, 20200.8, 20600], raster, 1000)["stimulation_raster_aligned"]
    assert not stimulation_alignment([10.2, 600, 200.8], raster, 1000)["stimulation_timestamps_finite_ordered"]
    assert not stimulation_alignment([10.2, 200.8], raster, 1000)["stimulation_raster_aligned"]


def trial(number, *, block="recordA", start=None, route="new"):
    return {"animal": "rat6", "released_group": "Disrupted", "date": "190101", "recording_block": block, "trial_number": number, "relative_path": f"{block}/{number}", "start_time_ms": number * 1000 if start is None else start, "end_time_ms": (number * 1000 if start is None else start) + 500, "route_label": route, "route_reason": "fixture", "optimized_new_success": route == "new", "record_kind": "trial_data", "active_checkpoint_phase": "new_checkpoint_active", "read_status": "ok", "log_sha256": f"{block}/{number}"}


def test_transitions_do_not_bridge_gaps_clock_resets_or_recording_blocks():
    rows = [trial(1), trial(2, route="obsolete"), trial(4), trial(5, start=0), trial(6, block="recordB")]
    t = adjacent_transitions(rows)
    assert len(t) == 1 and t[0]["next_route_label"] == "obsolete"
    assert not t[0]["primary_eligible"]
    rows = [trial(1), trial(2, route="unclassifiable")]
    assert adjacent_transitions(rows)[0]["next_route_label"] == "unclassifiable"


def test_block_audit_preserves_missing_initial_and_internal_trial_numbers():
    b = block_inventory([trial(2), trial(4)])[0]
    assert json.loads(b["missing_trial_numbers"]) == [1, 3]
    assert not b["complete_prior_experience_verified"]
    b = block_inventory([trial(1), trial(1)])[0]
    assert b["duplicate_trial_numbers"] == 1 and b["within_block_clock_overlaps"] == 1


def test_semantic_log_duplicates_are_not_independent_animals():
    a, b = trial(1), dict(trial(1), animal="rat7")
    duplicate = duplicate_inventory([a, b])[0]
    assert duplicate["cross_animal_duplicate"] and duplicate["n_records"] == 2


def test_source_modified_levenshtein_and_independent_implementation():
    assert modified_levenshtein("A", "A") == 0
    assert modified_levenshtein("", "ABC") == pytest.approx(.6)
    assert modified_levenshtein("A", "B") == pytest.approx(.2 / 1.13)
    assert modified_levenshtein("A", "Y") == pytest.approx(.4)  # cheaper delete+insert
    rng = np.random.default_rng(431)
    for _ in range(30):
        a = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXY"), size=15))
        b = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXY"), size=11))
        assert modified_levenshtein(a, b) == pytest.approx(reference_distance(a, b))
    with pytest.raises(ValueError):
        modified_levenshtein("lUV", "GBCDEa")


def test_filename_identity_is_not_condition_inference():
    r = file_identity(Path("190712_detourG01_trial001.npz"))
    assert r == {"date": "190712", "recording_block": "detourG01", "trial_number": 1}
    assert "phase" not in r and "condition" not in r
    with pytest.raises(ValueError):
        file_identity(Path("arbitrary.npz"))


def test_protocol_stops_before_native_contrast_and_uses_independent_validation_seed():
    protocol = json.loads((Path(__file__).parents[1] / "docs/igata_obsolete_route_protocol.json").read_text())
    assert protocol["validation"]["replicates_per_generator"] == 1000
    assert protocol["validation"]["development_seed"] != protocol["validation"]["validation_seed"]
    assert len(protocol["validation"]["generators"]) == 4
    assert not protocol["source_review"]["recording_block_order_documented"]
    assert not protocol["source_review"]["online_ripple_trigger_timestamps_released"]


def test_bad_npz_remains_explicit(tmp_path):
    path = tmp_path / "Disrupted/rat6/trial_data/190101_detourG01_trial001.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, log=np.ones((2, 4)))
    protocol = json.loads((Path(__file__).parents[1] / "docs/igata_obsolete_route_protocol.json").read_text())
    r = scan_file(path, tmp_path, protocol)
    assert r["read_status"] == "failed" and r["route_label"] == "unclassifiable"
    assert r["file_sha256"]
