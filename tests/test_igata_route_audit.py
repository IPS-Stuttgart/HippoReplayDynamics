import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_igata_obsolete_route_relapse import block_inventory, duplicate_inventory, scan_file
from scripts.igata_route_audit import adjacent_transitions, classify_route, field_string, file_identity, grid_labels, inspect_log, modified_levenshtein, stimulation_alignment
from scripts.verify_igata_route_audit import reference_distance
from scripts.launch_igata_route_audit import ALLOWED_HOSTS
from scripts import audit_igata_obsolete_route_relapse as driver
from scripts._provenance import file_sha256
from scripts.verify_igata_route_audit import verify


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
    assert file_identity(Path("190712_detourG01_return001.npz")) == r
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


def test_return_records_use_release_return_filename_pattern(tmp_path):
    path = tmp_path / "Delayed/rat12/return_data/190712_detourG01_return001.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, log=np.ones((2, 4)))
    protocol = json.loads((Path(__file__).parents[1] / "docs/igata_obsolete_route_protocol.json").read_text())
    row = scan_file(path, tmp_path, protocol)
    assert row["record_kind"] == "return_data" and row["trial_number"] == 1
    assert row["read_status"] == "failed"  # bad log, not a filename crash


def test_launcher_accepts_verified_server_hostname_not_unrelated_hosts():
    assert ALLOWED_HOSTS == {"gpuserver6000", "workstation2"}
    assert "gpuserver4090" not in ALLOWED_HOSTS


def save_valid_trial(dataset):
    path = dataset / "Disrupted/rat6/trial_data/190101_detourG01_trial001.npz"
    path.parent.mkdir(parents=True)
    letters = "lUVQLGBCDEa"
    xy = []
    for letter in letters:
        if letter in "la":
            xy.extend([[-100, 800] if letter == "l" else [1100, 100]] * 3)
        else:
            n = ord(letter) - 65
            xy.extend([[100 + n % 5 * 200, 100 + n // 5 * 200]] * 3)
    log = np.zeros((len(xy), 16))
    log[:, 0] = np.arange(len(xy)) * 40
    log[:, 1:3] = xy
    log[:, 3] = 1
    log[15:, 8] = 1
    log[-3:, 10] = 1
    raster = np.zeros(len(log) * 40)
    raster[[10, 200]] = 1
    np.savez(path, log=log, locus_string_list=np.array(list(letters)), locus_stay_frame_list=np.full(len(letters), 3), spike_hist=np.zeros((0, len(raster))), lfp=np.zeros(len(raster) * 2), stim_mat=raster, trial_stim=np.array([10.2, 200.8]))
    return path


def test_valid_release_fixture_scans_and_labels_new_route(tmp_path):
    path = save_valid_trial(tmp_path)
    protocol = json.loads((Path(__file__).parents[1] / "docs/igata_obsolete_route_protocol.json").read_text())
    row = scan_file(path, tmp_path, protocol)
    assert row["read_status"] == "ok", row
    assert row["native_string_matches_full_coordinates"] and row["native_dwell_covers_log"]
    assert row["route_label"] == "new" and row["optimized_new_success"]
    assert row["stimulation_raster_aligned"]


def test_literal_grid_flicker_does_not_reject_published_processed_route(tmp_path):
    path = save_valid_trial(tmp_path)
    with np.load(path, allow_pickle=False) as d:
        values = {k: d[k] for k in d.files}
    values["log"][4, 1] = 201  # brief raw crossing before the released U-to-V segment boundary
    np.savez(path, **values)
    protocol = json.loads((Path(__file__).parents[1] / "docs/igata_obsolete_route_protocol.json").read_text())
    row = scan_file(path, tmp_path, protocol)
    assert not row["native_string_matches_full_coordinates"]
    assert row["native_grid_processing_status"] == "source_string_differs_from_instantaneous_grid"
    assert row["route_label"] == "new"


def test_full_audit_stop_and_independent_verifier(tmp_path, monkeypatch):
    dataset, out = tmp_path / "dataset", tmp_path / "output"
    trial_path = save_valid_trial(dataset)
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(trial_path, trial_path.relative_to(dataset).as_posix())
    readme, geometry = dataset / "readme.txt", dataset / "virtual_maze_field.py"
    xml, pdf, txt = (tmp_path / name for name in ("main.xml", "si.pdf", "si.txt"))
    for path in (readme, geometry, xml, pdf, txt):
        path.write_text("synthetic source fixture")
    protocol = json.loads((Path(__file__).parents[1] / "docs/igata_obsolete_route_protocol.json").read_text())
    for key, path in (("release_readme_sha256", readme), ("release_geometry_sha256", geometry), ("supplement_pdf_sha256", pdf), ("supplement_text_sha256", txt)):
        protocol[key] = file_sha256(path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(sys, "argv", ["audit", "--dataset-root", str(dataset), "--archive", str(archive), "--source-main-xml", str(xml), "--source-si-pdf", str(pdf), "--source-si-text", str(txt), "--protocol", str(protocol_path), "--output-dir", str(out)])
    original = driver.build_script_provenance
    monkeypatch.setattr(driver, "build_script_provenance", lambda **kwargs: dict(original(**kwargs), git_dirty=False))
    driver.main()
    decision = json.loads((out / "decision.json").read_text())
    assert decision["decision"] == "stop_unverified_public_trial_design"
    assert decision["validation_replicates_run"] == 0 and not decision["biological_contrast_run"]
    result = verify(out, dataset)
    assert result["overall"] == "pass", result
    (out / "decision.json").write_text("{}")
    with pytest.raises(KeyError):
        verify(out, dataset)  # cannot bless a tampered or incomplete decision
