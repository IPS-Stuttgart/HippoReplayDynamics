import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from scripts.audit_odor_place_post_error_feasibility import acquire
from scripts.odor_place_feasibility_core import blocked_run_qc, clean, inventory, read_table, source_inventory, unique_clock_matches, verify, write_table
from scripts.odor_place_source_io import BoundedHTTPFile, atomic_json, check_space, digest, download_verified

PROTOCOL = json.loads((Path(__file__).parents[1] / "docs/odor_place_post_error_protocol.json").read_text())


def mat_char(handle, path, value):
    return handle.create_dataset(path, data=np.array([ord(c) for c in value], dtype="uint16")[:, None])


def mat_ref(handle, path, nodes):
    return handle.create_dataset(path, data=np.array([n.ref for n in nodes], dtype=h5py.ref_dtype)[:, None])


def source_fixture(path, *, conflict=False, duplicate=False):
    with h5py.File(path, "w") as handle:
        handle.create_group("SuperRat")
        refs = handle.create_group("refs")
        name = mat_char(handle, "refs/name", "CS33")
        day = handle.create_dataset("refs/day", data=[[3.]])
        long = handle.create_dataset("refs/long", data=[[1.]])
        epochs = handle.create_dataset("refs/epochs", data=[[2.]])
        trial = refs.create_group("trials")
        for k, v in {"sniffstart": [1., 4., 7.], "sniffend": [1.6, 4.6, 7.1], "rewardstart": [2., 5., 8.],
                     "rewardend": [3., 6., 9.], "starttime": [1., 4., 7.], "endtime": [3., 6., 9.],
                     "CorrIncorr10": [0., 1., 0.], "leftright10": [1., 0., 1.]}.items():
            trial.create_dataset(k, data=np.array(v)[None])
        trial.create_dataset("EpochInds", data=[[1., 4., 7.], [2., 2., 2.]])
        if conflict:
            del trial["leftright10"]
            trial.create_dataset("leftright10", data=[[1., 0.]])
        tracking = refs.create_group("tracking")
        tracking.create_dataset("data", data=np.zeros((6, 100)))
        chars = [mat_char(handle, "refs/field" + str(i), k) for i, k in enumerate(["time", "x", "y", "dir", "vel", "epoch"])]
        mat_ref(handle, "refs/tracking/fields", chars)
        files = refs.create_group("files")
        task = mat_char(handle, "refs/taskfile", "CS33runTrajBounds03.mat")
        mat_ref(handle, "refs/files/taskfile", [task])
        units = refs.create_group("units")
        for k, v in [("tet", 8), ("unitnum", 2)]:
            value = handle.create_dataset("refs/" + k, data=[[float(v)]])
            mat_ref(handle, "refs/units/" + k, [value])
        spikes = handle.create_dataset("refs/spikes", data=np.arange(5)[:, None])
        mat_ref(handle, "refs/units/ts", [spikes])
        mat_ref(handle, "refs/units/area", [mat_char(handle, "refs/area", "CA1")])
        for k, v in [("name", name), ("daynum", day), ("longTrack", long), ("RunEpochs", epochs),
                     ("trialdata", trial), ("tracking", tracking), ("files", files), ("units", units)]:
            mat_ref(handle, "SuperRat/" + k, [v, v] if duplicate else [v])


def test_actual_matlab_reference_schema_keeps_cue_unverified(tmp_path):
    path = tmp_path / "source.mat"
    source_fixture(path)
    records, trials, units = source_inventory(path)
    assert records[0]["animal"] == "CS33" and records[0]["source_day"] == 3
    assert records[0]["source_run_epochs"] == [2]
    assert len(trials) == 3 and len(units) == 1
    assert trials[0]["leftright10"] == 1 and trials[0]["CorrIncorr10"] == 0
    assert all(not t["cue_verified"] for t in trials)
    assert units[0]["source_tetrode"] == 8 and units[0]["source_cluster"] == 2
    assert "not_nwb_spike_verified" in units[0]["identity_status"]


def test_conflicting_source_annotations_are_explicit_not_reindexed(tmp_path):
    path = tmp_path / "source.mat"
    source_fixture(path, conflict=True)
    records, trials, _ = source_inventory(path)
    assert records[0]["source_trial_field_lengths"]["leftright10"] == 2
    assert records[0]["nonaligned_trial_columns"] == {"leftright10": [1., 0.]}
    assert all(t["leftright10"] is None and t["source_unaligned_fields"] == ["leftright10"] for t in trials)


def test_duplicate_source_days_fail(tmp_path):
    path = tmp_path / "source.mat"
    source_fixture(path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        source_inventory(path)


def test_clock_match_does_not_fit_offset_or_bridge_missing_trial():
    assert unique_clock_matches([1., 2., 4.], [1.001, 2.003, 3., 4.], .005) == [0, 1, None, 2]
    assert unique_clock_matches([1., 2., 4.], [1.02, 2.02, 4.02], .005) == [None, None, None]


@pytest.mark.parametrize("source,target,match", [([1, 1], [1], "Duplicate"), ([1], [1, 1], "Duplicate"),
                                                ([1], [1, 1.001], "many-to-one"), ([np.nan], [1], "Nonfinite")])
def test_ambiguous_clock_records_fail(source, target, match):
    with pytest.raises(ValueError, match=match):
        unique_clock_matches(source, target, .005)


def test_close_duplicate_source_matches_are_not_chosen_by_nearest():
    assert unique_clock_matches([1., 1.003], [1.001], .005) == [None]


def test_nonfinite_metrics_are_missing_not_zero():
    assert clean({"x": np.array([np.nan, np.inf, 3.]), "bytes": b"CA1"}) == {"x": [None, None, 3.], "bytes": "CA1"}


def test_empty_csv_keeps_schema(tmp_path):
    write_table(tmp_path, "zero", [], ["animal", "eligible", "failure_reason"])
    assert (tmp_path / "odor_place_post_error_zero.csv").read_text().strip() == "animal,eligible,failure_reason"
    with pytest.raises(ValueError, match="schema"):
        write_table(tmp_path, "no_schema", [])


def test_atomic_json_forbids_nan_and_checks_digest(tmp_path):
    p = tmp_path / "a.json"
    atomic_json(p, {"value": 1})
    assert len(digest(p)) == 64 and len(digest(p, "md5")) == 32
    with pytest.raises(ValueError):
        atomic_json(p, {"value": float("nan")})


def test_space_reserve_is_required(tmp_path):
    with patch("scripts.odor_place_source_io.shutil.disk_usage", return_value=SimpleNamespace(free=39)):
        with pytest.raises(ValueError, match="reserve"):
            check_space(tmp_path, 10, 30)
        assert check_space(tmp_path, 9, 30) == 39


def test_existing_wrong_file_is_never_overwritten(tmp_path):
    path = tmp_path / "bad.mat"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Existing file differs"):
        download_verified("https://unused.test", path, 3, "incorrect", "md5", 0)
    assert path.read_bytes() == b"bad"


def test_existing_hash_verified_download_is_resumable(tmp_path):
    path = tmp_path / "ok.mat"
    path.write_bytes(b"verified")
    result = download_verified("https://unused.test", path, path.stat().st_size, digest(path, "md5"), "md5", 0)
    assert result["published_digest_verified"] and result["sha256"] == digest(path)


class Response(io.BytesIO):
    def __init__(self, content, status, content_range):
        super().__init__(content)
        self.status, self.headers = status, {"Content-Range": content_range}


def test_bounded_http_never_falls_back_to_full_file():
    reader = BoundedHTTPFile("https://unused.test", 100, 10, block_size=4)
    with patch("urllib.request.urlopen", return_value=Response(b"full", 200, "")):
        with pytest.raises(ValueError, match="bounded"):
            reader.read(1)


def test_bounded_http_budget_seek_cache_and_partial_read():
    reader = BoundedHTTPFile("https://unused.test", 12, 8, block_size=4)
    with patch("urllib.request.urlopen", side_effect=[Response(b"abcd", 206, "bytes 0-3/12"), Response(b"efgh", 206, "bytes 4-7/12")]):
        assert reader.read(6) == b"abcdef"
        reader.seek(1)
        buf = bytearray(3)
        assert reader.readinto(buf) == 3 and bytes(buf) == b"bcd"
        reader.seek(8)
        with pytest.raises(ValueError, match="budget"):
            reader.read(1)
        assert reader.transferred_bytes == 8


def test_version_one_cannot_silently_acquire_version_three_zip(tmp_path):
    args = SimpleNamespace(dataset_root=tmp_path, source_archive=tmp_path / "Figure1-6.zip")
    with patch("scripts.audit_odor_place_post_error_feasibility.get_json", return_value={"version": 1, "files": [{"name": "ClaireDataShort-20-Apr-2022.mat"}]}):
        with pytest.raises(ValueError, match="pinned Figshare version"):
            acquire(args, PROTOCOL)


def test_frozen_protocol_has_no_association_or_future_encoder_inputs():
    assert PROTOCOL["scope"] == "feasibility_only_no_replay_behavior_association"
    assert PROTOCOL["infer_cue_from_choice_and_outcome"] is False
    assert PROTOCOL["run"]["preceding_same_file_only"] is True
    assert PROTOCOL["run"]["retain_zero_spike_bins"] is True
    assert PROTOCOL["run"]["bin_s"] == .02 and PROTOCOL["run"]["graph_bin_cm"] == 3
    assert PROTOCOL["no_association_stage"] and PROTOCOL["no_threshold_relaxation"]


def nwb_bytes(animal):
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle.create_dataset("general/subject/subject_id", data="Symanski-" + animal)
        handle.create_dataset("session_start_time", data="2025-07-22T00:00:00")
        handle.create_dataset("session_description", data="converted task")
        group = handle.create_group("intervals/epoch intervals")
        for key, value in {"epoch_type": [b"odorplace"], "epoch_type_index": [1], "start_time": [0.], "stop_time": [20.]}.items():
            group.create_dataset(key, data=value)
        group = handle.create_group("intervals/trials")
        for key, value in {"id": [0, 1, 2], "start_time": [1.001, 4.003, 7.1], "stop_time": [1.601, 4.603, 7.7],
                           "rewarded": [0, 1, 0], "reward_start_time": [np.nan, 5, np.nan], "reward_end_time": [np.nan, 6, np.nan]}.items():
            group.create_dataset(key, data=value)
        position = handle.create_group("processing/behavior/Position/SpatialSeries")
        position.create_dataset("data", data=np.zeros((3, 3))).attrs["unit"] = "centimeters; centimeters/second"
        position.create_dataset("reference_frame", data="center well")
        units = handle.create_group("units")
        units.create_dataset("id", data=[108])
        units.create_dataset("electrodes", data=[8])
        electrodes = handle.create_group("general/extracellular_ephys/electrodes")
        electrodes.create_dataset("id", data=[108, 110])
        electrodes.create_dataset("location", data=[b"CA1", b"PFC"])
        electrodes.create_dataset("group_name", data=[b"t8", b"t10"])
        handle.create_group("stimulus/presentation")
    return buffer.getvalue()


def actual_pipeline_fixture(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    metadata = dataset / "metadata"
    metadata.mkdir(parents=True)
    output.mkdir()
    source = dataset / "ClaireDataShort-20-Apr-2022.mat"
    source_fixture(source)
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, PROTOCOL)
    atomic_json(metadata / "source_verified.json", {"sha256": digest(source), "published_digest": digest(source, "md5"), "published_digest_verified": True})
    assets = []
    for animal in PROTOCOL["animals"]:
        asset = {"asset_id": animal, "path": animal + "/irrelevant_filename.nwb"}
        assets.append(asset)
        atomic_json(metadata / (animal + ".json"), {"contentUrl": ["https://dandiarchive.s3.amazonaws.com/" + animal],
                                                    "contentSize": len(nwb_bytes(animal)), "digest": {"dandi:sha2-256": "example"},
                                                    "wasAttributedTo": [{"schemaKey": "Participant", "identifier": "Symanski-" + animal}]})
    atomic_json(metadata / "dandi_assets.json", {"results": assets})
    args = SimpleNamespace(dataset_root=dataset, source_archive=source, protocol=protocol_path, output_dir=output)

    def reader(url, *_args):
        stream = io.BytesIO(nwb_bytes(url.rsplit("/", 1)[1]))
        stream.transferred_bytes = len(stream.getvalue())
        return stream

    return args, reader


def test_full_inventory_preserves_errors_silence_missing_cues_and_bad_clock(tmp_path):
    args, reader = actual_pipeline_fixture(tmp_path)
    with patch("scripts.odor_place_feasibility_core.BoundedHTTPFile", side_effect=reader):
        result = inventory(args, PROTOCOL)
    assert result["n_assets"] == 8 and result["n_nwb_trials"] == 24 and result["n_nwb_errors"] == 16
    assert result["full_nwb_files_downloaded"] == 0 and not result["ready_for_calibration"]
    trials = read_table(args.output_dir, "trial_inventory")
    assert sum(r["nwb_well_start_s"] == "" for r in trials) == 16
    assert all(r["cue_id"] == "" and r["eligible"] == "False" for r in trials)
    units = read_table(args.output_dir, "unit_crosswalk")
    assert all(r["nwb_electrode_reference_values"] == "[8]" and r["source_tetrode"] == "" for r in units)
    assert all(r["used_conversion_date_for_chronology"] == "False" for r in read_table(args.output_dir, "asset_inventory"))
    result = blocked_run_qc(args, PROTOCOL)
    assert result["status"] == "not_run_source_prerequisite_failed"
    assert all(r["balanced_accuracy"] == "" for r in read_table(args.output_dir, "run_validation"))
    assert all(r["n_ripples"] == "" for r in read_table(args.output_dir, "ripple_opportunities"))
    checked = verify(args, PROTOCOL)
    assert checked["independent_trial_accounting_passed"] and checked["n_errors"] == 16


def test_resume_checkpoint_is_hash_bound_and_deterministic(tmp_path):
    args, reader = actual_pipeline_fixture(tmp_path)
    with patch("scripts.odor_place_feasibility_core.BoundedHTTPFile", side_effect=reader):
        inventory(args, PROTOCOL)
    before = digest(args.output_dir / "odor_place_post_error_trial_crosswalk.csv")
    with patch("scripts.odor_place_feasibility_core.BoundedHTTPFile", side_effect=AssertionError("No redownload")):
        inventory(args, PROTOCOL)
    assert before == digest(args.output_dir / "odor_place_post_error_trial_crosswalk.csv")
    path = next((args.output_dir / "checkpoints").glob("*.json"))
    data = json.loads(path.read_text())
    data["packet"]["transferred_bytes"] += 1
    atomic_json(path, data)
    with pytest.raises(ValueError, match="checksum"):
        inventory(args, PROTOCOL)


def test_independent_verifier_rejects_changed_output_or_input(tmp_path):
    args, reader = actual_pipeline_fixture(tmp_path)
    with patch("scripts.odor_place_feasibility_core.BoundedHTTPFile", side_effect=reader):
        inventory(args, PROTOCOL)
    path = args.output_dir / "odor_place_post_error_trial_inventory.csv"
    original = path.read_text()
    path.write_text(original + "tampering\n")
    with pytest.raises(ValueError, match="Output hash"):
        verify(args, PROTOCOL)
    path.write_text(original)
    atomic_json(args.protocol, {**PROTOCOL, "seed": 1})
    with pytest.raises(ValueError, match="Input identity"):
        verify(args, PROTOCOL)


def test_cli_never_exposes_association_stage(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "scripts/audit_odor_place_post_error_feasibility.py", "analysis"], capture_output=True, text=True)
    assert result.returncode == 2 and "invalid choice" in result.stderr
