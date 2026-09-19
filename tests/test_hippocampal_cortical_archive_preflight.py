import hashlib
import io
import json

import h5py
import numpy as np
import pytest

from scripts.preflight_hippocampal_cortical_archive import (
    LFP,
    POSITION,
    fetch_asset,
    inventory,
    json_value,
    safe_path,
    validate_spikes,
    verify_file,
)


@pytest.fixture
def nwb(tmp_path):
    path = tmp_path / "recording.nwb"
    with h5py.File(path, "w") as f:
        f["general/subject/subject_id"] = "mouse"
        f["session_start_time"] = "2026-01-01T10:00:00+00:00"
        f["session_description"] = "Novel Maze Session with Pre and Post sleep"
        f["units/id"] = [10, 20, 30]
        f["units/cell_area"] = np.array(["CA1", "CA3", "RSC"], dtype=h5py.string_dtype())
        f["units/cell_type"] = np.array(["Pyramidal Cell"] * 3, dtype=h5py.string_dtype())
        f["units/spike_times"] = [11., 13., 12., 15., 13., 19.]
        f["units/spike_times_index"] = np.array([2, 4, 6], np.uint32)
        f[f"{POSITION}/timestamps"] = np.arange(12., 18.)
        d = f.create_dataset(f"{POSITION}/data", data=np.arange(12.).reshape(6, 2))
        d.attrs.update(unit="centimeters", conversion=0.01, offset=0.)
        f["processing/behavior/Speed/timestamps"] = np.arange(12., 18.)
        f[f"{LFP}/data"] = np.zeros((200, 1), np.int16)
        d = f.create_dataset(f"{LFP}/starting_time", data=10.)
        d.attrs["rate"] = 10.
        g = f.create_group("intervals/SleepStates")
        g.attrs.update(neurodata_type="TimeIntervals", description="Native states and ripples")
        g["start_time"] = [10., 20., 21.]
        g["stop_time"] = [20., 29., 21.2]
        g["state"] = np.array(["WAKE", "NREM", "Ripple"], dtype=h5py.string_dtype())
        f.create_dataset("object_references", data=[g.ref], dtype=h5py.ref_dtype)
    return path


def test_inventory_records_regions_and_preserves_unknown_context(nwb):
    row, detail = inventory(nwb)
    assert row["n_units"] == 3 and row["n_CA1_pyramidal"] == row["n_RSC_pyramidal"] == 1
    assert row["n_native_Ripple_intervals"] == 1
    assert row["basic_clock_checks_passed"]
    assert row["lfp_stop_s"] == 30  # Includes nonzero starting time.
    assert row["position_scaling_requires_review"]
    assert row["position_raw_x_span"] == 10
    assert row["non_sleep_interval_table_paths"] == ""
    assert not row["has_epochs_group"] and not row["has_trials_table"]
    assert row["context_validation_status"] == "requires_author_confirmed_epoch_assignment"
    assert any(x["path"] == "object_references" for x in detail["schema"])
    json.dumps(json_value(detail), allow_nan=False)


def test_clock_shift_is_not_silently_corrected(nwb):
    with h5py.File(nwb, "r+") as f:
        f[f"{POSITION}/timestamps"][:] += 100
    row, _ = inventory(nwb)
    assert not row["position_inside_lfp_clock"]
    assert not row["speed_position_clocks_equal"]
    assert not row["basic_clock_checks_passed"]


def test_existing_epochs_need_explicit_review(nwb):
    with h5py.File(nwb, "r+") as f:
        g = f.create_group("epochs")
        g.attrs["neurodata_type"] = "TimeIntervals"
        g["start_time"], g["stop_time"] = [12., 15.], [14., 17.]
        g["tags"] = np.array(["maze_one", "maze_two"], dtype=h5py.string_dtype())
    row, detail = inventory(nwb)
    assert row["has_epochs_group"]
    assert row["non_sleep_interval_table_paths"] == "epochs"
    assert row["context_validation_status"] == "requires_author_confirmed_epoch_assignment"
    assert detail["interval_tables"][0]["label_counts"]["tags"]["maze_one"] == 1


@pytest.mark.parametrize("path", ["../a", "/a", "a/../../b", "a\\b", "C:/a"])
def test_reject_unsafe_asset_paths(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_reject_symlink_escape(tmp_path):
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(tmp_path, "outside/asset.nwb")


@pytest.mark.parametrize("ids,indices,times", [
    ([1, 1], [1, 2], [0., 1.]),
    ([1, 2], [2, 1], [0., 1.]),
    ([1], [1], [0., 1.]),
    ([1], [2.], [0., 1.]),
    ([1], [2], [1., 0.]),
    ([1], [2], [0., float("nan")]),
    ([], [], []),
])
def test_reject_corrupt_spike_schema(ids, indices, times):
    with pytest.raises(ValueError):
        validate_spikes(ids, indices, times)


def test_allow_timestamp_reset_between_units():
    validate_spikes([1, 2], [2, 4], [1., 2., 0., 1.])


def test_verified_cache_and_identity_check(tmp_path):
    data = b"small fixture"
    p = tmp_path / "sub/session.nwb"
    p.parent.mkdir()
    p.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    asset = {"asset_id": "identifier", "path": "sub/session.nwb", "size": len(data)}
    metadata = {"identifier": "identifier", "path": asset["path"], "contentSize": len(data), "digest": {"dandi:sha2-256": digest}}
    assert fetch_asset(asset, metadata, tmp_path) == p
    with pytest.raises(ValueError):
        fetch_asset(asset, {**metadata, "identifier": "other"}, tmp_path)
    p.write_bytes(b"bad" + data[3:])
    with pytest.raises(ValueError, match="mismatch"):
        fetch_asset(asset, metadata, tmp_path, download=True)
    assert p.read_bytes().startswith(b"bad")  # Never overwrite a bad cache.


def test_failed_download_never_promoted(tmp_path, monkeypatch):
    asset = {"asset_id": "identifier", "path": "sub/session.nwb", "size": 3}
    metadata = {"identifier": "identifier", "path": asset["path"], "contentSize": 3, "digest": {"dandi:sha2-256": hashlib.sha256(b"yes").hexdigest()}}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        fetch_asset(asset, metadata, tmp_path, download=True)
    assert not (tmp_path / asset["path"]).exists()
    assert not list(tmp_path.rglob("*.part-*"))


def test_download_and_checksum(tmp_path, monkeypatch):
    data = b"yes"
    asset = {"asset_id": "identifier", "path": "sub/session.nwb", "size": len(data)}
    metadata = {"identifier": "identifier", "path": asset["path"], "contentSize": 3, "digest": {"dandi:sha2-256": hashlib.sha256(data).hexdigest()}}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(data))
    p = fetch_asset(asset, metadata, tmp_path, download=True)
    verify_file(p, 3, hashlib.sha256(data).hexdigest())
    assert p.read_bytes() == data
