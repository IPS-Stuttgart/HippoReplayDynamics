"""No inferred cross-file identity or anatomy from plausible-looking counts."""

import numpy as np
import pandas as pd
import pytest

from scripts.audit_dandi000978_cohort import file_relationships, unit_epoch_counts, verify_acquisition_record


def record(file, animal="rat", ids=(0, 1), refs=(4, 8), start=0, stop=10):
    return {"file": file, "animal": animal, "ids": list(ids), "refs": list(refs), "start": start, "stop": stop, "identifier": "same"}


def test_zt2_style_different_unit_counts_do_not_establish_identity():
    rows = file_relationships([record("early"), record("late", ids=(0, 1, 2), refs=(4, 7, 9), start=11, stop=20)])
    r = rows.iloc[0]
    assert r.clock_order == "nonoverlapping_successive_intervals"
    assert r.gap_s == 1
    assert r.n_common_ids_with_different_references == 1
    assert not r.same_unit_ids
    assert not r.cross_file_unit_identity_verified


def test_equal_ids_and_references_still_do_not_prove_cell_identity():
    r = file_relationships([record("a"), record("b", start=11, stop=20)]).iloc[0]
    assert r.same_unit_ids and r.same_unit_electrode_references
    assert not r.cross_file_unit_identity_verified


def test_different_animals_not_joined_and_overlapping_clocks_reported():
    assert file_relationships([record("a"), record("b", animal="other")]).empty
    r = file_relationships([record("a"), record("b", start=5, stop=12)]).iloc[0]
    assert r.clock_order == "overlapping_intervals" and r.gap_s == -5


def test_unit_epoch_counts_keep_empty_units_and_half_open_boundaries():
    arrays = {"unit_ids": np.array([8, 9, 10]), "spike_ends": np.array([3, 3, 5]), "spikes": np.array([0.0, 1.0, 2.0, 1.5, 2.0])}
    epochs = pd.DataFrame({"epoch_index": [0, 1], "start_time_s": [0.0, 1.0], "stop_time_s": [1.0, 2.0], "role_from_trial_presence": ["RUN", "no_trials_rest_candidate"]})
    rows = unit_epoch_counts(arrays, epochs)
    assert rows.n_spikes.tolist() == [1, 1, 0, 0, 0, 1]
    assert rows.unit_id.nunique() == 3


def fixture_record(tmp_path):
    asset = {"path": "rat/a.nwb", "size": 3, "digest": {"dandi:sha2-256": "a" * 64}, "url": "https://dandiarchive.s3.amazonaws.com/blobs/test"}
    p = tmp_path / "raw/rat/a.nwb"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"nwb")
    manifest = {"dataset": "000978", "version": "0.240511.0307", "assets": [asset]}
    status = {"status": "complete_verified", "verified": [{"path": asset["path"], "size": 3, "sha256": "a" * 64}]}
    return manifest, status


def test_verified_record_accepts_matching_sizes_but_is_not_a_rehash(tmp_path):
    manifest, status = fixture_record(tmp_path)
    verify_acquisition_record(tmp_path, manifest, status)


@pytest.mark.parametrize("fault", ["incomplete", "checksum", "duplicate", "missing", "size", "version"])
def test_bad_acquisition_records_fail_closed(tmp_path, fault):
    manifest, status = fixture_record(tmp_path)
    if fault == "incomplete":
        status["status"] = "downloading"
    elif fault == "checksum":
        status["verified"][0]["sha256"] = "b" * 64
    elif fault == "duplicate":
        status["verified"].append(status["verified"][0])
    elif fault == "missing":
        status["verified"] = []
    elif fault == "size":
        (tmp_path / "raw/rat/a.nwb").write_bytes(b"truncated")
    elif fault == "version":
        manifest["version"] = "draft"
    with pytest.raises(ValueError):
        verify_acquisition_record(tmp_path, manifest, status)
