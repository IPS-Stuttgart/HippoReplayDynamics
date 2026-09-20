"""Synthetic checks; no network access or large fixture download."""

import hashlib

import numpy as np
import pytest

from scripts.acquire_dandi000978 import destination, verify
from scripts.preflight_dandi000978 import RangeReader, assign_epochs, reference_audit


def test_epoch_assignment_requires_complete_containment():
    result = assign_epochs([1, 5, 9, 11], [2, 6, 11, 12], [0, 10], [9, 20])
    assert result.tolist() == [0, 0, -1, 1]


def test_overlapping_epochs_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        assign_epochs([1], [2], [0, 5], [6, 10])


def test_region_mapping_ambiguity_is_reported_not_silently_corrected():
    audit = reference_audit([10, 11], np.array([1, 2]), np.array(["PFC", "PFC", "CA1", "CA1"]), np.array(["tetrode1", "tetrode1", "tetrode2", "tetrode2"]))
    assert not audit.mapping_hypotheses_disagree.any()
    audit = reference_audit([10], np.array([2]), np.array(["PFC", "PFC", "CA1", "CA1"]), np.array(["tetrode1", "tetrode2", "tetrode3", "tetrode4"]))
    assert audit.mapping_hypotheses_disagree.iloc[0]
    assert audit.nwb_row_region.iloc[0] == "CA1"


def test_invalid_unit_reference_rejected():
    with pytest.raises(ValueError, match="Out-of-range"):
        reference_audit([0], np.array([4]), np.array(["CA1"]), np.array(["tetrode1"]))


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../../escape", "a\\b"])
def test_unsafe_download_path_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        destination(tmp_path, {"path": relative})


def test_published_checksum_required(tmp_path):
    path = tmp_path / "fixture"
    path.write_bytes(b"known-data")
    item = {"size": 10, "digest": {"dandi:sha2-256": hashlib.sha256(b"known-data").hexdigest()}}
    assert verify(path, item) == item["digest"]["dandi:sha2-256"]
    item["digest"]["dandi:sha2-256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        verify(path, item)


def test_range_reader_refuses_unbounded_payload():
    reader = RangeReader("https://example.invalid", 1024**3)
    with pytest.raises(ValueError, match="large"):
        reader.read()
    with pytest.raises(ValueError, match="negative"):
        reader.seek(-1)
