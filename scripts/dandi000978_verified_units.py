"""Read-only, source-explicit regional mappings for the pinned DANDI000978 release.

The private author attachment is an input, never a bundled fixture. No correction
is inferred from decoding performance or extended to unconfirmed animals.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AUTHOR_ARCHIVE_SHA256 = "8dd2507874069dc0caf236e602d7d79c567c8980944c6b3747f83daf6954eed6"
SUPPORTED_ANIMALS = ("JS14", "ZT2")
ZT2_CROSSWALKS = {
    "sub-JDS-SingleDay-ZT2_obj-u40err_behavior+ecephys.nwb": "ZT2idx1.csv",
    "sub-JDS-SingleDay-ZT2_obj-1dss6zi_behavior+ecephys.nwb": "ZT2idx2.csv",
}


def read_author_crosswalks(path):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != AUTHOR_ARCHIVE_SHA256:
        raise ValueError("Author archive differs from the pinned original")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise ValueError("Author archive CRC failed")
        return {filename: pd.read_csv(io.BytesIO(archive.read(f"dataForFlorian/{member}"))) for filename, member in ZT2_CROSSWALKS.items()}


def integer_vector(values, name):
    x = np.asarray(values)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"{name} must be a numeric vector")
    if not np.isfinite(x).all() or (x != np.floor(x)).any() or (x < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative integers")
    return x.astype(np.int64)


def verified_unit_table(animal, filename, ids, references, locations, groups, crosswalk=None):
    if animal not in SUPPORTED_ANIMALS:
        raise ValueError("Anatomical mapping not source-confirmed for this animal")
    ids = integer_vector(ids, "unit IDs")
    references = integer_vector(references, "electrode references")
    if len(ids) != len(references) or len(set(ids)) != len(ids):
        raise ValueError("Unit/reference coverage or identity invalid")
    by_tetrode = {}
    for name, location in zip(groups, locations, strict=True):
        match = re.fullmatch(r"tetrode(\d+)", str(name), re.IGNORECASE)
        if match is None:
            raise ValueError("Unrecognized electrode group name")
        by_tetrode.setdefault(int(match.group(1)), set()).add(str(location))
    if any(len(x) != 1 for x in by_tetrode.values()):
        raise ValueError("Conflicting anatomical locations for a tetrode")
    region = []
    for tetrode in references:
        choices = by_tetrode.get(int(tetrode), set())
        if len(choices) != 1 or next(iter(choices)) not in ("CA1", "PFC"):
            raise ValueError("Unresolved tetrode anatomy")
        region.append(next(iter(choices)))
    result = pd.DataFrame({"unit_id": ids, "tetrode_id": references, "region": region})
    if animal == "ZT2":
        if filename not in ZT2_CROSSWALKS or crosswalk is None:
            raise ValueError("File-specific author crosswalk required for ZT2")
        cols = ["Cell", "Tetrode", "tetCellDandiIdx3"]
        if not set(cols).issubset(crosswalk.columns):
            raise ValueError("Missing crosswalk columns")
        c = pd.DataFrame({k: integer_vector(crosswalk[k], k) for k in cols})
        if c.Cell.duplicated().any() or c.duplicated(cols[1:]).any():
            raise ValueError("Crosswalk must be one-to-one")
        if set(c.Cell) != set(ids) or (c[cols[1:]] <= 0).any().any():
            raise ValueError("Crosswalk coverage or tetrode/cluster IDs invalid")
        c = c.set_index("Cell").loc[ids]
        if not np.array_equal(c.Tetrode.to_numpy(), references):
            raise ValueError("Author tetrodes disagree with raw unit references")
        result["cluster_id"] = c.tetCellDandiIdx3.to_numpy()
        result["stable_unit_key"] = [f"ZT2:t{t}:c{k}" for t, k in zip(references, result.cluster_id, strict=True)]
        result["mapping_basis"] = "author_explicit_crosswalk_20260921"
        result["cross_file_source_identity_verified"] = True
    else:
        if filename != "sub-JDS-SingleDay-JS14_behavior+ecephys.nwb":
            raise ValueError("Unrecognized JS14 source file")
        result["cluster_id"] = pd.NA
        result["stable_unit_key"] = [f"JS14:file_local_unit:{i}" for i in ids]
        result["mapping_basis"] = "author_tetrode_ID_clarification_20260921_not_explicit_crosswalk"
        result["cross_file_source_identity_verified"] = False
    result["source_identity_is_not_sorting_stability_proof"] = True
    return result
