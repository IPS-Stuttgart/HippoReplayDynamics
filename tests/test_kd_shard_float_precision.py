from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import (  # noqa: E402
    _coerce_grid_index_array,
    _integer_metadata,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_integer_metadata_rejects_ambiguous_floating_identifiers(
    tmp_path: Path,
    dtype,
) -> None:
    path = tmp_path / "ambiguous_event_ids.npz"
    precision_bits = int(np.finfo(dtype).nmant) + 1
    ambiguous_boundary = 1 << precision_bits

    with pytest.raises(ValueError, match=r"event_ids.*exact-integer range"):
        _integer_metadata(
            np.asarray([ambiguous_boundary], dtype=dtype),
            key="event_ids",
            path=path,
            min_value=0,
        )


def test_integer_metadata_preserves_safe_extended_precision_identifier(
    tmp_path: Path,
) -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("platform longdouble does not exceed float64 precision")
    expected = 2**53 + 1
    if np.iinfo(np.intp).max < expected:
        pytest.skip("platform integer type cannot hold the regression identifier")

    values = _integer_metadata(
        np.asarray([np.longdouble(expected)], dtype=np.longdouble),
        key="event_ids",
        path=tmp_path / "longdouble_event_ids.npz",
        min_value=0,
    )

    assert values.dtype == np.dtype(np.intp)
    assert int(values[0]) == expected


def test_grid_index_coercion_rejects_ambiguous_float32_index(tmp_path: Path) -> None:
    precision_bits = int(np.finfo(np.float32).nmant) + 1
    shard = {
        "path": tmp_path / "ambiguous_grid_index.npz",
        "sd_indices": np.asarray([1 << precision_bits], dtype=np.float32),
    }

    with pytest.raises(ValueError, match=r"sd_indices.*exact-integer range"):
        _coerce_grid_index_array(shard, "sd_indices")
