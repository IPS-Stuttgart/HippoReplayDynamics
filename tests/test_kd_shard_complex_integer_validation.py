from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import (  # noqa: E402
    _coerce_grid_index_array,
    _integer_metadata,
    _integer_scalar_metadata,
)


def test_integer_metadata_rejects_complex_values(tmp_path: Path) -> None:
    path = tmp_path / "complex_metadata.npz"

    with pytest.raises(TypeError, match=r"event_ids.*complex values"):
        _integer_metadata(
            np.asarray([10.0 + 0.5j], dtype=np.complex128),
            key="event_ids",
            path=path,
            min_value=0,
        )


def test_integer_scalar_metadata_rejects_complex_values(tmp_path: Path) -> None:
    path = tmp_path / "complex_scalar_metadata.npz"

    with pytest.raises(TypeError, match=r"kd_n_bins.*complex values"):
        _integer_scalar_metadata(
            np.asarray(4.0 + 0.5j, dtype=np.complex128),
            key="kd_n_bins",
            path=path,
            min_value=1,
        )


def test_grid_index_validator_rejects_complex_values(tmp_path: Path) -> None:
    path = tmp_path / "complex_grid_indices.npz"
    shard = {
        "path": path,
        "sd_indices": np.asarray([0.0 + 0.5j], dtype=np.complex128),
    }

    with pytest.raises(TypeError, match=r"sd_indices.*complex values"):
        _coerce_grid_index_array(shard, "sd_indices")
