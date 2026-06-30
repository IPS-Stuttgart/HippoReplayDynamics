from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.encoding_select_cells_validation import (
    _PATCHED_FLAG,
    apply_encoding_select_cells_validation_patch,
)


def _encoding_model() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[1.0]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


def test_package_import_applies_select_cells_validation() -> None:
    assert getattr(EncodingModel.select_cells, _PATCHED_FLAG, False)
    with pytest.raises(ValueError, match="integer-valued"):
        _encoding_model().select_cells([1.5])
    with pytest.raises(TypeError, match="boolean"):
        _encoding_model().select_cells([True])


def test_encoding_select_cells_validation_patch_refreshes_replaced_method(monkeypatch) -> None:
    import hipporeplayimm.encoding as encoding

    def lossy_select_cells(self, cell_ids):
        np.asarray(sorted(set(cell_ids)), dtype=int)
        return self

    monkeypatch.setattr(EncodingModel, "select_cells", lossy_select_cells)
    monkeypatch.setattr(encoding, _PATCHED_FLAG, True, raising=False)

    apply_encoding_select_cells_validation_patch()

    assert getattr(EncodingModel.select_cells, _PATCHED_FLAG, False)
    with pytest.raises(ValueError, match="integer-valued"):
        _encoding_model().select_cells([1.5])
