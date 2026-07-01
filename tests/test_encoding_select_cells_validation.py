from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _encoding_model() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[2.0], [4.0]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(),
    )


def _large_id_encoding_model() -> tuple[EncodingModel, int]:
    base = 2**53
    model = EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[2.0], [4.0]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([base, base + 1], dtype=int),
        config=EncodingConfig(),
    )
    return model, base + 1


def test_select_cells_accepts_integer_valued_float_ids() -> None:
    selected = _encoding_model().select_cells(np.array([2.0, 1.0, 2.0], dtype=float))

    assert selected.cell_ids.tolist() == [1, 2]
    np.testing.assert_allclose(selected.rates_hz, np.array([[2.0], [4.0]], dtype=float))


@pytest.mark.parametrize("requested", [[1.9], [1.0000000005], [np.nan]])
def test_select_cells_rejects_invalid_float_ids_without_truncation(requested) -> None:
    with pytest.raises(ValueError, match="integer-valued|finite"):
        _encoding_model().select_cells(requested)


@pytest.mark.parametrize("requested", [[True], ["1"]])
def test_select_cells_rejects_non_numeric_requested_ids(requested) -> None:
    with pytest.raises(TypeError, match="boolean identifiers|integer-valued cell IDs"):
        _encoding_model().select_cells(requested)


def test_select_cells_keeps_unique_sorted_integer_ids_after_validation() -> None:
    selected = _encoding_model().select_cells([2, 1, 2])

    assert selected.cell_ids.tolist() == [1, 2]
    np.testing.assert_allclose(selected.rates_hz, np.array([[2.0], [4.0]], dtype=float))


def test_select_cells_preserves_large_integer_ids_without_float_rounding() -> None:
    model, large_cell_id = _large_id_encoding_model()

    selected = model.select_cells([large_cell_id])

    assert selected.cell_ids.tolist() == [large_cell_id]
    np.testing.assert_allclose(selected.rates_hz, np.array([[4.0]], dtype=float))
