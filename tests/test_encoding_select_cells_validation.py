import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _encoding_model() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rates_hz=np.array([[2.0], [4.0]]),
        occupancy_s=np.array([1.0]),
        cell_ids=np.array([1, 2]),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    "requested",
    [
        [1.0],
        [1.9],
        [np.nan],
        [True],
        ["1"],
    ],
)
def test_select_cells_rejects_non_integer_requested_ids(requested):
    with pytest.raises(TypeError, match="integer cell IDs"):
        _encoding_model().select_cells(requested)


def test_select_cells_keeps_unique_sorted_integer_ids_after_validation():
    selected = _encoding_model().select_cells([2, 1, 2])

    assert selected.cell_ids.tolist() == [1, 2]
    np.testing.assert_allclose(selected.rates_hz, np.array([[2.0], [4.0]]))
