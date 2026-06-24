import numpy as np

from hipporeplayimm.encoding import EncodingConfig, _make_grid


def test_make_grid_uses_first_two_coordinate_columns_when_extra_columns_are_present():
    config = EncodingConfig(bin_size_cm=1.0, arena_padding_cm=0.0)
    xy_with_metadata = np.array(
        [
            [0.0, 1.0, 99.0],
            [2.0, 3.0, 100.0],
        ],
        dtype=float,
    )

    x_edges, y_edges, centers = _make_grid(xy_with_metadata, config)

    np.testing.assert_allclose(x_edges, np.array([0.0, 1.0, 2.0], dtype=float))
    np.testing.assert_allclose(y_edges, np.array([1.0, 2.0, 3.0], dtype=float))
    np.testing.assert_allclose(
        centers,
        np.array(
            [
                [0.5, 1.5],
                [0.5, 2.5],
                [1.5, 1.5],
                [1.5, 2.5],
            ],
            dtype=float,
        ),
    )
