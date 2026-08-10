from __future__ import annotations

import numpy as np

import hipporeplayimm.clusterless as clusterless
import hipporeplayimm.encoding as encoding
import hipporeplayimm.kd_reference as kd_reference
import hipporeplayimm.position_validation as position_validation


def test_position_interpolation_rejects_queries_outside_measured_support():
    times = np.array([1.0, 2.0], dtype=float)
    xy = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    query_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)

    interpolated = encoding._interp_positions(times, xy, query_times)

    assert np.isnan(interpolated[[0, -1]]).all()
    np.testing.assert_allclose(
        interpolated[1:-1],
        np.array([[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]], dtype=float),
    )

    flat_bins = encoding._positions_to_flat_bins(
        interpolated,
        np.array([0.0, 20.0, 40.0], dtype=float),
        np.array([10.0, 30.0, 50.0], dtype=float),
    )
    np.testing.assert_array_equal(flat_bins, np.array([-1, 0, 3, 3, -1], dtype=int))


def test_position_interpolation_rejects_queries_inside_tracking_gaps():
    times = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2], dtype=float)
    xy = np.column_stack(
        [
            np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0], dtype=float),
            np.zeros(6, dtype=float),
        ]
    )
    query_times = np.array([0.15, 0.2, 0.21, 1.0, 2.99, 3.0, 3.05], dtype=float)

    interpolated = encoding._interp_positions(times, xy, query_times)

    assert np.isnan(interpolated[2:5]).all()
    np.testing.assert_allclose(
        interpolated[[0, 1, 5, 6]],
        np.array([[1.5, 0.0], [2.0, 0.0], [100.0, 0.0], [100.5, 0.0]]),
    )


def test_position_interpolation_patch_refreshes_preimported_aliases():
    assert clusterless._interp_positions is encoding._interp_positions
    assert kd_reference._interp_positions is encoding._interp_positions
    assert position_validation._interp_positions is encoding._interp_positions

    times = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2], dtype=float)
    xy = np.column_stack([np.arange(6, dtype=float), np.zeros(6, dtype=float)])
    for module in (clusterless, kd_reference, position_validation):
        assert np.isnan(module._interp_positions(times, xy, np.array([1.0]))).all()
