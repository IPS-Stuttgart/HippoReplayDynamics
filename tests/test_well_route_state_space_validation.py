import numpy as np
import pytest

from hipporeplayimm.well_route_state_space import routes_from_wells


def test_routes_from_wells_rejects_zero_coordinate_wells():
    with pytest.raises(ValueError, match="at least one coordinate column"):
        routes_from_wells(np.empty((2, 0)))


def test_routes_from_wells_rejects_nonfinite_wells():
    with pytest.raises(ValueError, match="well_locations must be finite"):
        routes_from_wells(np.array([[0.0, 0.0], [np.nan, 1.0]]))


def test_routes_from_wells_preserves_forward_only_routes():
    routes = routes_from_wells(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        include_reverse=False,
    )

    assert routes.shape == (3, 2, 2)
    np.testing.assert_allclose(
        routes,
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )
