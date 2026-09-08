import numpy as np

from scripts.audit_2d_geometry_predictive_dissociation import CRITERIA, screen
from scripts.verify_2d_geometry_predictive_dissociation import independent_geometry


def test_independent_geometry_random_paths_and_support():
    rng = np.random.default_rng(79)
    for _ in range(100):
        n = int(rng.integers(0, 100))
        xy = np.cumsum(rng.choice([0, 4, 8, 20, 40], size=(n, 2)), axis=0)
        counts = rng.integers(0, 3, size=(n, 3))
        for filtered, minimum in CRITERIA.values():
            a, b = screen(xy, counts, filtered, minimum), independent_geometry(xy, counts, filtered, minimum)
            assert a.keys() == b.keys()
            for key in a:
                if isinstance(a[key], str):
                    assert a[key] == b[key]
                else:
                    np.testing.assert_allclose(a[key], b[key], equal_nan=True)
