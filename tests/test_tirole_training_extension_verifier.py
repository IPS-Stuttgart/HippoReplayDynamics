import numpy as np

from scripts.expand_tirole_composition_training import test_likelihoods as lookup_likelihoods
from scripts.verify_tirole_training_extension import reference_lookup


def test_independent_lookup_matches_component_order_and_endpoints():
    rng = np.random.default_rng(8)
    components = rng.uniform(0.1, 1.0, (5, 2, 5))
    components /= components.sum(axis=-1, keepdims=True)
    q = rng.uniform(0, 1, (7, 5, 2))
    q[0, 0] = [0.0, 1.0]
    q[1, 1] = np.nan
    expected = lookup_likelihoods(q, components[None])[0]
    np.testing.assert_allclose(reference_lookup(q, components), expected, equal_nan=True)


def test_missing_training_components_are_preserved():
    q = np.full((3, 5, 2), 0.4)
    components = np.full((5, 2, 5), 0.2)
    components[2] = np.nan
    ll = reference_lookup(q, components)
    assert np.isnan(ll[:, 2]).all()
    assert np.isfinite(ll[:, [0, 1, 3, 4]]).all()
