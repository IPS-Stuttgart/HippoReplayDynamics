import numpy as np

from scripts.audit_kleinman_epoch_maps import decode_metrics
from scripts.validate_kleinman_run_decoder import fit_maps, interval_counts
from scripts.verify_kleinman_epoch_maps import independent_map, scalar_metrics


def test_independent_raw_spike_map_matches_interval_encoder_without_test_leakage():
    rng = np.random.default_rng(73)
    t = np.arange(0, 30, 0.02)
    trains = [np.sort(rng.uniform(0, t[-1], 1000)) for _ in range(6)]
    dt = np.diff(t)
    direction = np.arange(len(dt)) % 2
    spatial = (np.arange(len(dt)) // 2) % 20
    mask = np.arange(len(dt)) % 5 != 0
    rates, support, units = independent_map(trains, t, dt, direction, spatial, mask, 20)
    a, b, c, _ = fit_maps(interval_counts(trains, t), dt, direction, spatial, mask, 20)
    np.testing.assert_allclose(a, rates[:, units])
    np.testing.assert_array_equal(b, support)
    np.testing.assert_array_equal(c, units)


def test_scalar_posterior_matches_vectorized_likelihood():
    rng = np.random.default_rng(21)
    rates = rng.uniform(0.01, 30, (40, 9))
    counts = rng.poisson(3, (5, 9))
    support = rng.uniform(size=40) > 0.15
    centers = np.arange(1, 40, 2)
    truth = rng.uniform(0, 40, 5)
    direction = np.array([0, 1, 0, 1, 1])
    edges = np.arange(0, 42, 2)
    for arm in ["poisson", "composition"]:
        batch = decode_metrics(counts, rates, support, centers, truth, direction, edges, arm)
        for i in range(5):
            scalar = scalar_metrics(counts[i], rates, support, centers, truth[i], direction[i], edges, arm)
            for key, value in scalar.items():
                np.testing.assert_allclose(value, batch[key][i], atol=1e-10, equal_nan=True)
