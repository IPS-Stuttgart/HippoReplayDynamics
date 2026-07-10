import numpy as np

import hipporeplayimm.simulation_recovery as recovery
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


class _RecordingGenerator:
    def __init__(self) -> None:
        self.poisson_means: list[np.ndarray] = []
        self.negative_binomial_calls: list[tuple[float, np.ndarray]] = []

    def poisson(self, lam, size=None):
        assert size is None
        mean = np.asarray(lam, dtype=float)
        self.poisson_means.append(mean.copy())
        return np.ones(mean.shape, dtype=int)

    def negative_binomial(self, n, p, size=None):
        assert size is None
        probability = np.asarray(p, dtype=float)
        self.negative_binomial_calls.append((float(n), probability.copy()))
        return np.full(probability.shape, 2, dtype=int)


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.array([[10.0, 20.0], [30.0, 40.0]]),
        occupancy_s=np.ones(2),
        cell_ids=np.array([1, 2]),
        config=EncodingConfig(bin_size_cm=1.0),
    )


def test_simulated_counts_use_configured_negative_binomial_likelihood(monkeypatch):
    path = np.array([0, 1], dtype=int)
    monkeypatch.setattr(
        recovery,
        "simulate_latent_path",
        lambda *args, **kwargs: path.copy(),
    )
    rng = _RecordingGenerator()

    emissions, actual_path = recovery.simulate_replay_event(
        _encoding(),
        true_model="diffusion",
        n_time=2,
        dt=0.1,
        rng=rng,
        spike_rate_scale=2.0,
        negative_binomial_overdispersion=0.5,
    )

    np.testing.assert_array_equal(actual_path, path)
    np.testing.assert_array_equal(emissions.spike_counts, np.full((2, 2), 2))
    assert rng.poisson_means == []
    assert len(rng.negative_binomial_calls) == 2

    dispersion_size = 2.0
    expected_means = (
        np.array([2.0, 6.0]),
        np.array([4.0, 8.0]),
    )
    for (actual_size, actual_probability), expected_mean in zip(
        rng.negative_binomial_calls,
        expected_means,
    ):
        assert actual_size == dispersion_size
        np.testing.assert_allclose(
            actual_probability,
            dispersion_size / (dispersion_size + expected_mean),
        )


def test_zero_overdispersion_keeps_poisson_sampling(monkeypatch):
    path = np.array([0, 1], dtype=int)
    monkeypatch.setattr(
        recovery,
        "simulate_latent_path",
        lambda *args, **kwargs: path.copy(),
    )
    rng = _RecordingGenerator()

    emissions, _ = recovery.simulate_replay_event(
        _encoding(),
        true_model="diffusion",
        n_time=2,
        dt=0.1,
        rng=rng,
        negative_binomial_overdispersion=0.0,
    )

    np.testing.assert_array_equal(emissions.spike_counts, np.ones((2, 2), dtype=int))
    assert len(rng.poisson_means) == 2
    assert rng.negative_binomial_calls == []
