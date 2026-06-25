import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    run_session_simulation_recovery,
    simulate_latent_path,
)
from hipporeplayimm.state_space import StateSpaceDecoderConfig


def _small_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rates_hz=np.array([[10.0]]),
        occupancy_s=np.ones(1),
        cell_ids=np.array([1]),
        config=EncodingConfig(bin_size_cm=1.0),
    )


@pytest.mark.parametrize("bad_length", [0, -1, 1.5, np.nan, np.array([2])])
def test_simulate_latent_path_rejects_invalid_length(bad_length):
    generator = np.random.default_rng(1)
    with pytest.raises(ValueError, match="n_time must be positive"):
        simulate_latent_path(
            _small_encoding(),
            true_model="stationary",
            n_time=bad_length,
            dt=0.02,
            rng=generator,
        )


@pytest.mark.parametrize(
    ("true_model", "state_space"),
    [
        (
            "diffusion",
            StateSpaceDecoderConfig(diffusion_sigma_cm_sqrt_s=-1.0),
        ),
        (
            "diffusion",
            StateSpaceDecoderConfig(diffusion_sigma_cm_sqrt_s=np.nan),
        ),
        (
            "diffusion",
            StateSpaceDecoderConfig(diffusion_sigma_cm_sqrt_s=np.array([1.0])),
        ),
        (
            "momentum",
            StateSpaceDecoderConfig(momentum_initial_sigma_cm_sqrt_s=-1.0),
        ),
        (
            "momentum",
            StateSpaceDecoderConfig(momentum_sigma_cm_sqrt_s=np.inf),
        ),
        (
            "momentum",
            StateSpaceDecoderConfig(momentum_sigma_cm_sqrt_s=np.array([1.0])),
        ),
    ],
)
def test_simulate_latent_path_rejects_invalid_motion_sigmas(true_model, state_space):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        simulate_latent_path(
            _small_encoding(),
            true_model=true_model,
            n_time=3,
            dt=0.02,
            rng=np.random.default_rng(1),
            state_space=state_space,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("events_per_model", 0),
        ("events_per_model", 1.5),
        ("events_per_model", np.array([2])),
        ("max_template_events", 0),
        ("max_template_events", 1.5),
        ("max_template_events", np.array([2])),
        ("max_synthetic_events", np.nan),
        ("max_synthetic_events", np.array([2])),
    ],
)
def test_run_session_simulation_recovery_rejects_invalid_count_options(field, value):
    config = SimulationRecoveryConfig(**{field: value})

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        run_session_simulation_recovery("unused-root", "unused-session", config)
