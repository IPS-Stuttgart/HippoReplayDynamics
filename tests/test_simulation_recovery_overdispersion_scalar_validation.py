import numpy as np
import pytest

import hipporeplayimm.simulation_recovery as recovery
from hipporeplayimm.simulation_recovery_overdispersion import _finite_nonnegative_scalar


def _nested_zero_dimensional(value: object, *, depth: int = 2) -> np.ndarray:
    current = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = current
        current = wrapper
    return current


@pytest.mark.parametrize("boolean", [True, np.bool_(False)])
def test_overdispersion_rejects_nested_boolean_scalars(boolean: object) -> None:
    value = _nested_zero_dimensional(boolean)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _finite_nonnegative_scalar("negative_binomial_overdispersion", value)


def test_public_simulator_rejects_nested_boolean_overdispersion() -> None:
    value = _nested_zero_dimensional(True)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        recovery.simulate_replay_event(
            None,
            true_model="diffusion",
            n_time=1,
            dt=0.1,
            rng=np.random.default_rng(0),
            negative_binomial_overdispersion=value,
        )


def test_overdispersion_rejects_nested_nonscalar_array() -> None:
    value = _nested_zero_dimensional(np.array([0.5]))

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _finite_nonnegative_scalar("negative_binomial_overdispersion", value)


def test_overdispersion_preserves_nested_real_scalar() -> None:
    value = _nested_zero_dimensional(np.float64(0.5), depth=3)

    assert _finite_nonnegative_scalar("negative_binomial_overdispersion", value) == 0.5


def test_overdispersion_rejects_cyclic_zero_dimensional_wrapper() -> None:
    value = np.empty((), dtype=object)
    value[()] = value

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _finite_nonnegative_scalar("negative_binomial_overdispersion", value)
