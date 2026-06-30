import numpy as np
import pytest

from hipporeplayimm.kd_reference import random_effects_model_probabilities


def _log_evidence() -> np.ndarray:
    return np.array(
        [
            [0.0, -1.0],
            [-0.5, 0.0],
            [0.2, -0.2],
        ],
        dtype=float,
    )


def test_random_effects_rejects_empty_posterior_burnin() -> None:
    with pytest.raises(ValueError, match="burnin"):
        random_effects_model_probabilities(_log_evidence(), ["a", "b"], n_iterations=5, burnin=5)

    with pytest.raises(ValueError, match="burnin"):
        random_effects_model_probabilities(_log_evidence(), ["a", "b"], n_iterations=5, burnin=-1)


def test_random_effects_rejects_invalid_sampler_options() -> None:
    with pytest.raises(ValueError, match="prior"):
        random_effects_model_probabilities(_log_evidence(), ["a", "b"], prior=0.0)

    with pytest.raises(ValueError, match="prior"):
        random_effects_model_probabilities(_log_evidence(), ["a", "b"], prior=np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match="n_iterations"):
        random_effects_model_probabilities(_log_evidence(), ["a", "b"], n_iterations=True)

    with pytest.raises(ValueError, match="n_iterations"):
        random_effects_model_probabilities(_log_evidence(), ["a", "b"], n_iterations=0)


def test_random_effects_accepts_valid_sampler_options() -> None:
    rows = random_effects_model_probabilities(
        _log_evidence(),
        ["a", "b"],
        prior=1.0,
        n_iterations=np.int64(12),
        burnin=np.int64(3),
    )

    assert [row["model"] for row in rows] == ["a", "b"]
    assert all(np.isfinite(row["p_model"]) for row in rows)
    assert all(np.isfinite(row["p_exceedance"]) for row in rows)
