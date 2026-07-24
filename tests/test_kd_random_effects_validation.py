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


def test_random_effects_rejects_model_count_mismatches() -> None:
    with pytest.raises(ValueError, match="models length"):
        random_effects_model_probabilities(_log_evidence(), ["a"])

    with pytest.raises(ValueError, match="models length"):
        random_effects_model_probabilities(_log_evidence()[:, :1], ["a", "b"])


def test_random_effects_rejects_invalid_evidence_shape() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        random_effects_model_probabilities(np.array([0.0, -1.0], dtype=float), ["a", "b"])

    with pytest.raises(ValueError, match="at least one model"):
        random_effects_model_probabilities(np.empty((2, 0), dtype=float), [])

    with pytest.raises(TypeError, match="sequence"):
        random_effects_model_probabilities(_log_evidence(), "ab")


@pytest.mark.parametrize(
    "log_evidence",
    [
        np.array(
            [
                [0.0 + 1.0j, -1.0],
                [-0.5, 0.0],
            ]
        ),
        np.array(
            [
                [np.complex128(0.0 + 1.0j), -1.0],
                [-0.5, 0.0],
            ],
            dtype=object,
        ),
    ],
    ids=["complex-dtype", "object-wrapped-complex"],
)
def test_random_effects_rejects_complex_evidence_before_float_coercion(
    log_evidence: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="log_evidence.*complex"):
        random_effects_model_probabilities(log_evidence, ["a", "b"])


def test_random_effects_preserves_exact_impossible_model_evidence() -> None:
    log_evidence = np.vstack(
        [
            np.tile(np.array([[0.0, -np.inf]]), (20, 1)),
            np.array([[-np.inf, -np.inf]]),
        ]
    )

    rows = random_effects_model_probabilities(
        log_evidence,
        ["supported", "impossible"],
        prior=1.0,
        n_iterations=80,
        burnin=10,
    )
    by_model = {row["model"]: row for row in rows}

    assert all(np.isfinite(row["p_model"]) for row in rows)
    assert all(np.isfinite(row["p_exceedance"]) for row in rows)
    assert by_model["supported"]["p_model"] > 0.9
    assert by_model["impossible"]["p_model"] < 0.1
    assert by_model["supported"]["p_exceedance"] == 1.0
    assert by_model["impossible"]["p_exceedance"] == 0.0


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
