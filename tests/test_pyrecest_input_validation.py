from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleIMMModel, PyRecEstGoalParticleModel, _coerce_candidate_goals


def _single_bin_emissions(n_bins: int = 2) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, n_bins), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_pyrecest_score_rejects_bin_center_count_mismatch_before_optional_import() -> None:
    emissions = _single_bin_emissions(n_bins=2)

    with pytest.raises(ValueError, match="emissions.n_bins must match bin_centers rows"):
        PyRecEstGoalParticleModel().score(
            emissions,
            np.zeros((1, 2), dtype=float),
        )


def test_pyrecest_score_rejects_nonfinite_bin_centers_before_optional_import() -> None:
    emissions = _single_bin_emissions(n_bins=2)
    bin_centers = np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="bin_centers must be finite"):
        PyRecEstGoalParticleModel().score(emissions, bin_centers)


def test_pyrecest_candidate_goals_reject_nonfinite_values() -> None:
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match=r"candidate_goals must .* finite"):
        _coerce_candidate_goals(
            np.array([[0.0, np.nan]], dtype=float),
            bin_centers,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_particles": True}, "n_particles must be a positive integer"),
        ({"initial_velocity_sigma_cm_s": True}, "initial_velocity_sigma_cm_s must be finite and positive"),
        ({"jump_probability": False}, r"jump_probability must lie in \[0, 1\]"),
        ({"position_proposal_ess_threshold": np.bool_(True)}, r"position_proposal_ess_threshold must lie in \[0, 1\]"),
    ],
)
def test_pyrecest_particle_model_rejects_boolean_numeric_parameters(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PyRecEstGoalParticleModel(**kwargs)


def test_pyrecest_score_rejects_mutated_boolean_particle_count_before_optional_import() -> None:
    emissions = _single_bin_emissions(n_bins=2)
    model = PyRecEstGoalParticleModel()
    model.n_particles = True  # type: ignore[assignment]

    with pytest.raises(ValueError, match="n_particles must be a positive integer"):
        model.score(emissions, np.zeros((2, 2), dtype=float))


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"mode_stickiness": True}, ValueError, r"mode_stickiness must lie in \[0, 1\]"),
        ({"jump_fraction": False}, ValueError, r"jump_fraction must lie in \[0, 1\]"),
        ({"momentum_velocity_decay": True}, TypeError, "not boolean"),
    ],
)
def test_pyrecest_imm_model_rejects_boolean_numeric_parameters(
    kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        PyRecEstGoalParticleIMMModel(**kwargs)


def test_pyrecest_bin_center_patch_refreshes_stale_candidate_goal_helper(monkeypatch) -> None:
    import hipporeplayimm.pyrecest_models as pyrecest_models

    hipporeplayimm.apply_runtime_patches()

    def stale_coerce_candidate_goals(candidate_goals, bin_centers):
        return np.asarray(bin_centers, dtype=float)

    monkeypatch.setattr(pyrecest_models, "_coerce_candidate_goals", stale_coerce_candidate_goals)

    hipporeplayimm.apply_runtime_patches()

    goals = pyrecest_models._coerce_candidate_goals(None, np.array([0.0, 1.0], dtype=float))
    assert pyrecest_models._coerce_candidate_goals is not stale_coerce_candidate_goals
    assert goals.shape == (2, 1)
