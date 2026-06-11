from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.pyrecest_models import PyRecEstGoalParticleModel


def test_pyrecest_is_not_a_core_install_dependency() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    core_dependency_names = {
        _requirement_name(dependency)
        for dependency in pyproject["project"].get("dependencies", [])
    }
    pyrecest_extra_names = {
        _requirement_name(dependency)
        for dependency in pyproject["project"]["optional-dependencies"]["pyrecest"]
    }

    assert "pyrecest" not in core_dependency_names
    assert "pyrecest" in pyrecest_extra_names


def test_pyrecest_model_reports_install_hint_when_extra_is_missing() -> None:
    if importlib.util.find_spec("pyrecest") is not None:
        pytest.skip("PyRecEst is installed in this environment")

    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.3, 0.7]])),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        n_particles=8,
        random_seed=0,
    )

    with pytest.raises(RuntimeError, match="hipporeplayimm\\[pyrecest\\]"):
        model.score(emissions, np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_pyrecest_missing_extra_does_not_reset_numpy_rng() -> None:
    if importlib.util.find_spec("pyrecest") is not None:
        pytest.skip("PyRecEst is installed in this environment")

    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.3, 0.7]])),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    model = PyRecEstGoalParticleModel(
        candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]),
        n_particles=8,
        random_seed=0,
    )

    np.random.seed(12345)
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="hipporeplayimm\\[pyrecest\\]"):
        model.score(emissions, np.array([[0.0, 0.0], [1.0, 0.0]]))
    after = np.random.get_state()

    _assert_numpy_rng_states_equal(after, before)


def test_exact_sparse_momentum_runs_without_pyrecest_extra() -> None:
    if importlib.util.find_spec("pyrecest") is not None:
        pytest.skip("PyRecEst is installed in this environment")

    from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
    from hipporeplayimm.state_space import StateSpaceDecoderConfig

    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.08, 0.02],
                    [0.15, 0.65, 0.15, 0.05],
                    [0.05, 0.15, 0.65, 0.15],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.003, 0.006]),
        dt=0.003,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    model = SortedSpikeStateSpaceReplayModel(
        mode="momentum-exact-sparse",
        config=StateSpaceDecoderConfig(mode="momentum-exact-sparse"),
    )

    full = model.score(emissions, centers, return_trajectory=True)
    evidence_only = model.score(emissions, centers, return_trajectory=False)

    assert np.isfinite(full.log_likelihood)
    assert evidence_only.log_likelihood == pytest.approx(full.log_likelihood, abs=1e-12)
    assert full.trajectory_log_posterior is not None
    assert evidence_only.trajectory_log_posterior is None


def _requirement_name(requirement: str) -> str:
    return requirement.split(";", 1)[0].split("@", 1)[0].strip().lower()


def _assert_numpy_rng_states_equal(left: tuple[object, ...], right: tuple[object, ...]) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2] == right[2]
    assert left[3] == right[3]
    assert left[4] == right[4]
