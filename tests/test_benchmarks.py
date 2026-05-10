from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.benchmarks as benchmarks
from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    _build_models,
    _score_session,
    bootstrap_delta_ci,
)
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel


def test_benchmark_summary_and_bootstrap_ci():
    rows = pd.DataFrame(
        {
            "model": ["diffusion", "imm", "diffusion", "imm"],
            "heldout_log_likelihood": [-10.0, -9.0, -12.0, -10.0],
            "delta_vs_best_static": [0.0, 1.0, 0.0, 2.0],
            "bits_per_spike_vs_best_static": [0.0, 0.1, 0.0, 0.2],
        }
    )
    result = BenchmarkResult(rows)
    summary = result.summary()
    ci = bootstrap_delta_ci(rows, model="imm", n_bootstrap=100, random_seed=0)

    assert set(summary["model"]) == {"diffusion", "imm"}
    assert np.isfinite(ci[0])
    assert np.isfinite(ci[1])


def test_build_models_includes_opt_in_pyrecest_model():
    models = _build_models(
        BenchmarkConfig(
            models=("pyrecest-goal-particle",),
            pyrecest_particles=64,
            pyrecest_position_proposal_probability=0.5,
        )
    )

    assert set(models) == {"pyrecest-goal-particle"}
    assert models["pyrecest-goal-particle"].position_proposal_probability == 0.5


def test_build_models_includes_opt_in_pyrecest_imm_model():
    models = _build_models(
        BenchmarkConfig(models=("pyrecest-goal-particle-imm",), pyrecest_particles=64)
    )

    assert set(models) == {"pyrecest-goal-particle-imm"}


class _RecordingMomentumStateSpaceModel(SortedSpikeStateSpaceReplayModel):
    def __init__(self) -> None:
        super().__init__(mode="momentum")
        self.candidates = [np.array([0], dtype=int), np.array([1], dtype=int)]
        self.score_candidate_arguments: list[list[np.ndarray] | None] = []
        self.candidate_calls = 0

    def candidate_indices(self, emissions: LogEmissionTensor) -> list[np.ndarray]:
        del emissions
        self.candidate_calls += 1
        return self.candidates

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        del bin_centers
        self.score_candidate_arguments.append(candidate_indices)
        return EventScore(
            "recording-state-space-momentum",
            log_likelihood=float(emissions.n_spikes),
            n_time=emissions.n_time,
            n_spikes=emissions.n_spikes,
        )


def test_score_session_reuses_train_candidates_for_state_space_momentum(monkeypatch):
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.0, 0.0], [1.0, 0.0]]),
        rates_hz=np.ones((2, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(),
    )
    model = _RecordingMomentumStateSpaceModel()

    def fake_build_emissions(session, selected_encoding, event_index, emission_config):
        del session, event_index, emission_config
        return LogEmissionTensor(
            log_likelihood=np.log(np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)),
            spike_counts=np.zeros((2, selected_encoding.n_cells), dtype=int),
            times=np.array([0.0, 1.0]),
            dt=1.0,
            cell_ids=selected_encoding.cell_ids,
            n_spikes=int(selected_encoding.n_cells),
        )

    monkeypatch.setattr(benchmarks, "fit_place_field_encoding", lambda session, config: encoding)
    monkeypatch.setattr(benchmarks, "build_emissions", fake_build_emissions)
    monkeypatch.setattr(
        benchmarks,
        "_build_models",
        lambda config, session=None: {"sorted-spike-state-space-momentum": model},
    )
    session = SimpleNamespace(session_id="Rat1/Open1", ripple_indices_in_run=lambda: np.array([0], dtype=int))

    rows = _score_session(session, BenchmarkConfig(models=("sorted-spike-state-space-momentum",)))

    assert len(rows) == 1
    assert model.candidate_calls == 1
    assert len(model.score_candidate_arguments) == 2
    assert all(candidate_argument is model.candidates for candidate_argument in model.score_candidate_arguments)


def test_state_space_aliases_canonicalize_sorted_spike_model_name():
    models = _build_models(BenchmarkConfig(models=("state-space-diffusion",)))

    assert models["state-space-diffusion"].name == "sorted-spike-state-space-diffusion"
