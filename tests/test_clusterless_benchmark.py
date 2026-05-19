from types import SimpleNamespace

import numpy as np

from hipporeplayimm import benchmarks
from hipporeplayimm.clusterless import ClusterlessStateSpaceReplayModel
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig, LogEmissionTensor


class _DummyEncoding:
    cell_ids = np.array([1, 2], dtype=int)
    bin_centers = np.array([[0.0, 0.0], [10.0, 0.0]])

    def select_cells(self, cell_ids):
        return self


def test_clusterless_benchmark_scores_train_and_joint_with_train_encoder(monkeypatch):
    session = _two_cell_clusterless_session()
    train_encoder = SimpleNamespace(bin_centers=np.array([[0.0, 0.0], [10.0, 0.0]]))
    fitted_cell_sets = []
    emission_calls = []

    monkeypatch.setattr(benchmarks, "fit_place_field_encoding", lambda session, config: _DummyEncoding())
    monkeypatch.setattr(benchmarks, "build_emissions", lambda *args, **kwargs: _dummy_emissions(0))
    monkeypatch.setattr(
        benchmarks,
        "_build_models",
        lambda config, session=None: {
            "clusterless-state-space-stationary": ClusterlessStateSpaceReplayModel(mode="stationary")
        },
    )

    def fake_fit_clusterless_mark_encoding(session, config):
        assert session.spike_marks is not None
        fitted_cell_sets.append(tuple(np.unique(session.spike_marks.cell_ids).tolist()))
        return train_encoder

    def fake_build_clusterless_mark_emissions(session, encoding, ripple, config):
        assert session.spike_marks is not None
        emission_calls.append((tuple(np.unique(session.spike_marks.cell_ids).tolist()), encoding))
        return _dummy_emissions(session.spike_marks.times.size)

    def fake_score_train_joint_model(model, train_emissions, joint_emissions, bin_centers):
        assert bin_centers is train_encoder.bin_centers
        return (
            SimpleNamespace(log_likelihood=1.0, model_name=model.name, diagnostics={}),
            SimpleNamespace(log_likelihood=3.0, model_name=model.name, diagnostics={}),
        )

    monkeypatch.setattr(benchmarks, "fit_clusterless_mark_encoding", fake_fit_clusterless_mark_encoding)
    monkeypatch.setattr(benchmarks, "build_clusterless_mark_emissions", fake_build_clusterless_mark_emissions)
    monkeypatch.setattr(benchmarks, "_score_train_joint_model", fake_score_train_joint_model)

    rows = benchmarks._score_session(
        session,
        benchmarks.BenchmarkConfig(
            encoding=EncodingConfig(use_excitatory=False),
            test_cell_fraction=0.5,
            max_events_per_session=1,
            random_seed=0,
            models=("clusterless-state-space-stationary",),
        ),
    )

    assert len(rows) == 1
    assert len(fitted_cell_sets) == 1
    assert fitted_cell_sets[0] in {(1,), (2,)}
    assert len(emission_calls) == 2
    assert emission_calls[0][0] == fitted_cell_sets[0]
    assert emission_calls[1][0] == (1, 2)
    assert emission_calls[0][1] is train_encoder
    assert emission_calls[1][1] is train_encoder
    assert rows[0]["heldout_log_likelihood"] == 2.0


def _dummy_emissions(n_spikes: int) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.array([[n_spikes]], dtype=int),
        times=np.array([4.5], dtype=float),
        dt=1.0,
        cell_ids=np.array([0], dtype=int),
        n_spikes=int(n_spikes),
    )


def _two_cell_clusterless_session() -> ReplaySession:
    position_times = np.linspace(0.0, 5.0, 51)
    x = np.where(position_times < 2.5, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.2, 0.4, 0.6, 3.2, 3.4, 3.6, 4.2, 4.4])
    cell_ids = np.array([1, 1, 1, 2, 2, 2, 1, 2], dtype=int)
    marks = np.array([[0.0], [0.1], [-0.1], [10.0], [10.2], [9.8], [0.05], [9.9]])
    spikes = np.column_stack([mark_times, cell_ids])
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1], [1, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[4.0, 5.0, 4.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 5.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=marks,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
        ),
    )
