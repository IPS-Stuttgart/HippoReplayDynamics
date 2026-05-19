from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipporeplayimm import benchmarks


class _DummyEncoding:
    cell_ids = np.array([1, 2], dtype=int)
    bin_centers = np.array([[0.0, 0.0]], dtype=float)

    def select_cells(self, _cell_ids):
        return self


class _DummyScore:
    def __init__(self, log_likelihood: float):
        self.log_likelihood = log_likelihood
        self.model_name = "clusterless-state-space-diffusion"
        self.diagnostics = {}


class _DummyClusterlessModel:
    def score(self, emissions, _bin_centers):
        return _DummyScore(float(emissions.n_spikes))


def test_clusterless_heldout_encoder_is_fit_on_train_marks_only(monkeypatch):
    """Clusterless held-out scoring must not fit mark parameters on test marks."""

    clusterless_model = _DummyClusterlessModel()
    fit_roles: list[str] = []
    emission_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(benchmarks, "fit_place_field_encoding", lambda _session, _config: _DummyEncoding())
    monkeypatch.setattr(
        benchmarks,
        "_split_cells",
        lambda _cell_ids, _test_fraction, _random_seed: (np.array([1], dtype=int), np.array([2], dtype=int)),
    )
    monkeypatch.setattr(benchmarks, "_event_indices", lambda _session, _config: np.array([0], dtype=int))
    monkeypatch.setattr(
        benchmarks,
        "_build_models",
        lambda _config, _session=None: {"clusterless-state-space-diffusion": clusterless_model},
    )
    monkeypatch.setattr(benchmarks, "_is_clusterless_model", lambda model: model is clusterless_model)
    monkeypatch.setattr(
        benchmarks,
        "build_emissions",
        lambda _session, _encoding, _event_index, _config: SimpleNamespace(n_time=1, n_spikes=0),
    )

    def fake_session_with_mark_cell_subset(session, _cell_ids, *, role):
        return SimpleNamespace(session_id=f"{session.session_id}:{role}", role=role)

    monkeypatch.setattr(benchmarks, "_session_with_mark_cell_subset", fake_session_with_mark_cell_subset)

    def fake_fit_clusterless_mark_encoding(session, _config):
        fit_roles.append(session.role)
        return SimpleNamespace(bin_centers=np.array([[0.0, 0.0]], dtype=float), fit_role=session.role)

    monkeypatch.setattr(benchmarks, "fit_clusterless_mark_encoding", fake_fit_clusterless_mark_encoding)

    def fake_build_clusterless_mark_emissions(session, encoding, _event_index, _config):
        emission_calls.append((session.role, encoding.fit_role))
        n_spikes = 1 if session.role == "train" else 3
        return SimpleNamespace(n_time=1, n_spikes=n_spikes)

    monkeypatch.setattr(benchmarks, "build_clusterless_mark_emissions", fake_build_clusterless_mark_emissions)

    config = benchmarks.BenchmarkConfig(models=("clusterless-state-space-diffusion",))
    rows = benchmarks._score_session(SimpleNamespace(session_id="s"), config)

    assert fit_roles == ["train"]
    assert emission_calls == [("train", "train"), ("joint", "train")]
    assert rows[0]["heldout_log_likelihood"] == 2.0
    assert rows[0]["test_spikes"] == 2
