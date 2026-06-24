from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.ground_truth as ground_truth
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, LogEmissionTensor
from hipporeplayimm.models import EventScore


@pytest.mark.parametrize("include_bma", [False, "False", "0", 0])
def test_compare_scores_to_ground_truth_respects_disabled_bma(monkeypatch, include_bma) -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "event_index": [0, 0],
            "model": ["left", "right"],
            "requested_model": ["left", "right"],
            "status": ["success", "success"],
            "evidence_support": ["exact_full_grid", "exact_full_grid"],
            "log_evidence": [0.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "true_well_id": [np.nan],
            "true_well_x": [np.nan],
            "true_well_y": [np.nan],
            "valid_label": [False],
        }
    )
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.ones(1, dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )

    class FakeModel:
        def __init__(self, name: str):
            self.name = name

        def score(self, emissions, bin_centers):
            del bin_centers
            return EventScore(
                self.name,
                0.0,
                emissions.n_time,
                emissions.n_spikes,
                terminal_log_posterior=np.log(np.array([1.0], dtype=float)),
            )

    def fake_build_emissions(session, selected_encoding, event_index, emission_config):
        del session, selected_encoding, event_index, emission_config
        return LogEmissionTensor(
            log_likelihood=np.zeros((1, 1), dtype=float),
            spike_counts=np.zeros((1, 1), dtype=int),
            times=np.array([0.0], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=0,
        )

    monkeypatch.setattr(ground_truth, "load_open_field_sessions", lambda root: [SimpleNamespace(session_id="s1")])
    monkeypatch.setattr(ground_truth, "fit_place_field_encoding", lambda session, config: encoding)
    monkeypatch.setattr(ground_truth, "build_emissions", fake_build_emissions)
    monkeypatch.setattr(
        ground_truth,
        "infer_well_locations",
        lambda session, config=None: pd.DataFrame(columns=["well_id", "well_x", "well_y"]),
    )
    monkeypatch.setattr(
        ground_truth,
        "_build_models",
        lambda config, session=None: {"left": FakeModel("left"), "right": FakeModel("right")},
    )

    comparison = ground_truth.compare_scores_to_ground_truth(
        "unused-root",
        scores,
        ground_truth=labels,
        include_bayesian_model_average=include_bma,
    )

    assert set(comparison["model"]) == {"left", "right"}
