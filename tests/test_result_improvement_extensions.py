from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel,
    _sorted_spike_counts_for_edges,
    add_model_averaged_endpoint_columns,
)


class _PosteriorOnlyModel:
    def __init__(self, name: str, terminal_log_posterior: np.ndarray) -> None:
        self.name = name
        self.terminal_log_posterior = np.asarray(terminal_log_posterior, dtype=float)

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        del bin_centers
        return EventScore(
            model_name=self.name,
            log_likelihood=0.0,
            n_time=emissions.n_time,
            n_spikes=emissions.n_spikes,
            diagnostics={
                "decoded_endpoint_x": float(np.argmax(self.terminal_log_posterior)),
                "decoded_endpoint_y": 0.0,
            },
            terminal_log_posterior=self.terminal_log_posterior.copy(),
            trajectory_log_posterior=None,
        )


def test_model_averaged_endpoint_accepts_legacy_tables_without_comparability_column() -> None:
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "a",
                "log_evidence": 0.0,
                "model_probability": 0.25,
                "diagnostic_decoded_endpoint_x": 10.0,
                "diagnostic_decoded_endpoint_y": 0.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "b",
                "log_evidence": 1.0,
                "model_probability": 0.75,
                "diagnostic_decoded_endpoint_x": 20.0,
                "diagnostic_decoded_endpoint_y": 10.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    assert np.allclose(out["model_averaged_endpoint_x"], 17.5)
    assert np.allclose(out["model_averaged_endpoint_y"], 7.5)
    assert (out["model_averaged_endpoint_models"] == 2).all()


def test_model_averaged_endpoint_uses_only_comparable_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "exact",
                "log_evidence": 0.0,
                "model_probability": 1.0,
                "evidence_comparable": True,
                "diagnostic_decoded_endpoint_x": 3.0,
                "diagnostic_decoded_endpoint_y": 4.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "truncated",
                "log_evidence": 100.0,
                "model_probability": 1.0,
                "evidence_comparable": False,
                "diagnostic_decoded_endpoint_x": 999.0,
                "diagnostic_decoded_endpoint_y": 999.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    assert np.allclose(out["model_averaged_endpoint_x"], 3.0)
    assert np.allclose(out["model_averaged_endpoint_y"], 4.0)
    assert (out["model_averaged_endpoint_models"] == 1).all()


def test_model_averaged_endpoint_parses_string_false_comparable_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "exact",
                "log_evidence": 0.0,
                "model_probability": 1.0,
                "evidence_comparable": "True",
                "diagnostic_decoded_endpoint_x": 3.0,
                "diagnostic_decoded_endpoint_y": 4.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "truncated",
                "log_evidence": 100.0,
                "model_probability": 1.0,
                "evidence_comparable": "False",
                "diagnostic_decoded_endpoint_x": 999.0,
                "diagnostic_decoded_endpoint_y": 999.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    assert np.allclose(out["model_averaged_endpoint_x"], 3.0)
    assert np.allclose(out["model_averaged_endpoint_y"], 4.0)
    assert (out["model_averaged_endpoint_models"] == 1).all()


def test_sorted_spike_count_edges_respect_encoding_cell_order() -> None:
    class Session:
        spikes = np.array(
            [
                [0.05, 10.0],
                [0.15, 2.0],
                [0.15, 99.0],
            ],
            dtype=float,
        )

    class Encoding:
        # External or hand-built encodings should not have to sort cell IDs for
        # replay-calibrated emission counts to stay aligned with rates_hz rows.
        cell_ids = np.array([10, 2], dtype=int)
        n_cells = 2

    counts = _sorted_spike_counts_for_edges(
        Session(),
        Encoding(),
        np.array([0.0, 0.1, 0.2], dtype=float),
    )

    np.testing.assert_array_equal(
        counts,
        np.array(
            [
                [1, 0],
                [0, 1],
            ],
            dtype=int,
        ),
    )


def test_bidirectional_replay_model_diagnostics_match_mixture_posterior() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
    forward = _PosteriorOnlyModel("forward", np.array([0.0, -np.inf], dtype=float))
    reverse = _PosteriorOnlyModel("reverse", np.array([-np.inf, 0.0], dtype=float))

    score = BidirectionalReplayModel(forward, reverse, name="bidirectional").score(emissions, bin_centers)

    assert score.terminal_log_posterior is not None
    np.testing.assert_allclose(np.exp(score.terminal_log_posterior), np.array([0.5, 0.5]))
    assert np.isclose(score.diagnostics["decoded_endpoint_x"], 5.0)
    assert np.isclose(score.diagnostics["decoded_endpoint_y"], 0.0)
