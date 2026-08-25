from __future__ import annotations

import importlib

import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import evidence_reporting, simulation_recovery
from hipporeplayimm.simulation_best_row_flags import _PATCHED_FLAG


def _row(seed: int, model: str, log_evidence: float) -> dict[str, object]:
    true_model = "stationary"
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "simulation_random_seed": seed,
        "simulation_event_index": 0,
        "event_index": 0,
        "true_model": true_model,
        "expected_model": simulation_recovery.expected_scoring_model(true_model),
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }


def test_runtime_patch_refresh_restores_simulation_event_scope_after_reporting_reload() -> None:
    assert getattr(evidence_reporting.simulation_add_evidence_columns, _PATCHED_FLAG, False)
    assert getattr(evidence_reporting.simulation_event_best_rows, _PATCHED_FLAG, False)

    importlib.reload(evidence_reporting)

    # importlib.reload() retains dynamically added module attributes, so the
    # historical module-level sentinel remains true even though the source
    # functions have been replaced by fresh, unpatched definitions.
    assert getattr(evidence_reporting, _PATCHED_FLAG, False)
    assert not getattr(evidence_reporting.simulation_add_evidence_columns, _PATCHED_FLAG, False)
    assert not getattr(evidence_reporting.simulation_event_best_rows, _PATCHED_FLAG, False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(evidence_reporting.simulation_add_evidence_columns, _PATCHED_FLAG, False)
    assert getattr(evidence_reporting.simulation_event_best_rows, _PATCHED_FLAG, False)
    assert simulation_recovery.add_evidence_columns is evidence_reporting.simulation_add_evidence_columns
    assert simulation_recovery._event_best_rows is evidence_reporting.simulation_event_best_rows

    rows = pd.DataFrame(
        [
            _row(1, "sorted-spike-state-space-stationary", 0.0),
            _row(1, "sorted-spike-state-space-diffusion", -4.0),
            _row(2, "sorted-spike-state-space-stationary", -4.0),
            _row(2, "sorted-spike-state-space-diffusion", 0.0),
        ]
    )
    scored = evidence_reporting.simulation_add_evidence_columns(rows)
    probability_mass = scored.groupby("simulation_random_seed")["model_probability"].sum()
    assert probability_mass.to_dict() == pytest.approx({1: 1.0, 2: 1.0})

    best = evidence_reporting.simulation_event_best_rows(scored).sort_values(
        "simulation_random_seed"
    )
    assert best["simulation_random_seed"].tolist() == [1, 2]
    assert best["model"].tolist() == [
        "sorted-spike-state-space-stationary",
        "sorted-spike-state-space-diffusion",
    ]

    refreshed_add_evidence_columns = evidence_reporting.simulation_add_evidence_columns
    refreshed_best_rows = evidence_reporting.simulation_event_best_rows
    hipporeplayimm.apply_runtime_patches()
    assert evidence_reporting.simulation_add_evidence_columns is refreshed_add_evidence_columns
    assert evidence_reporting.simulation_event_best_rows is refreshed_best_rows
    assert simulation_recovery.add_evidence_columns is refreshed_add_evidence_columns
    assert simulation_recovery._event_best_rows is refreshed_best_rows
