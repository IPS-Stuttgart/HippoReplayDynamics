from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_plot_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "plot_replay_event_summary.py"
    spec = importlib.util.spec_from_file_location("plot_replay_event_summary_selection_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_selection_does_not_truncate_fractional_indices() -> None:
    module = _load_plot_module()
    scores = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "event_index": 7.5, "model": "fractional"},
            {"session": "Rat1/Open1", "event_index": 7, "model": "exact"},
        ]
    )

    selected = module._select_event_scores(scores, session="Rat1/Open1", event_index=7)

    assert selected["model"].tolist() == ["exact"]


def test_event_selection_ignores_missing_indices_from_other_sessions() -> None:
    module = _load_plot_module()
    scores = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "event_index": 7, "model": "exact"},
            {"session": "Rat2/Open1", "event_index": pd.NA, "model": "unrelated"},
        ]
    )

    selected = module._select_event_scores(scores, session="Rat1/Open1", event_index=7)

    assert selected["model"].tolist() == ["exact"]


def test_event_selection_rejects_malformed_indices_in_selected_session() -> None:
    module = _load_plot_module()
    scores = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": "not-an-index", "model": "bad"}]
    )

    with pytest.raises(ValueError, match="event_index values for session 'Rat1/Open1' must be numeric"):
        module._select_event_scores(scores, session="Rat1/Open1", event_index=7)
