from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.accuracy_upgrades import model_probability_diagnostics


def test_model_probability_diagnostics_ignores_nan_evidence_for_best_model() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["nan-evidence", "low", "high"],
            "log_evidence": [np.nan, 1.0, 3.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    diagnostics = model_probability_diagnostics(scores)

    assert diagnostics.shape[0] == 1
    assert diagnostics.loc[0, "models"] == 2
    assert diagnostics.loc[0, "best_model"] == "high"
    assert diagnostics.loc[0, "best_log_evidence"] == 3.0


def test_model_probability_diagnostics_handles_positive_infinite_evidence() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["inf-a", "inf-b", "finite"],
            "log_evidence": [np.inf, np.inf, 3.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    diagnostics = model_probability_diagnostics(scores)

    assert diagnostics.shape[0] == 1
    assert diagnostics.loc[0, "models"] == 3
    assert diagnostics.loc[0, "best_model"] == "inf-a"
    assert diagnostics.loc[0, "best_log_evidence"] == np.inf
    assert diagnostics.loc[0, "evidence_margin_to_second_best"] == 0.0
    assert diagnostics.loc[0, "best_model_probability"] == 0.5
    assert np.isclose(diagnostics.loc[0, "model_probability_entropy"], np.log(2.0))


def test_model_probability_diagnostics_handles_all_negative_infinite_evidence() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["left", "right"],
            "log_evidence": [-np.inf, -np.inf],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )

    diagnostics = model_probability_diagnostics(scores)

    assert diagnostics.shape[0] == 1
    assert diagnostics.loc[0, "models"] == 2
    assert diagnostics.loc[0, "best_model"] == "left"
    assert diagnostics.loc[0, "best_log_evidence"] == -np.inf
    assert diagnostics.loc[0, "evidence_margin_to_second_best"] == 0.0
    assert diagnostics.loc[0, "best_model_probability"] == 0.5
    assert np.isclose(diagnostics.loc[0, "model_probability_entropy"], np.log(2.0))
