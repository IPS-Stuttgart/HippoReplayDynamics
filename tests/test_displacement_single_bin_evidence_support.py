from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import (
    DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)
from hipporeplayimm.state_space import StateSpaceReplayModel


@pytest.mark.parametrize(
    ("mode", "diagnostic_key"),
    [
        (
            "displacement-momentum",
            "state_space_displacement_momentum_evidence_support",
        ),
        (
            "displacement-imm",
            "state_space_displacement_imm_evidence_support",
        ),
    ],
)
def test_displacement_path_models_mark_single_bin_evidence_degenerate(
    mode: str,
    diagnostic_key: str,
) -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.2, 0.5, 0.3]], dtype=float)),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.01,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )

    score = StateSpaceReplayModel(mode=mode).score(emissions, bin_centers)

    assert score.diagnostics[diagnostic_key] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert score.diagnostics[diagnostic_key.replace("evidence_support", "required_min_time_bins")] == 2

    scored = ensure_evidence_support_columns(
        pd.DataFrame(
            [
                {
                    "status": "success",
                    "model": score.model_name,
                    "log_evidence": score.log_likelihood,
                    f"diagnostic_{diagnostic_key}": score.diagnostics[diagnostic_key],
                }
            ]
        )
    )

    assert scored.loc[0, "evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])
