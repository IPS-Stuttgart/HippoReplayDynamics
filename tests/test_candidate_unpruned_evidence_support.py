from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns
from hipporeplayimm.models import CandidateKinematicModel


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.60, 0.30, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.10, 0.30, 0.60],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 2.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


def test_unpruned_candidate_model_reports_exact_full_grid_support() -> None:
    emissions = _emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    score = CandidateKinematicModel(
        mode="diffusion",
        top_k=0,
        diffusion_sigma_cm=1.0,
    ).score(emissions, centers)

    assert score.diagnostics["mean_candidate_log_mass"] == 0.0
    assert score.diagnostics["candidate_evidence_support"] == "exact_full_grid"

    frame = ensure_evidence_support_columns(
        pd.DataFrame(
            [
                {
                    "status": "success",
                    "model": score.model_name,
                    "diagnostic_candidate_evidence_support": score.diagnostics[
                        "candidate_evidence_support"
                    ],
                }
            ]
        )
    )
    assert frame.loc[0, "evidence_support"] == "exact_full_grid"
    assert bool(frame.loc[0, "evidence_comparable"])


def test_finite_candidate_limit_remains_conservatively_truncated() -> None:
    emissions = _emissions()
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    score = CandidateKinematicModel(
        mode="diffusion",
        top_k=2,
        diffusion_sigma_cm=1.0,
    ).score(emissions, centers)

    assert score.diagnostics["candidate_evidence_support"] == "truncated_full_grid"
