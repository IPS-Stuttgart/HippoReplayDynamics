from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_model_evidence_support as legacy_audit  # noqa: E402
import benchmark_kd_model_evidence as kd_benchmark  # noqa: E402
import compare_model_evidence_artifacts as artifact_compare  # noqa: E402

from hipporeplayimm.duration_dynamics import DurationFloat  # noqa: E402
from hipporeplayimm.evidence_reporting import (  # noqa: E402
    DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
    EXACT_EVIDENCE_SUPPORT,
    PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)


def test_pyrecest_rows_are_not_marked_exact_comparable() -> None:
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "pyrecest-goal-particle",
                "log_evidence": 1.0,
                "diagnostic_pyrecest_evidence_support": PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
            }
        ]
    )

    out = ensure_evidence_support_columns(rows)

    assert out.loc[0, "evidence_support"] == PYRECEST_PARTICLE_EVIDENCE_SUPPORT
    assert not bool(out.loc[0, "evidence_comparable"])


def test_legacy_audit_uses_canonical_support_inference_for_degenerate_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "diffusion",
                "log_evidence": 0.0,
                "diagnostic_candidate_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
            }
        ]
    )

    out = legacy_audit.ensure_evidence_support_columns(rows)

    assert out.loc[0, "evidence_support"] == DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    assert not bool(out.loc[0, "evidence_comparable"])


def test_artifact_exact_only_recomputes_relative_evidence_after_filtering(tmp_path: Path) -> None:
    score_path = tmp_path / "event_model_evidence.csv"
    pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "random",
                "log_evidence": 10.0,
                "relative_log_evidence": 0.0,
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "stationary",
                "log_evidence": 9.0,
                "relative_log_evidence": -1.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "diffusion",
                "log_evidence": 8.0,
                "relative_log_evidence": -2.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
            },
        ]
    ).to_csv(score_path, index=False)

    out = artifact_compare.load_scores(score_path, "run", exact_only=True)

    by_model = out.set_index("model")["relative_log_evidence"].to_dict()
    assert by_model == {"stationary": 0.0, "diffusion": -1.0}


def test_kd_momentum_chunk_uses_first_transition_duration_for_initial_velocity(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    def fake_diffusion_transition(n_bins, sd_meters, bin_size_cm, dt):
        recorded["initial_sd_meters"] = float(sd_meters)
        return np.eye(n_bins)

    def fake_momentum_transition(n_bins, sd_meters, decay, bin_size_cm, dt):
        recorded["transition_dt"] = dt
        return np.ones((n_bins, n_bins, n_bins), dtype=float)

    monkeypatch.setattr(kd_benchmark, "diffusion_transition_1d", fake_diffusion_transition)
    monkeypatch.setattr(kd_benchmark, "momentum_transition_1d", fake_momentum_transition)
    monkeypatch.setattr(kd_benchmark, "kd_momentum_log_evidence_from_transitions", lambda *args: 0.0)

    dt = DurationFloat(0.03, [0.01])
    emissions = SimpleNamespace(
        n_time=2,
        log_likelihood=np.zeros((2, 4), dtype=float),
        dt=dt,
    )

    kd_benchmark._score_momentum_param_chunk(
        (0, 0, 0.5, 1.0),
        [emissions],
        initial_sd_m_per_s=10.0,
        n_bins=2,
        bin_size_cm=6.0,
    )

    assert np.isclose(recorded["initial_sd_meters"], 0.1)
    assert recorded["transition_dt"] is dt
