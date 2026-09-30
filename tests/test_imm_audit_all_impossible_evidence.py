from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_first_order_imm_mode_usage import (  # noqa: E402
    build_first_order_imm_mode_usage_event_table,
)
from audit_imm_fragmented_hypotheses import (  # noqa: E402
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    build_event_table,
)


MODELS = (STATIONARY, DIFFUSION, FRAGMENTED, FIRST_ORDER_IMM, MOMENTUM_EXACT)


def _score(model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 7,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
    }


def test_imm_fragmented_audit_does_not_invent_all_impossible_winner() -> None:
    evidence = pd.DataFrame([_score(model, -np.inf) for model in MODELS])

    event = build_event_table(evidence, threshold=5.5).iloc[0]

    assert bool(event["exact_core_complete"])
    assert event["best_exact_core_model"] == ""
    assert np.isnan(float(event["best_exact_core_log_evidence"]))
    assert np.isnan(float(event["delta_imm_minus_fragmented"]))
    assert event["within_family_classification"] == "trajectory_family_ambiguous"


def test_first_order_mode_audit_does_not_invent_all_impossible_winner() -> None:
    evidence = pd.DataFrame([_score(model, -np.inf) for model in MODELS])

    event = build_first_order_imm_mode_usage_event_table(
        evidence,
        margin_threshold=5.5,
    ).iloc[0]

    assert bool(event["exact_core_complete"])
    assert event["best_exact_core_model"] == ""
    assert np.isnan(float(event["best_exact_core_margin_to_runner_up"]))
    assert not bool(event["first_order_imm_is_best_exact_core"])
    assert not bool(event["first_order_imm_confident_exact_core_best"])


def test_first_order_mode_audit_accepts_finite_vs_impossible_margin() -> None:
    evidence = pd.DataFrame(
        [
            _score(STATIONARY, -np.inf),
            _score(DIFFUSION, -np.inf),
            _score(FRAGMENTED, -np.inf),
            _score(FIRST_ORDER_IMM, 12.0),
            _score(MOMENTUM_EXACT, -np.inf),
        ]
    )

    event = build_first_order_imm_mode_usage_event_table(
        evidence,
        margin_threshold=5.5,
    ).iloc[0]

    assert event["best_exact_core_model"] == FIRST_ORDER_IMM
    assert np.isposinf(float(event["best_exact_core_margin_to_runner_up"]))
    assert bool(event["first_order_imm_is_best_exact_core"])
    assert bool(event["first_order_imm_confident_exact_core_best"])
    assert np.isposinf(float(event["trajectory_capable_minus_stationary"]))
    assert bool(event["trajectory_capable_confident_vs_stationary"])
