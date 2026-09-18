import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from build_replay_dynamics_axis import (  # noqa: E402
    _safe_softmax as axis_softmax,
    build_event_dynamics_axis,
)
from causal_trajectory_family_detector import (  # noqa: E402
    _safe_softmax as causal_softmax,
    causal_replay_detection_time_bin_table,
)
from off_swr_trajectory_discovery import _safe_softmax as off_swr_softmax  # noqa: E402
from replay_behavior_alignment import (  # noqa: E402
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    _safe_softmax as behavior_softmax,
    build_event_evidence_features,
)


SOFTMAX_IMPLEMENTATIONS = (
    axis_softmax,
    behavior_softmax,
    off_swr_softmax,
    causal_softmax,
)


@pytest.mark.parametrize("softmax", SOFTMAX_IMPLEMENTATIONS)
def test_log_evidence_softmax_treats_negative_infinity_as_zero_evidence(softmax):
    probabilities = softmax([0.0, -np.inf, -2.0])

    expected_denominator = 1.0 + np.exp(-2.0)
    assert probabilities == pytest.approx(
        [1.0 / expected_denominator, 0.0, np.exp(-2.0) / expected_denominator]
    )
    assert probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("softmax", SOFTMAX_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "values",
    (
        [0.0, np.nan],
        [0.0, np.inf],
        [-np.inf, -np.inf],
    ),
)
def test_log_evidence_softmax_rejects_undefined_normalizations(softmax, values):
    assert np.isnan(softmax(values)).all()


def _exact_core_rows(*, stationary: float) -> pd.DataFrame:
    log_evidence = {
        STATIONARY: stationary,
        DIFFUSION: 0.0,
        FRAGMENTED: -1.0,
        FIRST_ORDER_IMM: -2.0,
        MOMENTUM_EXACT: -3.0,
    }
    return pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_role": "real",
                "null_index": -1,
                "prefix_time_bin_index": 1,
                "model": model,
                "log_evidence": value,
                "status": "success",
                "evidence_comparable": True,
            }
            for model, value in log_evidence.items()
        ]
    )


def test_replay_behavior_features_preserve_negative_infinite_static_evidence():
    features = build_event_evidence_features(_exact_core_rows(stationary=-np.inf))

    row = features.iloc[0]
    assert row["posterior_static"] == 0.0
    assert row["trajectory_family_posterior_mass"] == pytest.approx(1.0)
    assert np.isposinf(row["trajectory_minus_nontrajectory_log_evidence"])


def test_replay_dynamics_axis_preserves_negative_infinite_static_evidence():
    axis = build_event_dynamics_axis(_exact_core_rows(stationary=-np.inf))

    row = axis.iloc[0]
    assert row["P_stationary"] == 0.0
    assert row["P_trajectory_family"] == pytest.approx(1.0)
    assert np.isposinf(row["trajectory_family_margin"])


def test_causal_detector_preserves_negative_infinite_static_evidence():
    table = causal_replay_detection_time_bin_table(
        _exact_core_rows(stationary=-np.inf),
        margin_threshold=5.0,
    )

    row = table.iloc[0]
    assert row["p_static"] == 0.0
    assert row["p_trajectory_family"] == pytest.approx(1.0)
    assert np.isposinf(row["trajectory_minus_nontrajectory_log_evidence"])
    assert row["causal_label"] == "trajectory_family"
