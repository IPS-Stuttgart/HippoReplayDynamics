from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import add_shuffle_p_values, shuffled_encoding


SHUFFLE_SUMMARY_COLUMNS = {
    "shuffle_p_value",
    "shuffle_log_evidence_median",
    "shuffle_log_evidence_mean",
    "shuffle_log_evidence_std",
    "shuffle_count",
}


def test_independent_spatial_permutation_handles_empty_cell_set():
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.empty((0, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )

    control = shuffled_encoding(
        encoding,
        mode="independent-spatial-permutation",
        random_seed=7,
    )

    assert control.rates_hz.shape == (0, 2)
    assert control.rates_hz.dtype == float
    np.testing.assert_array_equal(control.cell_ids, np.array([], dtype=int))
    np.testing.assert_allclose(control.occupancy_s, np.ones(2, dtype=float))


def test_add_shuffle_p_values_preserves_schema_when_control_scores_empty() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [12.5],
        }
    )
    control_scores = pd.DataFrame(columns=["session", "event_index", "model", "log_evidence"])

    out = add_shuffle_p_values(real_scores, control_scores)

    assert SHUFFLE_SUMMARY_COLUMNS.issubset(out.columns)
    assert np.isnan(out.loc[0, "shuffle_p_value"])
    assert np.isnan(out.loc[0, "shuffle_log_evidence_median"])
    assert np.isnan(out.loc[0, "shuffle_log_evidence_mean"])
    assert np.isnan(out.loc[0, "shuffle_log_evidence_std"])
    assert np.isnan(out.loc[0, "shuffle_count"])


def test_add_shuffle_p_values_preserves_schema_when_real_scores_empty() -> None:
    real_scores = pd.DataFrame(columns=["session", "event_index", "model", "log_evidence"])
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [11.0],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert out.empty
    assert SHUFFLE_SUMMARY_COLUMNS.issubset(out.columns)


def test_add_shuffle_p_values_returns_nan_for_nonfinite_real_log_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [np.nan],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["sorted-spike-state-space-first-order-imm", "sorted-spike-state-space-first-order-imm"],
            "log_evidence": [11.0, 12.0],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert np.isnan(out.loc[0, "shuffle_p_value"])
    assert out.loc[0, "shuffle_count"] == 2


def test_add_shuffle_p_values_ignores_nonfinite_control_log_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [10.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 5,
            "event_index": [3] * 5,
            "model": ["sorted-spike-state-space-first-order-imm"] * 5,
            "log_evidence": [12.0, 8.0, np.nan, np.inf, -np.inf],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert np.isclose(out.loc[0, "shuffle_p_value"], 2.0 / 3.0)
    assert out.loc[0, "shuffle_count"] == 2
    assert out.loc[0, "shuffle_log_evidence_median"] == 10.0
