from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _valid_row() -> dict[str, object]:
    return {
        "model": "diffusion",
        "status": "success",
        "n_spikes": 4,
        "n_time": 3,
        "mean_candidate_log_mass": 0.0,
        "terminal_posterior_entropy": 0.0,
    }


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("n_spikes", np.bool_(True)),
        ("n_time", np.complex128(3.0 + 4.0j)),
        ("mean_candidate_log_mass", np.complex128(0.0 + 1.0j)),
        ("terminal_posterior_entropy", np.complex128(0.0 + 1.0j)),
    ],
)
def test_nested_lossy_metrics_are_invalid_without_cast_warnings(
    column: str,
    value: object,
) -> None:
    row = _valid_row()
    row[column] = _nested_scalar(value)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flagged = add_event_reliability_flags(pd.DataFrame([row]))

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert flagged.loc[0, "event_reliability_reasons"] == "invalid_numeric_metric"


def test_nested_real_metrics_remain_valid() -> None:
    row = _valid_row()
    row.update(
        {
            "n_spikes": _nested_scalar(np.int64(4)),
            "n_time": _nested_scalar(np.int64(3)),
            "mean_candidate_log_mass": _nested_scalar(np.float64(0.0)),
            "terminal_posterior_entropy": _nested_scalar(np.float64(0.0)),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flagged = add_event_reliability_flags(pd.DataFrame([row]))

    assert bool(flagged.loc[0, "event_reliable"])
    assert not bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert flagged.loc[0, "event_reliability_reasons"] == ""
