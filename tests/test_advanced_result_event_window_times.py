from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def _valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_index": [0],
            "start": [1.0],
            "end": [1.02],
        }
    )


def test_event_window_variants_rejects_reversed_time_bounds() -> None:
    events = pd.DataFrame(
        {
            "event_index": [0],
            "start": [2.0],
            "end": [1.0],
        }
    )

    with pytest.raises(ValueError, match="end.*greater than start"):
        diagnostics.event_window_variants(events)


def test_event_window_variants_rejects_nonfinite_event_times() -> None:
    events = pd.DataFrame(
        {
            "event_index": [0],
            "start": [float("nan")],
            "end": [1.0],
        }
    )

    with pytest.raises(ValueError, match="start.*finite scalar"):
        diagnostics.event_window_variants(events)


def test_event_window_variants_rejects_negative_padding() -> None:
    with pytest.raises(ValueError, match="paddings_s"):
        diagnostics.event_window_variants(_valid_events(), paddings_s=(-0.01,))


def test_event_window_variants_rejects_nonpositive_min_duration() -> None:
    with pytest.raises(ValueError, match="min_duration_s"):
        diagnostics.event_window_variants(_valid_events(), min_duration_s=0.0)


def test_event_window_variants_valid_windows_remain_positive_duration() -> None:
    windows = diagnostics.event_window_variants(_valid_events(), paddings_s=(0.0, 0.01))

    assert not windows.empty
    assert (windows["window_end"] > windows["window_start"]).all()


def test_runtime_patches_restore_stale_event_window_validation_alias() -> None:
    def _legacy_event_window_variants(
        events: pd.DataFrame,
        *,
        start_col: str = "start",
        end_col: str = "end",
        event_id_col: str = "event_index",
        paddings_s=(0.0, 0.01, 0.02),
        min_duration_s: float = 0.003,
    ) -> pd.DataFrame:
        rows = []
        for _, event in events.iterrows():
            start = float(event[start_col])
            end = float(event[end_col])
            for padding in paddings_s:
                rows.append(
                    {
                        event_id_col: int(event[event_id_col]),
                        "window_variant": f"pad_{float(padding):.3f}s",
                        "window_start": max(start - float(padding), 0.0),
                        "window_end": end + float(padding),
                        "padding_s": float(padding),
                    }
                )
        return pd.DataFrame(rows)

    diagnostics.event_window_variants = _legacy_event_window_variants

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="end.*greater than start"):
        diagnostics.event_window_variants(
            pd.DataFrame(
                {
                    "event_index": [0],
                    "start": [2.0],
                    "end": [1.0],
                }
            )
        )
