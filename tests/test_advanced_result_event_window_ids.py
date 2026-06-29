from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
import hipporeplayimm.advanced_result_diagnostics as diagnostics


def _events(event_ids) -> pd.DataFrame:
    n_events = len(event_ids)
    return pd.DataFrame(
        {
            "event_index": event_ids,
            "start": np.linspace(0.1, 0.1 * n_events, n_events),
            "end": np.linspace(0.2, 0.1 * n_events + 0.1, n_events),
        }
    )


def test_event_window_variants_accepts_integral_event_identifiers() -> None:
    out = diagnostics.event_window_variants(
        _events([0, np.int64(2), 3.0]),
        paddings_s=(0.0,),
    )

    assert out["event_index"].tolist() == [0, 0, 2, 2, 3, 3]


@pytest.mark.parametrize("bad_event_id", [True, np.bool_(False), 1.5, np.nan, pd.NA, "2"])
def test_event_window_variants_rejects_silently_coerced_event_ids(bad_event_id) -> None:
    with pytest.raises(ValueError, match="event_index must contain integer event identifiers"):
        diagnostics.event_window_variants(_events([bad_event_id]), paddings_s=(0.0,))


def test_event_window_variant_patch_refreshes_after_stale_overwrite(monkeypatch) -> None:
    def stale_event_window_variants(*args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        return pd.DataFrame({"event_index": [int(True)]})

    monkeypatch.setattr(diagnostics, "_missing_group_metadata_patch_applied", True, raising=False)
    monkeypatch.setattr(diagnostics, "event_window_variants", stale_event_window_variants)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(diagnostics.event_window_variants, "_missing_group_metadata_event_window_variants_wrapper", False)
    with pytest.raises(ValueError, match="event_index must contain integer event identifiers"):
        diagnostics.event_window_variants(_events([True]), paddings_s=(0.0,))
