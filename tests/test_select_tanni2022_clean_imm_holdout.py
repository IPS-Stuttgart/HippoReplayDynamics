from __future__ import annotations

import pandas as pd

from scripts.select_tanni2022_clean_imm_holdout import (
    select_holdout_events,
    selection_gates,
)


def test_selection_is_balanced_pre_evidence_and_excludes_prior_events() -> None:
    rows = []
    for animal in ("RatA", "RatB"):
        for event_index in range(5):
            rows.append(
                {
                    "animal": animal,
                    "session": f"{animal}_session",
                    "event_index": event_index,
                    "window_start_time_s": float(event_index),
                    "window_end_time_s": float(event_index) + 0.2,
                    "peak_time_s": float(event_index) + 0.1,
                    "peak_ripple_z": 10.0 + event_index,
                    "n_spikes": 20 + event_index,
                    "n_active_cells": 5 + event_index,
                    "immobile": True,
                    "spike_supported": True,
                    "selected_for_decoding": True,
                    "event_definition": "synthetic",
                }
            )
    candidates = pd.DataFrame(rows)
    prior = candidates.iloc[[4]][["animal", "session", "event_index"]]

    selected, by_animal = select_holdout_events(
        candidates,
        prior,
        events_per_animal=2,
    )
    gates = selection_gates(
        selected,
        by_animal,
        events_per_animal=2,
        expected_animals=2,
    )

    assert len(selected) == 4
    assert not selected["excluded_prior_model_event"].any()
    assert selected.groupby("animal").size().eq(2).all()
    rat_a = selected[selected["animal"].eq("RatA")]
    assert rat_a["event_index"].tolist() == [3, 2]
    assert gates.set_index("gate").loc["overall_technical", "passed"]
    assert not any("evidence" in column for column in selected["selection_score_name"].unique())
