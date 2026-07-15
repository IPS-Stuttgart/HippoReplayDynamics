from __future__ import annotations

import pandas as pd

from hipporeplayimm.simulation_recovery import add_evidence_columns


def test_simulation_evidence_annotation_ignores_duplicate_input_indices() -> None:
    stationary = "sorted-spike-state-space-stationary"
    diffusion = "sorted-spike-state-space-diffusion"
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "simulation_random_seed": 1,
                "simulation_event_index": 0,
                "event_index": 0,
                "event_id": "evt-duplicate-index",
                "true_model": "stationary",
                "expected_model": stationary,
                "model": stationary,
                "log_evidence": 2.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "simulation_random_seed": 1,
                "simulation_event_index": 0,
                "event_index": 0,
                "event_id": "evt-duplicate-index",
                "true_model": "stationary",
                "expected_model": stationary,
                "model": diffusion,
                "log_evidence": 0.0,
                "n_time": 3,
                "n_spikes": 5,
            },
        ],
        index=[7, 7],
    )

    scored = add_evidence_columns(rows).set_index("model")

    assert int(scored["is_best_model"].sum()) == 1
    assert bool(scored.loc[stationary, "is_best_model"])
    assert not bool(scored.loc[diffusion, "is_best_model"])
    assert scored["best_model"].unique().tolist() == [stationary]
    assert scored["recovered_expected_model"].astype(bool).all()
