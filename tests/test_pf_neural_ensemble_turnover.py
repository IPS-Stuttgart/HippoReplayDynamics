from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.test_pf_neural_ensemble_turnover import (
    PRIMARY,
    build_event_medians,
    run_tests,
)


def _split_fixture() -> pd.DataFrame:
    rows = []
    for rat_index, rat in enumerate(("Rat1", "Rat2", "Rat3", "Rat4")):
        for event_index in range(5):
            for split_index in range(6):
                controls = [0.1 + 0.001 * split_index, 0.12, 0.08]
                boundary = 0.8 + 0.001 * event_index
                rows.append(
                    {
                        "session": f"{rat}/Open1",
                        "rat": rat,
                        "event_index": event_index,
                        "cell_split_index": split_index,
                        "status": "success",
                        "assembly_turnover_evaluable": True,
                        "assembly_boundary_heldout_turnover_hellinger": boundary,
                        "assembly_control_heldout_turnovers_json": json.dumps(controls),
                        "assembly_control_heldout_turnover_median": np.median(controls),
                        "heldout_assembly_turnover_excess": boundary - np.median(controls),
                        "assembly_boundary_switch_probability": 0.8,
                        "real_frozen_heldout_delta_imm_minus_fragmented": 5.0 + event_index,
                        "train_cell_count": 20 + rat_index,
                        "test_cell_count": 8,
                        "train_spikes": 30,
                        "test_spikes": 12,
                        "real_imm_train_posterior_entropy": 2.0,
                        "n_time": 30,
                        "heldout_replay_spikes_used_for_latent_inference": False,
                    }
                )
    return pd.DataFrame(rows)


def test_turnover_event_medians_and_exchangeability_test() -> None:
    splits = _split_fixture()
    events = build_event_medians(splits)

    tests, by_rat, loo, null = run_tests(
        splits,
        events,
        permutations=199,
        bootstraps=200,
        seed=5,
    )

    primary = tests[tests["role"].eq("primary")].iloc[0]
    assert len(events) == 20
    assert events["completed_evaluable_splits"].eq(6).all()
    assert events[PRIMARY].gt(0.6).all()
    assert primary["estimate"] > 0.6
    assert primary["permutation_p_value"] <= 0.05
    assert by_rat["positive_direction"].all()
    assert loo["positive_direction"].all()
    assert len(null) == 199
