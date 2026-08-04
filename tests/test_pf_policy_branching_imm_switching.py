from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.test_pf_policy_branching_imm_switching import (
    build_branching_field,
    event_branching_effects,
    run_branching_test,
)


def test_branching_field_computes_shannon_entropy() -> None:
    graph = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "excluded_cv_fold": 0,
                "from_x_bin": 1,
                "from_y_bin": 2,
                "from_x_cm": 15.0,
                "from_y_cm": 25.0,
                "transition_probability": probability,
                "observed_out_degree": 2,
            }
            for probability in (0.75, 0.25)
        ]
    )

    field = build_branching_field(graph)

    expected = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    assert len(field) == 1
    assert np.isclose(field.loc[0, "branch_entropy_nats"], expected)
    assert np.isclose(field.loc[0, "effective_outgoing_actions"], np.exp(expected))


def test_branching_test_detects_event_internal_alignment() -> None:
    rows = []
    for rat_index, rat in enumerate(("Rat1", "Rat2", "Rat3", "Rat4")):
        for event_index in range(4):
            entropy = np.linspace(0.0, 1.0, 12)
            switch = entropy + 0.001 * (rat_index + event_index) * np.arange(12)
            for transition_index, (x_value, y_value) in enumerate(zip(entropy, switch, strict=True)):
                rows.append(
                    {
                        "session": f"{rat}/Open1",
                        "rat": rat,
                        "event_index": event_index,
                        "transition_index": transition_index,
                        "branch_mapping_valid": True,
                        "branch_entropy_nats": x_value,
                        "stationary_continuous_switch_probability_given_nonfragmented": y_value,
                        "branch_mapping_distance_cm": 1.0,
                    }
                )
    events, offsets = event_branching_effects(pd.DataFrame(rows), minimum_transitions=8)
    test, by_rat, loo, null, _ = run_branching_test(
        events,
        offsets,
        permutations=199,
        bootstraps=200,
        seed=4,
    )

    assert events["event_evaluable"].all()
    assert np.allclose(events["branch_switch_spearman_r"], 1.0)
    assert test.loc[0, "estimate"] == 1.0
    assert test.loc[0, "permutation_p_value"] <= 0.05
    assert by_rat["positive_direction"].all()
    assert loo["positive_direction"].all()
    assert len(null) == 199
