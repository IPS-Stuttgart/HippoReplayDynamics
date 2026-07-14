from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_pfeiffer_imm_order_map_factorial import (
    FRAGMENTED,
    FIRST_ORDER_IMM,
    build_event_decisions,
    build_gates,
    rat_bootstrap,
    summarize,
)


def _scores(
    *,
    real_original: float,
    wrong_original: float,
    real_shuffled: list[float],
    wrong_shuffled: list[float],
    rats: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rat_index in range(rats):
        rat = f"Rat{rat_index + 1}"
        session = f"{rat}/Open1"
        event_index = 100 + rat_index
        for map_condition, original, shuffled in (
            ("real_map", real_original, real_shuffled),
            ("population_code_permuted", wrong_original, wrong_shuffled),
        ):
            rows.extend(
                _pair_rows(
                    session,
                    rat,
                    event_index,
                    map_condition,
                    "original",
                    -1,
                    original,
                )
            )
            for shuffle_index, delta in enumerate(shuffled):
                rows.extend(
                    _pair_rows(
                        session,
                        rat,
                        event_index,
                        map_condition,
                        "shuffled",
                        shuffle_index,
                        delta,
                    )
                )
    return pd.DataFrame(rows)


def _pair_rows(
    session: str,
    rat: str,
    event_index: int,
    map_condition: str,
    order_condition: str,
    shuffle_index: int,
    delta: float,
) -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "session": session,
            "rat": rat,
            "event_index": event_index,
            "event_group": "clean_imm",
            "map_condition": map_condition,
            "order_condition": order_condition,
            "shuffle_index": shuffle_index,
            "model": FRAGMENTED,
            "log_evidence": 10.0,
        },
        {
            "status": "success",
            "session": session,
            "rat": rat,
            "event_index": event_index,
            "event_group": "clean_imm",
            "map_condition": map_condition,
            "order_condition": order_condition,
            "shuffle_index": shuffle_index,
            "model": FIRST_ORDER_IMM,
            "log_evidence": 10.0 + delta,
        },
    ]


def test_interaction_uses_all_four_factorial_cells() -> None:
    scores = _scores(
        real_original=30.0,
        wrong_original=20.0,
        real_shuffled=[5.0, 7.0, 9.0],
        wrong_shuffled=[4.0, 6.0, 8.0],
        rats=1,
    )
    decisions = build_event_decisions(scores, expected_shuffles=3)
    row = decisions.iloc[0]

    assert row["real_order_advantage"] == 23.0
    assert row["wrong_order_advantage"] == 14.0
    assert row["order_by_map_interaction"] == 9.0
    assert row["n_paired_shuffles"] == 3
    assert row["factorial_complete"]


def test_near_zero_interaction_means_map_generic_ordering() -> None:
    scores = _scores(
        real_original=30.0,
        wrong_original=25.0,
        real_shuffled=[10.0, 10.0, 10.0],
        wrong_shuffled=[5.0, 5.0, 5.0],
    )
    decisions = build_event_decisions(scores, expected_shuffles=3)
    bootstrap = rat_bootstrap(decisions, replicates=100, seed=1)
    gates = build_gates(
        scores,
        decisions,
        bootstrap,
        expected_events=4,
        expected_shuffles=3,
    )

    verdict = gates.loc[gates["gate"].eq("factorial_verdict"), "observed"].iloc[0]
    assert np.allclose(decisions["order_by_map_interaction"], 0.0)
    assert verdict == "temporal_order_advantage_largely_map_generic"


def test_positive_interaction_passes_rat_bootstrap() -> None:
    scores = _scores(
        real_original=35.0,
        wrong_original=20.0,
        real_shuffled=[5.0, 6.0, 7.0],
        wrong_shuffled=[8.0, 9.0, 10.0],
    )
    decisions = build_event_decisions(scores, expected_shuffles=3)
    bootstrap = rat_bootstrap(decisions, replicates=100, seed=2)
    gates = build_gates(
        scores,
        decisions,
        bootstrap,
        expected_events=4,
        expected_shuffles=3,
    )

    verdict = gates.loc[gates["gate"].eq("factorial_verdict"), "observed"].iloc[0]
    assert verdict == "correct_map_strengthens_temporal_order_advantage"
    assert summarize(decisions)["median_order_by_map_interaction"].iloc[0] > 0.0


def test_missing_wrong_map_shuffle_fails_factorial() -> None:
    scores = _scores(
        real_original=30.0,
        wrong_original=20.0,
        real_shuffled=[5.0, 6.0, 7.0],
        wrong_shuffled=[4.0, 5.0, 6.0],
    )
    scores = scores[
        ~(
            scores["map_condition"].eq("population_code_permuted")
            & scores["order_condition"].eq("shuffled")
            & scores["shuffle_index"].eq(2)
        )
    ]
    decisions = build_event_decisions(scores, expected_shuffles=3)
    bootstrap = rat_bootstrap(decisions, replicates=20, seed=3)
    gates = build_gates(
        scores,
        decisions,
        bootstrap,
        expected_events=4,
        expected_shuffles=3,
    )

    assert not decisions["factorial_complete"].any()
    assert (
        gates.loc[gates["gate"].eq("factorial_verdict"), "observed"].iloc[0]
        == "incomplete_factorial"
    )
