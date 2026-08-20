from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from hipporeplayimm.benchmark_cell_split_metadata import (
    _compare_scores_with_cell_split_metadata,
)


def _unique_int_from_column(frame: pd.DataFrame, column: str, default: int) -> int:
    if column not in frame.columns:
        return int(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) == 0:
        return int(default)
    if len(values) != 1:
        raise ValueError(f"{column} contains multiple values")
    return int(values[0])


def _observed_model_seed(scores_frame: pd.DataFrame, *, base_seed: int = 7) -> int:
    observed: list[int] = []

    def build_models(config: object, *args: object, **kwargs: object) -> dict[str, object]:
        observed.append(int(getattr(config, "random_seed")))
        return {}

    gt = SimpleNamespace(
        _build_models=build_models,
        _cell_split_for_score_rows=lambda *args, **kwargs: None,
        _unique_int_from_column=_unique_int_from_column,
    )
    bench = SimpleNamespace()

    def compare_scores(root: object, scores: object, **kwargs: object) -> pd.DataFrame:
        gt._build_models(SimpleNamespace(random_seed=base_seed))
        return pd.DataFrame()

    _compare_scores_with_cell_split_metadata(
        compare_scores,
        bench,
        gt,
        ".",
        scores_frame,
        scores_frame,
        "random",
        4,
        {},
    )
    assert len(observed) == 1
    return observed[0]


def test_ground_truth_model_build_uses_saved_cell_split_seed() -> None:
    scores = pd.DataFrame({"benchmark_cell_split_seed": [123, 123]})

    assert _observed_model_seed(scores, base_seed=7) == 123


def test_ground_truth_model_build_falls_back_to_base_seed_for_legacy_rows() -> None:
    scores = pd.DataFrame({"model": ["legacy"]})

    assert _observed_model_seed(scores, base_seed=7) == 7
