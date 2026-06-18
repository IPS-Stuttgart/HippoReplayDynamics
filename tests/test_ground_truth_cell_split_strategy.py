from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.ground_truth as ground_truth
import hipporeplayimm.ground_truth_cell_split_strategy as split_strategy_patch


class _EncodingStub:
    cell_ids = np.array([1, 2, 3, 4], dtype=int)


def test_ground_truth_split_helper_uses_configured_strategy_when_metadata_lacks_cell_ids(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_split_cells_from_encoding(encoding, config, random_seed):
        captured["cell_ids"] = tuple(int(cell_id) for cell_id in encoding.cell_ids)
        captured["test_cell_fraction"] = float(config.test_cell_fraction)
        captured["cell_split_strategy"] = str(config.cell_split_strategy)
        captured["cell_split_strata"] = int(config.cell_split_strata)
        captured["random_seed"] = int(random_seed)
        return np.array([1, 3], dtype=int), np.array([2, 4], dtype=int)

    monkeypatch.setattr(
        split_strategy_patch,
        "_split_cells_from_encoding",
        fake_split_cells_from_encoding,
    )
    scores = pd.DataFrame(
        {
            "benchmark_test_cell_fraction": [0.5],
            "benchmark_random_seed": [3],
            "benchmark_cell_split_seed": [17],
        }
    )
    config = SimpleNamespace(
        test_cell_fraction=0.25,
        random_seed=3,
        cell_split_strategy="mean-rate",
        cell_split_strata=7,
    )

    train_cells, test_cells = ground_truth._cell_split_for_score_rows(
        scores,
        _EncodingStub(),
        config,
    )

    assert train_cells.tolist() == [1, 3]
    assert test_cells.tolist() == [2, 4]
    assert captured == {
        "cell_ids": (1, 2, 3, 4),
        "test_cell_fraction": 0.5,
        "cell_split_strategy": "mean-rate",
        "cell_split_strata": 7,
        "random_seed": 17,
    }


def test_ground_truth_split_helper_preserves_explicit_cell_ids(monkeypatch) -> None:
    def fail_split_cells_from_encoding(*_args, **_kwargs):
        raise AssertionError("explicit score-table cell IDs should not be regenerated")

    monkeypatch.setattr(
        split_strategy_patch,
        "_split_cells_from_encoding",
        fail_split_cells_from_encoding,
    )
    scores = pd.DataFrame(
        {
            "train_cell_ids": ["1,3"],
            "test_cell_ids": ["2,4"],
        }
    )
    config = SimpleNamespace(
        test_cell_fraction=0.5,
        random_seed=3,
        cell_split_strategy="peak-rate",
        cell_split_strata=7,
    )

    train_cells, test_cells = ground_truth._cell_split_for_score_rows(
        scores,
        _EncodingStub(),
        config,
    )

    assert train_cells.tolist() == [1, 3]
    assert test_cells.tolist() == [2, 4]
