from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth


def test_compare_ground_truth_scopes_cell_split_metadata_by_session(monkeypatch) -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s2"],
            "event_index": [0, 0],
            "model": ["random", "random"],
            "heldout_log_likelihood": [0.0, 0.0],
            "train_log_likelihood": [0.0, 0.0],
            "joint_log_likelihood": [0.0, 0.0],
            "benchmark_test_cell_fraction": [0.5, 0.5],
            "benchmark_random_seed": [11, 11],
            "benchmark_cell_split_seed": [13, 17],
            "benchmark_cell_split_strategy": ["random", "peak-rate"],
            "benchmark_cell_split_strata": [4, 6],
        }
    )
    labels = pd.DataFrame(
        {
            "session": ["s1", "s2"],
            "event_index": [0, 0],
            "true_well_id": [np.nan, np.nan],
            "true_well_x": [np.nan, np.nan],
            "true_well_y": [np.nan, np.nan],
            "valid_label": [False, False],
        }
    )
    captured_splits: list[tuple[str, int, str, int]] = []

    class FakeEncoding:
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        cell_ids = np.asarray([1, 2], dtype=int)

        def select_cells(self, cell_ids):
            return self

    class FakeModel:
        def score(self, emissions, bin_centers):
            del emissions, bin_centers
            return SimpleNamespace(
                terminal_log_posterior=np.log(np.asarray([0.5, 0.5], dtype=float)),
                trajectory_log_posterior=None,
            )

    def fake_build_models(config, session=None):
        del config, session
        return {"random": FakeModel()}

    def fake_split_cells_from_encoding(encoding, config, random_seed):
        del encoding
        captured_splits.append(
            (
                str(config.cell_split_strategy),
                int(config.cell_split_strata),
                float(config.test_cell_fraction),
                int(random_seed),
            )
        )
        return np.asarray([1], dtype=int), np.asarray([2], dtype=int)

    monkeypatch.setattr(ground_truth, "_load_or_generate_ground_truth", lambda *args, **kwargs: labels)
    monkeypatch.setattr(
        ground_truth,
        "load_open_field_sessions",
        lambda root: [SimpleNamespace(session_id="s1"), SimpleNamespace(session_id="s2")],
    )
    monkeypatch.setattr(ground_truth, "fit_place_field_encoding", lambda *args, **kwargs: FakeEncoding())
    monkeypatch.setattr(ground_truth, "build_emissions", lambda *args, **kwargs: SimpleNamespace(n_time=1))
    monkeypatch.setattr(
        ground_truth,
        "infer_well_locations",
        lambda *args, **kwargs: pd.DataFrame(columns=["well_id", "well_x", "well_y"]),
    )
    monkeypatch.setattr(ground_truth, "_build_models", fake_build_models)
    monkeypatch.setattr(benchmarks, "_split_cells_from_encoding", fake_split_cells_from_encoding)

    comparison = ground_truth.compare_scores_to_ground_truth("unused-root", scores)

    assert captured_splits == [
        ("random", 4, 0.5, 13),
        ("peak-rate", 6, 0.5, 17),
    ]
    assert set(comparison["session"]) == {"s1", "s2"}
