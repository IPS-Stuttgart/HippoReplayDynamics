from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth


def test_clusterless_ground_truth_uses_saved_cell_split_metadata(monkeypatch) -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "model": ["clusterless-state-space-diffusion"],
            "requested_model": ["clusterless-state-space-diffusion"],
            "heldout_log_likelihood": [0.0],
            "train_log_likelihood": [0.0],
            "joint_log_likelihood": [0.0],
            "benchmark_test_cell_fraction": [0.5],
            "benchmark_random_seed": [11],
            "benchmark_cell_split_seed": [17],
            "benchmark_cell_split_strategy": ["peak-rate"],
            "benchmark_cell_split_strata": [6],
        }
    )
    labels = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "true_well_id": [np.nan],
            "true_well_x": [np.nan],
            "true_well_y": [np.nan],
            "valid_label": [False],
        }
    )
    captured_configs = []
    captured_splits = []

    class FakeEncoding:
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        cell_ids = np.asarray([1, 2, 3, 4], dtype=int)

        def select_cells(self, cell_ids):
            del cell_ids
            return self

    class FakeClusterlessEncoding:
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        occupancy_s = np.ones(2, dtype=float)

    class FakeModel:
        def score(self, emissions, bin_centers):
            del emissions, bin_centers
            return SimpleNamespace(
                terminal_log_posterior=np.log(np.asarray([0.5, 0.5], dtype=float)),
                trajectory_log_posterior=None,
            )

    def fake_build_models(config, session=None):
        del session
        captured_configs.append(config)
        return {"clusterless-state-space-diffusion": FakeModel()}

    def fake_split_cells_from_encoding(encoding, config, random_seed):
        del encoding
        captured_splits.append((config, random_seed))
        return np.asarray([1, 2], dtype=int), np.asarray([3, 4], dtype=int)

    monkeypatch.setattr(
        ground_truth,
        "_load_or_generate_ground_truth",
        lambda *args, **kwargs: labels,
    )
    monkeypatch.setattr(
        ground_truth,
        "load_open_field_sessions",
        lambda root: [SimpleNamespace(session_id="s1")],
    )
    monkeypatch.setattr(
        ground_truth,
        "fit_place_field_encoding",
        lambda *args, **kwargs: FakeEncoding(),
    )
    monkeypatch.setattr(
        ground_truth,
        "_session_with_mark_cell_subset",
        lambda session, cell_ids, role: session,
    )
    monkeypatch.setattr(
        ground_truth,
        "fit_clusterless_mark_encoding",
        lambda *args, **kwargs: FakeClusterlessEncoding(),
    )
    monkeypatch.setattr(
        ground_truth,
        "build_clusterless_mark_emissions",
        lambda *args, **kwargs: SimpleNamespace(n_time=1),
    )
    monkeypatch.setattr(
        ground_truth,
        "infer_well_locations",
        lambda *args, **kwargs: pd.DataFrame(columns=["well_id", "well_x", "well_y"]),
    )
    monkeypatch.setattr(ground_truth, "_build_models", fake_build_models)
    monkeypatch.setattr(benchmarks, "_split_cells_from_encoding", fake_split_cells_from_encoding)

    comparison = ground_truth.compare_scores_to_ground_truth(
        "unused-root",
        scores,
        cell_split_strategy="random",
        cell_split_strata=4,
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].cell_split_strategy == "peak-rate"
    assert captured_configs[0].cell_split_strata == 6
    assert len(captured_splits) == 1
    split_config, random_seed = captured_splits[0]
    assert split_config.cell_split_strategy == "peak-rate"
    assert split_config.cell_split_strata == 6
    assert split_config.test_cell_fraction == 0.5
    assert random_seed == 17
    assert comparison["model"].tolist() == ["clusterless-state-space-diffusion"]
