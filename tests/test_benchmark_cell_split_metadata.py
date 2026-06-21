from __future__ import annotations

from dataclasses import is_dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth


def test_metadata_patch_preserves_cell_split_config_options() -> None:
    config = benchmarks.BenchmarkConfig(
        cell_split_strategy="mean-rate",
        cell_split_strata=6,
    )

    assert config.cell_split_strategy == "mean-rate"
    assert config.cell_split_strata == 6
    assert ground_truth.BenchmarkConfig is benchmarks.BenchmarkConfig
    assert hipporeplayimm.BenchmarkConfig is benchmarks.BenchmarkConfig


def test_benchmark_metadata_records_cell_split_config_options() -> None:
    config = benchmarks.BenchmarkConfig(
        cell_split_strategy="peak-rate",
        cell_split_strata=5,
    )

    metadata = benchmarks._benchmark_config_metadata(config)

    assert metadata["benchmark_cell_split_strategy"] == "peak-rate"
    assert metadata["benchmark_cell_split_strata"] == 5


def test_compare_ground_truth_recovers_cell_split_metadata_from_scores(monkeypatch) -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "model": ["random"],
            "heldout_log_likelihood": [0.0],
            "train_log_likelihood": [0.0],
            "joint_log_likelihood": [0.0],
            "benchmark_test_cell_fraction": [0.5],
            "benchmark_random_seed": [11],
            "benchmark_cell_split_seed": [13],
            "benchmark_cell_split_strategy": ["peak-rate"],
            "benchmark_cell_split_strata": [7],
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
        del session
        if is_dataclass(config):
            config = replace(config, models=("random",))
        else:
            values = dict(vars(config))
            values["models"] = ("random",)
            config = SimpleNamespace(**values)
        captured_configs.append(config)
        return {"random": FakeModel()}

    def fake_split_cells_from_encoding(encoding, config, random_seed):
        del encoding
        captured_splits.append((config, random_seed))
        return np.asarray([1], dtype=int), np.asarray([2], dtype=int)

    monkeypatch.setattr(ground_truth, "_load_or_generate_ground_truth", lambda *args, **kwargs: labels)
    monkeypatch.setattr(ground_truth, "load_open_field_sessions", lambda root: [SimpleNamespace(session_id="s1")])
    monkeypatch.setattr(ground_truth, "fit_place_field_encoding", lambda *args, **kwargs: FakeEncoding())
    monkeypatch.setattr(ground_truth, "build_emissions", lambda *args, **kwargs: SimpleNamespace(n_time=1))
    monkeypatch.setattr(
        ground_truth,
        "infer_well_locations",
        lambda *args, **kwargs: pd.DataFrame(columns=["well_id", "well_x", "well_y"]),
    )
    monkeypatch.setattr(ground_truth, "_build_models", fake_build_models)
    monkeypatch.setattr(benchmarks, "_split_cells_from_encoding", fake_split_cells_from_encoding)

    ground_truth.compare_scores_to_ground_truth(
        "unused-root",
        scores,
        cell_split_strategy="random",
        cell_split_strata=4,
    )

    assert len(captured_configs) == 1
    assert captured_configs[0].cell_split_strategy == "peak-rate"
    assert captured_configs[0].cell_split_strata == 7
    assert len(captured_splits) == 1
    split_config, random_seed = captured_splits[0]
    assert split_config.cell_split_strategy == "peak-rate"
    assert split_config.cell_split_strata == 7
    assert split_config.test_cell_fraction == 0.5
    assert random_seed == 13


def test_compare_ground_truth_decodes_each_benchmark_split_independently(monkeypatch) -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "event_index": [0, 0],
            "model": ["random", "random"],
            "heldout_log_likelihood": [0.0, 0.0],
            "train_log_likelihood": [0.0, 0.0],
            "joint_log_likelihood": [0.0, 0.0],
            "benchmark_test_cell_fraction": [0.5, 0.5],
            "benchmark_random_seed": [11, 11],
            "benchmark_cell_split_seed": [13, 14],
            "benchmark_cell_split_index": [0, 1],
            "benchmark_cell_split_strategy": ["peak-rate", "peak-rate"],
            "benchmark_cell_split_strata": [7, 7],
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
    captured_splits = []

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
        captured_splits.append((config, random_seed))
        return np.asarray([1], dtype=int), np.asarray([2], dtype=int)

    monkeypatch.setattr(ground_truth, "_load_or_generate_ground_truth", lambda *args, **kwargs: labels)
    monkeypatch.setattr(ground_truth, "load_open_field_sessions", lambda root: [SimpleNamespace(session_id="s1")])
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

    assert [seed for _, seed in captured_splits] == [13, 14]
    assert [config.cell_split_strategy for config, _ in captured_splits] == ["peak-rate", "peak-rate"]
    assert [config.cell_split_strata for config, _ in captured_splits] == [7, 7]
    assert set(comparison["benchmark_cell_split_index"].astype(int)) == {0, 1}


def test_compare_ground_truth_decodes_each_random_seed_independently(monkeypatch) -> None:
    scores = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "event_index": [0, 0],
            "model": ["random", "random"],
            "heldout_log_likelihood": [0.0, 0.0],
            "train_log_likelihood": [0.0, 0.0],
            "joint_log_likelihood": [0.0, 0.0],
            "benchmark_test_cell_fraction": [0.5, 0.5],
            "benchmark_random_seed": [11, 12],
            "benchmark_cell_split_seed": [11, 12],
            "benchmark_cell_split_index": [0, 0],
            "benchmark_cell_split_strategy": ["peak-rate", "peak-rate"],
            "benchmark_cell_split_strata": [7, 7],
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
    captured_splits = []

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
        captured_splits.append((config, random_seed))
        return np.asarray([1], dtype=int), np.asarray([2], dtype=int)

    monkeypatch.setattr(ground_truth, "_load_or_generate_ground_truth", lambda *args, **kwargs: labels)
    monkeypatch.setattr(ground_truth, "load_open_field_sessions", lambda root: [SimpleNamespace(session_id="s1")])
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

    assert [seed for _, seed in captured_splits] == [11, 12]
    assert [config.cell_split_strategy for config, _ in captured_splits] == ["peak-rate", "peak-rate"]
    assert [config.cell_split_strata for config, _ in captured_splits] == [7, 7]
    assert set(comparison["benchmark_random_seed"].astype(int)) == {11, 12}
