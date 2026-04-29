"""Parameter sweeps for replay model experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import BenchmarkConfig, BenchmarkResult, run_open_field_benchmark
from .ground_truth import (
    GroundTruthConfig,
    compare_scores_to_ground_truth,
    generate_behavioral_ground_truth,
)


PYRECEST_SWEEP_PARAMETER_COLUMNS = (
    "pyrecest_particles",
    "pyrecest_alpha",
    "pyrecest_beta",
    "pyrecest_process_noise_sigma_cm_s",
    "pyrecest_position_jump_sigma_cm",
    "pyrecest_jump_probability",
    "pyrecest_goal_reset_probability",
    "pyrecest_initial_velocity_sigma_cm_s",
)


@dataclass(frozen=True)
class PyRecEstSweepConfig:
    """Configuration for a PyRecEst goal-particle model sweep."""

    max_events_per_session: int | None = 1
    candidate_top_k: int = 64
    random_seed: int = 1
    event_epoch: str = "run"
    baseline_models: tuple[str, ...] = ("random", "stationary")
    particles: tuple[int, ...] = (128,)
    alphas: tuple[float, ...] = (0.80,)
    betas: tuple[float, ...] = (1.00,)
    process_noise_sigmas_cm_s: tuple[float, ...] = (60.0,)
    position_jump_sigmas_cm: tuple[float, ...] = (25.0,)
    jump_probabilities: tuple[float, ...] = (0.03,)
    goal_reset_probabilities: tuple[float, ...] = (0.02,)
    initial_velocity_sigmas_cm_s: tuple[float, ...] = (120.0,)
    include_ground_truth: bool = True
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)


@dataclass
class PyRecEstSweepResult:
    """Tables produced by a PyRecEst parameter sweep."""

    summary: pd.DataFrame
    event_scores: pd.DataFrame
    ground_truth_comparison: pd.DataFrame
    behavioral_ground_truth: pd.DataFrame | None


def run_pyrecest_parameter_sweep(
    root: str | Path,
    config: PyRecEstSweepConfig | None = None,
) -> PyRecEstSweepResult:
    """Run a cartesian sweep over PyRecEst goal-particle replay parameters."""

    config = PyRecEstSweepConfig() if config is None else config
    parameter_rows = pyrecest_parameter_grid(config)
    if not parameter_rows:
        raise ValueError("PyRecEst sweep parameter grid is empty")

    ground_truth = None
    if config.include_ground_truth:
        ground_truth = generate_behavioral_ground_truth(
            root,
            config=config.ground_truth,
            max_events_per_session=config.max_events_per_session,
        )

    summary_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []

    for sweep_id, parameters in enumerate(parameter_rows):
        benchmark_config = _benchmark_config(config, parameters)
        benchmark = run_open_field_benchmark(root, benchmark_config)
        scores = _with_sweep_columns(benchmark.rows, sweep_id, parameters)
        if not scores.empty:
            score_frames.append(scores)

        summary_row = _benchmark_summary_row(sweep_id, parameters, benchmark)
        if config.include_ground_truth:
            comparison = compare_scores_to_ground_truth(
                root,
                benchmark.rows,
                ground_truth=ground_truth,
                ground_truth_config=config.ground_truth,
                candidate_top_k=config.candidate_top_k,
                pyrecest_particles=int(parameters["pyrecest_particles"]),
                pyrecest_alpha=float(parameters["pyrecest_alpha"]),
                pyrecest_beta=float(parameters["pyrecest_beta"]),
                pyrecest_process_noise_sigma_cm_s=float(
                    parameters["pyrecest_process_noise_sigma_cm_s"]
                ),
                pyrecest_position_jump_sigma_cm=float(
                    parameters["pyrecest_position_jump_sigma_cm"]
                ),
                pyrecest_jump_probability=float(parameters["pyrecest_jump_probability"]),
                pyrecest_goal_reset_probability=float(
                    parameters["pyrecest_goal_reset_probability"]
                ),
                pyrecest_initial_velocity_sigma_cm_s=float(
                    parameters["pyrecest_initial_velocity_sigma_cm_s"]
                ),
            )
            comparison = _with_sweep_columns(comparison, sweep_id, parameters)
            if not comparison.empty:
                comparison_frames.append(comparison)
            summary_row.update(_ground_truth_summary_metrics(comparison))
        summary_rows.append(summary_row)

    return PyRecEstSweepResult(
        summary=pd.DataFrame(summary_rows),
        event_scores=_concat_or_empty(score_frames),
        ground_truth_comparison=_concat_or_empty(comparison_frames),
        behavioral_ground_truth=ground_truth,
    )


def write_pyrecest_sweep_outputs(result: PyRecEstSweepResult, output: str | Path) -> None:
    """Write sweep tables under an output directory."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_path / "sweep_summary.csv", index=False)
    result.event_scores.to_csv(output_path / "event_scores.csv", index=False)
    if result.behavioral_ground_truth is not None:
        result.behavioral_ground_truth.to_csv(
            output_path / "behavioral_ground_truth.csv",
            index=False,
        )
    if not result.ground_truth_comparison.empty:
        result.ground_truth_comparison.to_csv(
            output_path / "ground_truth_comparison.csv",
            index=False,
        )


def pyrecest_parameter_grid(config: PyRecEstSweepConfig) -> list[dict[str, float | int]]:
    """Return the cartesian product of PyRecEst sweep parameters."""

    _validate_nonempty(config.particles, "particles")
    _validate_nonempty(config.alphas, "alphas")
    _validate_nonempty(config.betas, "betas")
    _validate_nonempty(config.process_noise_sigmas_cm_s, "process_noise_sigmas_cm_s")
    _validate_nonempty(config.position_jump_sigmas_cm, "position_jump_sigmas_cm")
    _validate_nonempty(config.jump_probabilities, "jump_probabilities")
    _validate_nonempty(config.goal_reset_probabilities, "goal_reset_probabilities")
    _validate_nonempty(
        config.initial_velocity_sigmas_cm_s,
        "initial_velocity_sigmas_cm_s",
    )

    rows: list[dict[str, float | int]] = []
    for values in product(
        config.particles,
        config.alphas,
        config.betas,
        config.process_noise_sigmas_cm_s,
        config.position_jump_sigmas_cm,
        config.jump_probabilities,
        config.goal_reset_probabilities,
        config.initial_velocity_sigmas_cm_s,
    ):
        row = dict(zip(PYRECEST_SWEEP_PARAMETER_COLUMNS, values, strict=True))
        rows.append(row)
    return rows


def sorted_sweep_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort a sweep summary by the most useful available objective columns."""

    if summary.empty:
        return summary
    sort_columns = [
        column
        for column in ("goal_accuracy", "mean_heldout_log_likelihood")
        if column in summary.columns
    ]
    if not sort_columns:
        return summary
    return summary.sort_values(sort_columns, ascending=[False] * len(sort_columns))


def _benchmark_config(
    config: PyRecEstSweepConfig,
    parameters: dict[str, float | int],
) -> BenchmarkConfig:
    return BenchmarkConfig(
        max_events_per_session=config.max_events_per_session,
        candidate_top_k=config.candidate_top_k,
        random_seed=config.random_seed,
        event_epoch=config.event_epoch,
        models=tuple(dict.fromkeys((*config.baseline_models, "pyrecest-goal-particle"))),
        pyrecest_particles=int(parameters["pyrecest_particles"]),
        pyrecest_alpha=float(parameters["pyrecest_alpha"]),
        pyrecest_beta=float(parameters["pyrecest_beta"]),
        pyrecest_process_noise_sigma_cm_s=float(
            parameters["pyrecest_process_noise_sigma_cm_s"]
        ),
        pyrecest_position_jump_sigma_cm=float(
            parameters["pyrecest_position_jump_sigma_cm"]
        ),
        pyrecest_jump_probability=float(parameters["pyrecest_jump_probability"]),
        pyrecest_goal_reset_probability=float(
            parameters["pyrecest_goal_reset_probability"]
        ),
        pyrecest_initial_velocity_sigma_cm_s=float(
            parameters["pyrecest_initial_velocity_sigma_cm_s"]
        ),
    )


def _with_sweep_columns(
    frame: pd.DataFrame,
    sweep_id: int,
    parameters: dict[str, float | int],
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "sweep_id", sweep_id)
    for column, value in reversed(parameters.items()):
        output.insert(1, column, value)
    return output


def _benchmark_summary_row(
    sweep_id: int,
    parameters: dict[str, float | int],
    benchmark: BenchmarkResult,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sweep_id": sweep_id,
        **parameters,
        "events": 0,
        "mean_heldout_log_likelihood": np.nan,
        "mean_delta_vs_best_static": np.nan,
        "mean_bits_per_spike_vs_best_static": np.nan,
    }
    summary = benchmark.summary()
    if summary.empty:
        return row
    model_summary = summary[summary["model"] == "pyrecest-goal-particle"]
    if model_summary.empty:
        return row
    values = model_summary.iloc[0]
    row.update(
        {
            "events": int(values["events"]),
            "mean_heldout_log_likelihood": float(
                values["mean_heldout_log_likelihood"]
            ),
            "mean_delta_vs_best_static": float(values["mean_delta_vs_best_static"]),
            "mean_bits_per_spike_vs_best_static": float(
                values["mean_bits_per_spike_vs_best_static"]
            ),
        }
    )
    return row


def _ground_truth_summary_metrics(comparison: pd.DataFrame) -> dict[str, float | int]:
    if comparison.empty or "valid_label" not in comparison.columns:
        return {
            "valid_goal_rows": 0,
            "goal_accuracy": np.nan,
            "median_endpoint_error_cm": np.nan,
            "mean_true_well_posterior": np.nan,
            "median_true_well_rank": np.nan,
        }
    if "model" in comparison.columns:
        comparison = comparison[comparison["model"] == "pyrecest-goal-particle"]
    valid = comparison[comparison["valid_label"].fillna(False)]
    if valid.empty:
        return {
            "valid_goal_rows": 0,
            "goal_accuracy": np.nan,
            "median_endpoint_error_cm": np.nan,
            "mean_true_well_posterior": np.nan,
            "median_true_well_rank": np.nan,
        }
    return {
        "valid_goal_rows": int(valid.shape[0]),
        "goal_accuracy": float(valid["goal_correct"].mean()),
        "median_endpoint_error_cm": float(valid["endpoint_error_cm"].median()),
        "mean_true_well_posterior": float(valid["true_well_posterior"].mean()),
        "median_true_well_rank": float(valid["true_well_rank"].median()),
    }


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _validate_nonempty(values: tuple[object, ...], name: str) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")
