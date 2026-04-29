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
    "random_seed",
    "pyrecest_model",
    "pyrecest_particles",
    "pyrecest_alpha",
    "pyrecest_beta",
    "pyrecest_process_noise_sigma_cm_s",
    "pyrecest_position_jump_sigma_cm",
    "pyrecest_jump_probability",
    "pyrecest_goal_reset_probability",
    "pyrecest_position_proposal_probability",
    "pyrecest_initial_velocity_sigma_cm_s",
    "pyrecest_imm_mode_stickiness",
    "pyrecest_imm_stationary_velocity_decay",
    "pyrecest_imm_diffusion_velocity_decay",
    "pyrecest_imm_momentum_velocity_decay",
    "pyrecest_imm_jump_fraction",
    "pyrecest_imm_jump_velocity_decay",
)

PYRECEST_SWEEP_AGGREGATE_GROUP_COLUMNS = tuple(
    column for column in PYRECEST_SWEEP_PARAMETER_COLUMNS if column != "random_seed"
)

PYRECEST_SWEEP_AGGREGATE_METRIC_COLUMNS = (
    "events",
    "mean_heldout_log_likelihood",
    "mean_delta_vs_best_static",
    "mean_bits_per_spike_vs_best_static",
    "valid_goal_rows",
    "goal_accuracy",
    "median_endpoint_error_cm",
    "mean_true_well_posterior",
    "median_true_well_rank",
)

PYRECEST_SWEEP_PARETO_OBJECTIVES = {
    "goal_accuracy": "max",
    "mean_delta_vs_best_static": "max",
    "mean_bits_per_spike_vs_best_static": "max",
    "mean_true_well_posterior": "max",
    "median_endpoint_error_cm": "min",
    "median_true_well_rank": "min",
}

PYRECEST_SWEEP_AGGREGATE_PARETO_OBJECTIVES = {
    f"{column}_mean": direction
    for column, direction in PYRECEST_SWEEP_PARETO_OBJECTIVES.items()
}


@dataclass(frozen=True)
class PyRecEstSweepConfig:
    """Configuration for a PyRecEst goal-particle model sweep."""

    max_events_per_session: int | None = 1
    candidate_top_k: int = 64
    random_seed: int = 1
    random_seeds: tuple[int, ...] | None = None
    event_epoch: str = "run"
    baseline_models: tuple[str, ...] = ("random", "stationary")
    pyrecest_models: tuple[str, ...] = ("pyrecest-goal-particle",)
    particles: tuple[int, ...] = (128,)
    alphas: tuple[float, ...] = (0.80,)
    betas: tuple[float, ...] = (1.00,)
    process_noise_sigmas_cm_s: tuple[float, ...] = (60.0,)
    position_jump_sigmas_cm: tuple[float, ...] = (25.0,)
    jump_probabilities: tuple[float, ...] = (0.03,)
    goal_reset_probabilities: tuple[float, ...] = (0.02,)
    position_proposal_probabilities: tuple[float, ...] = (0.0,)
    initial_velocity_sigmas_cm_s: tuple[float, ...] = (120.0,)
    imm_mode_stickinesses: tuple[float, ...] = (0.95,)
    imm_stationary_velocity_decays: tuple[float, ...] = (0.0,)
    imm_diffusion_velocity_decays: tuple[float, ...] = (0.0,)
    imm_momentum_velocity_decays: tuple[float, ...] = (0.95,)
    imm_jump_fractions: tuple[float, ...] = (0.9,)
    imm_jump_velocity_decays: tuple[float, ...] = (0.25,)
    include_ground_truth: bool = True
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)


@dataclass
class PyRecEstSweepResult:
    """Tables produced by a PyRecEst parameter sweep."""

    summary: pd.DataFrame
    event_scores: pd.DataFrame
    ground_truth_comparison: pd.DataFrame
    behavioral_ground_truth: pd.DataFrame | None
    aggregate_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


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
                pyrecest_position_proposal_probability=float(
                    parameters["pyrecest_position_proposal_probability"]
                ),
                pyrecest_initial_velocity_sigma_cm_s=float(
                    parameters["pyrecest_initial_velocity_sigma_cm_s"]
                ),
                pyrecest_imm_mode_stickiness=float(
                    parameters["pyrecest_imm_mode_stickiness"]
                ),
                pyrecest_imm_stationary_velocity_decay=float(
                    parameters["pyrecest_imm_stationary_velocity_decay"]
                ),
                pyrecest_imm_diffusion_velocity_decay=float(
                    parameters["pyrecest_imm_diffusion_velocity_decay"]
                ),
                pyrecest_imm_momentum_velocity_decay=float(
                    parameters["pyrecest_imm_momentum_velocity_decay"]
                ),
                pyrecest_imm_jump_fraction=float(
                    parameters["pyrecest_imm_jump_fraction"]
                ),
                pyrecest_imm_jump_velocity_decay=float(
                    parameters["pyrecest_imm_jump_velocity_decay"]
                ),
                random_seed=int(parameters["random_seed"]),
            )
            comparison = _with_sweep_columns(comparison, sweep_id, parameters)
            if not comparison.empty:
                comparison_frames.append(comparison)
            summary_row.update(
                _ground_truth_summary_metrics(
                    comparison,
                    model=str(parameters["pyrecest_model"]),
                )
            )
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    return PyRecEstSweepResult(
        summary=summary,
        event_scores=_concat_or_empty(score_frames),
        ground_truth_comparison=_concat_or_empty(comparison_frames),
        behavioral_ground_truth=ground_truth,
        aggregate_summary=aggregate_sweep_summary(summary),
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
    pareto = pareto_sweep_summary(result.summary)
    if not pareto.empty:
        pareto.to_csv(output_path / "pareto_summary.csv", index=False)
    aggregate = (
        aggregate_sweep_summary(result.summary)
        if result.aggregate_summary.empty
        else result.aggregate_summary
    )
    if not aggregate.empty:
        aggregate.to_csv(output_path / "aggregate_summary.csv", index=False)
        pareto_aggregate = pareto_aggregate_sweep_summary(aggregate)
        if not pareto_aggregate.empty:
            pareto_aggregate.to_csv(
                output_path / "pareto_aggregate_summary.csv",
                index=False,
            )


def pyrecest_parameter_grid(config: PyRecEstSweepConfig) -> list[dict[str, object]]:
    """Return the cartesian product of PyRecEst sweep parameters."""

    random_seeds = config.random_seeds or (config.random_seed,)
    _validate_nonempty(random_seeds, "random_seeds")
    _validate_nonempty(config.pyrecest_models, "pyrecest_models")
    _validate_nonempty(config.particles, "particles")
    _validate_nonempty(config.alphas, "alphas")
    _validate_nonempty(config.betas, "betas")
    _validate_nonempty(config.process_noise_sigmas_cm_s, "process_noise_sigmas_cm_s")
    _validate_nonempty(config.position_jump_sigmas_cm, "position_jump_sigmas_cm")
    _validate_nonempty(config.jump_probabilities, "jump_probabilities")
    _validate_nonempty(config.goal_reset_probabilities, "goal_reset_probabilities")
    _validate_nonempty(
        config.position_proposal_probabilities,
        "position_proposal_probabilities",
    )
    _validate_nonempty(
        config.initial_velocity_sigmas_cm_s,
        "initial_velocity_sigmas_cm_s",
    )
    _validate_nonempty(config.imm_mode_stickinesses, "imm_mode_stickinesses")
    _validate_nonempty(
        config.imm_stationary_velocity_decays,
        "imm_stationary_velocity_decays",
    )
    _validate_nonempty(
        config.imm_diffusion_velocity_decays,
        "imm_diffusion_velocity_decays",
    )
    _validate_nonempty(
        config.imm_momentum_velocity_decays,
        "imm_momentum_velocity_decays",
    )
    _validate_nonempty(config.imm_jump_fractions, "imm_jump_fractions")
    _validate_nonempty(config.imm_jump_velocity_decays, "imm_jump_velocity_decays")

    rows: list[dict[str, object]] = []
    for values in product(
        random_seeds,
        config.pyrecest_models,
        config.particles,
        config.alphas,
        config.betas,
        config.process_noise_sigmas_cm_s,
        config.position_jump_sigmas_cm,
        config.jump_probabilities,
        config.goal_reset_probabilities,
        config.position_proposal_probabilities,
        config.initial_velocity_sigmas_cm_s,
        config.imm_mode_stickinesses,
        config.imm_stationary_velocity_decays,
        config.imm_diffusion_velocity_decays,
        config.imm_momentum_velocity_decays,
        config.imm_jump_fractions,
        config.imm_jump_velocity_decays,
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
        for column in (
            "goal_accuracy",
            "goal_accuracy_mean",
            "mean_heldout_log_likelihood",
            "mean_heldout_log_likelihood_mean",
        )
        if column in summary.columns
    ]
    if not sort_columns:
        return summary
    return summary.sort_values(sort_columns, ascending=[False] * len(sort_columns))


def pareto_sweep_summary(
    summary: pd.DataFrame,
    objectives: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return nondominated sweep rows across available objective columns."""

    if summary.empty:
        return summary
    objectives = PYRECEST_SWEEP_PARETO_OBJECTIVES if objectives is None else objectives
    objective_columns = [column for column in objectives if column in summary.columns]
    if not objective_columns:
        return sorted_sweep_summary(summary)

    values = _objective_matrix(summary, objectives, objective_columns)
    finite_objective = np.any(np.isfinite(values), axis=1)
    keep = np.zeros(summary.shape[0], dtype=bool)
    finite_indices = np.flatnonzero(finite_objective)
    for row_index in finite_indices:
        keep[row_index] = not _is_dominated(values[row_index], values[finite_indices])
    pareto = summary.loc[keep].copy()
    if pareto.empty:
        return pareto
    return sorted_sweep_summary(pareto)


def aggregate_sweep_summary(
    summary: pd.DataFrame,
    group_columns: tuple[str, ...] = PYRECEST_SWEEP_AGGREGATE_GROUP_COLUMNS,
    metric_columns: tuple[str, ...] = PYRECEST_SWEEP_AGGREGATE_METRIC_COLUMNS,
) -> pd.DataFrame:
    """Aggregate seed-level sweep rows by hyperparameter setting."""

    if summary.empty or "random_seed" not in summary.columns:
        return pd.DataFrame()
    available_groups = [column for column in group_columns if column in summary.columns]
    available_metrics = [column for column in metric_columns if column in summary.columns]
    if not available_groups or not available_metrics:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_values, group in summary.groupby(
        available_groups,
        dropna=False,
        sort=False,
    ):
        if len(available_groups) == 1:
            group_values = (group_values,)
        row = dict(zip(available_groups, group_values, strict=True))
        seed_values = _sorted_numeric_values(group["random_seed"])
        row["random_seed_count"] = len(seed_values)
        row["random_seeds"] = ",".join(str(seed) for seed in seed_values)
        row["sweep_replicates"] = int(group.shape[0])
        for column in available_metrics:
            row.update(_aggregate_metric(column, group[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_aggregate_sweep_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Return nondominated aggregate rows using mean objective columns."""

    return pareto_sweep_summary(
        aggregate,
        objectives=PYRECEST_SWEEP_AGGREGATE_PARETO_OBJECTIVES,
    )


def _benchmark_config(
    config: PyRecEstSweepConfig,
    parameters: dict[str, object],
) -> BenchmarkConfig:
    pyrecest_model = str(parameters["pyrecest_model"])
    return BenchmarkConfig(
        max_events_per_session=config.max_events_per_session,
        candidate_top_k=config.candidate_top_k,
        random_seed=int(parameters["random_seed"]),
        event_epoch=config.event_epoch,
        models=tuple(dict.fromkeys((*config.baseline_models, pyrecest_model))),
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
        pyrecest_position_proposal_probability=float(
            parameters["pyrecest_position_proposal_probability"]
        ),
        pyrecest_initial_velocity_sigma_cm_s=float(
            parameters["pyrecest_initial_velocity_sigma_cm_s"]
        ),
        pyrecest_imm_mode_stickiness=float(
            parameters["pyrecest_imm_mode_stickiness"]
        ),
        pyrecest_imm_stationary_velocity_decay=float(
            parameters["pyrecest_imm_stationary_velocity_decay"]
        ),
        pyrecest_imm_diffusion_velocity_decay=float(
            parameters["pyrecest_imm_diffusion_velocity_decay"]
        ),
        pyrecest_imm_momentum_velocity_decay=float(
            parameters["pyrecest_imm_momentum_velocity_decay"]
        ),
        pyrecest_imm_jump_fraction=float(parameters["pyrecest_imm_jump_fraction"]),
        pyrecest_imm_jump_velocity_decay=float(
            parameters["pyrecest_imm_jump_velocity_decay"]
        ),
    )


def _with_sweep_columns(
    frame: pd.DataFrame,
    sweep_id: int,
    parameters: dict[str, object],
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "sweep_id", sweep_id)
    for column, value in reversed(parameters.items()):
        output.insert(1, column, value)
    return output


def _benchmark_summary_row(
    sweep_id: int,
    parameters: dict[str, object],
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
    model_summary = summary[summary["model"] == str(parameters["pyrecest_model"])]
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


def _ground_truth_summary_metrics(
    comparison: pd.DataFrame,
    model: str = "pyrecest-goal-particle",
) -> dict[str, float | int]:
    if comparison.empty or "valid_label" not in comparison.columns:
        return {
            "valid_goal_rows": 0,
            "goal_accuracy": np.nan,
            "median_endpoint_error_cm": np.nan,
            "mean_true_well_posterior": np.nan,
            "median_true_well_rank": np.nan,
        }
    if "model" in comparison.columns:
        comparison = comparison[comparison["model"] == model]
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


def _aggregate_metric(column: str, values: pd.Series) -> dict[str, float | int]:
    finite_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(finite_values.size)
    if n == 0:
        return {
            f"{column}_n": 0,
            f"{column}_mean": np.nan,
            f"{column}_std": np.nan,
            f"{column}_ci95_low": np.nan,
            f"{column}_ci95_high": np.nan,
        }
    mean = float(np.mean(finite_values))
    std = float(np.std(finite_values, ddof=1)) if n > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        f"{column}_n": n,
        f"{column}_mean": mean,
        f"{column}_std": std,
        f"{column}_ci95_low": mean - half_width,
        f"{column}_ci95_high": mean + half_width,
    }


def _sorted_numeric_values(values: pd.Series) -> list[int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    return sorted(set(int(value) for value in numeric))


def _objective_matrix(
    summary: pd.DataFrame,
    objectives: dict[str, str],
    objective_columns: list[str],
) -> np.ndarray:
    columns = []
    for column in objective_columns:
        direction = objectives[column]
        values = pd.to_numeric(summary[column], errors="coerce").to_numpy(dtype=float)
        if direction == "max":
            columns.append(values)
        elif direction == "min":
            columns.append(-values)
        else:
            raise ValueError("objective directions must be 'max' or 'min'")
    return np.nan_to_num(np.column_stack(columns), nan=-np.inf)


def _is_dominated(candidate: np.ndarray, all_values: np.ndarray) -> bool:
    better_or_equal = np.all(all_values >= candidate, axis=1)
    strictly_better = np.any(all_values > candidate, axis=1)
    return bool(np.any(better_or_equal & strictly_better))


def _validate_nonempty(values: tuple[object, ...], name: str) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one value")
