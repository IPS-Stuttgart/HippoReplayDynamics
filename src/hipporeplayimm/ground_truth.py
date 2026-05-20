"""Behavioral proxy ground truth for open-field replay events."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .benchmarks import BenchmarkConfig, _build_models, _split_cells
from .data import ReplaySession, load_open_field_sessions
from .encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from .evidence_reporting import EXACT_EVIDENCE_SUPPORT, ensure_evidence_support_columns
from .state_space import StateSpaceReplayModel


@dataclass(frozen=True)
class GroundTruthConfig:
    well_arrival_window_s: float = 1.0
    visit_radius_cm: float = 10.0
    min_dwell_s: float = 0.2
    future_horizon_s: float = 30.0
    event_epoch: str = "run"


@dataclass(frozen=True)
class GroundTruthSensitivityConfig:
    """Parameter grid for behavioral-label sensitivity analysis."""

    visit_radii_cm: tuple[float, ...] = (7.5, 10.0, 12.5)
    min_dwells_s: tuple[float, ...] = (0.1, 0.2, 0.4)
    future_horizons_s: tuple[float, ...] = (15.0, 30.0, 60.0)
    well_arrival_window_s: float = 1.0
    event_epoch: str = "run"

    def ground_truth_configs(self) -> tuple[GroundTruthConfig, ...]:
        """Expand the sensitivity grid into concrete label configurations."""

        if not self.visit_radii_cm:
            raise ValueError("visit_radii_cm must contain at least one value")
        if not self.min_dwells_s:
            raise ValueError("min_dwells_s must contain at least one value")
        if not self.future_horizons_s:
            raise ValueError("future_horizons_s must contain at least one value")
        return tuple(
            GroundTruthConfig(
                well_arrival_window_s=float(self.well_arrival_window_s),
                visit_radius_cm=float(visit_radius_cm),
                min_dwell_s=float(min_dwell_s),
                future_horizon_s=float(future_horizon_s),
                event_epoch=self.event_epoch,
            )
            for visit_radius_cm, min_dwell_s, future_horizon_s in product(
                self.visit_radii_cm, self.min_dwells_s, self.future_horizons_s
            )
        )


def generate_behavioral_ground_truth(
    root: str | Path,
    config: GroundTruthConfig | None = None,
    max_events_per_session: int | None = None,
) -> pd.DataFrame:
    """Generate next-well behavioral labels for open-field ripples."""

    config = GroundTruthConfig() if config is None else config
    rows: list[dict[str, object]] = []
    for session in load_open_field_sessions(root):
        session_rows = label_session_behavioral_ground_truth(session, config)
        if max_events_per_session is not None:
            session_rows = session_rows.head(max_events_per_session)
        rows.extend(session_rows.to_dict("records"))
    return pd.DataFrame(rows)


def label_session_behavioral_ground_truth(
    session: ReplaySession,
    config: GroundTruthConfig | None = None,
) -> pd.DataFrame:
    """Label each run ripple with the first post-ripple well visit."""

    config = GroundTruthConfig() if config is None else config
    wells = infer_well_locations(session, config)
    event_indices = _event_indices(session, config.event_epoch)
    rows: list[dict[str, object]] = []
    for event_index in event_indices:
        ripple = session.ripple(int(event_index))
        active_goal_id = active_goal_at_time(session, ripple.peak)
        row: dict[str, object] = {
            "session": session.session_id,
            "event_index": int(event_index),
            "ripple_peak": float(ripple.peak),
            "active_goal_id": _nullable_int(active_goal_id),
        }
        if wells.empty:
            rows.append(_invalid_row(row, "no_well_locations"))
            continue
        visit = first_post_ripple_well_visit(
            session.position,
            wells,
            ripple.peak,
            visit_radius_cm=config.visit_radius_cm,
            min_dwell_s=config.min_dwell_s,
            future_horizon_s=config.future_horizon_s,
        )
        if visit is None:
            rows.append(_invalid_row(row, "no_visit_within_horizon"))
            continue
        row.update(
            {
                "true_well_id": int(visit["well_id"]),
                "true_well_x": float(visit["well_x"]),
                "true_well_y": float(visit["well_y"]),
                "arrival_time": float(visit["arrival_time"]),
                "time_to_arrival_s": float(visit["arrival_time"] - ripple.peak),
                "valid_label": True,
                "exclude_reason": "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def infer_well_locations(
    session: ReplaySession,
    config: GroundTruthConfig | None = None,
) -> pd.DataFrame:
    """Infer well coordinates from shifted well-fill times.

    The dataset stores the time at which a well was filled. In the open-field
    task, the animal reaches that filled well before the next fill event, so
    row ``i`` gets its coordinate estimate from positions just before
    ``Well_Sequence[i + 1, 0]``.
    """

    config = GroundTruthConfig() if config is None else config
    return infer_well_locations_from_arrays(
        session.position,
        session.well_sequence,
        well_arrival_window_s=config.well_arrival_window_s,
    )


def infer_well_locations_from_arrays(
    position: np.ndarray,
    well_sequence: np.ndarray | None,
    well_arrival_window_s: float = 1.0,
) -> pd.DataFrame:
    if well_sequence is None or len(well_sequence) < 2:
        return _empty_wells()
    position = _clean_position(position)
    estimates: list[dict[str, float | int]] = []
    for idx in range(len(well_sequence) - 1):
        well_id = int(well_sequence[idx, 1])
        next_fill_time = float(well_sequence[idx + 1, 0])
        in_window = (
            (position[:, 0] >= next_fill_time - well_arrival_window_s)
            & (position[:, 0] <= next_fill_time)
        )
        if int(np.sum(in_window)) < 3:
            continue
        estimates.append(
            {
                "well_id": well_id,
                "x": float(np.median(position[in_window, 1])),
                "y": float(np.median(position[in_window, 2])),
                "n_estimates": 1,
            }
        )
    if not estimates:
        return _empty_wells()
    estimates_frame = pd.DataFrame(estimates)
    grouped = (
        estimates_frame.groupby("well_id", as_index=False)
        .agg(
            well_x=("x", "median"),
            well_y=("y", "median"),
            n_estimates=("n_estimates", "sum"),
        )
        .sort_values("well_id")
        .reset_index(drop=True)
    )
    grouped["well_id"] = grouped["well_id"].astype(int)
    grouped["n_estimates"] = grouped["n_estimates"].astype(int)
    return grouped


def active_goal_at_time(session: ReplaySession, time_s: float) -> int | None:
    if session.well_sequence is None or session.well_sequence.size == 0:
        return None
    times = session.well_sequence[:, 0]
    idx = int(np.searchsorted(times, time_s, side="right") - 1)
    if idx < 0:
        return None
    return int(session.well_sequence[idx, 1])


def first_post_ripple_well_visit(
    position: np.ndarray,
    wells: pd.DataFrame,
    ripple_peak: float,
    *,
    visit_radius_cm: float,
    min_dwell_s: float,
    future_horizon_s: float,
) -> dict[str, float | int] | None:
    position = _clean_position(position)
    future = position[
        (position[:, 0] >= ripple_peak)
        & (position[:, 0] <= ripple_peak + future_horizon_s)
    ]
    if future.size == 0 or wells.empty:
        return None
    candidates: list[dict[str, float | int]] = []
    for well in wells.itertuples(index=False):
        center = np.array([float(well.well_x), float(well.well_y)])
        distances = np.sqrt(np.sum((future[:, 1:3] - center[None, :]) ** 2, axis=1))
        in_radius = distances <= visit_radius_cm
        for start_idx, end_idx in _true_runs(in_radius):
            dwell_s = float(future[end_idx - 1, 0] - future[start_idx, 0])
            if dwell_s >= min_dwell_s:
                candidates.append(
                    {
                        "well_id": int(well.well_id),
                        "well_x": float(well.well_x),
                        "well_y": float(well.well_y),
                        "arrival_time": float(future[start_idx, 0]),
                        "dwell_s": dwell_s,
                    }
                )
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item["arrival_time"]))
    return candidates[0]


def compare_scores_to_ground_truth(
    root: str | Path,
    scores: str | Path | pd.DataFrame,
    *,
    ground_truth: str | Path | pd.DataFrame | None = None,
    ground_truth_config: GroundTruthConfig | None = None,
    encoding_config: EncodingConfig | None = None,
    emission_config: EmissionConfig | None = None,
    test_cell_fraction: float = 0.25,
    candidate_top_k: int = 64,
    pyrecest_particles: int = 512,
    pyrecest_alpha: float = 0.80,
    pyrecest_beta: float = 1.00,
    pyrecest_process_noise_sigma_cm_s: float = 60.0,
    pyrecest_position_jump_sigma_cm: float = 25.0,
    pyrecest_jump_probability: float = 0.03,
    pyrecest_goal_reset_probability: float = 0.02,
    pyrecest_position_proposal_probability: float = 0.0,
    pyrecest_initial_velocity_sigma_cm_s: float = 120.0,
    pyrecest_imm_mode_stickiness: float = 0.95,
    pyrecest_imm_stationary_velocity_decay: float = 0.0,
    pyrecest_imm_diffusion_velocity_decay: float = 0.0,
    pyrecest_imm_momentum_velocity_decay: float = 0.95,
    pyrecest_imm_jump_fraction: float = 0.9,
    pyrecest_imm_jump_velocity_decay: float = 0.25,
    random_seed: int = 1,
    state_space_valid_occupancy_threshold_s: float = 0.0,
    include_bayesian_model_average: bool = True,
    bayesian_model_average_name: str = "bayesian-model-average",
) -> pd.DataFrame:
    """Merge event scores with next-well behavioral correctness metrics."""

    scores_frame = pd.read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
    gt_frame = _load_or_generate_ground_truth(root, ground_truth, ground_truth_config)
    if scores_frame.empty:
        return scores_frame
    scores_frame = ensure_evidence_support_columns(scores_frame)

    benchmark_decode = _score_table_is_heldout_benchmark(scores_frame)
    encoding_config = _encoding_config_for_scores(
        scores_frame,
        EncodingConfig() if encoding_config is None else encoding_config,
    )
    emission_config = _emission_config_for_scores(
        scores_frame,
        EmissionConfig() if emission_config is None else emission_config,
    )
    test_cell_fraction = _unique_float_from_column(
        scores_frame,
        "benchmark_test_cell_fraction",
        test_cell_fraction,
    )
    random_seed = _unique_int_from_column(
        scores_frame,
        "benchmark_random_seed",
        random_seed,
    )
    state_space_valid_occupancy_threshold_s = _unique_float_from_column(
        scores_frame,
        "state_space_valid_occupancy_threshold_s",
        state_space_valid_occupancy_threshold_s,
    )

    sessions = {session.session_id: session for session in load_open_field_sessions(root)}
    decoded_rows: list[dict[str, object]] = []
    average_score_rows: list[dict[str, object]] = []
    model_names = _model_names_for_scores(scores_frame)
    model_config = BenchmarkConfig(
        encoding=encoding_config,
        emissions=emission_config,
        test_cell_fraction=test_cell_fraction,
        candidate_top_k=candidate_top_k,
        pyrecest_particles=pyrecest_particles,
        pyrecest_alpha=pyrecest_alpha,
        pyrecest_beta=pyrecest_beta,
        pyrecest_process_noise_sigma_cm_s=pyrecest_process_noise_sigma_cm_s,
        pyrecest_position_jump_sigma_cm=pyrecest_position_jump_sigma_cm,
        pyrecest_jump_probability=pyrecest_jump_probability,
        pyrecest_goal_reset_probability=pyrecest_goal_reset_probability,
        pyrecest_position_proposal_probability=pyrecest_position_proposal_probability,
        pyrecest_initial_velocity_sigma_cm_s=pyrecest_initial_velocity_sigma_cm_s,
        pyrecest_imm_mode_stickiness=pyrecest_imm_mode_stickiness,
        pyrecest_imm_stationary_velocity_decay=pyrecest_imm_stationary_velocity_decay,
        pyrecest_imm_diffusion_velocity_decay=pyrecest_imm_diffusion_velocity_decay,
        pyrecest_imm_momentum_velocity_decay=pyrecest_imm_momentum_velocity_decay,
        pyrecest_imm_jump_fraction=pyrecest_imm_jump_fraction,
        pyrecest_imm_jump_velocity_decay=pyrecest_imm_jump_velocity_decay,
        state_space_valid_occupancy_threshold_s=state_space_valid_occupancy_threshold_s,
        random_seed=random_seed,
        models=model_names,
    )

    decode_group_columns = _decode_group_columns(scores_frame, benchmark_decode)
    for group_key, session_scores in scores_frame.groupby(decode_group_columns, sort=False):
        group_values = _group_key_values(decode_group_columns, group_key)
        session_id = str(group_values["session"])
        session = sessions.get(session_id)
        if session is None:
            continue
        models = _build_models(model_config, session=session)
        wells = infer_well_locations(session, ground_truth_config)
        encoding = fit_place_field_encoding(session, encoding_config)
        if benchmark_decode:
            train_cells, test_cells = _cell_split_for_score_rows(
                session_scores,
                encoding,
                model_config,
            )
            train_encoding = encoding.select_cells(train_cells)
            joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))
        for event_index, event_scores in session_scores.groupby("event_index", sort=False):
            if benchmark_decode:
                train_emissions = build_emissions(
                    session,
                    train_encoding,
                    int(event_index),
                    emission_config,
                )
                joint_emissions = build_emissions(
                    session,
                    joint_encoding,
                    int(event_index),
                    emission_config,
                )
                if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
                    continue
            else:
                emissions = build_emissions(session, encoding, int(event_index), emission_config)
                if emissions.n_time == 0:
                    continue
            average_components: list[tuple[str, float, np.ndarray]] = []
            for score_row in event_scores.itertuples(index=False):
                model_name = str(getattr(score_row, "model"))
                requested_model = _requested_model_name(score_row, model_name)
                model = models.get(requested_model) or models.get(model_name)
                if model is None:
                    continue
                if benchmark_decode:
                    score = _score_joint_for_ground_truth(
                        model,
                        train_emissions,
                        joint_emissions,
                        encoding.bin_centers,
                        occupancy_s=encoding.occupancy_s,
                    )
                else:
                    if isinstance(model, StateSpaceReplayModel):
                        score = model.score(
                            emissions,
                            encoding.bin_centers,
                            occupancy_s=encoding.occupancy_s,
                        )
                    else:
                        score = model.score(emissions, encoding.bin_centers)
                decoded_rows.append(
                    _decoded_row(
                        str(session_id),
                        int(event_index),
                        model_name,
                        score.terminal_log_posterior,
                        score.trajectory_log_posterior,
                        encoding.bin_centers,
                        wells,
                    )
                )
                log_evidence = _score_row_log_evidence(score_row)
                if (
                    include_bayesian_model_average
                    and log_evidence is not None
                    and _score_row_has_exact_comparable_evidence(score_row)
                    and score.terminal_log_posterior is not None
                ):
                    average_components.append((model_name, log_evidence, score.terminal_log_posterior))
            average_log_posterior = _bayesian_model_average_log_posterior(average_components)
            if average_log_posterior is not None:
                decoded_rows.append(
                    _decoded_row(
                        str(session_id),
                        int(event_index),
                        bayesian_model_average_name,
                        average_log_posterior,
                        None,
                        encoding.bin_centers,
                        wells,
                    )
                )
                average_score_rows.append(
                    _bayesian_model_average_score_row(
                        event_scores,
                        average_components,
                        bayesian_model_average_name,
                    )
                )
    if average_score_rows:
        scores_frame = pd.concat(
            [scores_frame, pd.DataFrame(average_score_rows)],
            ignore_index=True,
            sort=False,
        )
    decoded = pd.DataFrame(decoded_rows)
    comparison = scores_frame.merge(gt_frame, on=["session", "event_index"], how="left")
    comparison = comparison.merge(decoded, on=["session", "event_index", "model"], how="left")
    comparison = _add_ground_truth_metrics(comparison, decoded, gt_frame)
    return comparison


@dataclass(frozen=True)
class GroundTruthSensitivityResult:
    """Event-level and aggregate outputs from behavioral-label sensitivity."""

    rows: pd.DataFrame
    per_setting_summary: pd.DataFrame
    robustness_summary: pd.DataFrame

    def write(self, output: str | Path) -> None:
        """Write sensitivity outputs into a directory."""

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.rows.to_csv(output / "ground_truth_sensitivity_rows.csv", index=False)
        self.per_setting_summary.to_csv(
            output / "ground_truth_sensitivity_by_setting.csv",
            index=False,
        )
        self.robustness_summary.to_csv(
            output / "ground_truth_sensitivity_robustness.csv",
            index=False,
        )


def compare_scores_to_ground_truth_sensitivity(
    root: str | Path,
    scores: str | Path | pd.DataFrame,
    *,
    sensitivity_config: GroundTruthSensitivityConfig | None = None,
    encoding_config: EncodingConfig | None = None,
    emission_config: EmissionConfig | None = None,
    test_cell_fraction: float = 0.25,
    candidate_top_k: int = 64,
    pyrecest_particles: int = 512,
    pyrecest_alpha: float = 0.80,
    pyrecest_beta: float = 1.00,
    pyrecest_process_noise_sigma_cm_s: float = 60.0,
    pyrecest_position_jump_sigma_cm: float = 25.0,
    pyrecest_jump_probability: float = 0.03,
    pyrecest_goal_reset_probability: float = 0.02,
    pyrecest_position_proposal_probability: float = 0.0,
    pyrecest_initial_velocity_sigma_cm_s: float = 120.0,
    pyrecest_imm_mode_stickiness: float = 0.95,
    pyrecest_imm_stationary_velocity_decay: float = 0.0,
    pyrecest_imm_diffusion_velocity_decay: float = 0.0,
    pyrecest_imm_momentum_velocity_decay: float = 0.95,
    pyrecest_imm_jump_fraction: float = 0.9,
    pyrecest_imm_jump_velocity_decay: float = 0.25,
    random_seed: int = 1,
) -> GroundTruthSensitivityResult:
    """Evaluate behavioral-score robustness across label-parameter settings.

    Decoding is run once using the first grid setting. Each additional setting
    only regenerates the behavioral proxy labels and recomputes correctness,
    endpoint-error, and true-well-posterior metrics. This keeps the sensitivity
    analysis focused on label robustness rather than stochastic decoder noise.
    """

    sensitivity_config = (
        GroundTruthSensitivityConfig()
        if sensitivity_config is None
        else sensitivity_config
    )
    label_configs = sensitivity_config.ground_truth_configs()
    reference_config = label_configs[0]
    reference_comparison = compare_scores_to_ground_truth(
        root,
        scores,
        ground_truth_config=reference_config,
        encoding_config=encoding_config,
        emission_config=emission_config,
        test_cell_fraction=test_cell_fraction,
        candidate_top_k=candidate_top_k,
        pyrecest_particles=pyrecest_particles,
        pyrecest_alpha=pyrecest_alpha,
        pyrecest_beta=pyrecest_beta,
        pyrecest_process_noise_sigma_cm_s=pyrecest_process_noise_sigma_cm_s,
        pyrecest_position_jump_sigma_cm=pyrecest_position_jump_sigma_cm,
        pyrecest_jump_probability=pyrecest_jump_probability,
        pyrecest_goal_reset_probability=pyrecest_goal_reset_probability,
        pyrecest_position_proposal_probability=pyrecest_position_proposal_probability,
        pyrecest_initial_velocity_sigma_cm_s=pyrecest_initial_velocity_sigma_cm_s,
        pyrecest_imm_mode_stickiness=pyrecest_imm_mode_stickiness,
        pyrecest_imm_stationary_velocity_decay=pyrecest_imm_stationary_velocity_decay,
        pyrecest_imm_diffusion_velocity_decay=pyrecest_imm_diffusion_velocity_decay,
        pyrecest_imm_momentum_velocity_decay=pyrecest_imm_momentum_velocity_decay,
        pyrecest_imm_jump_fraction=pyrecest_imm_jump_fraction,
        pyrecest_imm_jump_velocity_decay=pyrecest_imm_jump_velocity_decay,
        random_seed=random_seed,
    )
    score_decode_base = _ground_truth_sensitivity_score_decode_base(
        reference_comparison
    )
    frames: list[pd.DataFrame] = []
    for label_config in label_configs:
        gt_frame = generate_behavioral_ground_truth(root, config=label_config)
        comparison = score_decode_base.merge(
            gt_frame,
            on=["session", "event_index"],
            how="left",
        )
        comparison = _add_ground_truth_metrics(
            comparison,
            decoded=pd.DataFrame(),
            gt_frame=gt_frame,
        )
        comparison = _add_ground_truth_sensitivity_parameter_columns(
            comparison,
            label_config,
        )
        frames.append(comparison)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    per_setting_summary = summarize_ground_truth_sensitivity_by_setting(rows)
    robustness_summary = summarize_ground_truth_sensitivity(rows)
    return GroundTruthSensitivityResult(
        rows=rows,
        per_setting_summary=per_setting_summary,
        robustness_summary=robustness_summary,
    )


def summarize_ground_truth_sensitivity_by_setting(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize behavioral metrics for each model and label setting."""

    columns = [
        "model",
        *_GROUND_TRUTH_SENSITIVITY_PARAMETER_COLUMNS,
        "rows",
        "goal_accuracy",
        "active_goal_accuracy",
        "median_endpoint_error_cm",
        "mean_true_well_posterior",
    ]
    if frame.empty or "valid_label" not in frame.columns or "goal_correct" not in frame.columns:
        return pd.DataFrame(columns=columns)
    valid = frame[_valid_label_mask(frame["valid_label"], frame.index).to_numpy(dtype=bool)]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    if "active_goal_correct" not in valid.columns:
        valid = valid.copy()
        valid["active_goal_correct"] = np.nan
    return (
        valid.groupby(["model", *_GROUND_TRUTH_SENSITIVITY_PARAMETER_COLUMNS], as_index=False)
        .agg(
            rows=("event_index", "count"),
            goal_accuracy=("goal_correct", "mean"),
            active_goal_accuracy=("active_goal_correct", "mean"),
            median_endpoint_error_cm=("endpoint_error_cm", "median"),
            mean_true_well_posterior=("true_well_posterior", "mean"),
        )
        .sort_values(["model", *_GROUND_TRUTH_SENSITIVITY_PARAMETER_COLUMNS])
    )


def summarize_ground_truth_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-setting summaries into robustness ranges per model."""

    per_setting = summarize_ground_truth_sensitivity_by_setting(frame)
    columns = [
        "model",
        "settings",
        "min_rows",
        "max_rows",
        "min_goal_accuracy",
        "median_goal_accuracy",
        "max_goal_accuracy",
        "goal_accuracy_range",
        "min_active_goal_accuracy",
        "median_active_goal_accuracy",
        "max_active_goal_accuracy",
        "active_goal_accuracy_range",
        "min_median_endpoint_error_cm",
        "median_median_endpoint_error_cm",
        "max_median_endpoint_error_cm",
        "min_mean_true_well_posterior",
        "median_mean_true_well_posterior",
        "max_mean_true_well_posterior",
    ]
    if per_setting.empty:
        return pd.DataFrame(columns=columns)
    return (
        per_setting.groupby("model", as_index=False)
        .agg(
            settings=("goal_accuracy", "count"),
            min_rows=("rows", "min"),
            max_rows=("rows", "max"),
            min_goal_accuracy=("goal_accuracy", "min"),
            median_goal_accuracy=("goal_accuracy", "median"),
            max_goal_accuracy=("goal_accuracy", "max"),
            goal_accuracy_range=(
                "goal_accuracy",
                lambda values: float(values.max() - values.min()),
            ),
            min_active_goal_accuracy=("active_goal_accuracy", "min"),
            median_active_goal_accuracy=("active_goal_accuracy", "median"),
            max_active_goal_accuracy=("active_goal_accuracy", "max"),
            active_goal_accuracy_range=(
                "active_goal_accuracy",
                lambda values: float(values.max() - values.min()),
            ),
            min_median_endpoint_error_cm=("median_endpoint_error_cm", "min"),
            median_median_endpoint_error_cm=("median_endpoint_error_cm", "median"),
            max_median_endpoint_error_cm=("median_endpoint_error_cm", "max"),
            min_mean_true_well_posterior=("mean_true_well_posterior", "min"),
            median_mean_true_well_posterior=("mean_true_well_posterior", "median"),
            max_mean_true_well_posterior=("mean_true_well_posterior", "max"),
        )
        .sort_values("model")
    )


_GROUND_TRUTH_COLUMNS_FOR_SENSITIVITY = {
    "ripple_peak",
    "active_goal_id",
    "true_well_id",
    "true_well_x",
    "true_well_y",
    "arrival_time",
    "time_to_arrival_s",
    "valid_label",
    "exclude_reason",
    "goal_correct",
    "endpoint_error_cm",
    "true_well_posterior",
    "true_well_rank",
}

_GROUND_TRUTH_SENSITIVITY_PARAMETER_COLUMNS = (
    "visit_radius_cm",
    "min_dwell_s",
    "future_horizon_s",
    "well_arrival_window_s",
    "event_epoch",
)


def _ground_truth_sensitivity_score_decode_base(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            column
            for column in _GROUND_TRUTH_COLUMNS_FOR_SENSITIVITY
            if column in frame.columns
        ]
    )


def _add_ground_truth_sensitivity_parameter_columns(
    frame: pd.DataFrame,
    config: GroundTruthConfig,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["visit_radius_cm"] = float(config.visit_radius_cm)
    frame["min_dwell_s"] = float(config.min_dwell_s)
    frame["future_horizon_s"] = float(config.future_horizon_s)
    frame["well_arrival_window_s"] = float(config.well_arrival_window_s)
    frame["event_epoch"] = config.event_epoch
    return frame


def _score_table_is_heldout_benchmark(scores_frame: pd.DataFrame) -> bool:
    return {
        "heldout_log_likelihood",
        "train_log_likelihood",
        "joint_log_likelihood",
    }.issubset(scores_frame.columns)


def _decode_group_columns(scores_frame: pd.DataFrame, benchmark_decode: bool) -> list[str]:
    columns = ["session"]
    if benchmark_decode and "benchmark_cell_split_index" in scores_frame.columns:
        columns.append("benchmark_cell_split_index")
    return columns


def _group_key_values(columns: list[str], group_key: object) -> dict[str, object]:
    if len(columns) == 1:
        values = (group_key,)
    else:
        values = tuple(group_key) if isinstance(group_key, tuple) else (group_key,)
    if len(values) != len(columns):
        raise ValueError("group key shape does not match group columns")
    return dict(zip(columns, values))


def _model_names_for_scores(scores_frame: pd.DataFrame) -> tuple[str, ...]:
    names: list[str] = []
    for column in ("requested_model", "model"):
        if column not in scores_frame.columns:
            continue
        for value in scores_frame[column].dropna():
            model_name = str(value)
            if model_name and model_name not in names:
                names.append(model_name)
    return tuple(names)


def _requested_model_name(score_row: object, fallback: str) -> str:
    value = getattr(score_row, "requested_model", None)
    if value is None or pd.isna(value):
        return fallback
    return str(value)


def _score_row_log_evidence(score_row: object) -> float | None:
    value = _score_row_value(score_row, "log_evidence", None)
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _score_row_has_exact_comparable_evidence(score_row: object) -> bool:
    status = _score_row_value(score_row, "status", "success")
    if pd.notna(status) and str(status) != "success":
        return False
    comparable = _score_row_value(score_row, "evidence_comparable", False)
    if comparable is None or pd.isna(comparable):
        return False
    return _parse_bool(comparable)


def _score_row_value(score_row: object, column: str, default: object = None) -> object:
    if isinstance(score_row, pd.Series):
        return score_row.get(column, default)
    return getattr(score_row, column, default)


def _bayesian_model_average_log_posterior(
    components: list[tuple[str, float, np.ndarray]],
) -> np.ndarray | None:
    """Return the evidence-weighted posterior mixture for exact model scores."""

    if len(components) < 2:
        return None
    _, log_evidences, log_posteriors = _unpack_bayesian_model_average_components(components)
    if len({posterior.shape for posterior in log_posteriors}) != 1:
        raise ValueError("Bayesian model average requires posterior arrays with matching shapes")
    log_weights = log_evidences - logsumexp(log_evidences)
    normalized_posteriors = np.vstack(
        [posterior - logsumexp(posterior) for posterior in log_posteriors]
    )
    mixture = logsumexp(normalized_posteriors + log_weights[:, None], axis=0)
    return mixture - logsumexp(mixture)


def _bayesian_model_average_score_row(
    event_scores: pd.DataFrame,
    components: list[tuple[str, float, np.ndarray]],
    model_name: str,
) -> dict[str, object]:
    """Build a traceable score-table row for a posterior model average.

    The model average is a predictive posterior mixture, not a new generative
    model to rank against its components. Keep its evidence non-comparable while
    retaining the component evidences and posterior model weights for auditing.
    """

    names, log_evidences, _ = _unpack_bayesian_model_average_components(components)
    weights = np.exp(log_evidences - logsumexp(log_evidences))
    row: dict[str, object]
    if event_scores.empty:
        row = {}
    else:
        row = event_scores.iloc[0].to_dict()
    for column in list(row):
        if column.startswith("diagnostic_"):
            row[column] = np.nan
    row.update(
        {
            "model": model_name,
            "requested_model": model_name,
            "model_family": "model-average",
            "status": "success",
            "log_evidence": np.nan,
            "relative_log_evidence": np.nan,
            "model_probability": np.nan,
            "is_best_model": False,
            "evidence_support": EXACT_EVIDENCE_SUPPORT,
            "evidence_comparable": False,
            "bma_component_count": int(len(names)),
            "bma_component_models": ",".join(names),
            "bma_component_log_evidences": ",".join(f"{value:.12g}" for value in log_evidences),
            "bma_component_weights": ",".join(f"{value:.12g}" for value in weights),
        }
    )
    return row


def _unpack_bayesian_model_average_components(
    components: list[tuple[str, float, np.ndarray]],
) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    names = [str(name) for name, _, _ in components]
    log_evidences = np.asarray([float(value) for _, value, _ in components], dtype=float)
    log_posteriors = [np.asarray(posterior, dtype=float) for _, _, posterior in components]
    return names, log_evidences, log_posteriors


def _score_joint_for_ground_truth(
    model,
    train_emissions,
    joint_emissions,
    bin_centers: np.ndarray,
    occupancy_s=None,
):
    if isinstance(model, StateSpaceReplayModel):
        candidates = model.candidate_indices(train_emissions, bin_centers)
        return model.score(joint_emissions, bin_centers, candidate_indices=candidates, occupancy_s=occupancy_s)
    if hasattr(model, "candidate_indices"):
        candidates = model.candidate_indices(train_emissions)
        return model.score(joint_emissions, bin_centers, candidate_indices=candidates)
    return model.score(joint_emissions, bin_centers)


def _cell_split_for_score_rows(
    session_scores: pd.DataFrame,
    encoding,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    train_cells = _cell_ids_from_score_column(session_scores, "train_cell_ids")
    test_cells = _cell_ids_from_score_column(session_scores, "test_cell_ids")
    if train_cells is not None or test_cells is not None:
        if train_cells is None or test_cells is None:
            raise ValueError("score rows must provide both train_cell_ids and test_cell_ids")
        if train_cells.size == 0 or test_cells.size == 0:
            raise ValueError("train_cell_ids and test_cell_ids must both be non-empty")
        if np.intersect1d(train_cells, test_cells).size:
            raise ValueError("train_cell_ids and test_cell_ids must not overlap")
        _validate_cell_ids_in_encoding(train_cells, encoding, "train_cell_ids")
        _validate_cell_ids_in_encoding(test_cells, encoding, "test_cell_ids")
        return np.sort(train_cells), np.sort(test_cells)

    test_cell_fraction = _unique_float_from_column(
        session_scores,
        "benchmark_test_cell_fraction",
        config.test_cell_fraction,
    )
    benchmark_random_seed = _unique_int_from_column(
        session_scores,
        "benchmark_random_seed",
        config.random_seed,
    )
    random_seed = _unique_int_from_column(
        session_scores,
        "benchmark_cell_split_seed",
        benchmark_random_seed,
    )
    return _split_cells(encoding.cell_ids, test_cell_fraction, random_seed)


def _cell_ids_from_score_column(
    session_scores: pd.DataFrame,
    column: str,
) -> np.ndarray | None:
    if column not in session_scores.columns:
        return None
    parsed: list[tuple[int, ...]] = []
    for value in session_scores[column]:
        ids = _parse_cell_ids(value)
        if ids is not None:
            parsed.append(tuple(int(cell_id) for cell_id in ids))
    if not parsed:
        return None
    unique = set(parsed)
    if len(unique) != 1:
        raise ValueError(f"{column} differs within a session score table")
    return np.asarray(next(iter(unique)), dtype=int)


def _parse_cell_ids(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=int)
    if isinstance(value, (list, tuple, set)):
        return np.asarray(list(value), dtype=int)
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.strip("[]()").replace(",", " ")
    return np.asarray([int(float(piece)) for piece in text.split()], dtype=int)


def _validate_cell_ids_in_encoding(cell_ids: np.ndarray, encoding, column: str) -> None:
    missing = np.setdiff1d(cell_ids, encoding.cell_ids)
    if missing.size:
        missing_text = ",".join(str(int(cell_id)) for cell_id in missing)
        raise ValueError(f"{column} contains cell IDs not present in the encoding: {missing_text}")


def _encoding_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: EncodingConfig,
) -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=_unique_float_from_column(scores_frame, "encoding_bin_size_cm", fallback.bin_size_cm),
        smoothing_sigma_bins=_unique_float_from_column(
            scores_frame,
            "encoding_smoothing_sigma_bins",
            fallback.smoothing_sigma_bins,
        ),
        min_speed_cm_s=_unique_float_from_column(
            scores_frame,
            "encoding_min_speed_cm_s",
            fallback.min_speed_cm_s,
        ),
        min_occupancy_s=_unique_float_from_column(
            scores_frame,
            "encoding_min_occupancy_s",
            fallback.min_occupancy_s,
        ),
        rate_floor_hz=_unique_float_from_column(
            scores_frame,
            "encoding_rate_floor_hz",
            fallback.rate_floor_hz,
        ),
        arena_padding_cm=_unique_float_from_column(
            scores_frame,
            "encoding_arena_padding_cm",
            fallback.arena_padding_cm,
        ),
        use_excitatory=_unique_bool_from_column(
            scores_frame,
            "encoding_use_excitatory",
            fallback.use_excitatory,
        ),
    )


def _emission_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: EmissionConfig,
) -> EmissionConfig:
    return EmissionConfig(
        time_bin_s=_unique_float_from_columns(
            scores_frame,
            ("emission_time_bin_s", "time_bin_s"),
            fallback.time_bin_s,
        ),
        spike_rate_scale=_unique_float_from_columns(
            scores_frame,
            ("emission_spike_rate_scale", "spike_rate_scale"),
            fallback.spike_rate_scale,
        ),
        likelihood_temperature=_unique_float_from_columns(
            scores_frame,
            ("emission_likelihood_temperature", "likelihood_temperature"),
            fallback.likelihood_temperature,
        ),
        negative_binomial_overdispersion=_unique_float_from_columns(
            scores_frame,
            ("emission_negative_binomial_overdispersion", "negative_binomial_overdispersion"),
            fallback.negative_binomial_overdispersion,
        ),
    )


def _unique_float_from_column(frame: pd.DataFrame, column: str, default: float) -> float:
    return _unique_float_from_columns(frame, (column,), default)


def _unique_float_from_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    default: float,
) -> float:
    values: list[float] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(float(value))
    if not values:
        return float(default)
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)


def _unique_int_from_column(frame: pd.DataFrame, column: str, default: int) -> int:
    values: list[int] = []
    if column in frame.columns:
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(int(float(value)))
    if not values:
        return int(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{column} contains multiple values")
    return int(first)


def _unique_bool_from_column(frame: pd.DataFrame, column: str, default: bool) -> bool:
    values: list[bool] = []
    if column in frame.columns:
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(_parse_bool(value))
    if not values:
        return bool(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{column} contains multiple values")
    return bool(first)


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _decoded_row(
    session_id: str,
    event_index: int,
    model_name: str,
    terminal_log_posterior: np.ndarray | None,
    trajectory_log_posterior: np.ndarray | None,
    bin_centers: np.ndarray,
    wells: pd.DataFrame,
) -> dict[str, object]:
    terminal_log_posterior, trajectory_log_posterior = _coerce_posterior_trajectory(
        terminal_log_posterior, trajectory_log_posterior
    )
    if terminal_log_posterior is None or trajectory_log_posterior is None:
        return {
            "session": session_id,
            "event_index": event_index,
            "model": model_name,
        }
    terminal_log_posterior = terminal_log_posterior - logsumexp(terminal_log_posterior)
    posterior = np.exp(terminal_log_posterior)
    endpoint = posterior @ bin_centers
    decoded_well = assign_endpoint_to_well(endpoint, wells)
    initial_log_posterior = trajectory_log_posterior[0]
    initial_posterior = np.exp(initial_log_posterior)
    initial_endpoint = initial_posterior @ bin_centers
    initial_decoded_well = assign_endpoint_to_well(initial_endpoint, wells)
    masses = well_posterior_masses(
        terminal_log_posterior,
        bin_centers,
        wells,
    )
    trajectory_masses = trajectory_well_posterior_masses(
        trajectory_log_posterior,
        bin_centers,
        wells,
    )
    max_well_id = _decoded_well_id_from_masses(
        {well_id: float(summary["max"]) for well_id, summary in trajectory_masses.items()}
    )
    trajectory_mean_well_id = _decoded_well_id_from_masses(
        {well_id: float(summary["trajectory"]) for well_id, summary in trajectory_masses.items()}
    )
    integrated_log_posterior = logsumexp(trajectory_log_posterior, axis=0) - np.log(
        trajectory_log_posterior.shape[0]
    )
    integrated_posterior = np.exp(integrated_log_posterior)
    integrated_endpoint = integrated_posterior @ bin_centers
    integrated_decoded_well = assign_endpoint_to_well(integrated_endpoint, wells)
    row: dict[str, object] = {
        "session": session_id,
        "event_index": event_index,
        "model": model_name,
        "decoded_endpoint_x": float(endpoint[0]),
        "decoded_endpoint_y": float(endpoint[1]),
        "decoded_well_id": np.nan if decoded_well is None else int(decoded_well["well_id"]),
        "initial_decoded_endpoint_x": float(initial_endpoint[0]),
        "initial_decoded_endpoint_y": float(initial_endpoint[1]),
        "initial_decoded_well_id": (
            np.nan if initial_decoded_well is None else int(initial_decoded_well["well_id"])
        ),
        "max_over_time_decoded_well_id": max_well_id,
        "decoded_max_posterior_well_id": max_well_id,
        "trajectory_mean_decoded_well_id": trajectory_mean_well_id,
        "decoded_integrated_endpoint_x": float(integrated_endpoint[0]),
        "decoded_integrated_endpoint_y": float(integrated_endpoint[1]),
        "decoded_integrated_well_id": (
            np.nan
            if integrated_decoded_well is None
            else int(integrated_decoded_well["well_id"])
        ),
    }
    for well_id, mass in masses.items():
        row[f"well_{well_id}_posterior"] = float(mass)
    for well_id, summary in trajectory_masses.items():
        row[f"initial_well_{well_id}_posterior"] = float(summary["initial"])
        row[f"max_well_{well_id}_posterior"] = float(summary["max"])
        row[f"well_{well_id}_max_posterior"] = float(summary["max"])
        row[f"max_well_{well_id}_posterior_time_bin"] = int(summary["max_time_index"])
        row[f"trajectory_well_{well_id}_posterior"] = float(summary["trajectory"])
        row[f"well_{well_id}_integrated_posterior"] = float(summary["trajectory"])
    return row


def _coerce_posterior_trajectory(
    terminal_log_posterior: np.ndarray | None,
    trajectory_log_posterior: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if trajectory_log_posterior is None:
        if terminal_log_posterior is None:
            return None, None
        terminal = _normalize_log_posterior(terminal_log_posterior)
        return terminal, terminal[None, :]
    trajectory = np.asarray(trajectory_log_posterior, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.ndim != 2:
        raise ValueError("trajectory_log_posterior must be one- or two-dimensional")
    trajectory = trajectory - logsumexp(trajectory, axis=1, keepdims=True)
    if terminal_log_posterior is None:
        return trajectory[-1], trajectory
    return _normalize_log_posterior(terminal_log_posterior), trajectory


def _normalize_log_posterior(log_posterior: np.ndarray) -> np.ndarray:
    values = np.asarray(log_posterior, dtype=float)
    return values - logsumexp(values)


def _decoded_well_id_from_masses(masses: dict[int, float]) -> int | float:
    if not masses:
        return np.nan
    return int(max(masses.items(), key=lambda item: item[1])[0])


def assign_endpoint_to_well(endpoint_xy: np.ndarray, wells: pd.DataFrame) -> dict[str, float | int] | None:
    if wells.empty:
        return None
    centers = wells[["well_x", "well_y"]].to_numpy(dtype=float)
    distances = np.sqrt(np.sum((centers - np.asarray(endpoint_xy, dtype=float)[None, :]) ** 2, axis=1))
    idx = int(np.argmin(distances))
    well = wells.iloc[idx]
    return {
        "well_id": int(well["well_id"]),
        "well_x": float(well["well_x"]),
        "well_y": float(well["well_y"]),
        "distance_cm": float(distances[idx]),
    }


def well_posterior_masses(
    terminal_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    wells: pd.DataFrame,
    radius_cm: float = 10.0,
) -> dict[int, float]:
    if wells.empty:
        return {}
    normalized = _normalize_log_posterior(terminal_log_posterior)
    posterior = np.exp(normalized)
    masses: dict[int, float] = {}
    for well_id, in_radius in _well_bin_masks(bin_centers, wells, radius_cm).items():
        masses[well_id] = float(np.sum(posterior[in_radius]))
    return masses


def trajectory_well_posterior_masses(
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    wells: pd.DataFrame,
    radius_cm: float = 10.0,
) -> dict[int, dict[str, float | int]]:
    if wells.empty:
        return {}
    trajectory = np.asarray(trajectory_log_posterior, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.ndim != 2:
        raise ValueError("trajectory_log_posterior must be one- or two-dimensional")
    normalized = trajectory - logsumexp(trajectory, axis=1, keepdims=True)
    posterior = np.exp(normalized)
    summaries: dict[int, dict[str, float | int]] = {}
    for well_id, in_radius in _well_bin_masks(bin_centers, wells, radius_cm).items():
        time_series = np.sum(posterior[:, in_radius], axis=1)
        max_time_index = int(np.argmax(time_series))
        summaries[well_id] = {
            "initial": float(time_series[0]),
            "max": float(time_series[max_time_index]),
            "max_time_index": max_time_index,
            "trajectory": float(np.mean(time_series)),
        }
    return summaries


def _well_bin_masks(bin_centers: np.ndarray, wells: pd.DataFrame, radius_cm: float) -> dict[int, np.ndarray]:
    well_ids = [int(well.well_id) for well in wells.itertuples(index=False)]
    centers = wells[["well_x", "well_y"]].to_numpy(dtype=float)
    distances = np.sqrt(
        np.sum((centers[:, None, :] - bin_centers[None, :, :]) ** 2, axis=2)
    )
    mask_matrix = distances <= float(radius_cm)
    for row_index in range(mask_matrix.shape[0]):
        if not np.any(mask_matrix[row_index]):
            mask_matrix[row_index, int(np.argmin(distances[row_index]))] = True
    for bin_index in range(mask_matrix.shape[1]):
        owners = np.flatnonzero(mask_matrix[:, bin_index])
        if owners.size > 1:
            nearest = owners[int(np.argmin(distances[owners, bin_index]))]
            mask_matrix[owners, bin_index] = False
            mask_matrix[nearest, bin_index] = True
    masks: dict[int, np.ndarray] = {}
    for well_id, row in zip(well_ids, mask_matrix, strict=True):
        masks[well_id] = row
    return masks


def _add_ground_truth_metrics(
    comparison: pd.DataFrame,
    decoded: pd.DataFrame,
    gt_frame: pd.DataFrame,
) -> pd.DataFrame:
    del decoded, gt_frame
    true_ids = comparison.get("true_well_id")
    decoded_ids = comparison.get("decoded_well_id")
    valid = comparison.get("valid_label")
    if true_ids is None or decoded_ids is None:
        return comparison
    valid_bool = _valid_label_mask(valid, comparison.index)
    for decoded_col, output_col in (
        ("decoded_well_id", "goal_correct"),
        ("initial_decoded_well_id", "initial_goal_correct"),
        ("max_over_time_decoded_well_id", "max_over_time_goal_correct"),
        ("decoded_max_posterior_well_id", "goal_correct_max_posterior"),
        ("trajectory_mean_decoded_well_id", "trajectory_mean_goal_correct"),
        ("decoded_integrated_well_id", "goal_correct_integrated"),
    ):
        if decoded_col in comparison.columns:
            comparison[output_col] = np.where(
                valid_bool,
                comparison[decoded_col].astype("float64") == true_ids.astype("float64"),
                np.nan,
            )
    active_goal_ids = comparison.get("active_goal_id")
    if active_goal_ids is not None:
        active_valid = valid_bool & active_goal_ids.notna()
        comparison["active_goal_correct"] = np.where(
            active_valid,
            decoded_ids.astype("float64") == active_goal_ids.astype("float64"),
            np.nan,
        )
    if {"decoded_endpoint_x", "decoded_endpoint_y", "true_well_x", "true_well_y"}.issubset(
        comparison.columns
    ):
        dx = comparison["decoded_endpoint_x"].astype(float) - comparison["true_well_x"].astype(float)
        dy = comparison["decoded_endpoint_y"].astype(float) - comparison["true_well_y"].astype(float)
        comparison["endpoint_error_cm"] = np.where(valid_bool, np.sqrt(dx * dx + dy * dy), np.nan)
    _add_true_well_summary_columns(
        comparison,
        valid_bool,
        column_prefix="",
        posterior_output="true_well_posterior",
        rank_output="true_well_rank",
    )
    _add_true_well_summary_columns(
        comparison,
        valid_bool,
        column_prefix="initial",
        posterior_output="true_initial_well_posterior",
        rank_output="true_initial_well_rank",
    )
    _add_true_well_summary_columns(
        comparison,
        valid_bool,
        column_prefix="max",
        posterior_output="true_max_well_posterior",
        rank_output="true_max_well_rank",
    )
    _add_true_well_summary_columns(
        comparison,
        valid_bool,
        column_prefix="trajectory",
        posterior_output="true_trajectory_well_posterior",
        rank_output="true_trajectory_well_rank",
    )
    if {
        "decoded_integrated_endpoint_x",
        "decoded_integrated_endpoint_y",
        "true_well_x",
        "true_well_y",
    }.issubset(comparison.columns):
        dx = comparison["decoded_integrated_endpoint_x"].astype(float) - comparison["true_well_x"].astype(float)
        dy = comparison["decoded_integrated_endpoint_y"].astype(float) - comparison["true_well_y"].astype(float)
        comparison["integrated_endpoint_error_cm"] = np.where(valid_bool, np.sqrt(dx * dx + dy * dy), np.nan)
    _add_true_well_metric_columns(
        comparison,
        valid_bool,
        suffix="_max_posterior",
        posterior_output="true_well_max_posterior",
        rank_output="true_well_max_rank",
    )
    _add_true_well_metric_columns(
        comparison,
        valid_bool,
        suffix="_integrated_posterior",
        posterior_output="true_well_integrated_posterior",
        rank_output="true_well_integrated_rank",
    )
    _add_active_well_summary_columns(comparison)
    return comparison


def _add_true_well_summary_columns(
    comparison: pd.DataFrame,
    valid_bool: pd.Series,
    *,
    column_prefix: str,
    posterior_output: str,
    rank_output: str,
) -> None:
    posterior_columns = _well_posterior_columns(comparison, column_prefix)
    true_masses: list[float] = []
    true_ranks: list[float] = []
    for is_valid, row in zip(valid_bool.to_numpy(dtype=bool), comparison.itertuples(index=False)):
        if not is_valid:
            true_masses.append(np.nan)
            true_ranks.append(np.nan)
            continue
        true_well_value = getattr(row, "true_well_id", np.nan)
        if pd.isna(true_well_value):
            true_masses.append(np.nan)
            true_ranks.append(np.nan)
            continue
        true_well_id = int(true_well_value)
        true_col = _well_posterior_column(column_prefix, true_well_id)
        mass = float(getattr(row, true_col, np.nan)) if true_col in comparison.columns else np.nan
        masses = [
            float(getattr(row, col))
            for col in posterior_columns
            if pd.notna(getattr(row, col))
        ]
        rank = np.nan
        if pd.notna(mass) and masses:
            rank = 1 + int(np.sum(np.asarray(masses) > mass))
        true_masses.append(mass)
        true_ranks.append(rank)
    comparison[posterior_output] = true_masses
    comparison[rank_output] = true_ranks


def _add_true_well_metric_columns(
    comparison: pd.DataFrame,
    valid_bool: pd.Series,
    *,
    suffix: str,
    posterior_output: str,
    rank_output: str,
) -> None:
    posterior_columns = _well_metric_columns(comparison, suffix)
    true_masses: list[float] = []
    true_ranks: list[float] = []
    for is_valid, row in zip(valid_bool.to_numpy(dtype=bool), comparison.itertuples(index=False)):
        if not is_valid:
            true_masses.append(np.nan)
            true_ranks.append(np.nan)
            continue
        true_well_value = getattr(row, "true_well_id", np.nan)
        if pd.isna(true_well_value):
            true_masses.append(np.nan)
            true_ranks.append(np.nan)
            continue
        true_well_id = int(true_well_value)
        true_col = f"well_{true_well_id}{suffix}"
        mass = float(getattr(row, true_col, np.nan)) if true_col in comparison.columns else np.nan
        masses = [
            float(getattr(row, col))
            for col in posterior_columns
            if pd.notna(getattr(row, col))
        ]
        rank = np.nan
        if pd.notna(mass) and masses:
            rank = 1 + int(np.sum(np.asarray(masses) > mass))
        true_masses.append(mass)
        true_ranks.append(rank)
    comparison[posterior_output] = true_masses
    comparison[rank_output] = true_ranks


def _add_active_well_summary_columns(comparison: pd.DataFrame) -> None:
    if "active_goal_id" not in comparison.columns:
        return
    for column_prefix, output_col in (
        ("", "active_well_posterior"),
        ("initial", "active_initial_well_posterior"),
        ("max", "active_max_well_posterior"),
        ("trajectory", "active_trajectory_well_posterior"),
    ):
        if not _well_posterior_columns(comparison, column_prefix):
            continue
        values: list[float] = []
        for row in comparison.itertuples(index=False):
            active_value = getattr(row, "active_goal_id", np.nan)
            if pd.isna(active_value):
                values.append(np.nan)
                continue
            active_col = _well_posterior_column(column_prefix, int(active_value))
            values.append(float(getattr(row, active_col, np.nan)) if active_col in comparison.columns else np.nan)
        comparison[output_col] = values
    if {"true_trajectory_well_posterior", "active_trajectory_well_posterior"}.issubset(
        comparison.columns
    ):
        comparison["true_vs_active_trajectory_posterior_margin"] = (
            comparison["true_trajectory_well_posterior"]
            - comparison["active_trajectory_well_posterior"]
        )


def _well_posterior_columns(comparison: pd.DataFrame, column_prefix: str) -> list[str]:
    prefix = "well_" if not column_prefix else f"{column_prefix}_well_"
    columns: list[str] = []
    for col in comparison.columns:
        if not col.startswith(prefix) or not col.endswith("_posterior"):
            continue
        well_id = col[len(prefix) : -len("_posterior")]
        if well_id.isdigit():
            columns.append(col)
    return columns


def _well_metric_columns(comparison: pd.DataFrame, suffix: str) -> list[str]:
    columns: list[str] = []
    for col in comparison.columns:
        if not col.startswith("well_") or not col.endswith(suffix):
            continue
        well_id = col[len("well_") : -len(suffix)]
        if well_id.isdigit():
            columns.append(col)
    return columns


def _well_posterior_column(column_prefix: str, well_id: int) -> str:
    return f"well_{well_id}_posterior" if not column_prefix else f"{column_prefix}_well_{well_id}_posterior"


def _valid_label_mask(valid: pd.Series | None, index: pd.Index) -> pd.Series:
    """Return a boolean validity mask, treating missing labels as invalid."""

    if valid is None:
        return pd.Series(False, index=index, dtype=bool)
    parsed: list[bool] = []
    for value in valid:
        if pd.isna(value):
            parsed.append(False)
        else:
            parsed.append(_parse_bool(value))
    return pd.Series(parsed, index=index, dtype=bool)


def _event_indices(session: ReplaySession, event_epoch: str) -> np.ndarray:
    if event_epoch == "run":
        return session.ripple_indices_in_run()
    if event_epoch == "all":
        return np.arange(session.ripple_count, dtype=int)
    raise ValueError("event_epoch must be 'run' or 'all'")


def _load_or_generate_ground_truth(
    root: str | Path,
    ground_truth: str | Path | pd.DataFrame | None,
    config: GroundTruthConfig | None,
) -> pd.DataFrame:
    if isinstance(ground_truth, pd.DataFrame):
        return ground_truth.copy()
    if ground_truth is not None:
        return pd.read_csv(ground_truth)
    return generate_behavioral_ground_truth(root, config)


def _invalid_row(row: dict[str, object], reason: str) -> dict[str, object]:
    row.update(
        {
            "true_well_id": np.nan,
            "true_well_x": np.nan,
            "true_well_y": np.nan,
            "arrival_time": np.nan,
            "time_to_arrival_s": np.nan,
            "valid_label": False,
            "exclude_reason": reason,
        }
    )
    return row


def _empty_wells() -> pd.DataFrame:
    return pd.DataFrame(columns=["well_id", "well_x", "well_y", "n_estimates"])


def _clean_position(position: np.ndarray) -> np.ndarray:
    arr = np.asarray(position, dtype=float)
    keep = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    return arr[keep]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(changes[0::2], changes[1::2])]


def _nullable_int(value: int | None) -> int | float:
    return np.nan if value is None else int(value)
