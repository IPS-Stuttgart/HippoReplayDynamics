"""Behavioral proxy ground truth for open-field replay events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .benchmarks import BenchmarkConfig, _build_models, _split_cells
from .data import ReplaySession, load_open_field_sessions
from .encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from .goal_state_space_integration import (
    DEFAULT_GOAL_DRIFT_SPEED_CM_S,
    DEFAULT_GOAL_MAX_STEP_SIGMA,
    DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
)
from .state_space_model import StateSpaceDecoderConfig


@dataclass(frozen=True)
class GroundTruthConfig:
    well_arrival_window_s: float = 1.0
    visit_radius_cm: float = 10.0
    min_dwell_s: float = 0.2
    future_horizon_s: float = 30.0
    event_epoch: str = "run"


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
    state_space_config: StateSpaceDecoderConfig | None = None,
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
    goal_state_space_transition_sigma_cm_sqrt_s: float = DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
    goal_state_space_drift_speed_cm_s: float = DEFAULT_GOAL_DRIFT_SPEED_CM_S,
    goal_state_space_max_step_sigma: float = DEFAULT_GOAL_MAX_STEP_SIGMA,
    random_seed: int = 1,
) -> pd.DataFrame:
    """Merge event scores with next-well behavioral correctness metrics."""

    scores_frame = pd.read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
    gt_frame = _load_or_generate_ground_truth(root, ground_truth, ground_truth_config)
    if scores_frame.empty:
        return scores_frame

    benchmark_decode = _score_table_is_heldout_benchmark(scores_frame)
    encoding_config = _encoding_config_for_scores(
        scores_frame,
        EncodingConfig() if encoding_config is None else encoding_config,
    )
    emission_config = _emission_config_for_scores(
        scores_frame,
        EmissionConfig() if emission_config is None else emission_config,
    )
    state_space_config = _state_space_config_for_scores(
        scores_frame,
        StateSpaceDecoderConfig() if state_space_config is None else state_space_config,
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
    goal_state_space_transition_sigma_cm_sqrt_s = _unique_float_from_column(
        scores_frame,
        "goal_state_space_transition_sigma_cm_sqrt_s",
        goal_state_space_transition_sigma_cm_sqrt_s,
    )
    goal_state_space_drift_speed_cm_s = _unique_float_from_column(
        scores_frame,
        "goal_state_space_drift_speed_cm_s",
        goal_state_space_drift_speed_cm_s,
    )
    goal_state_space_max_step_sigma = _unique_float_from_column(
        scores_frame,
        "goal_state_space_max_step_sigma",
        goal_state_space_max_step_sigma,
    )

    sessions = {session.session_id: session for session in load_open_field_sessions(root)}
    decoded_rows: list[dict[str, object]] = []
    model_names = _model_names_for_scores(scores_frame)
    model_config = BenchmarkConfig(
        encoding=encoding_config,
        emissions=emission_config,
        test_cell_fraction=test_cell_fraction,
        candidate_top_k=candidate_top_k,
        state_space=state_space_config,
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
        goal_state_space_transition_sigma_cm_sqrt_s=goal_state_space_transition_sigma_cm_sqrt_s,
        goal_state_space_drift_speed_cm_s=goal_state_space_drift_speed_cm_s,
        goal_state_space_max_step_sigma=goal_state_space_max_step_sigma,
        random_seed=random_seed,
        models=model_names,
    )

    for session_id, session_scores in scores_frame.groupby("session", sort=False):
        session = sessions.get(str(session_id))
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
                    )
                else:
                    score = model.score(emissions, encoding.bin_centers)
                decoded_rows.append(
                    _decoded_row(
                        str(session_id),
                        int(event_index),
                        model_name,
                        score.terminal_log_posterior,
                        encoding.bin_centers,
                        wells,
                    )
                )
    decoded = pd.DataFrame(decoded_rows)
    comparison = scores_frame.merge(gt_frame, on=["session", "event_index"], how="left")
    comparison = comparison.merge(decoded, on=["session", "event_index", "model"], how="left")
    comparison = _add_ground_truth_metrics(comparison, decoded, gt_frame)
    return comparison


def _score_table_is_heldout_benchmark(scores_frame: pd.DataFrame) -> bool:
    return {
        "heldout_log_likelihood",
        "train_log_likelihood",
        "joint_log_likelihood",
    }.issubset(scores_frame.columns)


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


def _score_joint_for_ground_truth(
    model,
    train_emissions,
    joint_emissions,
    bin_centers: np.ndarray,
):
    if hasattr(model, "candidate_indices"):
        try:
            candidates = model.candidate_indices(train_emissions, bin_centers)
        except TypeError:
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
    random_seed = _unique_int_from_column(
        session_scores,
        "benchmark_random_seed",
        config.random_seed,
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
        time_bin_s=_unique_float_from_column(
            scores_frame,
            "emission_time_bin_s",
            fallback.time_bin_s,
        )
    )


def _state_space_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: StateSpaceDecoderConfig,
) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        stationary_sigma_cm=_unique_float_from_column(scores_frame, "state_space_stationary_sigma_cm", fallback.stationary_sigma_cm),
        diffusion_sigma_cm_sqrt_s=_unique_float_from_column(scores_frame, "state_space_diffusion_sigma_cm_sqrt_s", fallback.diffusion_sigma_cm_sqrt_s),
        max_step_sigma=_unique_float_from_column(scores_frame, "state_space_max_step_sigma", fallback.max_step_sigma),
        imm_mode_stickiness=_unique_float_from_column(scores_frame, "state_space_imm_mode_stickiness", fallback.imm_mode_stickiness),
        momentum_sigma_cm_sqrt_s=_unique_float_from_column(scores_frame, "state_space_momentum_sigma_cm_sqrt_s", fallback.momentum_sigma_cm_sqrt_s),
        momentum_initial_sigma_cm_sqrt_s=_unique_float_from_column(
            scores_frame,
            "state_space_momentum_initial_sigma_cm_sqrt_s",
            fallback.momentum_initial_sigma_cm_sqrt_s,
        ),
        momentum_velocity_decay=_unique_float_from_column(scores_frame, "state_space_momentum_velocity_decay", fallback.momentum_velocity_decay),
        momentum_candidate_top_k=_unique_int_from_column(scores_frame, "state_space_momentum_candidate_top_k", fallback.momentum_candidate_top_k),
        momentum_predicted_candidate_top_k=_unique_int_from_column(scores_frame, "state_space_momentum_predicted_candidate_top_k", fallback.momentum_predicted_candidate_top_k),
    )


def _unique_float_from_column(frame: pd.DataFrame, column: str, default: float) -> float:
    values: list[float] = []
    if column in frame.columns:
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(float(value))
    if not values:
        return float(default)
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{column} contains multiple values")
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
    bin_centers: np.ndarray,
    wells: pd.DataFrame,
) -> dict[str, object]:
    if terminal_log_posterior is None:
        return {
            "session": session_id,
            "event_index": event_index,
            "model": model_name,
        }
    posterior = np.exp(terminal_log_posterior)
    endpoint = posterior @ bin_centers
    decoded_well = assign_endpoint_to_well(endpoint, wells)
    masses = well_posterior_masses(
        terminal_log_posterior,
        bin_centers,
        wells,
    )
    row: dict[str, object] = {
        "session": session_id,
        "event_index": event_index,
        "model": model_name,
        "decoded_endpoint_x": float(endpoint[0]),
        "decoded_endpoint_y": float(endpoint[1]),
        "decoded_well_id": np.nan if decoded_well is None else int(decoded_well["well_id"]),
    }
    for well_id, mass in masses.items():
        row[f"well_{well_id}_posterior"] = float(mass)
    return row


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
    normalized = terminal_log_posterior - logsumexp(terminal_log_posterior)
    posterior = np.exp(normalized)
    masses: dict[int, float] = {}
    for well in wells.itertuples(index=False):
        center = np.array([float(well.well_x), float(well.well_y)])
        distances = np.sqrt(np.sum((bin_centers - center[None, :]) ** 2, axis=1))
        in_radius = distances <= radius_cm
        if not np.any(in_radius):
            in_radius[int(np.argmin(distances))] = True
        masses[int(well.well_id)] = float(np.sum(posterior[in_radius]))
    return masses


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
    comparison["goal_correct"] = np.where(
        valid_bool,
        decoded_ids.astype("float64") == true_ids.astype("float64"),
        np.nan,
    )
    dx = comparison["decoded_endpoint_x"].astype(float) - comparison["true_well_x"].astype(float)
    dy = comparison["decoded_endpoint_y"].astype(float) - comparison["true_well_y"].astype(float)
    comparison["endpoint_error_cm"] = np.where(valid_bool, np.sqrt(dx * dx + dy * dy), np.nan)
    true_masses: list[float] = []
    true_ranks: list[float] = []
    posterior_columns = [col for col in comparison.columns if col.startswith("well_") and col.endswith("_posterior")]
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
        true_col = f"well_{true_well_id}_posterior"
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
    comparison["true_well_posterior"] = true_masses
    comparison["true_well_rank"] = true_ranks
    return comparison


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
