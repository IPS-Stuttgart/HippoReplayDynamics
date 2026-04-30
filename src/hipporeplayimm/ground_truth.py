"""Behavioral proxy ground truth for open-field replay events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .benchmarks import BenchmarkConfig, _build_models
from .data import ReplaySession, load_open_field_sessions
from .encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding


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
) -> pd.DataFrame:
    """Merge event scores with next-well behavioral correctness metrics."""

    scores_frame = pd.read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
    gt_frame = _load_or_generate_ground_truth(root, ground_truth, ground_truth_config)
    if scores_frame.empty:
        return scores_frame
    sessions = {session.session_id: session for session in load_open_field_sessions(root)}
    decoded_rows: list[dict[str, object]] = []
    model_names = tuple(str(model_name) for model_name in scores_frame["model"].dropna().unique())
    model_config = BenchmarkConfig(
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
        models=model_names,
    )
    encoding_config = EncodingConfig() if encoding_config is None else encoding_config
    emission_config = EmissionConfig() if emission_config is None else emission_config

    for session_id, session_scores in scores_frame.groupby("session", sort=False):
        session = sessions.get(str(session_id))
        if session is None:
            continue
        models = _build_models(model_config, session=session)
        wells = infer_well_locations(session, ground_truth_config)
        encoding = fit_place_field_encoding(session, encoding_config)
        for event_index, event_scores in session_scores.groupby("event_index", sort=False):
            emissions = build_emissions(session, encoding, int(event_index), emission_config)
            for score_row in event_scores.itertuples(index=False):
                model_name = str(getattr(score_row, "model"))
                model = models.get(model_name)
                if model is None:
                    continue
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
    valid_bool = valid.fillna(False).astype(bool) if valid is not None else pd.Series(False, index=comparison.index)
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
    for row in comparison.itertuples(index=False):
        if not bool(getattr(row, "valid_label", False)):
            true_masses.append(np.nan)
            true_ranks.append(np.nan)
            continue
        true_well_id = int(getattr(row, "true_well_id"))
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
