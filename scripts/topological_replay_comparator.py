#!/usr/bin/env python3
"""Compare Euclidean and topology-aware replay trajectory evidence.

This script keeps topology-aware replay as an opt-in analysis layer.  It scores
the usual Euclidean sorted-spike diffusion row beside three occupied-state
variants:

* valid-state Euclidean diffusion over occupied bins;
* grid-walk diffusion over the occupied-bin graph;
* geodesic diffusion whose transition kernel uses shortest-path distance over
  the occupied-bin graph instead of straight-line distance.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from benchmark_model_evidence import (
    _check_session,
    _events,
    _postprocess_evidence_scores,
    _session_path,
    _summary,
)
from hipporeplayimm.accuracy_upgrades import (
    ValidStateConfig,
    ValidStateDiffusionReplayModel,
    ValidStateGridReplayModel,
    restrict_emissions_to_mask,
    valid_state_mask_from_encoding,
)
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, LogEmissionTensor, fit_place_field_encoding
from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT
from hipporeplayimm.models import EventScore, LOG_ZERO, _posterior_diagnostics
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.state_space_first_order import _forward_backward_first_order
from hipporeplayimm.state_space_utils import _mean_entropy

EVIDENCE_OUTPUT = "topological_state_space_model_evidence.csv"
COMPARISON_OUTPUT = "euclidean_vs_topological_trajectory_summary.csv"
GATE_OUTPUT = "topological_replay_gate_summary.csv"

EUCLIDEAN_DIFFUSION_MODEL = "sorted-spike-state-space-diffusion"
TOPO_VALID_DIFFUSION_MODEL = "topological-valid-state-diffusion"
TOPO_GRID_WALK_MODEL = "topological-grid-walk"
TOPO_GEODESIC_MODEL = "topological-geodesic-diffusion"
TOPOLOGICAL_MODELS = (
    TOPO_VALID_DIFFUSION_MODEL,
    TOPO_GRID_WALK_MODEL,
    TOPO_GEODESIC_MODEL,
)
DEFAULT_MODELS = " ".join((EUCLIDEAN_DIFFUSION_MODEL, *TOPOLOGICAL_MODELS))
_EVENT_KEY_CANDIDATES = (
    "session",
    "event_index",
    "window_role",
    "event_window_variant",
    "window_index",
    "null_index",
    "matched_null_rank",
)


@dataclass
class TopologicalGeodesicReplayModel:
    """Exact first-order diffusion over shortest paths through occupied bins."""

    valid_mask: np.ndarray
    grid_shape: tuple[int, int]
    sigma_cm: float = 5.0
    max_distance_sigma: float = 4.0
    diagonal_neighbors: bool = True
    stay_probability: float = 0.0
    name: str = TOPO_GEODESIC_MODEL

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        mask = _coerce_mask(self.valid_mask, emissions.n_bins)
        restricted = restrict_emissions_to_mask(emissions, mask)
        transition, graph_stats = geodesic_transition_matrix(
            self.grid_shape,
            mask,
            bin_centers,
            sigma_cm=float(self.sigma_cm),
            max_distance_sigma=float(self.max_distance_sigma),
            diagonal_neighbors=bool(self.diagonal_neighbors),
            stay_probability=float(self.stay_probability),
        )
        logp, trajectory = _forward_backward_first_order(restricted.log_likelihood, transition)
        full_trajectory = _expand_log_trajectory(trajectory, mask, emissions.n_bins)
        terminal = full_trajectory[-1]
        diagnostics: dict[str, float | int | str] = {
            "topological_transition": "occupied_graph_geodesic_gaussian",
            "topological_state_support": "exact_valid_state_grid",
            "topological_sigma_cm": float(self.sigma_cm),
            "topological_max_distance_sigma": float(self.max_distance_sigma),
            "topological_diagonal_neighbors": int(bool(self.diagonal_neighbors)),
            "topological_stay_probability": float(self.stay_probability),
            "valid_state_bins": int(np.sum(mask)),
            "valid_state_fraction": float(np.mean(mask)),
            "mean_trajectory_posterior_entropy": _mean_entropy(full_trajectory),
            **graph_stats,
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=full_trajectory,
        )


def geodesic_transition_matrix(
    grid_shape: tuple[int, int],
    valid_mask: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    max_distance_sigma: float = 4.0,
    diagonal_neighbors: bool = True,
    stay_probability: float = 0.0,
) -> tuple[csr_matrix, dict[str, float | int]]:
    """Return a column-stochastic Gaussian kernel over graph geodesic distance."""

    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    mask = _coerce_mask(valid_mask, nx * ny)
    centers = np.asarray(bin_centers, dtype=float)
    if centers.shape[0] != nx * ny:
        raise ValueError("bin_centers must contain one row per grid bin")
    sigma_cm = float(sigma_cm)
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    max_distance_sigma = float(max_distance_sigma)
    if max_distance_sigma <= 0.0:
        raise ValueError("max_distance_sigma must be positive")
    stay_probability = float(stay_probability)
    if not 0.0 <= stay_probability < 1.0:
        raise ValueError("stay_probability must lie in [0, 1)")

    valid_flat = np.flatnonzero(mask)
    if valid_flat.size == 0:
        raise ValueError("valid_mask must contain at least one valid state")
    if valid_flat.size == 1:
        return csr_matrix(([1.0], ([0], [0])), shape=(1, 1)), {
            "topological_graph_edges": 0,
            "topological_graph_components": 1,
            "topological_largest_component_fraction": 1.0,
            "topological_mean_finite_geodesic_cm": 0.0,
        }

    compact_index = {int(flat): idx for idx, flat in enumerate(valid_flat)}
    graph = _occupied_grid_graph(
        grid_shape,
        mask,
        centers,
        compact_index,
        diagonal_neighbors=diagonal_neighbors,
    )
    components, labels = connected_components(graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=components)
    largest_component_fraction = float(component_sizes.max() / valid_flat.size) if component_sizes.size else 0.0
    distances = shortest_path(graph, directed=False, unweighted=False)
    finite_nonzero = distances[np.isfinite(distances) & (distances > 0.0)]

    radius = sigma_cm * max_distance_sigma
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src in range(valid_flat.size):
        dist = distances[:, src]
        keep = np.isfinite(dist) & (dist <= radius)
        if not np.any(keep):
            keep[src] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * (dist[dst] / sigma_cm) ** 2)
        weights /= float(weights.sum())
        if stay_probability > 0.0:
            weights *= 1.0 - stay_probability
            if src in dst:
                weights[np.flatnonzero(dst == src)[0]] += stay_probability
            else:
                dst = np.append(dst, src)
                weights = np.append(weights, stay_probability)
        rows.extend(int(value) for value in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)

    transition = csr_matrix((data, (rows, cols)), shape=(valid_flat.size, valid_flat.size))
    return transition, {
        "topological_graph_edges": int(graph.nnz // 2),
        "topological_graph_components": int(components),
        "topological_largest_component_fraction": largest_component_fraction,
        "topological_mean_finite_geodesic_cm": float(np.mean(finite_nonzero)) if finite_nonzero.size else 0.0,
    }


def write_topological_replay_outputs(
    evidence: pd.DataFrame,
    outdir: str | Path,
    *,
    margin_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    """Write the three paper-facing topology comparator tables."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    evidence_out = evidence.copy()
    summary = euclidean_vs_topological_trajectory_summary(evidence_out, margin_threshold=margin_threshold)
    gates = topological_replay_gate_summary(evidence_out, summary, margin_threshold=margin_threshold)
    evidence_out.to_csv(output / EVIDENCE_OUTPUT, index=False)
    summary.to_csv(output / COMPARISON_OUTPUT, index=False)
    gates.to_csv(output / GATE_OUTPUT, index=False)
    return {
        EVIDENCE_OUTPUT: evidence_out,
        COMPARISON_OUTPUT: summary,
        GATE_OUTPUT: gates,
    }


def _as_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    return default


def _bool_column(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].map(lambda value: _as_bool(value, default=default)).astype(bool)


def euclidean_vs_topological_trajectory_summary(
    evidence: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
    baseline_model: str = EUCLIDEAN_DIFFUSION_MODEL,
) -> pd.DataFrame:
    """Summarize paired event-level topology-minus-Euclidean evidence deltas."""

    if evidence.empty:
        return _empty_comparison_summary()
    key_columns = _event_key_columns(evidence)
    ok = evidence[evidence.get("status", "success").eq("success")].copy()
    if "evidence_comparable" in ok:
        ok = ok[_bool_column(ok, "evidence_comparable")]
    if ok.empty or not key_columns:
        return _empty_comparison_summary()

    pivot = ok.pivot_table(index=key_columns, columns="model", values="log_evidence", aggfunc="first")
    rows: list[dict[str, object]] = []
    for model in TOPOLOGICAL_MODELS:
        rows.append(_comparison_row(pivot, key_columns, baseline_model, model, model, margin_threshold))
    available_topology = [model for model in TOPOLOGICAL_MODELS if model in pivot]
    if baseline_model in pivot and available_topology:
        values = pivot[[baseline_model, *available_topology]].dropna()
        if values.empty:
            rows.append(_empty_comparison_row("best_topological_vs_euclidean", baseline_model, "best_topological"))
        else:
            best_topological = values[available_topology].max(axis=1)
            delta = best_topological - values[baseline_model]
            rows.append(
                _summarize_delta(
                    "best_topological_vs_euclidean",
                    baseline_model,
                    "best_topological",
                    delta,
                    values.index,
                    key_columns,
                    margin_threshold,
                )
            )
    else:
        rows.append(_empty_comparison_row("best_topological_vs_euclidean", baseline_model, "best_topological"))
    return pd.DataFrame(rows)


def topological_replay_gate_summary(
    evidence: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return infrastructure gates plus non-required scientific diagnostics."""

    scored = evidence[evidence.get("status", "success").eq("success")] if not evidence.empty else pd.DataFrame()
    models = set(scored["model"].astype(str)) if "model" in scored else set()
    required: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    def add_gate(
        gate: str,
        passed: bool,
        value: object,
        threshold: object,
        details: str,
        *,
        required_for_overall: bool,
    ) -> None:
        row = {
            "gate": gate,
            "passed": bool(passed),
            "required_for_overall": bool(required_for_overall),
            "value": value,
            "threshold": threshold,
            "details": details,
        }
        (required if required_for_overall else diagnostics).append(row)

    add_gate(
        "evidence_rows_present",
        not evidence.empty,
        int(len(evidence)),
        ">0",
        "topological state-space evidence table has rows",
        required_for_overall=True,
    )
    add_gate(
        "euclidean_baseline_scored",
        EUCLIDEAN_DIFFUSION_MODEL in models,
        int((scored["model"].astype(str).eq(EUCLIDEAN_DIFFUSION_MODEL)).sum()) if "model" in scored else 0,
        ">0",
        "Euclidean sorted-spike diffusion baseline is present",
        required_for_overall=True,
    )
    topology_rows = scored[scored["model"].astype(str).isin(TOPOLOGICAL_MODELS)] if "model" in scored else pd.DataFrame()
    add_gate(
        "topological_models_scored",
        not topology_rows.empty,
        int(len(topology_rows)),
        ">0",
        "At least one topology-aware model was scored",
        required_for_overall=True,
    )
    max_paired = int(comparison["paired_events"].max()) if "paired_events" in comparison and not comparison.empty else 0
    add_gate(
        "paired_euclidean_topological_events_present",
        max_paired > 0,
        max_paired,
        ">0",
        "At least one event has paired Euclidean and topology-aware evidence",
        required_for_overall=True,
    )
    coverage_column = _first_existing_column(
        topology_rows,
        ("diagnostic_valid_state_fraction", "diagnostic_topological_valid_state_fraction"),
    )
    coverage_present = coverage_column != "" and topology_rows[coverage_column].notna().any()
    add_gate(
        "valid_state_coverage_present",
        coverage_present,
        float(topology_rows[coverage_column].median()) if coverage_present else np.nan,
        "finite median",
        "Topology rows report occupied-state coverage diagnostics",
        required_for_overall=True,
    )

    geodesic = _comparison_for(comparison, TOPO_GEODESIC_MODEL)
    geodesic_win_fraction = float(geodesic["topological_win_fraction"]) if geodesic is not None else np.nan
    add_gate(
        "topological_geodesic_beats_euclidean_majority",
        bool(np.isfinite(geodesic_win_fraction) and geodesic_win_fraction > 0.5),
        geodesic_win_fraction,
        ">0.5",
        "Scientific diagnostic only: geodesic topology wins most paired events",
        required_for_overall=False,
    )
    geodesic_confident = int(geodesic["confident_topological_wins"]) if geodesic is not None else 0
    add_gate(
        "topological_geodesic_confident_wins_present",
        geodesic_confident > 0,
        geodesic_confident,
        f"delta>{margin_threshold:g}",
        "Scientific diagnostic only: at least one confident geodesic win",
        required_for_overall=False,
    )

    overall = all(bool(row["passed"]) for row in required)
    rows = [
        {
            "gate": "overall",
            "passed": bool(overall),
            "required_for_overall": True,
            "value": int(sum(bool(row["passed"]) for row in required)),
            "threshold": f"{len(required)}/{len(required)} required gates",
            "details": "Topology comparator outputs are structurally usable",
        },
        *required,
        *diagnostics,
    ]
    return pd.DataFrame(rows)


def _occupied_grid_graph(
    grid_shape: tuple[int, int],
    valid_mask: np.ndarray,
    bin_centers: np.ndarray,
    compact_index: dict[int, int],
    *,
    diagonal_neighbors: bool,
) -> csr_matrix:
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal_neighbors:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for flat in np.flatnonzero(valid_mask):
        x = int(flat // ny)
        y = int(flat % ny)
        src = compact_index[int(flat)]
        for dx, dy in offsets:
            xx, yy = x + dx, y + dy
            if 0 <= xx < nx and 0 <= yy < ny:
                neighbor_flat = xx * ny + yy
                if not valid_mask[neighbor_flat]:
                    continue
                dst = compact_index[int(neighbor_flat)]
                distance = float(np.linalg.norm(bin_centers[neighbor_flat] - bin_centers[flat]))
                rows.append(dst)
                cols.append(src)
                data.append(max(distance, np.finfo(float).tiny))
    return csr_matrix((data, (rows, cols)), shape=(len(compact_index), len(compact_index)))


def _comparison_row(
    pivot: pd.DataFrame,
    key_columns: Sequence[str],
    baseline_model: str,
    topological_model: str,
    comparison: str,
    margin_threshold: float,
) -> dict[str, object]:
    if baseline_model not in pivot or topological_model not in pivot:
        return _empty_comparison_row(comparison, baseline_model, topological_model)
    values = pivot[[baseline_model, topological_model]].dropna()
    if values.empty:
        return _empty_comparison_row(comparison, baseline_model, topological_model)
    delta = values[topological_model] - values[baseline_model]
    return _summarize_delta(
        comparison,
        baseline_model,
        topological_model,
        delta,
        values.index,
        key_columns,
        margin_threshold,
    )


def _summarize_delta(
    comparison: str,
    baseline_model: str,
    topological_model: str,
    delta: pd.Series,
    event_index: pd.Index,
    key_columns: Sequence[str],
    margin_threshold: float,
) -> dict[str, object]:
    wins = delta > 0.0
    confident = delta > float(margin_threshold)
    rat_medians = _rat_median_deltas(delta, event_index, key_columns)
    return {
        "comparison": comparison,
        "baseline_model": baseline_model,
        "topological_model": topological_model,
        "paired_events": int(delta.shape[0]),
        "topological_wins": int(wins.sum()),
        "topological_win_fraction": float(wins.mean()) if delta.shape[0] else np.nan,
        "confident_topological_wins": int(confident.sum()),
        "confident_topological_win_fraction": float(confident.mean()) if delta.shape[0] else np.nan,
        "mean_delta_log_evidence": float(delta.mean()),
        "median_delta_log_evidence": float(delta.median()),
        "min_delta_log_evidence": float(delta.min()),
        "max_delta_log_evidence": float(delta.max()),
        "rat_level_median_delta_positive_count": int((rat_medians > 0.0).sum()) if not rat_medians.empty else 0,
        "rats_with_paired_events": int(rat_medians.shape[0]),
        "margin_threshold": float(margin_threshold),
    }


def _rat_median_deltas(delta: pd.Series, event_index: pd.Index, key_columns: Sequence[str]) -> pd.Series:
    if "session" not in key_columns:
        return pd.Series(dtype=float)
    if isinstance(event_index, pd.MultiIndex):
        session_values = event_index.get_level_values(key_columns.index("session")).astype(str)
    else:
        session_values = pd.Index([str(value) for value in event_index], name="session")
    rats = pd.Series(session_values, index=delta.index).astype(str).str.split("/", n=1).str[0]
    return delta.groupby(rats).median()


def _empty_comparison_row(comparison: str, baseline_model: str, topological_model: str) -> dict[str, object]:
    return {
        "comparison": comparison,
        "baseline_model": baseline_model,
        "topological_model": topological_model,
        "paired_events": 0,
        "topological_wins": 0,
        "topological_win_fraction": np.nan,
        "confident_topological_wins": 0,
        "confident_topological_win_fraction": np.nan,
        "mean_delta_log_evidence": np.nan,
        "median_delta_log_evidence": np.nan,
        "min_delta_log_evidence": np.nan,
        "max_delta_log_evidence": np.nan,
        "rat_level_median_delta_positive_count": 0,
        "rats_with_paired_events": 0,
        "margin_threshold": np.nan,
    }


def _empty_comparison_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _empty_comparison_row(model, EUCLIDEAN_DIFFUSION_MODEL, model)
            for model in (*TOPOLOGICAL_MODELS, "best_topological")
        ]
    )


def _comparison_for(comparison: pd.DataFrame, topological_model: str) -> pd.Series | None:
    if comparison.empty or "topological_model" not in comparison:
        return None
    rows = comparison[comparison["topological_model"].astype(str).eq(topological_model)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column in df:
            return column
    return ""


def _event_key_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in _EVENT_KEY_CANDIDATES if column in df]


def _coerce_mask(mask: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.shape[0] != int(n_bins):
        raise ValueError("valid_mask length must match the number of spatial bins")
    return out


def _expand_log_trajectory(trajectory: np.ndarray, valid_mask: np.ndarray, n_bins: int) -> np.ndarray:
    full = np.full((trajectory.shape[0], int(n_bins)), LOG_ZERO, dtype=float)
    full[:, valid_mask] = trajectory
    return full


def _model_family(model: str) -> str:
    if model == EUCLIDEAN_DIFFUSION_MODEL or model in TOPOLOGICAL_MODELS:
        return "trajectory"
    return "other"


def _state_space_config(args: argparse.Namespace, mode: str) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode=mode,
        stationary_sigma_cm=float(args.state_space_stationary_sigma_cm),
        diffusion_sigma_cm_sqrt_s=float(args.state_space_diffusion_sigma_cm_sqrt_s),
        max_step_sigma=float(args.state_space_max_step_sigma),
        valid_occupancy_threshold_s=float(args.state_space_valid_occupancy_threshold_s),
    )


def _topological_sigma_cm(args: argparse.Namespace) -> float:
    direct = float(args.topological_transition_sigma_cm)
    if direct > 0.0:
        return direct
    return float(args.topological_transition_sigma_cm_sqrt_s) * float(np.sqrt(args.time_bin_s))


def _models(args: argparse.Namespace, encoding) -> dict[str, object]:
    names = [name for name in args.models.replace(",", " ").split() if name]
    missing = sorted(set(names) - {EUCLIDEAN_DIFFUSION_MODEL, *TOPOLOGICAL_MODELS})
    if missing:
        raise ValueError(f"unknown models: {missing}; available: {sorted({EUCLIDEAN_DIFFUSION_MODEL, *TOPOLOGICAL_MODELS})}")

    valid_mask = valid_state_mask_from_encoding(
        encoding,
        ValidStateConfig(
            min_occupancy_s=float(args.valid_state_min_occupancy_s),
            keep_top_occupancy_fraction=args.valid_state_top_occupancy_fraction,
        ),
    )
    sigma_cm = _topological_sigma_cm(args)
    available = {
        EUCLIDEAN_DIFFUSION_MODEL: SortedSpikeStateSpaceReplayModel(
            mode="diffusion",
            config=_state_space_config(args, "diffusion"),
            name=EUCLIDEAN_DIFFUSION_MODEL,
        ),
        TOPO_VALID_DIFFUSION_MODEL: ValidStateDiffusionReplayModel(
            valid_mask,
            sigma_cm=sigma_cm,
            max_step_sigma=float(args.topological_max_distance_sigma),
            name=TOPO_VALID_DIFFUSION_MODEL,
        ),
        TOPO_GRID_WALK_MODEL: ValidStateGridReplayModel(
            valid_mask,
            grid_shape=encoding.grid_shape,
            diagonal_neighbors=bool(args.topological_diagonal_neighbors),
            stay_probability=float(args.topological_stay_probability),
            name=TOPO_GRID_WALK_MODEL,
        ),
        TOPO_GEODESIC_MODEL: TopologicalGeodesicReplayModel(
            valid_mask,
            grid_shape=encoding.grid_shape,
            sigma_cm=sigma_cm,
            max_distance_sigma=float(args.topological_max_distance_sigma),
            diagonal_neighbors=bool(args.topological_diagonal_neighbors),
            stay_probability=float(args.topological_stay_probability),
            name=TOPO_GEODESIC_MODEL,
        ),
    }
    return {name: available[name] for name in dict.fromkeys(names)}


def _run_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "bin_size_cm": float(args.bin_size_cm),
        "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
        "min_speed_cm_s": float(args.min_speed_cm_s),
        "min_occupancy_s": float(args.min_occupancy_s),
        "rate_floor_hz": float(args.rate_floor_hz),
        "time_bin_s": float(args.time_bin_s),
        "spike_rate_scale": float(args.spike_rate_scale),
        "emission_likelihood_temperature": float(args.emission_likelihood_temperature),
        "emission_negative_binomial_overdispersion": float(args.emission_negative_binomial_overdispersion),
        "sorted_spike_emission_model": str(args.sorted_spike_emission_model),
        "replay_gain_mode": str(args.replay_gain_mode),
        "replay_gain_prior_count": float(args.replay_gain_prior_count),
        "replay_gain_max_gain": float(args.replay_gain_max_gain),
        "negative_binomial_dispersion": float(args.negative_binomial_dispersion),
        "state_space_valid_occupancy_threshold_s": float(args.state_space_valid_occupancy_threshold_s),
        "state_space_stationary_sigma_cm": float(args.state_space_stationary_sigma_cm),
        "state_space_diffusion_sigma_cm_sqrt_s": float(args.state_space_diffusion_sigma_cm_sqrt_s),
        "state_space_max_step_sigma": float(args.state_space_max_step_sigma),
        "valid_state_min_occupancy_s": float(args.valid_state_min_occupancy_s),
        "valid_state_top_occupancy_fraction": ""
        if args.valid_state_top_occupancy_fraction is None
        else float(args.valid_state_top_occupancy_fraction),
        "topological_transition_sigma_cm": float(_topological_sigma_cm(args)),
        "topological_transition_sigma_cm_sqrt_s": float(args.topological_transition_sigma_cm_sqrt_s),
        "topological_max_distance_sigma": float(args.topological_max_distance_sigma),
        "topological_diagonal_neighbors": bool(args.topological_diagonal_neighbors),
        "topological_stay_probability": float(args.topological_stay_probability),
        "margin_threshold": float(args.margin_threshold),
    }


def _score(args: argparse.Namespace) -> pd.DataFrame:
    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session)
    if args.max_events is not None:
        event_ids = event_ids[: int(args.max_events)]

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=float(args.bin_size_cm),
            smoothing_sigma_bins=float(args.smoothing_sigma_bins),
            min_speed_cm_s=float(args.min_speed_cm_s),
            min_occupancy_s=float(args.min_occupancy_s),
            rate_floor_hz=float(args.rate_floor_hz),
        ),
    )
    models = _models(args, encoding)
    emissions_cfg = EmissionConfig(
        time_bin_s=float(args.time_bin_s),
        spike_rate_scale=float(args.spike_rate_scale),
        likelihood_temperature=float(args.emission_likelihood_temperature),
        negative_binomial_overdispersion=float(args.emission_negative_binomial_overdispersion),
    )
    calibration = ReplayEmissionCalibration(
        gain_mode=str(args.replay_gain_mode),
        gain_prior_count=float(args.replay_gain_prior_count),
        max_gain=float(args.replay_gain_max_gain),
        emission_model=str(args.sorted_spike_emission_model),
        negative_binomial_dispersion=float(args.negative_binomial_dispersion),
    )
    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        event = session.ripple(int(event_id))
        emissions = build_sorted_emissions_with_replay_calibration(
            session,
            encoding,
            event,
            emissions_cfg,
            calibration=calibration,
        )
        if emissions.n_time == 0:
            continue
        for requested_model, model in models.items():
            start = time.perf_counter()
            try:
                result = score_replay_model_compat(
                    model,
                    emissions,
                    encoding.bin_centers,
                    occupancy_s=encoding.occupancy_s,
                )
                row: dict[str, object] = {
                    "status": "success",
                    "session": session.session_id,
                    "event_index": int(event_id),
                    "model": str(result.model_name),
                    "requested_model": requested_model,
                    "model_family": _model_family(str(result.model_name)),
                    "log_evidence": float(result.log_likelihood),
                    "n_time": int(result.n_time),
                    "n_spikes": int(result.n_spikes),
                    "runtime_s": float(time.perf_counter() - start),
                    "error": "",
                    "evidence_support": EXACT_EVIDENCE_SUPPORT,
                    **_run_settings(args),
                }
                metadata = getattr(emissions, "metadata", {}) or {}
                row.update({f"emission_{key}": value for key, value in metadata.items()})
                row.update({f"diagnostic_{key}": value for key, value in result.diagnostics.items()})
                rows.append(row)
                print(f"Scored {session.session_id} event {event_id} with {requested_model}", flush=True)
            except Exception as exc:
                rows.append(
                    {
                        "status": "failure",
                        "session": session.session_id,
                        "event_index": int(event_id),
                        "model": requested_model,
                        "requested_model": requested_model,
                        "model_family": _model_family(requested_model),
                        "log_evidence": np.nan,
                        "n_time": int(emissions.n_time),
                        "n_spikes": int(emissions.n_spikes),
                        "runtime_s": float(time.perf_counter() - start),
                        "error": f"{type(exc).__name__}: {exc}",
                        **_run_settings(args),
                    }
                )
                if not args.continue_on_error:
                    raise
    return _postprocess_evidence_scores(pd.DataFrame(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Euclidean and topology-aware replay evidence.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-25")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=EncodingConfig().min_occupancy_s)
    parser.add_argument("--rate-floor-hz", type=float, default=EncodingConfig().rate_floor_hz)
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument(
        "--sorted-spike-emission-model",
        choices=("poisson", "negative-binomial", "gamma-poisson"),
        default="poisson",
    )
    parser.add_argument("--replay-gain-mode", choices=("none", "event", "cell", "event-cell"), default="none")
    parser.add_argument("--replay-gain-prior-count", type=float, default=10.0)
    parser.add_argument("--replay-gain-max-gain", type=float, default=20.0)
    parser.add_argument("--negative-binomial-dispersion", type=float, default=50.0)
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--valid-state-min-occupancy-s", type=float, default=0.02)
    parser.add_argument("--valid-state-top-occupancy-fraction", type=float, default=None)
    parser.add_argument(
        "--topological-transition-sigma-cm",
        type=float,
        default=0.0,
        help="Direct per-bin topology transition sigma. Use <=0 to derive it from --topological-transition-sigma-cm-sqrt-s.",
    )
    parser.add_argument("--topological-transition-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--topological-max-distance-sigma", type=float, default=4.0)
    parser.add_argument("--topological-diagonal-neighbors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--topological-stay-probability", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--output", default="results/topological-replay")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    evidence = _score(args)
    if evidence.empty:
        raise RuntimeError("No topology comparator scores were generated.")
    print(_summary(evidence).to_string(index=False))
    outputs = write_topological_replay_outputs(
        evidence,
        args.output,
        margin_threshold=float(args.margin_threshold),
    )
    print("\nEuclidean-vs-topological summary:")
    print(outputs[COMPARISON_OUTPUT].to_string(index=False))
    print("\nTopology gates:")
    print(outputs[GATE_OUTPUT].to_string(index=False))
    print(f"\nRows: {len(evidence)}")
    print(f"Failures: {int((evidence['status'] != 'success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
