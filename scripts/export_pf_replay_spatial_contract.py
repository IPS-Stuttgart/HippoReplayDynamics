#!/usr/bin/env python3
"""Freeze causal raw PF replay emissions and pre-replay behavioral fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance, git_metadata  # noqa: E402
from hipporeplayimm.data import ReplaySession, RippleEvent, load_replay_session  # noqa: E402
from hipporeplayimm.encoding import (  # noqa: E402
    EmissionConfig,
    EncodingConfig,
    EncodingModel,
    build_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.replay_spatial_export import (  # noqa: E402
    BehaviorFieldConfig,
    SPATIAL_CANDIDATE_NAMES,
    build_pre_replay_candidate_fields,
)
from hipporeplayimm.smoothing_trace import (  # noqa: E402
    SMOOTHING_TRACE_SCHEMA_VERSION,
    TRANSITION_CONVENTION,
)


PREDICTOR_OUTPUT = "replay_spatial_predictors.npz"
MANIFEST_OUTPUT = "replay_spatial_manifest.json"
EVENT_AUDIT_OUTPUT = "replay_spatial_event_audit.csv"
SUMMARY_OUTPUT = "replay_spatial_export_summary.md"
SCHEMA_VERSION = "bayesian-ach.replay-spatial.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class EventSelectionConfig:
    """Decoder-independent RUN-ripple selection fixed before spatial scoring."""

    events_per_session: int = 20
    minimum_raw_power: float = 0.0
    minimum_training_duration_s: float = 300.0
    minimum_completed_routes: int = 6
    ranking_metric: str = "raw_power"

    def validate(self) -> None:
        if self.events_per_session < 1:
            raise ValueError("events_per_session must be positive")
        if not np.isfinite(self.minimum_raw_power):
            raise ValueError("minimum_raw_power must be finite")
        if (
            not np.isfinite(self.minimum_training_duration_s)
            or self.minimum_training_duration_s <= 0.0
        ):
            raise ValueError("minimum_training_duration_s must be finite and positive")
        if self.minimum_completed_routes < 2:
            raise ValueError("minimum_completed_routes must be at least two")
        if self.ranking_metric != "raw_power":
            raise ValueError("only raw_power LFP ranking is supported")

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PointSpreadConfig:
    """Strictly pre-event temporal holdout for empirical decoder resolution."""

    holdout_window_s: float = 60.0
    time_bin_s: float = 0.100
    quantile: float = 0.68
    minimum_valid_bins: int = 20
    minimum_cells: int = 1

    def validate(self) -> None:
        if not np.isfinite(self.holdout_window_s) or self.holdout_window_s <= 0.0:
            raise ValueError("holdout_window_s must be finite and positive")
        if not np.isfinite(self.time_bin_s) or self.time_bin_s <= 0.0:
            raise ValueError("time_bin_s must be finite and positive")
        if not 0.5 <= self.quantile < 1.0:
            raise ValueError("quantile must lie in [0.5, 1)")
        if self.minimum_valid_bins < 2 or self.minimum_cells < 1:
            raise ValueError("point-spread minimum counts are invalid")

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ExportedEvent:
    event_id: str
    rat: str
    session: str
    event_index: int
    event_start_s: float
    event_end_s: float
    history_cutoff_s: float
    decoder_training_cutoff_s: float
    field_available_s: np.ndarray
    log_emissions: np.ndarray
    log_emission_offsets: np.ndarray
    spatial_coordinates: np.ndarray
    nuisance_base: np.ndarray
    candidate_fields: np.ndarray
    candidate_available: np.ndarray
    decoder_point_spread_cm: float
    well_masses: dict[str, float]
    audit: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_digest(*objects: object) -> str:
    payload = json.dumps(
        [asdict(value) for value in objects],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_position(session: ReplaySession) -> np.ndarray:
    position = np.asarray(session.position, dtype=float)
    keep = np.isfinite(position[:, :3]).all(axis=1)
    position = position[keep, :3]
    order = np.argsort(position[:, 0], kind="stable")
    return position[order]


def _times_in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(times), dtype=bool)
    for start, end in np.asarray(intervals, dtype=float):
        selected |= (times >= float(start)) & (times <= float(end))
    return selected


def _speed(position: np.ndarray) -> np.ndarray:
    if len(position) < 2:
        return np.zeros(len(position), dtype=float)
    return np.hypot(
        np.gradient(position[:, 1], position[:, 0]),
        np.gradient(position[:, 2], position[:, 0]),
    )


def _outside_ripples(session: ReplaySession, times: np.ndarray) -> np.ndarray:
    outside = np.ones(len(times), dtype=bool)
    for event in np.asarray(session.ripple_events, dtype=float):
        if len(event) >= 2 and np.isfinite(event[:2]).all():
            outside &= ~((times >= event[0]) & (times <= event[1]))
    return outside


def _current_location(
    session: ReplaySession,
    cutoff_s: float,
) -> tuple[np.ndarray, float]:
    position = _finite_position(session)
    prefix = position[position[:, 0] <= float(cutoff_s)]
    if len(prefix) == 0:
        raise ValueError("no tracked position is available before replay")
    return np.asarray(prefix[-1, 1:3], dtype=float), float(prefix[-1, 0])


def select_lfp_only_events(
    session: ReplaySession,
    routes: pd.DataFrame,
    config: EventSelectionConfig,
) -> list[dict[str, object]]:
    """Select RUN ripples without inspecting decoded content or future outcomes."""

    config.validate()
    position = _finite_position(session)
    if len(position) < 2:
        raise ValueError(f"{session.session_id}: insufficient position data")
    first_time = float(position[0, 0])
    candidates: list[dict[str, object]] = []
    for event_index in session.ripple_indices_in_run():
        event = session.ripple(int(event_index))
        completed_routes = int(
            np.sum(
                pd.to_numeric(
                    routes["movement_end_time_s"],
                    errors="coerce",
                )
                <= event.start
            )
        )
        if event.start - first_time < config.minimum_training_duration_s:
            continue
        if completed_routes < config.minimum_completed_routes:
            continue
        if not np.isfinite(event.raw_power):
            continue
        if event.raw_power < config.minimum_raw_power:
            continue
        candidates.append(
            {
                "event_index": int(event_index),
                "event_start_s": float(event.start),
                "event_end_s": float(event.end),
                "selection_metric": float(event.raw_power),
                "completed_routes_at_selection": completed_routes,
            }
        )
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["selection_metric"]),
            int(row["event_index"]),
        ),
    )[: config.events_per_session]
    if len(selected) != config.events_per_session:
        raise ValueError(
            f"{session.session_id}: LFP-only selection found {len(selected)} "
            f"eligible events, expected {config.events_per_session}"
        )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    return selected


def estimate_prefix_decoder_point_spread_cm(
    session: ReplaySession,
    event_start_s: float,
    encoding_config: EncodingConfig,
    emission_config: EmissionConfig,
    config: PointSpreadConfig,
) -> tuple[float, dict[str, object]]:
    """Estimate spatial resolution on a strictly pre-event temporal RUN holdout."""

    config.validate()
    event_cutoff = float(np.nextafter(float(event_start_s), -np.inf))
    training_cutoff = event_cutoff - config.holdout_window_s
    encoding = fit_place_field_encoding(
        session,
        encoding_config,
        training_end_s=training_cutoff,
    )
    if encoding.n_cells < config.minimum_cells:
        raise ValueError("point-spread calibration has too few prefix-trained cells")
    holdout = RippleEvent(
        start=training_cutoff,
        end=event_cutoff,
        peak=0.5 * (training_cutoff + event_cutoff),
        raw_power=0.0,
        z_power_session=0.0,
        z_power_epoch=0.0,
    )
    calibration_emissions = build_emissions(
        session,
        encoding,
        holdout,
        replace(emission_config, time_bin_s=config.time_bin_s),
    )
    map_bins = np.argmax(calibration_emissions.log_likelihood, axis=1)
    decoded = encoding.bin_centers[map_bins]
    position = _finite_position(session)
    actual = np.column_stack(
        [
            np.interp(calibration_emissions.times, position[:, 0], position[:, axis])
            for axis in (1, 2)
        ]
    )
    speed = np.interp(calibration_emissions.times, position[:, 0], _speed(position))
    valid = _times_in_intervals(calibration_emissions.times, session.run_times)
    valid &= speed >= encoding_config.min_speed_cm_s
    valid &= _outside_ripples(session, calibration_emissions.times)
    if int(np.sum(valid)) < config.minimum_valid_bins:
        raise ValueError("point-spread calibration has too few held-out RUN bins")
    errors = np.linalg.norm(decoded[valid] - actual[valid], axis=1)
    point_spread = float(np.quantile(errors, config.quantile))
    if not np.isfinite(point_spread) or point_spread <= 0.0:
        raise ValueError("point-spread calibration did not produce a positive error")
    return point_spread, {
        "point_spread_training_cutoff_s": training_cutoff,
        "point_spread_holdout_end_s": event_cutoff,
        "point_spread_valid_bins": int(np.sum(valid)),
        "point_spread_quantile": config.quantile,
    }


def _well_locations_and_masses(
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    coordinates: np.ndarray,
    shifted_log_emissions: np.ndarray,
    session_id: str,
) -> dict[str, float]:
    locations: list[np.ndarray] = []
    identifiers: list[str] = []
    for well_id in sorted(set(routes["destination_well_id"].astype(int))):
        selected_routes = routes[
            routes["destination_well_id"].astype(int).eq(well_id)
        ]["route_id"].astype(str)
        endpoints = []
        for route_id in selected_routes:
            points = route_points[
                route_points["route_id"].astype(str).eq(route_id)
            ].sort_values("point_index")
            if not points.empty:
                endpoints.append(points[["x_cm", "y_cm"]].to_numpy(dtype=float)[-1])
        if endpoints:
            locations.append(np.median(np.stack(endpoints), axis=0))
            identifiers.append(f"{session_id}|well-{well_id}")
    if not locations:
        raise ValueError("cannot derive well masses without historical well locations")
    posterior = np.exp(
        shifted_log_emissions
        - logsumexp(shifted_log_emissions, axis=1, keepdims=True)
    )
    grid_mass = posterior.mean(axis=0)
    nearest = np.argmin(
        np.sum(
            (
                coordinates[:, None, :]
                - np.asarray(locations, dtype=float)[None, :, :]
            )
            ** 2,
            axis=2,
        ),
        axis=1,
    )
    masses = np.asarray(
        [np.sum(grid_mass[nearest == index]) for index in range(len(locations))],
        dtype=float,
    )
    masses /= masses.sum()
    return {
        identifier: float(mass)
        for identifier, mass in zip(identifiers, masses, strict=True)
    }


def export_event(
    session: ReplaySession,
    session_routes: pd.DataFrame,
    session_route_points: pd.DataFrame,
    selection: dict[str, object],
    encoding_config: EncodingConfig,
    emission_config: EmissionConfig,
    behavior_config: BehaviorFieldConfig,
    point_spread_config: PointSpreadConfig,
) -> ExportedEvent:
    event_index = int(selection["event_index"])
    event = session.ripple(event_index)
    cutoff = float(np.nextafter(event.start, -np.inf))
    encoding = fit_place_field_encoding(
        session,
        encoding_config,
        training_end_s=cutoff,
    )
    emissions = build_emissions(
        session,
        encoding,
        event_index,
        emission_config,
    )
    if emissions.metadata.get("decoder_training_schedule") != (
        "event_specific_prefix_refit"
    ):
        raise ValueError("replay emission decoder is not an event-specific prefix refit")
    for key in (
        "decoder_training_position_max_s",
        "decoder_training_spike_max_s",
    ):
        value = emissions.metadata.get(key)
        if value is not None and float(value) > cutoff:
            raise ValueError(f"{key} crosses the replay cutoff")

    active = np.asarray(encoding.occupancy_s > 0.0, dtype=bool)
    if int(np.sum(active)) < 2:
        raise ValueError("event encoding has fewer than two occupied spatial bins")
    coordinates = np.asarray(encoding.bin_centers[active], dtype=float)
    raw = np.asarray(emissions.log_likelihood[:, active], dtype=float)
    offsets = np.max(raw, axis=1)
    shifted = raw - offsets[:, None]
    location, location_time = _current_location(session, cutoff)
    behavior = build_pre_replay_candidate_fields(
        session_routes,
        session_route_points,
        coordinates,
        event_start_s=event.start,
        current_location_xy=location,
        current_location_time_s=location_time,
        config=behavior_config,
    )
    nuisance = np.asarray(encoding.occupancy_s[active], dtype=float)
    nuisance += max(float(np.mean(nuisance)) * 1e-6, np.finfo(float).tiny)
    nuisance /= nuisance.sum()
    point_spread, point_spread_audit = estimate_prefix_decoder_point_spread_cm(
        session,
        event.start,
        encoding_config,
        emission_config,
        point_spread_config,
    )
    completed_routes = session_routes[
        pd.to_numeric(
            session_routes["movement_end_time_s"],
            errors="coerce",
        )
        <= event.start
    ]
    completed_ids = set(completed_routes["route_id"].astype(str))
    completed_points = session_route_points[
        session_route_points["route_id"].astype(str).isin(completed_ids)
    ]
    well_masses = _well_locations_and_masses(
        completed_routes,
        completed_points,
        coordinates,
        shifted,
        session.session_id,
    )
    audit = {
        **selection,
        **point_spread_audit,
        "session": session.session_id,
        "rat": session.rat,
        "event_id": f"{session.session_id}:event-{event_index}",
        "decoder_training_cutoff_s": cutoff,
        "decoder_training_position_max_s": encoding.training_position_max_s,
        "decoder_training_spike_max_s": encoding.training_spike_max_s,
        "encoding_cells": encoding.n_cells,
        "active_spatial_bins": int(np.sum(active)),
        "replay_time_bins": emissions.n_time,
        "revision_total_weight": behavior.revision_total_weight,
        "revision_snippet_count": behavior.revision_snippet_count,
        **{
            f"candidate_available_{name}": bool(behavior.available[index])
            for index, name in enumerate(SPATIAL_CANDIDATE_NAMES)
        },
    }
    return ExportedEvent(
        event_id=str(audit["event_id"]),
        rat=session.rat,
        session=session.session_id,
        event_index=event_index,
        event_start_s=float(event.start),
        event_end_s=float(event.end),
        history_cutoff_s=behavior.history_cutoff_s,
        decoder_training_cutoff_s=cutoff,
        field_available_s=behavior.available_s,
        log_emissions=shifted,
        log_emission_offsets=offsets,
        spatial_coordinates=coordinates,
        nuisance_base=nuisance,
        candidate_fields=behavior.fields,
        candidate_available=behavior.available,
        decoder_point_spread_cm=point_spread,
        well_masses=well_masses,
        audit=audit,
    )


def _pack_events(events: list[ExportedEvent]) -> dict[str, np.ndarray]:
    if not events:
        raise ValueError("at least one exported event is required")
    n_events = len(events)
    n_time = max(len(event.log_emissions) for event in events)
    n_bins = max(len(event.spatial_coordinates) for event in events)
    n_candidates = len(SPATIAL_CANDIDATE_NAMES)
    log_emissions = np.full((n_events, n_time, n_bins), -np.inf, dtype=float)
    offsets = np.zeros((n_events, n_time), dtype=float)
    time_mask = np.zeros((n_events, n_time), dtype=bool)
    active = np.zeros((n_events, n_bins), dtype=bool)
    coordinates = np.full((n_events, n_bins, 2), np.nan, dtype=float)
    nuisance = np.zeros((n_events, n_bins), dtype=float)
    fields = np.zeros((n_events, n_candidates, n_bins), dtype=float)
    available = np.zeros((n_events, n_candidates), dtype=bool)
    available_s = np.zeros((n_events, n_candidates), dtype=float)
    well_ids = tuple(sorted({key for event in events for key in event.well_masses}))
    well_mass = np.zeros((n_events, len(well_ids)), dtype=float)
    well_lookup = {well_id: index for index, well_id in enumerate(well_ids)}

    for index, event in enumerate(events):
        event_time, event_bins = event.log_emissions.shape
        log_emissions[index, :event_time, :event_bins] = event.log_emissions
        offsets[index, :event_time] = event.log_emission_offsets
        time_mask[index, :event_time] = True
        active[index, :event_bins] = True
        coordinates[index, :event_bins] = event.spatial_coordinates
        nuisance[index, :event_bins] = event.nuisance_base
        fields[index, :, :event_bins] = event.candidate_fields
        available[index] = event.candidate_available
        available_s[index] = event.field_available_s
        for well_id, mass in event.well_masses.items():
            well_mass[index, well_lookup[well_id]] = mass

    return {
        "event_ids": np.asarray([event.event_id for event in events], dtype=str),
        "rat_ids": np.asarray([event.rat for event in events], dtype=str),
        "session_ids": np.asarray([event.session for event in events], dtype=str),
        "event_start_s": np.asarray(
            [event.event_start_s for event in events],
            dtype=float,
        ),
        "event_end_s": np.asarray(
            [event.event_end_s for event in events],
            dtype=float,
        ),
        "history_cutoff_s": np.asarray(
            [event.history_cutoff_s for event in events],
            dtype=float,
        ),
        "decoder_training_cutoff_s": np.asarray(
            [event.decoder_training_cutoff_s for event in events],
            dtype=float,
        ),
        "field_available_s": available_s,
        "log_emissions": log_emissions,
        "log_emission_offsets": offsets,
        "time_mask": time_mask,
        "active_spatial_mask": active,
        "spatial_coordinates": coordinates,
        "decoder_point_spread_cm": np.asarray(
            [event.decoder_point_spread_cm for event in events],
            dtype=float,
        ),
        "nuisance_base": nuisance,
        "candidate_fields": fields,
        "candidate_available": available,
        "candidate_names": np.asarray(SPATIAL_CANDIDATE_NAMES, dtype=str),
        "well_masses": well_mass,
        "well_ids": np.asarray(well_ids, dtype=str),
    }


def run_export(
    *,
    dataset_root: str | Path,
    route_segments_csv: str | Path,
    route_points_csv: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    dataset_sha256: str,
    selection_config: EventSelectionConfig | None = None,
    encoding_config: EncodingConfig | None = None,
    emission_config: EmissionConfig | None = None,
    behavior_config: BehaviorFieldConfig | None = None,
    point_spread_config: PointSpreadConfig | None = None,
) -> dict[str, Path]:
    selection_config = (
        EventSelectionConfig() if selection_config is None else selection_config
    )
    encoding_config = EncodingConfig() if encoding_config is None else encoding_config
    emission_config = EmissionConfig() if emission_config is None else emission_config
    behavior_config = (
        BehaviorFieldConfig() if behavior_config is None else behavior_config
    )
    point_spread_config = (
        PointSpreadConfig() if point_spread_config is None else point_spread_config
    )
    selection_config.validate()
    behavior_config.validate()
    point_spread_config.validate()
    if _SHA256.fullmatch(str(dataset_sha256)) is None:
        raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")

    routes = pd.read_csv(route_segments_csv)
    route_points = pd.read_csv(route_points_csv)
    sessions = tuple(sorted(set(routes["session"].astype(str))))
    dataset = Path(dataset_root)
    events: list[ExportedEvent] = []
    for session_id in sessions:
        session = load_replay_session(dataset / Path(session_id))
        session_routes = routes[routes["session"].astype(str).eq(session_id)].copy()
        session_route_ids = set(session_routes["route_id"].astype(str))
        session_points = route_points[
            route_points["route_id"].astype(str).isin(session_route_ids)
        ].copy()
        selections = select_lfp_only_events(
            session,
            session_routes,
            selection_config,
        )
        for selection in selections:
            events.append(
                export_event(
                    session,
                    session_routes,
                    session_points,
                    selection,
                    encoding_config,
                    emission_config,
                    behavior_config,
                    point_spread_config,
                )
            )

    arrays = _pack_events(events)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictor_path = output / PREDICTOR_OUTPUT
    np.savez_compressed(predictor_path, **arrays)
    predictor_sha256 = _sha256_file(predictor_path)
    git = git_metadata(ROOT)
    producer_commit = str(git.get("code_commit", ""))
    if _COMMIT.fullmatch(producer_commit) is None:
        raise ValueError("producer must run from a committed Git checkout")
    manifest = {
        "producer_repository": "IPS-Stuttgart/HippoReplayDynamics",
        "producer_commit": producer_commit,
        "dataset_id": str(dataset_id),
        "dataset_sha256": str(dataset_sha256),
        "trace_schema_version": SMOOTHING_TRACE_SCHEMA_VERSION,
        "transition_convention": TRANSITION_CONVENTION,
        "candidate_evidence_cutoff": "strict_pre_replay",
        "likelihood_domain": "max_shifted_log_emission_plus_offset",
        "decoder_training_schedule": "event_specific_prefix_refit",
        "decoder_point_spread_schedule": "pre_event_temporal_holdout_run_68pct",
        "well_mass_source": "raw_log_emission_posterior",
        "behavior_latent_state": "compact_destination_well",
        "behavior_observation_schedule": (
            "tracked_position_and_well_visits_pre_replay"
        ),
        "state_to_spatial_mapping": "pre_replay_route_kernel",
        "event_selection_schedule": "lfp_raw_peak_power_top_n_per_session",
        "event_selection_parameters_sha256": selection_config.sha256(),
        "behavior_field_parameters_sha256": behavior_config.sha256(),
        "decoder_parameters_sha256": _configuration_digest(
            encoding_config,
            emission_config,
            point_spread_config,
        ),
        "spatial_coordinate_units": "cm",
        "replay_feedback_used": False,
        "outcomes_in_predictor": False,
        "schema_version": SCHEMA_VERSION,
        "predictor_file": predictor_path.name,
        "predictor_sha256": predictor_sha256,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_path = output / EVENT_AUDIT_OUTPUT
    pd.DataFrame([event.audit for event in events]).to_csv(audit_path, index=False)
    provenance = build_script_provenance(
        input_paths={
            "route_segments": route_segments_csv,
            "route_points": route_points_csv,
        },
        cwd=ROOT,
    )
    summary_path = output / SUMMARY_OUTPUT
    common = np.all(arrays["candidate_available"], axis=1)
    summary_path.write_text(
        "\n".join(
            [
                "# PF replay spatial export",
                "",
                f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}",
                f"- Events: {len(events)}",
                f"- Sessions: {len(sessions)}",
                f"- Rats: {len(set(arrays['rat_ids'].astype(str)))}",
                f"- Complete-case events: {int(np.sum(common))}",
                "- Event selection: raw LFP peak power only; decoded content was not inspected.",
                "- Decoder training: event-specific strict prefix refit.",
                "- Later outcomes/replay feedback: absent.",
                f"- Predictor SHA-256: {predictor_sha256}",
                f"- Producer commit: {producer_commit}",
                f"- Git dirty at run: {provenance.get('git_dirty')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        PREDICTOR_OUTPUT: predictor_path,
        MANIFEST_OUTPUT: manifest_path,
        EVENT_AUDIT_OUTPUT: audit_path,
        SUMMARY_OUTPUT: summary_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--route-segments", required=True)
    parser.add_argument("--route-points", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="PfeifferFoster-open-field-2013")
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--events-per-session", type=int, default=20)
    parser.add_argument("--minimum-raw-power", type=float, default=0.0)
    parser.add_argument("--minimum-training-duration-s", type=float, default=300.0)
    parser.add_argument("--minimum-completed-routes", type=int, default=6)
    parser.add_argument("--point-spread-window-s", type=float, default=60.0)
    parser.add_argument("--max-events", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_events:
        raise ValueError(
            "--max-events is diagnostic-only and cannot produce a frozen artifact"
        )
    run_export(
        dataset_root=args.dataset_root,
        route_segments_csv=args.route_segments,
        route_points_csv=args.route_points,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_sha256=args.dataset_sha256,
        selection_config=EventSelectionConfig(
            events_per_session=args.events_per_session,
            minimum_raw_power=args.minimum_raw_power,
            minimum_training_duration_s=args.minimum_training_duration_s,
            minimum_completed_routes=args.minimum_completed_routes,
        ),
        point_spread_config=PointSpreadConfig(
            holdout_window_s=args.point_spread_window_s,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
