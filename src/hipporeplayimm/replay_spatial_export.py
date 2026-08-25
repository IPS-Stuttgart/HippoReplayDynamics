"""Causal behavioral candidate fields for the PF replay spatial contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .smoothing_trace import first_order_smoothing_trace

SPATIAL_CANDIDATE_NAMES = (
    "smoothing_revision",
    "online_surprise",
    "posterior_content",
    "current_location",
    "recency",
    "prospective",
    "td_error",
)
_FLOAT_TOL = 1e-10


@dataclass(frozen=True, slots=True)
class BehaviorFieldConfig:
    """Predeclared compact-well smoother and spatial projection settings."""

    observation_sigma_cm: float = 15.0
    terminal_label_error: float = 0.02
    transition_pseudocount: float = 0.5
    revision_recency_tau_s: float = 300.0
    candidate_recency_tau_s: float = 300.0
    current_location_sigma_cm: float = 15.0
    route_kernel_sigma_cm: float = 10.0
    template_points: int = 21
    filter_prefix_fraction: float = 0.5
    minimum_revision_weight: float = 1e-8
    td_learning_rate: float = 0.2
    td_discount: float = 0.9
    td_clip: float = 5.0

    def validate(self) -> None:
        positive = (
            self.observation_sigma_cm,
            self.transition_pseudocount,
            self.revision_recency_tau_s,
            self.candidate_recency_tau_s,
            self.current_location_sigma_cm,
            self.route_kernel_sigma_cm,
            self.minimum_revision_weight,
            self.td_clip,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("positive behavior-field settings must be finite")
        if self.template_points < 5:
            raise ValueError("template_points must be at least five")
        if not 0.0 < self.terminal_label_error < 0.5:
            raise ValueError("terminal_label_error must lie in (0, 0.5)")
        if not 0.0 < self.filter_prefix_fraction < 1.0:
            raise ValueError("filter_prefix_fraction must lie in (0, 1)")
        if not 0.0 < self.td_learning_rate <= 1.0:
            raise ValueError("td_learning_rate must lie in (0, 1]")
        if not 0.0 <= self.td_discount <= 1.0:
            raise ValueError("td_discount must lie in [0, 1]")

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BehavioralSpatialFields:
    fields: NDArray[np.float64]
    available: NDArray[np.bool_]
    available_s: NDArray[np.float64]
    history_cutoff_s: float
    revision_total_weight: float
    revision_snippet_count: int
    well_ids: tuple[int, ...]
    filtered_probabilities: tuple[NDArray[np.float64], ...]
    smoothed_probabilities: tuple[NDArray[np.float64], ...]
    snippet_end_s: NDArray[np.float64]


def _unit_mass(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    total = float(np.sum(array))
    return np.asarray(array / total, dtype=np.float64) if total > 0.0 else array


def _unit_absolute_mass(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    total = float(np.sum(np.abs(array)))
    return np.asarray(array / total, dtype=np.float64) if total > 0.0 else array


def _resample_path(points: pd.DataFrame, n_points: int) -> NDArray[np.float64] | None:
    ordered = points.sort_values("point_index")
    if len(ordered) < 2:
        return None
    xy = ordered[["x_cm", "y_cm"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(xy)):
        return None
    if "arc_fraction" in ordered:
        fraction = ordered["arc_fraction"].to_numpy(dtype=float)
    else:
        distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distance)))
        if cumulative[-1] <= 0.0:
            return None
        fraction = cumulative / cumulative[-1]
    keep = np.concatenate(([True], np.diff(fraction) > 1e-12))
    fraction = fraction[keep]
    xy = xy[keep]
    if len(fraction) < 2 or fraction[-1] <= fraction[0]:
        return None
    target = np.linspace(float(fraction[0]), float(fraction[-1]), n_points)
    return np.column_stack(
        [np.interp(target, fraction, xy[:, axis]) for axis in range(2)]
    )


def _route_path(
    route_points: pd.DataFrame,
    route_id: str,
    n_points: int,
) -> NDArray[np.float64] | None:
    return _resample_path(
        route_points[route_points["route_id"].astype(str).eq(str(route_id))],
        n_points,
    )


def _template_path(
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    *,
    origin_well: int,
    destination_well: int,
    n_points: int,
) -> NDArray[np.float64] | None:
    same_destination = routes[
        routes["destination_well_id"].astype(int).eq(int(destination_well))
    ]
    same_pair = same_destination[
        same_destination["origin_well_id"].astype(int).eq(int(origin_well))
    ]
    selected = same_pair if not same_pair.empty else same_destination
    paths = [
        path
        for route_id in selected["route_id"].astype(str)
        if (
            path := _route_path(route_points, route_id, n_points)
        ) is not None
    ]
    if not paths:
        return None
    return np.asarray(np.mean(np.stack(paths), axis=0), dtype=np.float64)


def _path_kernel(
    spatial_coordinates: NDArray[np.float64],
    path: NDArray[np.float64],
    sigma_cm: float,
) -> NDArray[np.float64]:
    distance_squared = np.min(
        np.sum(
            (
                spatial_coordinates[:, None, :]
                - np.asarray(path, dtype=float)[None, :, :]
            )
            ** 2,
            axis=2,
        ),
        axis=1,
    )
    return _unit_mass(np.exp(-0.5 * distance_squared / float(sigma_cm) ** 2))


def _point_kernel(
    spatial_coordinates: NDArray[np.float64],
    point: NDArray[np.float64],
    sigma_cm: float,
) -> NDArray[np.float64]:
    distance_squared = np.sum(
        (spatial_coordinates - np.asarray(point, dtype=float)[None, :]) ** 2,
        axis=1,
    )
    return _unit_mass(np.exp(-0.5 * distance_squared / float(sigma_cm) ** 2))


def _transition_prior(
    routes: pd.DataFrame,
    origin_well: int,
    states: tuple[int, ...],
    pseudocount: float,
) -> NDArray[np.float64]:
    counts = np.full(len(states), float(pseudocount), dtype=float)
    selected = routes[routes["origin_well_id"].astype(int).eq(int(origin_well))]
    destinations = selected["destination_well_id"].astype(int).to_numpy()
    for index, state in enumerate(states):
        counts[index] += float(np.sum(destinations == state))
    return _unit_mass(counts)


def _state_templates_and_mapping(
    training_routes: pd.DataFrame,
    route_points: pd.DataFrame,
    states: tuple[int, ...],
    origin_well: int,
    spatial_coordinates: NDArray[np.float64],
    config: BehaviorFieldConfig,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    templates: list[NDArray[np.float64]] = []
    mappings: list[NDArray[np.float64]] = []
    for state in states:
        template = _template_path(
            training_routes,
            route_points,
            origin_well=origin_well,
            destination_well=state,
            n_points=config.template_points,
        )
        if template is None:
            return None
        templates.append(template)
        mappings.append(
            _path_kernel(
                spatial_coordinates,
                template,
                config.route_kernel_sigma_cm,
            )
        )
    return templates, np.asarray(mappings, dtype=np.float64)


def _static_filter_and_smoother(
    prior: NDArray[np.float64],
    observed_path: NDArray[np.float64],
    templates: list[NDArray[np.float64]],
    states: tuple[int, ...],
    actual_destination: int,
    config: BehaviorFieldConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    distance_squared = np.asarray(
        [
            np.sum((observed_path - template) ** 2, axis=1)
            for template in templates
        ],
        dtype=float,
    )
    log_likelihood = -0.5 * distance_squared / config.observation_sigma_cm**2
    prefix_count = max(
        1,
        min(
            len(observed_path) - 1,
            int(np.ceil(config.filter_prefix_fraction * len(observed_path))),
        ),
    )
    terminal = np.full(
        len(states),
        np.log(config.terminal_label_error / max(len(states) - 1, 1)),
        dtype=float,
    )
    terminal[states.index(actual_destination)] = np.log(
        1.0 - config.terminal_label_error
    )
    emissions = np.vstack((log_likelihood.T, terminal[None, :]))
    trace = first_order_smoothing_trace(
        emissions,
        np.eye(len(states), dtype=float),
        initial_probabilities=prior,
    )
    prefix_index = prefix_count - 1
    return (
        np.asarray(trace.filtered_probabilities[prefix_index], dtype=np.float64),
        np.asarray(trace.smoothed_probabilities[prefix_index], dtype=np.float64),
    )


def _categorical_kl(
    posterior: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> float:
    positive = posterior > 0.0
    return float(np.sum(posterior[positive] * np.log(posterior[positive] / prior[positive])))


def _nonconstant(field: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(field)) and np.std(field) > 1e-12)


def build_pre_replay_candidate_fields(
    routes: pd.DataFrame,
    route_points: pd.DataFrame,
    spatial_coordinates: NDArray[np.float64],
    *,
    event_start_s: float,
    current_location_xy: NDArray[np.float64],
    current_location_time_s: float,
    config: BehaviorFieldConfig | None = None,
) -> BehavioralSpatialFields:
    """Build seven frozen fields from completed pre-replay RUN traversals only."""

    config = BehaviorFieldConfig() if config is None else config
    config.validate()
    coordinates = np.asarray(spatial_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("spatial_coordinates must have shape (at least two bins, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("spatial_coordinates must be finite")
    event_start = float(event_start_s)
    location_time = float(current_location_time_s)
    location = np.asarray(current_location_xy, dtype=float)
    if location.shape != (2,) or not np.all(np.isfinite(location)):
        raise ValueError("current_location_xy must contain finite x/y coordinates")
    if (
        not np.isfinite(event_start)
        or not np.isfinite(location_time)
        or location_time > event_start + _FLOAT_TOL
    ):
        raise ValueError("current location must be finite and available before replay")

    required_route_columns = {
        "route_id",
        "origin_well_id",
        "destination_well_id",
        "movement_start_time_s",
        "movement_end_time_s",
        "interval_end_time_s",
    }
    required_point_columns = {"route_id", "point_index", "x_cm", "y_cm"}
    if not required_route_columns.issubset(routes.columns):
        raise ValueError("route table is missing required columns")
    if not required_point_columns.issubset(route_points.columns):
        raise ValueError("route-point table is missing required columns")

    evidence_cutoff = float(np.nextafter(event_start, -np.inf))
    completed = routes[
        pd.to_numeric(routes["interval_end_time_s"], errors="coerce")
        <= evidence_cutoff
    ].copy()
    completed = completed.sort_values(
        [
            "interval_end_time_s",
            "movement_end_time_s",
            "movement_start_time_s",
            "route_id",
        ]
    )
    n_bins = len(coordinates)
    fields = np.zeros((len(SPATIAL_CANDIDATE_NAMES), n_bins), dtype=float)
    available = np.zeros(len(SPATIAL_CANDIDATE_NAMES), dtype=bool)
    available_s = np.full(len(SPATIAL_CANDIDATE_NAMES), location_time, dtype=float)
    filtered_rows: list[NDArray[np.float64]] = []
    smoothed_rows: list[NDArray[np.float64]] = []
    snippet_ends: list[float] = []
    revision_total_weight = 0.0
    revision_snippet_count = 0
    history_cutoff = location_time

    revision = np.zeros(n_bins, dtype=float)
    surprise_field = np.zeros(n_bins, dtype=float)
    content_field = np.zeros(n_bins, dtype=float)
    recency_field = np.zeros(n_bins, dtype=float)
    td_field = np.zeros(n_bins, dtype=float)
    values: dict[int, float] = {}

    for route in completed.itertuples(index=False):
        route_start = float(route.movement_start_time_s)
        route_end = float(route.movement_end_time_s)
        route_available = float(route.interval_end_time_s)
        history_cutoff = max(history_cutoff, route_available)
        training = completed[
            pd.to_numeric(
                completed["interval_end_time_s"],
                errors="coerce",
            )
            < route_start - _FLOAT_TOL
        ]
        states = tuple(
            sorted(
                set(
                    pd.to_numeric(
                        training["destination_well_id"],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                )
            )
        )
        actual_destination = int(route.destination_well_id)
        origin = int(route.origin_well_id)
        observed = _route_path(
            route_points,
            str(route.route_id),
            config.template_points,
        )
        age_weight = np.exp(
            -max(event_start - route_available, 0.0)
            / config.candidate_recency_tau_s
        )

        actual_kernel = None
        if observed is not None:
            actual_kernel = _path_kernel(
                coordinates,
                observed,
                config.route_kernel_sigma_cm,
            )
            recency_field += age_weight * actual_kernel

        reward = 1.0
        delta = (
            reward
            + config.td_discount * values.get(actual_destination, 0.0)
            - values.get(origin, 0.0)
        )
        values[origin] = values.get(origin, 0.0) + config.td_learning_rate * delta
        if actual_kernel is not None:
            td_field += (
                age_weight
                * float(np.clip(delta, -config.td_clip, config.td_clip))
                * actual_kernel
            )

        if (
            observed is None
            or len(states) < 2
            or actual_destination not in states
        ):
            continue
        template_result = _state_templates_and_mapping(
            training,
            route_points,
            states,
            origin,
            coordinates,
            config,
        )
        if template_result is None:
            continue
        templates, mapping = template_result
        prior = _transition_prior(
            training,
            origin,
            states,
            config.transition_pseudocount,
        )
        filtered, smoothed = _static_filter_and_smoother(
            prior,
            observed,
            templates,
            states,
            actual_destination,
            config,
        )
        kl = _categorical_kl(smoothed, filtered)
        revision_weight = kl * np.exp(
            -max(event_start - route_available, 0.0)
            / config.revision_recency_tau_s
        )
        revision += revision_weight * ((smoothed - filtered) @ mapping)
        surprise_field += age_weight * (-np.log(prior[states.index(actual_destination)])) * mapping[
            states.index(actual_destination)
        ]
        content_field += age_weight * (smoothed @ mapping)
        revision_total_weight += revision_weight
        revision_snippet_count += 1
        filtered_rows.append(filtered)
        smoothed_rows.append(smoothed)
        snippet_ends.append(route_available)

    fields[0] = _unit_absolute_mass(revision)
    fields[1] = _unit_mass(surprise_field)
    fields[2] = _unit_mass(content_field)
    fields[3] = _point_kernel(
        coordinates,
        location,
        config.current_location_sigma_cm,
    )
    fields[4] = _unit_mass(recency_field)
    fields[6] = _unit_absolute_mass(td_field)

    if not completed.empty:
        last = completed.iloc[-1]
        states = tuple(
            sorted(
                set(
                    pd.to_numeric(
                        completed["destination_well_id"],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                )
            )
        )
        if len(states) >= 2:
            origin = int(last["destination_well_id"])
            mapped = _state_templates_and_mapping(
                completed,
                route_points,
                states,
                origin,
                coordinates,
                config,
            )
            if mapped is not None:
                _, mapping = mapped
                fields[5] = (
                    _transition_prior(
                        completed,
                        origin,
                        states,
                        config.transition_pseudocount,
                    )
                    @ mapping
                )

    available[0] = (
        revision_total_weight > config.minimum_revision_weight
        and revision_snippet_count > 0
        and _nonconstant(fields[0])
    )
    for index in range(1, len(SPATIAL_CANDIDATE_NAMES)):
        available[index] = _nonconstant(fields[index])
    if not completed.empty:
        route_cutoff = float(completed["interval_end_time_s"].max())
        available_s[[0, 1, 2, 4, 5, 6]] = route_cutoff
    available_s[3] = location_time
    if np.any(available_s > event_start + _FLOAT_TOL):
        raise ValueError("candidate evidence crosses the replay start")

    well_ids = tuple(
        sorted(
            set(
                pd.to_numeric(
                    completed["destination_well_id"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )
        )
    )
    return BehavioralSpatialFields(
        fields=np.asarray(fields, dtype=np.float64),
        available=np.asarray(available, dtype=np.bool_),
        available_s=np.asarray(available_s, dtype=np.float64),
        history_cutoff_s=float(history_cutoff),
        revision_total_weight=float(revision_total_weight),
        revision_snippet_count=int(revision_snippet_count),
        well_ids=well_ids,
        filtered_probabilities=tuple(filtered_rows),
        smoothed_probabilities=tuple(smoothed_rows),
        snippet_end_s=np.asarray(snippet_ends, dtype=np.float64),
    )


__all__ = [
    "BehaviorFieldConfig",
    "BehavioralSpatialFields",
    "SPATIAL_CANDIDATE_NAMES",
    "build_pre_replay_candidate_fields",
]
