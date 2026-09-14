"""Auditable recording inputs for replay coverage and recovery experiments.

No replay-content filter is applied here. Position gaps are excluded from RUN
encoding; the original clocks and all candidate windows remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ReplaySession, load_replay_session
from .encoding import (
    EncodingConfig,
    _frame_durations,
    _speed_cm_s,
    _times_in_intervals,
    fit_place_field_encoding,
)

TANNI_POSITION = "/acquisition/timeseries/recording1/tracking/ProcessedPos"
TANNI_SPIKES = "/acquisition/timeseries/recording1/spikes"
TANNI_GENERAL = "/general/data_collection/Settings/General"


@dataclass(frozen=True)
class CoverageInputConfig:
    bin_size_cm: float = 8.0
    smoothing_sigma_bins: float = 1.5
    min_speed_cm_s: float = 10.0
    min_occupancy_s: float = 0.05
    maximum_position_gap_s: float = 0.1
    base_bin_s: float = 0.005
    min_running_spikes: int = 30
    max_running_rate_hz: float = 4.0
    min_peak_rate_hz: float = 2.0
    min_split_half_stability: float = 0.25


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape and contiguous content, not an ambiguous byte payload."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    if value.dtype.hasobject:
        strings = [v.decode("utf-8") if isinstance(v, bytes) else v for v in value.reshape(-1)]
        if not all(isinstance(v, str) for v in strings):
            raise ValueError("non-string object arrays have no portable content hash")
        digest.update(json.dumps(strings, ensure_ascii=True).encode("ascii"))
    else:
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def align_tanni_clusters(
    times: np.ndarray, labels: np.ndarray, keep: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Handle labels for either all detections or only idx_keep detections."""
    times = np.asarray(times, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if not np.isfinite(labels).all() or np.any(labels != np.floor(labels)):
        raise ValueError("cluster labels must be finite integers")
    if keep is None:
        if labels.size != times.size:
            raise ValueError("cluster/timestamp length mismatch without idx_keep")
    else:
        keep = np.asarray(keep).reshape(-1)
        if keep.size != times.size or not np.isin(keep, [0, 1]).all():
            raise ValueError("idx_keep must be a Boolean mask over timestamps")
        keep = keep.astype(bool)
        if labels.size == times.size:
            labels = labels[keep]
        elif labels.size != int(keep.sum()):
            raise ValueError("cluster/timestamp/idx_keep length mismatch")
        times = times[keep]
    valid = np.isfinite(times) & (labels > 1)
    return times[valid], labels[valid].astype(np.int64)


def load_tanni_coverage_session(path: Path) -> tuple[ReplaySession, dict]:
    """Read only position, manual clusters, and arena metadata from native NWB.

    Provenance hashes every consumed HDF5 dataset. Unused waveform/LFP payloads
    are NOT represented as whole-file hashes. Cluster 0/unassigned and 1/noise
    are excluded, matching this dataset's manual_1 convention.
    """
    import h5py

    path = Path(path).resolve()
    before = path.stat()
    consumed = {}
    with h5py.File(path, "r") as handle:
        def read(key):
            value = np.asarray(handle[key][()])
            consumed[key] = {
                "shape": list(value.shape), "dtype": value.dtype.str,
                "sha256": array_sha256(value),
            }
            return value

        position = np.asarray(read(TANNI_POSITION), dtype=float)[:, :3]
        size = np.asarray(read(f"{TANNI_GENERAL}/arena_size"), dtype=float).reshape(-1)
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("invalid Tanni arena dimensions")
        animal_value = read(f"{TANNI_GENERAL}/animal").reshape(-1)[0]
        animal = animal_value.decode() if isinstance(animal_value, bytes) else str(animal_value)
        chunks = []
        for name in sorted(handle[TANNI_SPIKES]):
            match = re.fullmatch(r"electrode(\d+)", name)
            if match is None:
                raise ValueError(f"unexpected spike group {name}")
            group = f"{TANNI_SPIKES}/{name}"
            label_key = f"{group}/clustering/manual_1"
            if label_key not in handle:
                raise ValueError(f"manual_1 clustering missing: {name}")
            times, labels = align_tanni_clusters(
                read(f"{group}/timestamps"), read(label_key),
                read(f"{group}/idx_keep") if f"{group}/idx_keep" in handle else None,
            )
            if np.any(labels >= 1000):
                raise ValueError("cluster ID exceeds collision-free tetrode*1000 convention")
            ids = int(match.group(1)) * 1000 + labels
            chunks.append(np.column_stack((times, ids)))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("NWB input changed while being read")
    spikes = np.concatenate(chunks) if chunks else np.empty((0, 2))
    spikes = spikes[np.argsort(spikes[:, 0], kind="stable")]
    ids = np.unique(spikes[:, 1].astype(int))
    finite_times = position[np.isfinite(position[:, 0]), 0]
    if finite_times.size < 2:
        raise ValueError("position clock missing")
    empty_intervals = np.empty((0, 2))
    bounds = np.array([[0.0, 0.0], size])
    session = ReplaySession(
        rat=animal, name=path.parent.name, path=path.parent,
        position=position, spikes=spikes,
        tetrode_cell_ids=np.column_stack((ids // 1000, ids % 1000)),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int), ripple_events=np.empty((0, 6)),
        run_times=np.array([[finite_times.min(), finite_times.max()]]),
        sleep_box_immobile_times=empty_intervals, sleep_times=empty_intervals,
        rem_times=empty_intervals, well_sequence=None,
        metadata={"source_dataset": "tanni2022", "arena_bounds_cm": bounds.tolist()},
    )
    return session, {
        "source_path": str(path), "source_size_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "hash_scope": "consumed_HDF5_datasets_not_entire_NWB",
        "consumed_datasets": consumed, "arena_bounds_cm": bounds.tolist(),
        "arena_bounds_source": "native_arena_size_metadata_origin_zero",
        "cell_type_source": "manual_sorted_clusters_no_waveform_cell_type_assignment",
    }


def load_pf_coverage_session(path: Path) -> tuple[ReplaySession, dict]:
    names = ["Spike_Data.mat", "Position_Data.mat", "Epochs.mat", "Experiment_Information.mat", "Ripple_Events.mat"]
    initial_stats = {name: (Path(path) / name).stat() for name in names if (Path(path) / name).exists()}
    session = load_replay_session(path)
    consumed = {}
    for name in names:
        source = Path(path) / name
        if not source.exists():
            continue
        digest = hashlib.sha256()
        before = initial_stats[name]
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("MAT input changed while being hashed")
        consumed[str(source.resolve())] = {"size_bytes": before.st_size, "sha256": digest.hexdigest()}
    return session, {
        "source_path": str(Path(path).resolve()), "hash_scope": "relevant_whole_MAT_files",
        "consumed_files": consumed,
        "arena_bounds_cm": None,
        "arena_bounds_source": "unknown_do_not_substitute_tracking_extent_for_walls",
        "cell_type_source": "provided_excitatory_neuron_IDs",
    }


def prepare_position_support(
    session: ReplaySession, maximum_gap_s: float, bounds: np.ndarray | None = None,
) -> tuple[ReplaySession, dict]:
    """Exclude gap-edge frames and prohibit RUN spike interpolation across gaps."""
    if not np.isfinite(maximum_gap_s) or maximum_gap_s <= 0:
        raise ValueError("maximum position gap must be positive")
    original = np.asarray(session.position, dtype=float)[:, :3]
    position = original[np.isfinite(original[:, 0])].copy()
    if not len(position):
        raise ValueError("no finite position times")
    # A clock reset is not repaired by sorting the samples or shifting spikes.
    increasing = np.r_[True, position[1:, 0] > np.maximum.accumulate(position[:-1, 0])]
    position = position[increasing]
    valid = np.isfinite(position[:, 1:3]).all(axis=1)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("invalid arena bounds")
        valid &= ((position[:, 1:3] >= bounds[0]) & (position[:, 1:3] <= bounds[1])).all(axis=1)
    position[~valid, 1:3] = np.nan
    links = valid[:-1] & valid[1:] & (np.diff(position[:, 0]) <= maximum_gap_s)
    starts = np.flatnonzero(valid & ~np.r_[False, links])
    stops = np.flatnonzero(valid & ~np.r_[links, False])
    safe_intervals = []
    for start, stop in zip(starts, stops, strict=True):
        # Removing each segment's end frames also removes inflated frame dwell
        # times and finite-difference speed estimates at a missing-data gap.
        if stop - start < 3:
            continue
        lo, hi = position[start + 1, 0], position[stop - 1, 0]
        for run_start, run_end in np.asarray(session.run_times).reshape(-1, 2):
            a, b = max(lo, run_start), min(hi, run_end)
            if b > a:
                safe_intervals.append((a, b))
    safe_intervals = np.asarray(safe_intervals, dtype=float).reshape(-1, 2)
    if not len(safe_intervals):
        raise ValueError("no contiguous tracking-supported RUN intervals")
    finite_position = position[valid]
    return replace(session, position=finite_position, run_times=safe_intervals), {
        "position_samples_raw": len(original), "position_samples_clock_retained": len(position),
        "position_samples_valid": int(valid.sum()),
        "position_clock_rejected": int(len(original) - len(position)),
        "position_valid_fraction": float(valid.mean()),
        "position_start_s": float(position[0, 0]), "position_end_s": float(position[-1, 0]),
        "tracking_supported_run_intervals": len(safe_intervals),
        "tracking_extent_cm": [finite_position[:, 1:3].min(axis=0).tolist(), finite_position[:, 1:3].max(axis=0).tolist()],
    }


def canonical_candidates(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Retain all high-MUA candidates, without using any decoder outputs."""
    if dataset == "tanni2022":
        source = frame.loc[frame.mua_method.eq("pooled_spike_density")].copy()
        names = {"mua_event_index": "event_index", "core_start_time_s": "start_s", "core_end_time_s": "end_s", "peak_time_s": "peak_s"}
    elif dataset == "pfeiffer_foster":
        source = frame.copy()
        if not source.event_definition.eq("immobile_run_high_mua").all():
            raise ValueError("unexpected PF event definition")
        names = {"event_start_s": "start_s", "event_end_s": "end_s", "event_peak_s": "peak_s"}
    else:
        raise ValueError("unknown dataset")
    source = source.rename(columns=names)
    result = source[["animal", "session", "event_index", "start_s", "end_s", "peak_s"]].copy()
    result.insert(0, "dataset", dataset)
    for name in ["event_index", "start_s", "end_s", "peak_s"]:
        result[name] = pd.to_numeric(result[name], errors="raise")
    if not np.isfinite(result[["event_index", "start_s", "end_s", "peak_s"]].to_numpy(float)).all():
        raise ValueError("nonfinite candidate identity or timing")
    if result[["animal", "session"]].isna().any().any():
        raise ValueError("missing candidate session identity")
    if (result.event_index != np.floor(result.event_index)).any() or (result.event_index < 0).any():
        raise ValueError("candidate event IDs must be nonnegative integers")
    result["event_index"] = result.event_index.astype(int)
    if (result.end_s <= result.start_s).any() or ((result.peak_s < result.start_s) | (result.peak_s > result.end_s)).any():
        raise ValueError("invalid candidate interval or peak")
    if result.duplicated(["dataset", "animal", "session", "event_index"]).any():
        raise ValueError("duplicate candidate keys")
    result["event_definition"] = "pooled_high_MUA_core_mean_return_boundaries"
    return result.sort_values(["animal", "session", "event_index"]).reset_index(drop=True)


def index_spike_times(spikes: np.ndarray, cell_ids: np.ndarray) -> dict[int, np.ndarray]:
    return {int(cell): np.sort(spikes[spikes[:, 1] == cell, 0]) for cell in cell_ids}


def count_candidate_bins(spikes, cell_ids, start_s, end_s, base_s):
    """Count half-open bins; preserve an explicitly labeled partial final bin."""
    if not np.isfinite([start_s, end_s, base_s]).all() or end_s <= start_s or base_s <= 0:
        raise ValueError("invalid event interval/bin size")
    ids = np.asarray(cell_ids, dtype=int)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("cell IDs must be unique")
    n_full = int(np.floor((end_s - start_s) / base_s + 1e-8))
    edges = start_s + np.arange(n_full + 1) * base_s
    if end_s - edges[-1] > 1e-9:
        edges = np.r_[edges, end_s]
    else:
        edges[-1] = end_s
    counts = np.zeros((len(edges) - 1, len(ids)), dtype=np.int32)
    indexed = spikes if isinstance(spikes, Mapping) else index_spike_times(spikes, ids)
    for column, cell in enumerate(ids):
        times = indexed[int(cell)]
        counts[:, column] = np.diff(np.searchsorted(times, edges, side="left"))
    return edges, counts


def fit_coverage_population(session: ReplaySession, config: CoverageInputConfig) -> tuple[dict, pd.DataFrame, dict]:
    """Fit all sorted units, retaining a separate pre-evidence unit-QC mask."""
    fit_config = EncodingConfig(
        bin_size_cm=config.bin_size_cm, smoothing_sigma_bins=config.smoothing_sigma_bins,
        min_speed_cm_s=config.min_speed_cm_s, min_occupancy_s=config.min_occupancy_s,
        rate_floor_hz=1e-4, arena_padding_cm=0.0, use_excitatory=False,
        exclude_ripple_intervals=False,
    )
    full = fit_place_field_encoding(session, fit_config)
    valid = full.occupancy_s >= config.min_occupancy_s
    if valid.sum() < 2:
        raise ValueError("insufficient occupied RUN bins")
    midpoint = float(np.mean([session.run_times[:, 0].min(), session.run_times[:, 1].max()]))
    first_intervals = [(a, min(b, midpoint)) for a, b in session.run_times if a < midpoint]
    second_intervals = [(max(a, midpoint), b) for a, b in session.run_times if b > midpoint]
    first = fit_place_field_encoding(replace(session, run_times=np.asarray(first_intervals).reshape(-1, 2)), fit_config)
    second = fit_place_field_encoding(replace(session, run_times=np.asarray(second_intervals).reshape(-1, 2)), fit_config)
    common = (first.occupancy_s >= config.min_occupancy_s) & (second.occupancy_s >= config.min_occupancy_s)
    t = session.position[:, 0]
    speed = _speed_cm_s(t, session.position[:, 1:3])
    moving = _times_in_intervals(t, session.run_times) & (speed >= config.min_speed_cm_s)
    running_duration = float(np.sum(_frame_durations(t)[moving]))
    spike_times, spike_ids = session.spikes.T
    running = _times_in_intervals(spike_times, session.run_times) & (np.interp(spike_times, t, speed) >= config.min_speed_cm_s)
    rows = []
    for index, cell in enumerate(full.cell_ids):
        n = int(np.count_nonzero(running & (spike_ids == cell)))
        mean_rate = n / running_duration if running_duration else np.nan
        peak_rate = float(full.rates_hz[index, valid].max())
        a, b = first.rates_hz[index, common], second.rates_hz[index, common]
        stability = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else np.nan
        untyped_sorted_source = session.metadata.get("source_dataset") in {"tanni2022", "blackstad_moser", "autopi_ca1"}
        source_allowed = untyped_sorted_source or bool(cell in session.excitatory_neurons)
        unit_pass = bool(source_allowed and n >= config.min_running_spikes and mean_rate <= config.max_running_rate_hz and peak_rate >= config.min_peak_rate_hz and np.isfinite(stability) and stability >= config.min_split_half_stability)
        rows.append({
            "cell_id": int(cell), "source_cell_type_allowed": source_allowed,
            "running_spikes": n, "mean_running_rate_hz": mean_rate,
            "peak_rate_hz": peak_rate, "split_half_stability": stability,
            "unit_qc_passed": unit_pass,
        })
    units = pd.DataFrame(rows)
    arrays = {
        "rates_hz": full.rates_hz, "rates_first_half_hz": first.rates_hz,
        "rates_second_half_hz": second.rates_hz,
        "occupancy_s": full.occupancy_s, "occupancy_first_half_s": first.occupancy_s,
        "occupancy_second_half_s": second.occupancy_s,
        "bin_centers_cm": full.bin_centers, "x_edges_cm": full.x_edges,
        "y_edges_cm": full.y_edges, "cell_ids": full.cell_ids,
        "valid_spatial_bins": valid, "unit_qc_mask": units.unit_qc_passed.to_numpy(bool),
        "position": session.position, "supported_run_intervals": session.run_times,
        "spikes": session.spikes,
    }
    metadata = {
        "settings": asdict(config), "encoding_config": asdict(fit_config),
        "encoding_cells_all_sorted": full.n_cells, "encoding_cells_qc": int(units.unit_qc_passed.sum()),
        "encoding_bins": full.n_bins, "encoding_bins_occupied": int(valid.sum()),
        "running_duration_s": running_duration, "rate_map_half_split_time_s": midpoint,
        "position_speed_estimator": "canonical_unsmoothed_centered_finite_difference",
        "running_speed_p50_cm_s": float(np.median(speed[moving])) if moving.any() else None,
        "running_speed_p95_cm_s": float(np.quantile(speed[moving], .95)) if moving.any() else None,
        "running_speed_p99_cm_s": float(np.quantile(speed[moving], .99)) if moving.any() else None,
        "running_speed_max_cm_s": float(np.max(speed[moving])) if moving.any() else None,
        "running_frames_over_200_cm_s_fraction": float(np.mean(speed[moving] > 200)) if moving.any() else None,
        "unit_selection_uses_replay_content": False,
    }
    return arrays, units, metadata
