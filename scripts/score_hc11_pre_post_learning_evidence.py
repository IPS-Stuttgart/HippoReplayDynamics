#!/usr/bin/env python3
"""Score matched hc-11 PRE/POST NREM ripple events for learning-dependent dynamics.

The existing hc-11 geometry pilot scores POST events only.  This campaign keeps
that scorer's topology-aware one-dimensional observation and transition models,
but matches PRE and POST ripple events on pre-evidence event-strength covariates
and evaluates the same windows with all, slow-firing, and fast-firing encoding
populations. The primary slow/fast split uses PRE-NREM firing rate to avoid
outcome leakage; an overall-session split reproduces the rate definition used
in the rate-stratified Grosmark and Buzsaki (2016) analysis.

This script performs original-order scoring only.  Map, time-order, and held-out
controls are deliberately separate so a technically successful PRE/POST run is
not silently promoted to a validated learning claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zlib

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
import score_hc11_webshare_native_ripple_evidence as hc11  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/mnt/lexar4tb/datasets/hc11_grosmark_buzsaki/webshare_processed")
DEFAULT_OUTPUT_DIR = Path("results/hc11-pre-post-learning-evidence")

SELECTION_OUTPUT = "hc11_pre_post_event_selection.csv"
EVENT_QC_OUTPUT = "hc11_pre_post_event_detection_qc.csv"
UNIT_OUTPUT = "hc11_pre_post_encoding_unit_qc.csv"
DECODER_OUTPUT = "hc11_pre_post_decoder_qc.csv"
EVIDENCE_OUTPUT = "hc11_pre_post_event_model_evidence.csv"
DECISION_OUTPUT = "hc11_pre_post_event_decisions.csv"
SESSION_OUTPUT = "hc11_pre_post_by_session.csv"
CONTRAST_OUTPUT = "hc11_pre_post_learning_contrasts.csv"
GATE_OUTPUT = "hc11_pre_post_gate_summary.csv"
MANIFEST_OUTPUT = "hc11_pre_post_manifest.json"
SUMMARY_OUTPUT = "hc11_pre_post_summary.md"

PHASES = ("PRE", "POST")
POPULATIONS = ("all", "slow_firing", "fast_firing")
ORDERED_MODELS = ("diffusion", "first_order_imm")
NONORDERED_MODELS = ("stationary", "fragmented")
MATCH_FEATURES = ("duration_ms", "n_spikes", "n_active_units")
EVENT_COLUMNS = (
    "phase",
    "event_id",
    "start_time_s",
    "end_time_s",
    "peak_time_s",
    "duration_ms",
    "peak_ripple_power_z",
    "lfp_ripple_count",
    "population_n_spikes",
    "population_n_active_units",
    "peak_mua_z",
    "n_spikes",
    "n_active_units",
    "event_definition",
)


@dataclass(frozen=True)
class RippleEventSession:
    animal: str
    session: str
    session_dir: Path
    ripple_event_path: Path
    event_source: str


def _manifest_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _manifest_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def resolve_ripple_event_sessions(
    dataset_root: Path,
    ripple_event_manifest: Path | None,
) -> list[RippleEventSession]:
    """Resolve native or explicitly validated generated ripple event tables."""

    root = Path(dataset_root).resolve()
    if ripple_event_manifest is None:
        return [
            RippleEventSession(
                animal=session_dir.parent.name,
                session=session_dir.name,
                session_dir=session_dir,
                ripple_event_path=session_dir / f"{session_dir.name}.ripplesNREM.event.mat",
                event_source="published_native",
            )
            for session_dir in hc11.discover_native_ripple_sessions(root)
        ]

    manifest_path = Path(ripple_event_manifest).resolve()
    table = pd.read_csv(manifest_path)
    required = {
        "animal",
        "session",
        "ripple_event_path",
        "event_source",
        "detector_qc_passed",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"ripple event manifest missing columns: {missing}")
    rows: list[RippleEventSession] = []
    seen: set[str] = set()
    for row in table.itertuples(index=False):
        animal = str(row.animal)
        session = str(row.session)
        if session in seen:
            raise ValueError(f"ripple event manifest contains duplicate session {session}")
        if not _manifest_bool(row.detector_qc_passed):
            raise ValueError(f"{session}: detector_qc_passed is false")
        session_dir = root / animal / session
        ripple_path = _manifest_path(row.ripple_event_path, manifest_path.parent)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"{session}: processed session directory missing: {session_dir}")
        if not ripple_path.is_file():
            raise FileNotFoundError(f"{session}: ripple event table missing: {ripple_path}")
        rows.append(
            RippleEventSession(
                animal=animal,
                session=session,
                session_dir=session_dir,
                ripple_event_path=ripple_path,
                event_source=str(row.event_source),
            )
        )
        seen.add(session)
    return sorted(rows, key=lambda row: (row.animal, row.session))


def _position_struct(session_dir: Path):
    base = session_dir.name
    return loadmat(
        session_dir / f"{base}.position.behavior.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["position"]


def _sleep_state_struct(session_dir: Path):
    base = session_dir.name
    return loadmat(
        session_dir / f"{base}.SleepState.states.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["SleepState"]


def phase_intervals(session_dir: Path) -> dict[str, np.ndarray]:
    position = _position_struct(session_dir)
    return {
        "PRE": hc11.as_intervals(position.Epochs.PREEpoch),
        "POST": hc11.as_intervals(position.Epochs.POSTEpoch),
    }


def nrem_intervals(session_dir: Path) -> np.ndarray:
    return hc11.as_intervals(_sleep_state_struct(session_dir).ints.NREMstate)


def interval_duration(intervals: np.ndarray) -> float:
    values = np.asarray(intervals, dtype=float).reshape(-1, 2)
    return float(np.sum(values[:, 1] - values[:, 0])) if values.size else 0.0


def count_spikes_in_window(values: np.ndarray, start_s: float, end_s: float) -> int:
    left = int(np.searchsorted(values, start_s, side="left"))
    right = int(np.searchsorted(values, end_s, side="right"))
    return right - left


def load_ripple_catalog(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        if "ripplesNREM" in handle:
            group = handle["ripplesNREM"]
        elif "ripples" in handle:
            group = handle["ripples"]
        else:
            raise ValueError(f"{path}: missing ripplesNREM or ripples group")
        times = np.asarray(group["times"], dtype=float)
        peaks = (
            np.asarray(group["peaks"], dtype=float).ravel()
            if "peaks" in group
            else np.array([], dtype=float)
        )
        powers = (
            np.asarray(group["peakNormedPower"], dtype=float).ravel()
            if "peakNormedPower" in group
            else np.array([], dtype=float)
        )
    if times.shape[0] == 2:
        times = times.T
    elif times.ndim != 2 or times.shape[1] != 2:
        raise ValueError(f"{path}: ripplesNREM/times must have two columns")
    if peaks.size == 0:
        peaks = np.mean(times, axis=1)
    if len(peaks) != len(times):
        raise ValueError(f"{path}: ripple peaks and times have different lengths")
    if len(powers) != len(times):
        powers = np.full(len(times), np.nan)
    return pd.DataFrame(
        {
            "lfp_ripple_id": np.arange(len(times), dtype=int),
            "start_time_s": times[:, 0],
            "end_time_s": times[:, 1],
            "peak_time_s": peaks,
            "peak_ripple_power_z": powers,
        }
    )


def pyramidal_unit_ids(session_dir: Path, spikes: hc11.SpikeData) -> tuple[int, ...]:
    base = session_dir.name
    cell_class = loadmat(
        session_dir / f"{base}.CellClass.cellinfo.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["CellClass"]
    unit_ids = np.asarray(cell_class.UID, dtype=int).ravel()
    pyramidal = np.asarray(cell_class.pE, dtype=bool).ravel()
    available = set(spikes.unit_ids)
    return tuple(
        int(unit_id)
        for unit_id, is_pyramidal in zip(unit_ids, pyramidal, strict=True)
        if bool(is_pyramidal) and int(unit_id) in available
    )


def _smoothed_population_rate(
    spike_times: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    bin_s: float,
    smoothing_s: float,
) -> np.ndarray:
    n_bins = max(int(np.ceil((float(end_s) - float(start_s)) / float(bin_s))), 1)
    edges = float(start_s) + np.arange(n_bins + 1, dtype=float) * float(bin_s)
    counts, _ = np.histogram(spike_times, bins=edges)
    return gaussian_filter1d(
        counts.astype(float),
        sigma=float(smoothing_s) / float(bin_s),
        mode="constant",
    )


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1),
            strict=True,
        )
    )


def detect_population_synchrony_events(
    session_dir: Path,
    spikes: hc11.SpikeData,
    selected_unit_ids: tuple[int, ...],
    ripple_event_path: Path,
    *,
    min_event_spikes: int,
    min_event_active_units: int,
    mua_bin_s: float,
    mua_smoothing_s: float,
    mua_threshold_sd: float,
    min_duration_s: float,
    max_duration_s: float,
) -> pd.DataFrame:
    """Reconstruct the paper's MUA-event-plus-LFP-ripple replay windows."""

    pyramidal_ids = pyramidal_unit_ids(session_dir, spikes)
    if len(pyramidal_ids) < 5:
        raise ValueError(f"{session_dir.name}: fewer than five CA1 pyramidal cells")
    all_pyramidal_spikes = np.sort(
        np.concatenate([spikes.times_by_unit[unit_id] for unit_id in pyramidal_ids])
    )
    nrem = nrem_intervals(session_dir)
    phases = phase_intervals(session_dir)
    phase_intervals_nrem: list[tuple[str, float, float]] = []
    for phase in PHASES:
        for start_s, end_s in _intersect_intervals(nrem, phases[phase]):
            phase_intervals_nrem.append((phase, float(start_s), float(end_s)))
    smoothed: list[tuple[str, float, np.ndarray]] = []
    total = 0
    value_sum = 0.0
    square_sum = 0.0
    for phase, start_s, end_s in phase_intervals_nrem:
        values = _smoothed_population_rate(
            all_pyramidal_spikes,
            start_s,
            end_s,
            bin_s=mua_bin_s,
            smoothing_s=mua_smoothing_s,
        )
        smoothed.append((phase, start_s, values))
        total += len(values)
        value_sum += float(values.sum())
        square_sum += float(np.dot(values, values))
    if total < 2:
        raise ValueError(f"{session_dir.name}: too little PRE/POST NREM for MUA detection")
    mean = value_sum / total
    variance = max((square_sum - total * mean * mean) / (total - 1), 0.0)
    sd = float(np.sqrt(variance))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"{session_dir.name}: invalid NREM MUA standard deviation")
    trigger = mean + float(mua_threshold_sd) * sd
    ripples = load_ripple_catalog(ripple_event_path)
    active_threshold = max(
        int(min_event_active_units),
        int(np.ceil(0.10 * len(selected_unit_ids))),
    )
    rows: list[dict[str, object]] = []
    for phase, interval_start_s, values in smoothed:
        for start_index, stop_index in _boolean_runs(values > mean):
            local = values[start_index:stop_index]
            if local.size == 0 or float(np.max(local)) < trigger:
                continue
            start_s = interval_start_s + start_index * float(mua_bin_s)
            end_s = interval_start_s + stop_index * float(mua_bin_s)
            duration_s = end_s - start_s
            if duration_s < float(min_duration_s) or duration_s > float(max_duration_s):
                continue
            contained = ripples[
                ripples["peak_time_s"].ge(start_s)
                & ripples["peak_time_s"].le(end_s)
            ]
            if contained.empty:
                continue
            pyramidal_counts = [
                count_spikes_in_window(spikes.times_by_unit[unit_id], start_s, end_s)
                for unit_id in pyramidal_ids
            ]
            encoding_counts = [
                count_spikes_in_window(spikes.times_by_unit[unit_id], start_s, end_s)
                for unit_id in selected_unit_ids
            ]
            n_spikes = int(np.sum(encoding_counts))
            n_active = int(np.sum(np.asarray(encoding_counts) > 0))
            if int(np.sum(np.asarray(pyramidal_counts) > 0)) < 5:
                continue
            if n_spikes < int(min_event_spikes) or n_active < active_threshold:
                continue
            peak_index = start_index + int(np.argmax(local))
            rows.append(
                {
                    "phase": phase,
                    "event_id": len(rows),
                    "start_time_s": start_s,
                    "end_time_s": end_s,
                    "peak_time_s": interval_start_s + peak_index * float(mua_bin_s),
                    "duration_ms": 1000.0 * duration_s,
                    "peak_ripple_power_z": float(
                        contained["peak_ripple_power_z"].max()
                    ),
                    "lfp_ripple_count": int(len(contained)),
                    "population_n_spikes": int(np.sum(pyramidal_counts)),
                    "population_n_active_units": int(
                        np.sum(np.asarray(pyramidal_counts) > 0)
                    ),
                    "peak_mua_z": float((np.max(local) - mean) / sd),
                    "n_spikes": n_spikes,
                    "n_active_units": n_active,
                    "event_definition": "paper_population_synchrony_with_lfp_ripple",
                }
            )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def load_phase_events(
    session_dir: Path,
    spikes: hc11.SpikeData,
    unit_ids: tuple[int, ...],
    *,
    ripple_event_path: Path | None = None,
    phase: str,
    min_event_spikes: int,
    min_event_active_units: int,
) -> pd.DataFrame:
    """Load native NREM ripples in one sleep phase without model-based selection."""

    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    base = session_dir.name
    event_path = (
        Path(ripple_event_path)
        if ripple_event_path is not None
        else session_dir / f"{base}.ripplesNREM.event.mat"
    )
    epochs = phase_intervals(session_dir)[phase]
    nrem = nrem_intervals(session_dir)
    catalog = load_ripple_catalog(event_path)

    rows: list[dict[str, object]] = []
    for event in catalog.itertuples(index=False):
        event_id = int(event.lfp_ripple_id)
        start_s = float(event.start_time_s)
        end_s = float(event.end_time_s)
        peak_s = float(event.peak_time_s)
        if not hc11.times_in_intervals(np.array([peak_s]), epochs)[0]:
            continue
        if not hc11.times_in_intervals(np.array([peak_s]), nrem)[0]:
            continue
        counts = [
            count_spikes_in_window(spikes.times_by_unit[int(unit_id)], float(start_s), float(end_s))
            for unit_id in unit_ids
        ]
        n_spikes = int(np.sum(counts))
        n_active = int(np.sum(np.asarray(counts) > 0))
        if n_spikes < int(min_event_spikes) or n_active < int(min_event_active_units):
            continue
        rows.append(
            {
                "phase": phase,
                "event_id": int(event_id),
                "start_time_s": float(start_s),
                "end_time_s": float(end_s),
                "peak_time_s": peak_s,
                "duration_ms": 1000.0 * float(end_s - start_s),
                "peak_ripple_power_z": (
                    float(event.peak_ripple_power_z)
                    if np.isfinite(event.peak_ripple_power_z)
                    else np.nan
                ),
                "n_spikes": n_spikes,
                "n_active_units": n_active,
                "event_definition": "lfp_ripple_envelope",
            }
        )
    return pd.DataFrame(rows)


def _candidate_pool(
    frame: pd.DataFrame,
    pool_size: int,
    rng: np.random.Generator,
    *,
    strategy: str,
) -> pd.DataFrame:
    if len(frame) <= pool_size:
        return frame.sort_values("event_id", kind="mergesort").copy()
    if strategy == "random":
        selected = np.sort(rng.choice(frame.index.to_numpy(), size=pool_size, replace=False))
        return frame.loc[selected].sort_values("event_id", kind="mergesort").copy()
    if strategy == "high_information":
        return (
            frame.sort_values(
                ["n_active_units", "n_spikes", "duration_ms", "event_id"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            .head(pool_size)
            .copy()
        )
    raise ValueError("strategy must be random or high_information")


def _standardized_match_features(pre: pd.DataFrame, post: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([pre[list(MATCH_FEATURES)], post[list(MATCH_FEATURES)]], ignore_index=True).astype(float)
    transformed = np.log1p(np.maximum(combined.to_numpy(), 0.0))
    median = np.nanmedian(transformed, axis=0)
    scale = np.nanpercentile(transformed, 75, axis=0) - np.nanpercentile(transformed, 25, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    standardized = (transformed - median) / scale
    return standardized[: len(pre)], standardized[len(pre) :]


def match_pre_post_events(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    *,
    max_pairs: int,
    seed: int,
    pool_multiplier: int = 10,
    pool_strategy: str = "random",
) -> pd.DataFrame:
    """Return deterministic, strength-matched PRE/POST events.

    Candidate pools are sampled without model evidence, then optimally matched
    on duration, spike count, and active-unit count.  Keeping selection separate
    from evidence prevents an apparent learning effect from being selected by the
    replay models it is later meant to test.
    """

    if pre.empty or post.empty or max_pairs <= 0:
        return pd.DataFrame()
    n_pairs = min(int(max_pairs), len(pre), len(post))
    pool_size = max(n_pairs, int(pool_multiplier) * n_pairs)
    rng = np.random.default_rng(int(seed))
    pre_pool = _candidate_pool(pre, min(pool_size, len(pre)), rng, strategy=pool_strategy)
    post_pool = _candidate_pool(post, min(pool_size, len(post)), rng, strategy=pool_strategy)
    pre_features, post_features = _standardized_match_features(pre_pool, post_pool)
    cost = np.sqrt(np.sum((pre_features[:, None, :] - post_features[None, :, :]) ** 2, axis=2))
    pre_rows, post_rows = linear_sum_assignment(cost)
    assignments = sorted(
        zip(pre_rows, post_rows, strict=True),
        key=lambda pair: (float(cost[pair]), int(pre_pool.iloc[pair[0]]["event_id"]), int(post_pool.iloc[pair[1]]["event_id"])),
    )[:n_pairs]
    rows: list[pd.Series] = []
    for pair_id, (pre_index, post_index) in enumerate(assignments, start=1):
        distance = float(cost[pre_index, post_index])
        for phase, source_index, source in (
            ("PRE", pre_index, pre_pool),
            ("POST", post_index, post_pool),
        ):
            row = source.iloc[source_index].copy()
            row["phase"] = phase
            row["match_pair_id"] = int(pair_id)
            row["match_distance"] = distance
            row["selection_rule"] = f"pre_evidence_{pool_strategy}_pool_strength_matched"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["match_pair_id", "phase"], kind="mergesort").reset_index(drop=True)


def _intersect_intervals(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for left_start, left_end in np.asarray(left, dtype=float).reshape(-1, 2):
        for right_start, right_end in np.asarray(right, dtype=float).reshape(-1, 2):
            start = max(float(left_start), float(right_start))
            end = min(float(left_end), float(right_end))
            if end > start:
                rows.append((start, end))
    return np.asarray(rows, dtype=float).reshape(-1, 2) if rows else np.empty((0, 2), dtype=float)


def offline_firing_rate_groups(
    session_dir: Path,
    spikes: hc11.SpikeData,
    unit_ids: tuple[int, ...],
    *,
    scope: str = "combined_offline_nrem",
) -> tuple[dict[str, tuple[int, ...]], pd.DataFrame]:
    """Split encoding cells at the within-session median offline NREM rate.

    ``overall_session`` follows the original rate-stratified analysis.
    ``pre_nrem`` freezes membership before novel experience and is the
    leakage-resistant learning primary. ``combined_offline_nrem`` is retained
    as a legacy offline-sleep sensitivity.
    """

    nrem = nrem_intervals(session_dir)
    epochs = phase_intervals(session_dir)
    if scope == "overall_session":
        phase_ends = [
            float(np.max(values[:, 1]))
            for values in epochs.values()
            if np.asarray(values).size
        ]
        offline = np.asarray([[0.0, max(phase_ends)]], dtype=float)
    elif scope == "combined_offline_nrem":
        offline = np.vstack(
            [
                _intersect_intervals(nrem, epochs["PRE"]),
                _intersect_intervals(nrem, epochs["POST"]),
            ]
        )
    elif scope == "pre_nrem":
        offline = _intersect_intervals(nrem, epochs["PRE"])
    else:
        raise ValueError(
            "scope must be overall_session, combined_offline_nrem, or pre_nrem"
        )
    duration_s = interval_duration(offline)
    rows: list[dict[str, object]] = []
    for unit_id in unit_ids:
        values = spikes.times_by_unit[int(unit_id)]
        count = int(np.sum(hc11.times_in_intervals(values, offline)))
        rows.append(
            {
                "unit_id": int(unit_id),
                "offline_nrem_spikes": count,
                "offline_nrem_duration_s": duration_s,
                "offline_nrem_rate_hz": count / max(duration_s, np.finfo(float).eps),
            }
        )
    table = pd.DataFrame(rows).sort_values(["offline_nrem_rate_hz", "unit_id"], kind="mergesort").reset_index(drop=True)
    split = (len(table) + 1) // 2
    slow = tuple(int(value) for value in table.iloc[:split]["unit_id"])
    fast = tuple(int(value) for value in table.iloc[split:]["unit_id"])
    group_by_unit = {unit_id: "slow_firing" for unit_id in slow}
    group_by_unit.update({unit_id: "fast_firing" for unit_id in fast})
    table["firing_rate_group"] = table["unit_id"].map(group_by_unit)
    table["firing_rate_group_scope"] = scope
    table["firing_rate_split_method"] = f"within_session_median_{scope}_rate"
    return {"all": unit_ids, "slow_firing": slow, "fast_firing": fast}, table


def subset_encoding_map(encoding: hc11.EncodingMap, unit_ids: tuple[int, ...]) -> hc11.EncodingMap:
    positions = {int(unit_id): index for index, unit_id in enumerate(encoding.unit_ids)}
    indices = np.asarray([positions[int(unit_id)] for unit_id in unit_ids], dtype=int)
    return replace(
        encoding,
        unit_ids=tuple(int(unit_id) for unit_id in unit_ids),
        rates_hz=np.asarray(encoding.rates_hz, dtype=float)[indices],
    )


def population_encodings(
    encodings: dict[str, list[hc11.EncodingMap]],
    groups: dict[str, tuple[int, ...]],
) -> dict[str, list[hc11.EncodingMap]]:
    primary = encodings[hc11.PRIMARY_ENCODING_VARIANT]
    return {
        population: [subset_encoding_map(encoding, unit_ids) for encoding in primary]
        for population, unit_ids in groups.items()
        if len(unit_ids) >= 2
    }


def event_decisions(evidence: pd.DataFrame, margin_threshold: float) -> pd.DataFrame:
    keys = ["animal", "session", "geometry", "phase", "match_pair_id", "event_id", "population"]
    rows: list[dict[str, object]] = []
    for key, group in evidence[evidence["status"].eq("success")].groupby(keys, sort=True):
        values = group.set_index("model")["log_evidence"]
        if not set(hc11.MODELS).issubset(values.index):
            continue
        best_model = str(values.idxmax())
        ordered_model = max(ORDERED_MODELS, key=lambda model: float(values[model]))
        nonordered_model = max(NONORDERED_MODELS, key=lambda model: float(values[model]))
        ordered_margin = float(values[ordered_model] - values[nonordered_model])
        imm_margin = float(values["first_order_imm"] - values["fragmented"])
        imm_row = group[group["model"].eq("first_order_imm")].iloc[0]
        content = bool(
            float(imm_row["mean_nonstationary_mode_probability"]) >= 0.5
            and float(imm_row["posterior_net_displacement_cm"]) >= 10.0
        )
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "best_model": best_model,
                "best_ordered_model": ordered_model,
                "best_nonordered_model": nonordered_model,
                "ordered_minus_nonordered": ordered_margin,
                "imm_minus_fragmented": imm_margin,
                "ordered_confident": ordered_margin >= float(margin_threshold),
                "nonordered_confident": ordered_margin <= -float(margin_threshold),
                "imm_confident_over_fragmented": imm_margin >= float(margin_threshold),
                "posterior_content_positive": content,
                "imm_trajectory_active_candidate": bool(
                    best_model == "first_order_imm"
                    and ordered_margin >= float(margin_threshold)
                    and imm_margin >= float(margin_threshold)
                    and content
                ),
                "mean_nonstationary_mode_probability": float(imm_row["mean_nonstationary_mode_probability"]),
                "fraction_time_map_nonstationary": float(imm_row["fraction_time_map_nonstationary"]),
                "posterior_expected_path_length_cm": float(imm_row["posterior_expected_path_length_cm"]),
                "posterior_net_displacement_cm": float(imm_row["posterior_net_displacement_cm"]),
                "n_spikes": int(imm_row["n_spikes"]),
                "n_active_units": int(imm_row["n_active_units"]),
                "n_encoding_units": int(imm_row["n_encoding_units"]),
                "duration_ms": float(imm_row["duration_ms"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_session(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["animal", "session", "geometry", "population", "phase"]
    for key, group in decisions.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "events": int(len(group)),
                "ordered_confident_count": int(group["ordered_confident"].sum()),
                "ordered_confident_fraction": float(group["ordered_confident"].mean()),
                "imm_trajectory_active_candidate_count": int(group["imm_trajectory_active_candidate"].sum()),
                "imm_trajectory_active_candidate_fraction": float(group["imm_trajectory_active_candidate"].mean()),
                "median_ordered_minus_nonordered": float(group["ordered_minus_nonordered"].median()),
                "median_imm_minus_fragmented": float(group["imm_minus_fragmented"].median()),
                "median_nonstationary_mode_probability": float(group["mean_nonstationary_mode_probability"].median()),
                "median_net_displacement_cm": float(group["posterior_net_displacement_cm"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def learning_contrasts(decisions: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "ordered_minus_nonordered",
        "imm_minus_fragmented",
        "mean_nonstationary_mode_probability",
        "posterior_net_displacement_cm",
        "ordered_confident",
        "imm_trajectory_active_candidate",
    )
    index = ["animal", "session", "geometry", "population", "match_pair_id"]
    wide = decisions.pivot_table(index=index, columns="phase", values=list(metrics), aggfunc="first")
    rows: list[dict[str, object]] = []
    for key, values in wide.iterrows():
        if not all((metric, phase) in wide.columns for metric in metrics for phase in PHASES):
            continue
        row = dict(zip(index, key, strict=True))
        for metric in metrics:
            row[f"pre_{metric}"] = float(values[(metric, "PRE")])
            row[f"post_{metric}"] = float(values[(metric, "POST")])
            row[f"post_minus_pre_{metric}"] = float(values[(metric, "POST")] - values[(metric, "PRE")])
        rows.append(row)
    return pd.DataFrame(rows)


def gate_summary(
    sessions: list[Path],
    decoder: pd.DataFrame,
    selection: pd.DataFrame,
    evidence: pd.DataFrame,
    decisions: pd.DataFrame,
    contrasts: pd.DataFrame,
    max_pairs_per_session: int,
) -> pd.DataFrame:
    selected_sessions = int(selection["session"].nunique()) if not selection.empty else 0
    selected_animals = int(selection["animal"].nunique()) if not selection.empty else 0
    phase_counts = selection.groupby(["session", "phase"]).size() if not selection.empty else pd.Series(dtype=int)
    expected_models = len(hc11.MODELS)
    complete = (
        evidence[evidence["status"].eq("success")]
        .groupby(["session", "phase", "event_id", "population"])["model"]
        .nunique()
        if not evidence.empty
        else pd.Series(dtype=int)
    )
    checks = [
        ("ripple_event_sessions_present", len(sessions) > 0, f"sessions={len(sessions)}"),
        ("pre_and_post_events_present", bool(not phase_counts.empty and set(phase_counts.index.get_level_values("phase")) == set(PHASES)), f"phase_groups={len(phase_counts)}"),
        ("matched_pairs_present", bool(not selection.empty and selection["match_pair_id"].notna().all()), f"selected_rows={len(selection)}"),
        ("multiple_animals_present", selected_animals >= 2, f"animals={selected_animals}"),
        ("all_discovered_sessions_represented", selected_sessions == len(sessions) and len(sessions) > 0, f"sessions={selected_sessions}/{len(sessions)}"),
        ("decoder_outputs_complete", len(decoder) == len(sessions) and len(sessions) > 0, f"rows={len(decoder)}/{len(sessions)}"),
        ("required_models_complete", bool(len(complete) > 0 and (complete == expected_models).all()), f"complete={int((complete == expected_models).sum())}/{len(complete)}"),
        ("no_scoring_failures", bool(not evidence.empty and evidence["status"].eq("success").all()), f"failures={int((~evidence['status'].eq('success')).sum()) if not evidence.empty else 0}"),
        ("all_three_populations_scored", set(decisions.get("population", [])) == set(POPULATIONS), f"populations={sorted(set(decisions.get('population', [])))}"),
        ("paired_learning_contrasts_present", not contrasts.empty, f"contrast_rows={len(contrasts)}"),
        ("target_pair_cap_recorded", max_pairs_per_session > 0, f"max_pairs_per_session={max_pairs_per_session}"),
    ]
    technical = all(passed for _, passed, _ in checks)
    checks.extend(
        [
            ("overall_technical", technical, "original-order matched PRE/POST scoring only"),
            ("map_specificity_control_present", False, "required before biological learning claim"),
            ("time_order_control_present", False, "required before biological learning claim"),
            ("heldout_prediction_present", False, "required before biological learning claim"),
            ("learning_dependent_trajectory_dynamics_supported", False, "not evaluated by original-order scorer alone"),
        ]
    )
    return pd.DataFrame([{"gate": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks])


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.ripple_event_manifest).resolve() if args.ripple_event_manifest else None
    event_sessions = resolve_ripple_event_sessions(dataset_root, manifest_path)
    sessions = [item.session_dir for item in event_sessions]

    selection_frames: list[pd.DataFrame] = []
    event_qc_rows: list[dict[str, object]] = []
    unit_frames: list[pd.DataFrame] = []
    decoder_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []

    for event_session in event_sessions:
        session_dir = event_session.session_dir
        animal = event_session.animal
        session = event_session.session
        track = hc11.load_track_samples(session_dir)
        spikes = hc11.load_spikes(session_dir)
        encodings, unit_qc = hc11.build_session_encodings(
            track,
            spikes,
            position_bin_size_cm=args.position_bin_size_cm,
            min_run_speed_cm_s=args.min_run_speed_cm_s,
            min_run_spikes=args.min_run_spikes,
            min_spatial_information=args.min_spatial_information,
            min_peak_rate_hz=args.min_peak_rate_hz,
            min_encoding_units=args.min_encoding_units,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
        )
        selected_units = encodings["pooled"][0].unit_ids
        groups, rate_groups = offline_firing_rate_groups(
            session_dir,
            spikes,
            selected_units,
            scope=args.rate_group_scope,
        )
        group_maps = population_encodings(encodings, groups)
        unit_qc = unit_qc.merge(rate_groups, on="unit_id", how="left")
        unit_qc.insert(0, "session", session)
        unit_qc.insert(0, "animal", animal)
        unit_frames.append(unit_qc)

        decoder_metrics = hc11.decode_crossvalidated(
            track,
            spikes,
            selected_units,
            position_bin_size_cm=args.position_bin_size_cm,
            min_run_speed_cm_s=args.min_run_speed_cm_s,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            n_folds=args.decoder_folds,
            decode_window_s=args.decoder_window_s,
            max_decode_bins=args.decoder_max_bins,
        )
        decoder_rows.append(
            {
                "animal": animal,
                "session": session,
                "geometry": track.topology,
                "maze_type": track.maze_type,
                "track_length_cm": track.track_length_cm,
                "total_ca1_units": len(spikes.unit_ids),
                "encoding_units": len(selected_units),
                "slow_firing_units": len(groups["slow_firing"]),
                "fast_firing_units": len(groups["fast_firing"]),
                **decoder_metrics,
            }
        )

        if args.event_definition == "paper_population_synchrony":
            detected_events = detect_population_synchrony_events(
                session_dir,
                spikes,
                selected_units,
                event_session.ripple_event_path,
                min_event_spikes=args.min_event_spikes,
                min_event_active_units=args.min_event_active_units,
                mua_bin_s=args.mua_bin_s,
                mua_smoothing_s=args.mua_smoothing_s,
                mua_threshold_sd=args.mua_threshold_sd,
                min_duration_s=args.min_synchrony_duration_s,
                max_duration_s=args.max_synchrony_duration_s,
            )
            phase_events = {
                phase: detected_events[detected_events["phase"].eq(phase)].copy()
                for phase in PHASES
            }
        else:
            phase_events = {
                phase: load_phase_events(
                    session_dir,
                    spikes,
                    selected_units,
                    ripple_event_path=event_session.ripple_event_path,
                    phase=phase,
                    min_event_spikes=args.min_event_spikes,
                    min_event_active_units=args.min_event_active_units,
                )
                for phase in PHASES
            }
        stable_seed = int(args.selection_seed) + int(zlib.crc32(session.encode("utf-8")))
        selected = match_pre_post_events(
            phase_events["PRE"],
            phase_events["POST"],
            max_pairs=args.max_pairs_per_session,
            seed=stable_seed,
            pool_multiplier=args.match_pool_multiplier,
            pool_strategy=args.selection_strategy,
        )
        phase_nrem = {
            phase: interval_duration(
                _intersect_intervals(
                    nrem_intervals(session_dir),
                    phase_intervals(session_dir)[phase],
                )
            )
            for phase in PHASES
        }
        expected_definition = (
            "paper_population_synchrony_with_lfp_ripple"
            if args.event_definition == "paper_population_synchrony"
            else "lfp_ripple_envelope"
        )
        for phase in PHASES:
            candidates = phase_events[phase]
            chosen = (
                selected[selected["phase"].eq(phase)]
                if not selected.empty
                else selected
            )
            event_qc_rows.append(
                {
                    "animal": animal,
                    "session": session,
                    "geometry": track.topology,
                    "event_source": event_session.event_source,
                    "event_definition": (
                        str(candidates["event_definition"].iloc[0])
                        if not candidates.empty
                        else expected_definition
                    ),
                    "phase": phase,
                    "nrem_duration_s": phase_nrem[phase],
                    "eligible_events": len(candidates),
                    "eligible_events_per_nrem_hour": (
                        3600.0 * len(candidates) / phase_nrem[phase]
                        if phase_nrem[phase] > 0.0
                        else np.nan
                    ),
                    "selected_events": len(chosen),
                    "median_duration_ms": (
                        float(candidates["duration_ms"].median())
                        if not candidates.empty
                        else np.nan
                    ),
                    "median_n_spikes": (
                        float(candidates["n_spikes"].median())
                        if not candidates.empty
                        else np.nan
                    ),
                    "median_n_active_units": (
                        float(candidates["n_active_units"].median())
                        if not candidates.empty
                        else np.nan
                    ),
                }
            )
        if selected.empty:
            continue
        selected.insert(0, "geometry", track.topology)
        selected.insert(0, "maze_type", track.maze_type)
        selected.insert(0, "scoring_event_padding_s", float(args.event_padding_s))
        selected.insert(0, "scoring_time_bin_s", float(args.time_bin_s))
        selected.insert(0, "ripple_event_path", str(event_session.ripple_event_path))
        selected.insert(0, "event_source", event_session.event_source)
        selected.insert(0, "session", session)
        selected.insert(0, "animal", animal)
        selection_frames.append(selected)

        for event in selected.itertuples(index=False):
            score_start_s = max(0.0, float(event.start_time_s) - float(args.event_padding_s))
            score_end_s = float(event.end_time_s) + float(args.event_padding_s)
            edges = hc11.event_bin_edges(score_start_s, score_end_s, args.time_bin_s)
            for population, unit_ids in groups.items():
                if population not in group_maps:
                    continue
                counts = hc11.spike_count_matrix(spikes, unit_ids, edges)
                started = time.perf_counter()
                try:
                    scores = hc11.score_encoding_variant(
                        counts,
                        edges,
                        group_maps[population],
                        topology=track.topology,
                        track_length_cm=track.track_length_cm,
                        diffusion_sigma_cm_sqrt_s=args.diffusion_sigma_cm_sqrt_s,
                        stationary_sigma_cm=args.stationary_sigma_cm,
                        max_step_sigma=args.max_step_sigma,
                        imm_mode_stickiness=args.imm_mode_stickiness,
                    )
                    runtime = time.perf_counter() - started
                    for model, score in scores.items():
                        diagnostics = {
                            "mean_stationary_mode_probability": np.nan,
                            "mean_nonstationary_mode_probability": np.nan,
                            "fraction_time_map_nonstationary": np.nan,
                            "posterior_expected_path_length_cm": np.nan,
                            "posterior_net_displacement_cm": np.nan,
                            "posterior_path_speed_cm_s": np.nan,
                        }
                        if model == "first_order_imm":
                            diagnostics = hc11.imm_content_diagnostics(
                                np.asarray(score["posterior"]),
                                np.asarray(score["mode_posterior"]),
                                group_maps[population][0].bin_centers_cm,
                                track.topology,
                                track.track_length_cm,
                                score_end_s - score_start_s,
                            )
                        evidence_rows.append(
                            {
                                "animal": animal,
                                "session": session,
                                "geometry": track.topology,
                                "event_source": event_session.event_source,
                                "event_definition": str(event.event_definition),
                                "phase": str(event.phase),
                                "match_pair_id": int(event.match_pair_id),
                                "event_id": int(event.event_id),
                                "population": population,
                                "model": model,
                                "model_family": "ordered" if model in ORDERED_MODELS else "nonordered",
                                "log_evidence": float(score["log_evidence"]),
                                "status": "success",
                                "failure_reason": "",
                                "runtime_s": runtime / len(hc11.MODELS),
                                "duration_ms": 1000.0 * (score_end_s - score_start_s),
                                "n_time_bins": len(edges) - 1,
                                "n_spikes": int(counts.sum()),
                                "n_active_units": int(np.sum(counts.sum(axis=0) > 0)),
                                "n_encoding_units": len(unit_ids),
                                "match_distance": float(event.match_distance),
                                **diagnostics,
                            }
                        )
                except Exception as exc:
                    runtime = time.perf_counter() - started
                    for model in hc11.MODELS:
                        evidence_rows.append(
                            {
                                "animal": animal,
                                "session": session,
                                "geometry": track.topology,
                                "event_source": event_session.event_source,
                                "event_definition": str(event.event_definition),
                                "phase": str(event.phase),
                                "match_pair_id": int(event.match_pair_id),
                                "event_id": int(event.event_id),
                                "population": population,
                                "model": model,
                                "model_family": "ordered" if model in ORDERED_MODELS else "nonordered",
                                "log_evidence": np.nan,
                                "status": "failure",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "runtime_s": runtime / len(hc11.MODELS),
                                "duration_ms": 1000.0 * (score_end_s - score_start_s),
                                "n_time_bins": len(edges) - 1,
                                "n_spikes": int(counts.sum()),
                                "n_active_units": int(np.sum(counts.sum(axis=0) > 0)),
                                "n_encoding_units": len(unit_ids),
                                "match_distance": float(event.match_distance),
                            }
                        )

    selection = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    event_qc = pd.DataFrame(event_qc_rows)
    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    decoder = pd.DataFrame(decoder_rows)
    evidence = pd.DataFrame(evidence_rows)
    decisions = event_decisions(evidence, args.margin_threshold) if not evidence.empty else pd.DataFrame()
    by_session = summarize_by_session(decisions) if not decisions.empty else pd.DataFrame()
    contrasts = learning_contrasts(decisions) if not decisions.empty else pd.DataFrame()
    gates = gate_summary(sessions, decoder, selection, evidence, decisions, contrasts, args.max_pairs_per_session)

    outputs = {
        SELECTION_OUTPUT: selection,
        EVENT_QC_OUTPUT: event_qc,
        UNIT_OUTPUT: units,
        DECODER_OUTPUT: decoder,
        EVIDENCE_OUTPUT: evidence,
        DECISION_OUTPUT: decisions,
        SESSION_OUTPUT: by_session,
        CONTRAST_OUTPUT: contrasts,
        GATE_OUTPUT: gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "hc-11_Grosmark_Buzsaki_Webshare",
        "analysis": "matched_PRE_POST_NREM_ripple_learning_evidence",
        "event_definition": (
            "paper_population_synchrony_with_lfp_ripple"
            if args.event_definition == "paper_population_synchrony"
            else "lfp_ripple_envelope"
        ),
        "models": list(hc11.MODELS),
        "ordered_models": list(ORDERED_MODELS),
        "nonordered_models": list(NONORDERED_MODELS),
        "populations": list(POPULATIONS),
        "claim_boundary": "original-order matched PRE/POST evidence; map, time-order, and held-out gates still required",
        "sessions": [
            {
                "animal": item.animal,
                "session": item.session,
                "ripple_event_path": str(item.ripple_event_path),
                "event_source": item.event_source,
            }
            for item in event_sessions
        ],
        "parameters": {key: value for key, value in vars(args).items() if key not in {"dataset_root", "output_dir"}},
        **build_script_provenance(
            input_paths={
                "dataset_root": dataset_root,
                "ripple_event_manifest": manifest_path,
            }
        ),
    }
    (output_dir / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / SUMMARY_OUTPUT).write_text(build_summary(by_session, contrasts, gates), encoding="utf-8")
    return {
        "selection": selection,
        "event_qc": event_qc,
        "units": units,
        "decoder": decoder,
        "evidence": evidence,
        "decisions": decisions,
        "by_session": by_session,
        "contrasts": contrasts,
        "gates": gates,
    }


def build_summary(by_session: pd.DataFrame, contrasts: pd.DataFrame, gates: pd.DataFrame) -> str:
    technical = bool(gates.loc[gates["gate"].eq("overall_technical"), "passed"].iloc[0]) if not gates.empty else False
    lines = [
        "# hc-11 matched PRE/POST learning evidence",
        "",
        f"Original-order technical status: **{'pass' if technical else 'fail'}**.",
        "",
        "This run matches PRE and POST validated NREM ripple events without using replay-model evidence, then scores all, slow-firing, and fast-firing encoding populations.",
        "Fragmented is treated as nonordered spatial reactivation, not as virtual movement.",
        "",
        "## Session summaries",
        "",
        "```text",
        by_session.to_string(index=False) if not by_session.empty else "No session summaries.",
        "```",
        "",
        "## Matched-event contrasts",
        "",
        "```text",
        contrasts.groupby("population").median(numeric_only=True).to_string() if not contrasts.empty else "No matched contrasts.",
        "```",
        "",
        "## Claim boundary",
        "",
        "These original-order contrasts are descriptive. A learning-dependent trajectory claim requires map-specificity, whole-bin time-order, and held-out-cell predictive gates on the same frozen events.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--ripple-event-manifest",
        help=(
            "Optional CSV mapping animal/session to validated native or generated ripple-event "
            "tables. Generated tables are never installed into the dataset implicitly."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-pairs-per-session", type=int, default=20)
    parser.add_argument("--match-pool-multiplier", type=int, default=10)
    parser.add_argument(
        "--selection-strategy",
        choices=("random", "high_information"),
        default="random",
        help="Choose candidate pools using only pre-evidence event covariates before PRE/POST matching.",
    )
    parser.add_argument(
        "--rate-group-scope",
        choices=("overall_session", "combined_offline_nrem", "pre_nrem"),
        default="pre_nrem",
        help="Use leakage-resistant PRE-NREM groups by default; overall_session matches the paper.",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--event-definition",
        choices=("paper_population_synchrony", "lfp_ripple_envelope"),
        default="paper_population_synchrony",
    )
    parser.add_argument("--mua-bin-s", type=float, default=0.001)
    parser.add_argument("--mua-smoothing-s", type=float, default=0.015)
    parser.add_argument("--mua-threshold-sd", type=float, default=3.0)
    parser.add_argument("--min-synchrony-duration-s", type=float, default=0.100)
    parser.add_argument("--max-synchrony-duration-s", type=float, default=0.500)
    parser.add_argument(
        "--time-bin-s",
        type=float,
        default=0.020,
        help="Replay decoder bin width; 20 ms matches Grosmark and Buzsaki (2016).",
    )
    parser.add_argument("--event-padding-s", type=float, default=0.0)
    parser.add_argument("--position-bin-size-cm", type=float, default=4.0)
    parser.add_argument("--min-run-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--min-run-spikes", type=int, default=20)
    parser.add_argument("--min-spatial-information", type=float, default=0.1)
    parser.add_argument("--min-peak-rate-hz", type=float, default=1.0)
    parser.add_argument("--min-encoding-units", type=int, default=5)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.5)
    parser.add_argument("--min-event-spikes", type=int, default=8)
    parser.add_argument("--min-event-active-units", type=int, default=5)
    parser.add_argument("--decoder-folds", type=int, default=5)
    parser.add_argument("--decoder-window-s", type=float, default=0.25)
    parser.add_argument("--decoder-max-bins", type=int, default=1000)
    parser.add_argument("--diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--max-step-sigma", type=float, default=4.0)
    parser.add_argument("--imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    print(
        f"Scored {len(outputs['decisions'])} phase-event-population decisions; "
        f"wrote outputs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
