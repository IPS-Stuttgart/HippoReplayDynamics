#!/usr/bin/env python3
"""Aggregate KD non-momentum scores and sharded momentum-grid scores."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_kd_model_evidence import _add_evidence_columns, _family, _write
from hipporeplayimm.kd_reference import (
    best_grid_params,
    empirical_grid_prior,
    marginalize_grid_log_evidence,
    random_effects_model_probabilities,
)


def _np_scalar(value, *, key: str, path: Path) -> object:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Momentum shard {key} must be scalar metadata in {path}: got shape {array.shape}")
    return array.item()


def _integer_metadata(value, *, key: str, path: Path, min_value: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"Momentum shard {key} must be one-dimensional: {path}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"Momentum shard {key} must contain integer values, not booleans: {path}")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"Momentum shard {key} must contain numeric integer values: {path}")
    intp_info = np.iinfo(np.dtype(np.intp))
    if np.issubdtype(raw.dtype, np.integer):
        out_of_range = any(int(item) < intp_info.min or int(item) > intp_info.max for item in raw.ravel())
        if out_of_range:
            raise ValueError(f"Momentum shard {key} must fit into NumPy integer range: {path}")
        values = raw.astype(np.intp, copy=True)
    else:
        numeric = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"Momentum shard {key} must contain finite integer values: {path}")
        if not np.all((numeric >= intp_info.min) & (numeric <= intp_info.max)):
            raise ValueError(f"Momentum shard {key} must fit into NumPy integer range: {path}")
        values = numeric.astype(np.intp, copy=True)
    if np.any(values < int(min_value)):
        qualifier = "nonnegative" if int(min_value) == 0 else "positive"
        raise ValueError(f"Momentum shard {key} must contain {qualifier} integer values: {path}")
    return values


def _load_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as shard:
        event_ids = _integer_metadata(shard["event_ids"], key="event_ids", path=path, min_value=0)
        n_time = _integer_metadata(shard["n_time"], key="n_time", path=path, min_value=1)
        n_spikes = _integer_metadata(shard["n_spikes"], key="n_spikes", path=path, min_value=0)
        if n_time.shape != event_ids.shape:
            raise ValueError(f"Momentum shard n_time must match event_ids shape in {path}: {n_time.shape} vs {event_ids.shape}")
        if n_spikes.shape != event_ids.shape:
            raise ValueError(f"Momentum shard n_spikes must match event_ids shape in {path}: {n_spikes.shape} vs {event_ids.shape}")
        return {
            "path": path,
            "session": str(_np_scalar(shard["session"], key="session", path=path)),
            "event_ids": event_ids,
            "n_time": n_time,
            "n_spikes": n_spikes,
            "sd_meters": np.array(shard["sd_meters"], dtype=float, copy=True),
            "decay": np.array(shard["decay"], dtype=float, copy=True),
            "sd_indices": np.array(shard["sd_indices"], copy=True),
            "decay_indices": np.array(shard["decay_indices"], copy=True),
            "values": np.array(shard["values"], dtype=float, copy=True),
            "runtime_s": float(_np_scalar(shard["runtime_s"], key="runtime_s", path=path)),
            "kd_grid_preset": str(_np_scalar(shard["kd_grid_preset"], key="kd_grid_preset", path=path)),
            "kd_time_bin_ms": float(_np_scalar(shard["kd_time_bin_ms"], key="kd_time_bin_ms", path=path)),
            "kd_bin_size_cm": float(_np_scalar(shard["kd_bin_size_cm"], key="kd_bin_size_cm", path=path)),
            "kd_n_bins": int(_np_scalar(shard["kd_n_bins"], key="kd_n_bins", path=path)),
            "kd_n_jobs": int(_np_scalar(shard["kd_n_jobs"], key="kd_n_jobs", path=path)),
            "kd_event_chunk_size": int(_np_scalar(shard["kd_event_chunk_size"], key="kd_event_chunk_size", path=path)),
            "kd_spike_rate_scale": float(_np_scalar(shard["kd_spike_rate_scale"], key="kd_spike_rate_scale", path=path))
            if "kd_spike_rate_scale" in shard.files
            else 1.0,
        }


def _check_same(reference: dict[str, object], shard: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        if shard[key] != reference[key]:
            raise ValueError(f"Momentum shard metadata differs for {key}: {shard['path']}")


def _coerce_grid_index_array(shard: dict[str, object], key: str) -> np.ndarray:
    raw = np.asarray(shard[key])
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"Momentum shard {key} must contain integer grid indices, not booleans: {shard['path']}")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"Momentum shard {key} must contain numeric integer grid indices: {shard['path']}")
    intp_info = np.iinfo(np.dtype(np.intp))
    if np.issubdtype(raw.dtype, np.integer):
        out_of_intp_range = any(
            int(value) < intp_info.min or int(value) > intp_info.max
            for value in raw.ravel()
        )
        if out_of_intp_range:
            raise ValueError(f"Momentum shard {key} must fit into NumPy integer index range: {shard['path']}")
        return raw.astype(np.intp, copy=False)
    values = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"Momentum shard {key} must contain finite integer grid indices: {shard['path']}")
    if not np.all((values >= intp_info.min) & (values <= intp_info.max)):
        raise ValueError(f"Momentum shard {key} must fit into NumPy integer index range: {shard['path']}")
    return values.astype(np.intp, copy=False)


def _validate_grid_indices(
    shard: dict[str, object],
    *,
    sd_count: int,
    decay_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sd_indices = _coerce_grid_index_array(shard, "sd_indices")
    decay_indices = _coerce_grid_index_array(shard, "decay_indices")
    if sd_indices.ndim != 1 or decay_indices.ndim != 1:
        raise ValueError(f"Momentum shard grid indices must be one-dimensional: {shard['path']}")
    if sd_indices.shape != decay_indices.shape:
        raise ValueError(
            f"Momentum shard sd_indices and decay_indices length mismatch in {shard['path']}: "
            f"{sd_indices.shape} vs {decay_indices.shape}"
        )
    bad_sd = (sd_indices < 0) | (sd_indices >= int(sd_count))
    if np.any(bad_sd):
        bad_values = sd_indices[bad_sd]
        raise ValueError(
            f"Momentum shard sd_indices out of range [0, {int(sd_count)}) in {shard['path']}: "
            f"{bad_values[:10].tolist()}"
        )
    bad_decay = (decay_indices < 0) | (decay_indices >= int(decay_count))
    if np.any(bad_decay):
        bad_values = decay_indices[bad_decay]
        raise ValueError(
            f"Momentum shard decay_indices out of range [0, {int(decay_count)}) in {shard['path']}: "
            f"{bad_values[:10].tolist()}"
        )
    return sd_indices, decay_indices


def _load_momentum_grid(shard_paths: list[Path]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if not shard_paths:
        raise FileNotFoundError("No momentum shard files matched.")

    shards = [_load_npz(path) for path in shard_paths]
    first = shards[0]
    sd_meters = first["sd_meters"]
    decay = first["decay"]
    event_ids = np.asarray(sorted({int(event_id) for shard in shards for event_id in shard["event_ids"]}), dtype=int)
    if event_ids.size == 0:
        raise ValueError("Momentum shard files did not contain any events.")
    event_row = {int(event_id): row_index for row_index, event_id in enumerate(event_ids)}
    grid = np.full((event_ids.shape[0], sd_meters.shape[0], decay.shape[0]), np.nan, dtype=float)
    n_time = np.full(event_ids.shape[0], -1, dtype=int)
    n_spikes = np.full(event_ids.shape[0], -1, dtype=int)
    runtime_s_by_event = np.zeros(event_ids.shape[0], dtype=float)
    metadata: dict[str, object] = {
        "session": first["session"],
        "event_ids": event_ids,
        "kd_grid_preset": first["kd_grid_preset"],
        "kd_time_bin_ms": first["kd_time_bin_ms"],
        "kd_bin_size_cm": first["kd_bin_size_cm"],
        "kd_n_bins": first["kd_n_bins"],
        "kd_n_jobs": first["kd_n_jobs"],
        "kd_event_chunk_size": first["kd_event_chunk_size"],
        "kd_spike_rate_scale": first["kd_spike_rate_scale"],
    }
    for shard in shards:
        _check_same(
            first,
            shard,
            (
                "session",
                "kd_grid_preset",
                "kd_time_bin_ms",
                "kd_bin_size_cm",
                "kd_n_bins",
                "kd_n_jobs",
                "kd_event_chunk_size",
                "kd_spike_rate_scale",
            ),
        )
        if not np.array_equal(shard["sd_meters"], sd_meters) or not np.array_equal(shard["decay"], decay):
            raise ValueError(f"Momentum grid differs in {shard['path']}")
        sd_indices, decay_indices = _validate_grid_indices(
            shard,
            sd_count=sd_meters.shape[0],
            decay_count=decay.shape[0],
        )
        values = shard["values"]
        if values.shape != (shard["event_ids"].shape[0], sd_indices.shape[0]):
            raise ValueError(f"Unexpected momentum values shape in {shard['path']}: {values.shape}")
        per_event_runtime = float(shard["runtime_s"]) / max(shard["event_ids"].shape[0], 1)
        for local_row, event_id in enumerate(shard["event_ids"]):
            row = event_row[int(event_id)]
            if n_time[row] == -1:
                n_time[row] = int(shard["n_time"][local_row])
                n_spikes[row] = int(shard["n_spikes"][local_row])
            elif n_time[row] != int(shard["n_time"][local_row]) or n_spikes[row] != int(shard["n_spikes"][local_row]):
                raise ValueError(f"Event metadata differs for event {int(event_id)} in {shard['path']}")
            runtime_s_by_event[row] = max(runtime_s_by_event[row], per_event_runtime)
            for column, (sd_index, decay_index) in enumerate(zip(sd_indices, decay_indices, strict=True)):
                target = grid[row, int(sd_index), int(decay_index)]
                value = float(values[local_row, column])
                if np.isfinite(target) and not np.isclose(target, value):
                    raise ValueError(f"Conflicting momentum score for event {int(event_id)}, grid ({sd_index}, {decay_index})")
                grid[row, int(sd_index), int(decay_index)] = value
    missing = np.argwhere(~np.isfinite(grid))
    if missing.size:
        raise ValueError(f"Momentum shards did not cover {missing.shape[0]} event/grid entries.")
    metadata["n_time"] = n_time
    metadata["n_spikes"] = n_spikes
    metadata["runtime_s_by_event"] = runtime_s_by_event
    return grid, {"sd_meters": sd_meters, "decay": decay}, metadata


def _momentum_rows(log_evidence: np.ndarray, metadata: dict[str, object]) -> list[dict[str, object]]:
    event_ids = np.asarray(metadata["event_ids"], dtype=int)
    n_time = np.asarray(metadata["n_time"], dtype=int)
    n_spikes = np.asarray(metadata["n_spikes"], dtype=int)
    runtime_s_by_event = np.asarray(metadata["runtime_s_by_event"], dtype=float)
    rows = []
    for row_index, event_id in enumerate(event_ids):
        rows.append(
            {
                "status": "success",
                "session": metadata["session"],
                "event_index": int(event_id),
                "model": "momentum",
                "model_family": _family("momentum"),
                "log_evidence": float(log_evidence[row_index]),
                "n_time": int(n_time[row_index]),
                "n_spikes": int(n_spikes[row_index]),
                "runtime_s": float(runtime_s_by_event[row_index]),
                "error": "",
                "kd_grid_preset": metadata["kd_grid_preset"],
                "kd_time_bin_ms": metadata["kd_time_bin_ms"],
                "kd_bin_size_cm": metadata["kd_bin_size_cm"],
                "kd_n_bins": metadata["kd_n_bins"],
                "kd_n_jobs": metadata["kd_n_jobs"],
                "kd_event_chunk_size": metadata["kd_event_chunk_size"],
                "kd_spike_rate_scale": metadata["kd_spike_rate_scale"],
            }
        )
    return rows


def aggregate(base_dir: Path, shard_glob: str, outdir: Path) -> None:
    base_scores = pd.read_csv(base_dir / "event_model_evidence.csv")
    base_grid_params = pd.read_csv(base_dir / "gridsearch_best_params.csv")
    base_marginalized = pd.read_csv(base_dir / "marginalized_model_evidence.csv")
    shard_paths = sorted(Path(path) for path in glob.glob(shard_glob))
    momentum_grid, momentum_params, metadata = _load_momentum_grid(shard_paths)
    base_event_ids = set(base_scores["event_index"].astype(int).unique())
    momentum_event_ids = set(np.asarray(metadata["event_ids"], dtype=int))
    if base_event_ids != momentum_event_ids:
        missing_in_base = sorted(momentum_event_ids - base_event_ids)
        missing_in_momentum = sorted(base_event_ids - momentum_event_ids)
        raise ValueError(
            "Base scores and momentum shards cover different events. "
            f"Missing in base: {missing_in_base[:20]}; missing in momentum: {missing_in_momentum[:20]}"
        )
    prior, _ = empirical_grid_prior(momentum_params, momentum_grid)
    momentum_evidence = marginalize_grid_log_evidence(momentum_grid, prior)
    momentum_scores = pd.DataFrame(_momentum_rows(momentum_evidence, metadata))
    combined = _add_evidence_columns(pd.concat([base_scores, momentum_scores], ignore_index=True))
    event_ids = np.asarray(metadata["event_ids"], dtype=int)
    momentum_best = pd.DataFrame(best_grid_params("momentum", event_ids, momentum_params, momentum_grid))
    grid_params = pd.concat([base_grid_params, momentum_best], ignore_index=True)
    momentum_marginalized = momentum_scores[["session", "event_index", "model", "log_evidence"]]
    marginalized = pd.concat([base_marginalized, momentum_marginalized], ignore_index=True)
    models = list(dict.fromkeys(combined["model"].astype(str)))
    pivot = combined.pivot_table(index=["session", "event_index"], columns="model", values="log_evidence", aggfunc="first")
    random_effects = pd.DataFrame(random_effects_model_probabilities(pivot[models].to_numpy(float), models))
    _write(combined, grid_params, marginalized, random_effects, outdir)


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate sharded KD momentum-grid evidence with base KD model scores.")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--shard-glob", required=True)
    p.add_argument("--output", default="results/kd-model-evidence")
    args = p.parse_args()
    aggregate(Path(args.base_dir), args.shard_glob, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
