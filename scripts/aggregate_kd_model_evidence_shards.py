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


def _load_momentum_grid(shard_paths: list[Path]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if not shard_paths:
        raise FileNotFoundError("No momentum shard files matched.")
    first = np.load(shard_paths[0], allow_pickle=False)
    event_ids = first["event_ids"]
    sd_meters = first["sd_meters"]
    decay = first["decay"]
    grid = np.full((event_ids.shape[0], sd_meters.shape[0], decay.shape[0]), np.nan, dtype=float)
    runtimes = []
    metadata: dict[str, object] = {
        "session": str(first["session"]),
        "event_ids": event_ids,
        "n_time": first["n_time"],
        "n_spikes": first["n_spikes"],
        "kd_grid_preset": str(first["kd_grid_preset"]),
        "kd_time_bin_ms": float(first["kd_time_bin_ms"]),
        "kd_bin_size_cm": float(first["kd_bin_size_cm"]),
        "kd_n_bins": int(first["kd_n_bins"]),
        "kd_n_jobs": int(first["kd_n_jobs"]),
        "kd_event_chunk_size": int(first["kd_event_chunk_size"]),
    }
    for path in shard_paths:
        shard = np.load(path, allow_pickle=False)
        if not np.array_equal(shard["event_ids"], event_ids):
            raise ValueError(f"Event IDs differ in {path}")
        if not np.array_equal(shard["sd_meters"], sd_meters) or not np.array_equal(shard["decay"], decay):
            raise ValueError(f"Momentum grid differs in {path}")
        values = shard["values"]
        for column, (sd_index, decay_index) in enumerate(zip(shard["sd_indices"], shard["decay_indices"], strict=True)):
            grid[:, int(sd_index), int(decay_index)] = values[:, column]
        runtimes.append(float(shard["runtime_s"]))
    missing = np.argwhere(~np.isfinite(grid))
    if missing.size:
        raise ValueError(f"Momentum shards did not cover {missing.shape[0]} event/grid entries.")
    metadata["runtime_s"] = max(runtimes) if runtimes else 0.0
    return grid, {"sd_meters": sd_meters, "decay": decay}, metadata


def _momentum_rows(log_evidence: np.ndarray, metadata: dict[str, object]) -> list[dict[str, object]]:
    event_ids = np.asarray(metadata["event_ids"], dtype=int)
    n_time = np.asarray(metadata["n_time"], dtype=int)
    n_spikes = np.asarray(metadata["n_spikes"], dtype=int)
    runtime_s = float(metadata["runtime_s"]) / max(event_ids.shape[0], 1)
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
                "runtime_s": runtime_s,
                "error": "",
                "kd_grid_preset": metadata["kd_grid_preset"],
                "kd_time_bin_ms": metadata["kd_time_bin_ms"],
                "kd_bin_size_cm": metadata["kd_bin_size_cm"],
                "kd_n_bins": metadata["kd_n_bins"],
                "kd_n_jobs": metadata["kd_n_jobs"],
                "kd_event_chunk_size": metadata["kd_event_chunk_size"],
            }
        )
    return rows


def aggregate(base_dir: Path, shard_glob: str, outdir: Path) -> None:
    base_scores = pd.read_csv(base_dir / "event_model_evidence.csv")
    base_grid_params = pd.read_csv(base_dir / "gridsearch_best_params.csv")
    base_marginalized = pd.read_csv(base_dir / "marginalized_model_evidence.csv")
    shard_paths = sorted(Path(path) for path in glob.glob(shard_glob))
    momentum_grid, momentum_params, metadata = _load_momentum_grid(shard_paths)
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
