#!/usr/bin/env python3
"""Krause-Drugowitsch-aligned replay model-evidence benchmark."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.data import load_replay_session
from hipporeplayimm.kd_reference import (
    KDEncodingConfig,
    KD_MODELS,
    NONTRAJECTORY_MODELS,
    TRAJECTORY_MODELS,
    adjusted_momentum_parameters,
    best_grid_params,
    build_kd_emissions,
    diffusion_transition_1d,
    empirical_grid_prior,
    fit_kd_place_field_encoding,
    grid_config_for_preset,
    kd_random_log_evidence,
    kd_diffusion_log_evidence_from_transition,
    kd_momentum_log_evidence_from_transitions,
    kd_stationary_gaussian_log_evidence_from_transitions,
    kd_stationary_log_evidence,
    marginalize_grid_log_evidence,
    momentum_transition_1d,
    random_effects_model_probabilities,
    stationary_gaussian_transition_1d,
)


_REQUIRED = ("Position_Data.mat", "Ripple_Events.mat", "Spike_Data.mat", "Epochs.mat")
_ALIASES = {"stationary_gaussian": "stationary-gaussian"}


def _session_path(root: str | Path, session: str) -> Path:
    parts = session.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', e.g. 'Rat1/Open1'")
    return Path(root) / parts[0] / parts[1]


def _check_session(path: Path) -> None:
    missing = [name for name in _REQUIRED if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Requested session {path} is missing: {', '.join(missing)}")


def _ints(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(x) for x in item.split("-", 1)]
            if hi < lo:
                raise ValueError(f"descending range: {item}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("no events selected")
    return sorted(dict.fromkeys(values))


def _events(spec: str, session, max_events: int | None = None) -> list[int]:
    s = spec.strip().lower()
    if s == "all":
        out = list(range(session.ripple_count))
    elif s == "run":
        out = [int(x) for x in session.ripple_indices_in_run()]
    elif s.startswith("run:"):
        run = [int(x) for x in session.ripple_indices_in_run()]
        out = []
        for ordinal in _ints(s.split(":", 1)[1]):
            if ordinal < 0 or ordinal >= len(run):
                raise IndexError(f"run ordinal {ordinal} outside 0..{len(run) - 1}")
            out.append(run[ordinal])
        out = sorted(dict.fromkeys(out))
    else:
        out = _ints(spec)
        bad = [event_id for event_id in out if event_id < 0 or event_id >= session.ripple_count]
        if bad:
            raise IndexError(f"event IDs outside 0..{session.ripple_count - 1}: {bad}")
    return out[:max_events] if max_events is not None else out


def _models(spec: str) -> list[str]:
    names = []
    for raw in spec.replace(",", " ").split():
        name = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name:
            names.append(name)
    names = list(dict.fromkeys(names))
    unknown = sorted(set(names) - set(KD_MODELS))
    if unknown:
        raise ValueError(f"unknown KD models: {unknown}; available: {list(KD_MODELS)}")
    if not names:
        raise ValueError("no models selected")
    return names


def _family(model: str) -> str:
    if model in TRAJECTORY_MODELS:
        return "trajectory"
    if model in NONTRAJECTORY_MODELS:
        return "nontrajectory"
    return "other"


def _chunks(length: int, chunk_size: int):
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def _score_momentum_grid(
    emissions_by_event,
    sd_grid: np.ndarray,
    decay_grid: np.ndarray,
    *,
    initial_sd_m_per_s: float,
    n_bins: int,
    bin_size_cm: float,
    n_jobs: int,
    event_chunk_size: int,
) -> np.ndarray:
    grid_values = np.empty((len(emissions_by_event), len(sd_grid), len(decay_grid)), dtype=float)
    event_chunk_size = max(1, min(event_chunk_size, len(emissions_by_event)))
    n_jobs = max(1, n_jobs)
    for chunk_number, (chunk_start, chunk_stop) in enumerate(_chunks(len(emissions_by_event), event_chunk_size), start=1):
        chunk = emissions_by_event[chunk_start:chunk_stop]
        tasks = [
            (sd_index, decay_index, float(sd), float(decay))
            for sd_index, sd in enumerate(sd_grid)
            for decay_index, decay in enumerate(decay_grid)
        ]
        if n_jobs == 1:
            for sd_index, decay_index, values in (
                _score_momentum_param_chunk(task, chunk, initial_sd_m_per_s, n_bins, bin_size_cm)
                for task in tasks
            ):
                grid_values[chunk_start:chunk_stop, sd_index, decay_index] = values
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(_score_momentum_param_chunk, task, chunk, initial_sd_m_per_s, n_bins, bin_size_cm)
                    for task in tasks
                ]
                for future in as_completed(futures):
                    sd_index, decay_index, values = future.result()
                    grid_values[chunk_start:chunk_stop, sd_index, decay_index] = values
        print(
            f"  Momentum chunk {chunk_number}: events {chunk_start}-{chunk_stop - 1}, {len(tasks)} grid points",
            flush=True,
        )
    return grid_values


def _score_momentum_param_chunk(task, emissions_chunk, initial_sd_m_per_s: float, n_bins: int, bin_size_cm: float):
    sd_index, decay_index, sd, decay = task
    values = np.empty(len(emissions_chunk), dtype=float)
    initial_cache: dict[float, np.ndarray] = {}
    transition_cache: dict[float, np.ndarray] = {}
    for event_index, emissions in enumerate(emissions_chunk):
        if emissions.n_time == 1:
            values[event_index] = kd_random_log_evidence(emissions.log_likelihood)
            continue
        dt = float(emissions.dt)
        if dt not in initial_cache:
            initial_sd_meters = initial_sd_m_per_s * dt
            initial_cache[dt] = diffusion_transition_1d(n_bins, initial_sd_meters, bin_size_cm, dt=1.0)
        if dt not in transition_cache:
            adjusted_decay = decay
            adjusted_sd = sd
            if adjusted_decay > 1.0:
                adjusted_decay, adjusted_sd = adjusted_momentum_parameters(adjusted_decay, adjusted_sd, dt)
            transition_cache[dt] = momentum_transition_1d(n_bins, adjusted_sd, adjusted_decay, bin_size_cm, dt)
        values[event_index] = kd_momentum_log_evidence_from_transitions(
            emissions.log_likelihood,
            n_bins,
            initial_cache[dt],
            transition_cache[dt],
        )
    return sd_index, decay_index, values


def _score(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.likelihood != "poisson":
        raise ValueError("KD alignment currently supports only --likelihood poisson")
    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session, args.max_events)
    models = _models(args.models)
    grid = grid_config_for_preset(args.grid_preset)
    encoding = fit_kd_place_field_encoding(
        session,
        KDEncodingConfig(
            bin_size_cm=args.bin_size_cm,
            n_bins_x=args.n_bins,
            n_bins_y=args.n_bins,
            smoothing_sigma_cm=args.place_field_smoothing_cm,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    emissions_by_event = [
        build_kd_emissions(session, encoding, int(event_id), time_bin_s=args.time_bin_ms / 1000.0)
        for event_id in event_ids
    ]
    rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    marginalized_rows: list[dict[str, object]] = []

    for model in models:
        start = time.perf_counter()
        if model == "random":
            values = np.array([kd_random_log_evidence(emissions.log_likelihood) for emissions in emissions_by_event])
        elif model == "stationary":
            values = np.array([kd_stationary_log_evidence(emissions.log_likelihood) for emissions in emissions_by_event])
        elif model == "stationary-gaussian":
            grid_values = np.empty((len(emissions_by_event), len(grid.stationary_gaussian_sd_meters)), dtype=float)
            for sd_index, sd in enumerate(grid.stationary_gaussian_sd_meters):
                transition = stationary_gaussian_transition_1d(args.n_bins, sd, args.bin_size_cm)
                for event_index, emissions in enumerate(emissions_by_event):
                    grid_values[event_index, sd_index] = kd_stationary_gaussian_log_evidence_from_transitions(
                        emissions.log_likelihood,
                        args.n_bins,
                        args.n_bins,
                        transition,
                    )
            prior, _ = empirical_grid_prior({"sd_meters": grid.stationary_gaussian_sd_meters}, grid_values)
            values = marginalize_grid_log_evidence(grid_values, prior)
            grid_rows.extend(best_grid_params(model, event_ids, {"sd_meters": grid.stationary_gaussian_sd_meters}, grid_values))
        elif model == "diffusion":
            grid_values = np.empty((len(emissions_by_event), len(grid.diffusion_sd_meters)), dtype=float)
            transition_cache: dict[tuple[float, float], np.ndarray] = {}
            for sd_index, sd in enumerate(grid.diffusion_sd_meters):
                for event_index, emissions in enumerate(emissions_by_event):
                    key = (float(sd), float(emissions.dt))
                    if key not in transition_cache:
                        transition_cache[key] = diffusion_transition_1d(args.n_bins, float(sd), args.bin_size_cm, emissions.dt)
                    grid_values[event_index, sd_index] = kd_diffusion_log_evidence_from_transition(
                        emissions.log_likelihood,
                        args.n_bins,
                        args.n_bins,
                        transition_cache[key],
                    )
            prior, _ = empirical_grid_prior({"sd_meters": grid.diffusion_sd_meters}, grid_values)
            values = marginalize_grid_log_evidence(grid_values, prior)
            grid_rows.extend(best_grid_params(model, event_ids, {"sd_meters": grid.diffusion_sd_meters}, grid_values))
        elif model == "momentum":
            grid_values = _score_momentum_grid(
                emissions_by_event,
                grid.momentum_sd_meters,
                grid.momentum_decay,
                initial_sd_m_per_s=grid.momentum_initial_sd_m_per_s,
                n_bins=args.n_bins,
                bin_size_cm=args.bin_size_cm,
                n_jobs=args.n_jobs,
                event_chunk_size=args.event_chunk_size,
            )
            grid_params = {"sd_meters": grid.momentum_sd_meters, "decay": grid.momentum_decay}
            prior, _ = empirical_grid_prior(grid_params, grid_values)
            values = marginalize_grid_log_evidence(grid_values, prior)
            grid_rows.extend(best_grid_params(model, event_ids, grid_params, grid_values))
        else:
            raise AssertionError(model)
        runtime = time.perf_counter() - start
        for event_id, emissions, log_evidence in zip(event_ids, emissions_by_event, values, strict=True):
            rows.append(
                {
                    "status": "success",
                    "session": session.session_id,
                    "event_index": int(event_id),
                    "model": model,
                    "model_family": _family(model),
                    "log_evidence": float(log_evidence),
                    "n_time": int(emissions.n_time),
                    "n_spikes": int(emissions.n_spikes),
                    "runtime_s": float(runtime / max(len(event_ids), 1)),
                    "error": "",
                    "kd_grid_preset": args.grid_preset,
                    "kd_time_bin_ms": float(args.time_bin_ms),
                    "kd_bin_size_cm": float(args.bin_size_cm),
                    "kd_n_bins": int(args.n_bins),
                    "kd_n_jobs": int(args.n_jobs),
                    "kd_event_chunk_size": int(args.event_chunk_size),
                }
            )
            marginalized_rows.append(
                {
                    "session": session.session_id,
                    "event_index": int(event_id),
                    "model": model,
                    "log_evidence": float(log_evidence),
                }
            )
        print(f"Scored {session.session_id} with KD {model}: {len(event_ids)} events", flush=True)
    df = _add_evidence_columns(pd.DataFrame(rows))
    random_effects = pd.DataFrame(
        random_effects_model_probabilities(
            df.pivot_table(index=["session", "event_index"], columns="model", values="log_evidence", aggfunc="first")[models].to_numpy(float),
            models,
        )
    )
    return df, pd.DataFrame(grid_rows), pd.DataFrame(marginalized_rows), random_effects


def _add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, g in df.groupby(["session", "event_index"], sort=False):
        g = g.copy()
        vals = g["log_evidence"].to_numpy(float)
        maxv = float(np.max(vals))
        probs = np.exp(vals - logsumexp(vals))
        best = str(g.iloc[int(np.argmax(vals))]["model"])
        g["relative_log_evidence"] = vals - maxv
        g["model_probability"] = probs
        g["is_best_model"] = g["model"] == best
        g["best_model"] = best
        for family, col in (("trajectory", "best_trajectory_model"), ("nontrajectory", "best_nontrajectory_model")):
            subset = g[g["model_family"] == family]
            if subset.empty:
                g[col] = ""
                g[f"delta_vs_{family}_best"] = np.nan
            else:
                bidx = int(np.argmax(subset["log_evidence"].to_numpy(float)))
                bname = str(subset.iloc[bidx]["model"])
                blog = float(subset.iloc[bidx]["log_evidence"])
                g[col] = bname
                g[f"delta_vs_{family}_best"] = g["log_evidence"] - blog
        groups.append(g)
    return pd.concat(groups, ignore_index=True).sort_values(["event_index", "model"]).reset_index(drop=True)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["model", "model_family"], as_index=False).agg(
        events=("event_index", "count"),
        wins=("is_best_model", "sum"),
        mean_log_evidence=("log_evidence", "mean"),
        median_log_evidence=("log_evidence", "median"),
        mean_relative_log_evidence=("relative_log_evidence", "mean"),
        median_relative_log_evidence=("relative_log_evidence", "median"),
        mean_model_probability=("model_probability", "mean"),
        median_model_probability=("model_probability", "median"),
        mean_runtime_s=("runtime_s", "mean"),
    ).sort_values(["wins", "mean_log_evidence"], ascending=[False, False])


def _counts(df: pd.DataFrame) -> pd.DataFrame:
    base = df.drop_duplicates(["session", "event_index"])
    rows = []
    for col in ("best_model", "best_trajectory_model", "best_nontrajectory_model"):
        vc = base[col].value_counts().rename_axis("model").reset_index(name="events")
        vc["comparison"] = col
        rows.extend(vc.to_dict("records"))
    return pd.DataFrame(rows)[["comparison", "model", "events"]]


def _write(df: pd.DataFrame, grid_params: pd.DataFrame, marginalized: pd.DataFrame, random_effects: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "event_model_evidence.csv", index=False)
    _summary(df).to_csv(outdir / "model_evidence_summary.csv", index=False)
    _counts(df).to_csv(outdir / "best_model_counts.csv", index=False)
    grid_params.to_csv(outdir / "gridsearch_best_params.csv", index=False)
    marginalized.to_csv(outdir / "marginalized_model_evidence.csv", index=False)
    random_effects.to_csv(outdir / "random_effects_model_probabilities.csv", index=False)
    for metric in ("log_evidence", "relative_log_evidence", "model_probability"):
        df.pivot_table(index=["session", "event_index"], columns="model", values=metric, aggfunc="first").reset_index().to_csv(
            outdir / f"event_model_pivot_{metric}.csv",
            index=False,
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Run a KD-aligned replay model-evidence benchmark.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--events", default="run")
    p.add_argument("--models", default="random stationary stationary-gaussian diffusion momentum")
    p.add_argument("--time-bin-ms", type=float, default=3.0)
    p.add_argument("--bin-size-cm", type=float, default=4.0)
    p.add_argument("--n-bins", type=int, default=50)
    p.add_argument("--likelihood", choices=("poisson",), default="poisson")
    p.add_argument("--grid-preset", choices=("kd", "smoke"), default="kd")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--event-chunk-size", type=int, default=16)
    p.add_argument("--place-field-smoothing-cm", type=float, default=4.0)
    p.add_argument("--min-speed-cm-s", type=float, default=5.0)
    p.add_argument("--output", default="results/kd-model-evidence")
    args = p.parse_args()
    df, grid_params, marginalized, random_effects = _score(args)
    if df.empty:
        raise RuntimeError("No scores were generated.")
    print(_summary(df).to_string(index=False))
    print("\nBest-model counts:")
    print(_counts(df).to_string(index=False))
    print("\nRandom-effects model probabilities:")
    print(random_effects.to_string(index=False))
    print(f"\nRows: {len(df)}")
    _write(df, grid_params, marginalized, random_effects, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
