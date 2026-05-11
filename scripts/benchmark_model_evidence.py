#!/usr/bin/env python3
"""Session-scoped full-event replay model-evidence benchmark.

This is an approximate model-evidence diagnostic using the repository's current
model ``score`` methods. It is meant as a positive-control step toward the
Krause/Drugowitsch-style Bayesian model-comparison result, not as an exact
reproduction of their Zenodo analysis code.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import (
    EmissionConfig,
    EncodingConfig,
    build_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.models import CandidateKinematicModel, RandomModel, StationaryModel
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel

_REQUIRED = ("Position_Data.mat", "Ripple_Events.mat", "Spike_Data.mat", "Epochs.mat")
_TRAJ = {
    "diffusion",
    "momentum",
    "imm",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-jump",
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-imm",
}
_NONTRAJ = {
    "random",
    "stationary",
    "stationary-gaussian",
    "sorted-spike-state-space-stationary",
}
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


def _events(spec: str, session) -> list[int]:
    s = spec.strip().lower()
    if s == "all":
        return list(range(session.ripple_count))
    if s == "run":
        return [int(x) for x in session.ripple_indices_in_run()]
    if s.startswith("run:"):
        run = [int(x) for x in session.ripple_indices_in_run()]
        out = []
        for ordinal in _ints(s.split(":", 1)[1]):
            if ordinal < 0 or ordinal >= len(run):
                raise IndexError(f"run ordinal {ordinal} outside 0..{len(run) - 1}")
            out.append(run[ordinal])
        return sorted(dict.fromkeys(out))
    out = _ints(spec)
    bad = [e for e in out if e < 0 or e >= session.ripple_count]
    if bad:
        raise IndexError(f"event IDs outside 0..{session.ripple_count - 1}: {bad}")
    return out


def _models(args) -> dict[str, object]:
    names = []
    for raw in args.models.replace(",", " ").split():
        name = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name:
            names.append(name)
    if not names:
        raise ValueError("no models selected")
    available = {
        "random": RandomModel(),
        "stationary": StationaryModel(),
        "stationary-gaussian": CandidateKinematicModel(
            mode="stationary", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="stationary-gaussian"),
        "diffusion": CandidateKinematicModel(
            mode="diffusion", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="diffusion"),
        "momentum": CandidateKinematicModel(
            mode="momentum", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="momentum"),
        "imm": CandidateKinematicModel(
            mode="imm", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="imm"),
        "sorted-spike-state-space-stationary": SortedSpikeStateSpaceReplayModel(mode="stationary"),
        "sorted-spike-state-space-diffusion": SortedSpikeStateSpaceReplayModel(mode="diffusion"),
        "sorted-spike-state-space-fragmented": SortedSpikeStateSpaceReplayModel(mode="fragmented"),
        "sorted-spike-state-space-jump": SortedSpikeStateSpaceReplayModel(mode="jump"),
        "sorted-spike-state-space-momentum": SortedSpikeStateSpaceReplayModel(mode="momentum"),
        "sorted-spike-state-space-imm": SortedSpikeStateSpaceReplayModel(mode="imm"),
    }
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown models: {missing}; available: {sorted(available)}")
    return {name: available[name] for name in dict.fromkeys(names)}


def _family(model: str) -> str:
    if model in _TRAJ:
        return "trajectory"
    if model in _NONTRAJ:
        return "nontrajectory"
    return "other"


def _score(args) -> pd.DataFrame:
    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    models = _models(args)
    emissions_cfg = EmissionConfig(time_bin_s=args.time_bin_s)
    rows: list[dict[str, object]] = []

    for event_id in event_ids:
        emissions = build_emissions(session, encoding, int(event_id), emissions_cfg)
        if emissions.n_time == 0:
            continue
        for name, model in models.items():
            start = time.perf_counter()
            try:
                if isinstance(model, CandidateKinematicModel):
                    cand = model.candidate_indices(emissions)
                    result = model.score(emissions, encoding.bin_centers, candidate_indices=cand)
                else:
                    result = model.score(emissions, encoding.bin_centers)
                model_name = str(result.model_name)
                row = {
                    "status": "success", "session": session.session_id, "event_index": int(event_id),
                    "model": model_name, "requested_model": name, "model_family": _family(model_name), "log_evidence": float(result.log_likelihood),
                    "n_time": int(result.n_time), "n_spikes": int(result.n_spikes),
                    "runtime_s": float(time.perf_counter() - start), "error": "",
                    "bin_size_cm": float(args.bin_size_cm),
                    "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
                    "min_speed_cm_s": float(args.min_speed_cm_s),
                    "time_bin_s": float(args.time_bin_s),
                }
                row.update({f"diagnostic_{k}": v for k, v in result.diagnostics.items()})
                rows.append(row)
                print(f"Scored {session.session_id} event {event_id} with {name}", flush=True)
            except Exception as exc:
                rows.append({
                    "status": "failure", "session": session.session_id, "event_index": int(event_id),
                    "model": name, "requested_model": name, "model_family": _family(name), "log_evidence": np.nan,
                    "n_time": int(emissions.n_time), "n_spikes": int(emissions.n_spikes),
                    "runtime_s": float(time.perf_counter() - start), "error": f"{type(exc).__name__}: {exc}",
                    "bin_size_cm": float(args.bin_size_cm),
                    "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
                    "min_speed_cm_s": float(args.min_speed_cm_s),
                    "time_bin_s": float(args.time_bin_s),
                })
                if not args.continue_on_error:
                    raise
    return _add_evidence_columns(pd.DataFrame(rows))


def _add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    groups = []
    for _, g in df.groupby(["session", "event_index"], sort=False):
        g = g.copy()
        ok = g["status"] == "success"
        s = g[ok]
        if s.empty:
            g["relative_log_evidence"] = np.nan
            g["model_probability"] = np.nan
            g["is_best_model"] = False
            g["best_model"] = ""
            groups.append(g)
            continue
        vals = s["log_evidence"].to_numpy(float)
        maxv = float(np.max(vals))
        probs = np.exp(vals - logsumexp(vals))
        best = str(s.iloc[int(np.argmax(vals))]["model"])
        g["relative_log_evidence"] = np.nan
        g["model_probability"] = np.nan
        g.loc[s.index, "relative_log_evidence"] = vals - maxv
        g.loc[s.index, "model_probability"] = probs
        g["is_best_model"] = g["model"] == best
        g["best_model"] = best
        for family, col in (("trajectory", "best_trajectory_model"), ("nontrajectory", "best_nontrajectory_model")):
            subset = s[s["model_family"] == family]
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
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    out = ok.groupby(["model", "model_family"], as_index=False).agg(
        events=("event_index", "count"), wins=("is_best_model", "sum"),
        mean_log_evidence=("log_evidence", "mean"), median_log_evidence=("log_evidence", "median"),
        mean_relative_log_evidence=("relative_log_evidence", "mean"),
        median_relative_log_evidence=("relative_log_evidence", "median"),
        mean_model_probability=("model_probability", "mean"),
        median_model_probability=("model_probability", "median"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    out["win_fraction"] = out["wins"] / out["events"].clip(lower=1)
    return out.sort_values(["wins", "mean_log_evidence"], ascending=[False, False])


def _counts(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    base = ok.drop_duplicates(["session", "event_index"])
    rows = []
    for col in ("best_model", "best_trajectory_model", "best_nontrajectory_model"):
        vc = base[col].value_counts().rename_axis("model").reset_index(name="events")
        vc["comparison"] = col
        rows.extend(vc.to_dict("records"))
    return pd.DataFrame(rows)[["comparison", "model", "events"]]


def _write(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "event_model_evidence.csv", index=False)
    _summary(df).to_csv(outdir / "model_evidence_summary.csv", index=False)
    _counts(df).to_csv(outdir / "best_model_counts.csv", index=False)
    ok = df[df["status"] == "success"]
    for metric in ("log_evidence", "relative_log_evidence", "model_probability"):
        ok.pivot_table(index=["session", "event_index"], columns="model", values=metric, aggfunc="first").reset_index().to_csv(outdir / f"event_model_pivot_{metric}.csv", index=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Run a session-scoped replay model-evidence benchmark.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--events", default="0-25")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--models", default="random stationary stationary-gaussian diffusion momentum imm")
    p.add_argument("--candidate-top-k", type=int, default=64)
    p.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    p.add_argument("--diffusion-sigma-cm", type=float, default=12.0)
    p.add_argument("--momentum-sigma-cm", type=float, default=12.0)
    p.add_argument("--velocity-decay", type=float, default=0.95)
    p.add_argument("--mode-stickiness", type=float, default=0.94)
    p.add_argument("--time-bin-s", type=float, default=0.02)
    p.add_argument("--bin-size-cm", type=float, default=4.0)
    p.add_argument("--smoothing-sigma-bins", type=float, default=1.5)
    p.add_argument("--min-speed-cm-s", type=float, default=5.0)
    p.add_argument("--output", default="results/model-evidence")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()
    df = _score(args)
    if df.empty:
        raise RuntimeError("No scores were generated.")
    print(_summary(df).to_string(index=False))
    print("\nBest-model counts:")
    print(_counts(df).to_string(index=False))
    print(f"\nRows: {len(df)}")
    print(f"Failures: {int((df['status'] != 'success').sum())}")
    _write(df, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
