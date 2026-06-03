#!/usr/bin/env python3
"""Cell-split held-out predictive control for replay trajectory evidence."""

from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmark_model_evidence import _check_session, _events, _session_path
from benchmark_model_evidence_improved import _family, _models, _run_settings
from hipporeplayimm.benchmarks import _score_train_joint_model, _session_mark_diagnostics, _split_cells
from hipporeplayimm.clusterless import ClusterlessStateSpaceReplayModel
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, fit_place_field_encoding
from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)
from spike_matched_event_window_null import DEFAULT_MATCHED_NULL_MODELS, _add_model_arguments

DEFAULT_CELL_SPLIT_HELDOUT_MODELS = DEFAULT_MATCHED_NULL_MODELS
DEFAULT_SPLIT_COUNT = 20
DEFAULT_TEST_CELL_FRACTION = 0.5
DEFAULT_MARGIN_THRESHOLD = 5.5
DEFAULT_REQUIRED_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)
DEFAULT_EXACT_TRAJECTORY_MODELS = (
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-exact-sparse",
)


def score_cell_split_heldout(args: argparse.Namespace) -> pd.DataFrame:
    """Score replay events with train-cell evidence and held-out-cell likelihood."""

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
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
    )
    models = _models(args, session, encoding=encoding)
    unsupported = [name for name, model in models.items() if isinstance(model, ClusterlessStateSpaceReplayModel)]
    if unsupported:
        raise ValueError(
            "cell-split held-out scoring currently supports sorted-spike/state-space models only; "
            f"clusterless model(s) requested: {unsupported}"
        )

    emissions_cfg = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )
    sorted_calibration = ReplayEmissionCalibration(
        gain_mode=args.replay_gain_mode,
        gain_prior_count=args.replay_gain_prior_count,
        max_gain=args.replay_gain_max_gain,
        emission_model=args.sorted_spike_emission_model,
        negative_binomial_dispersion=args.negative_binomial_dispersion,
    )

    rows: list[dict[str, object]] = []
    for split_index in range(int(args.n_splits)):
        split_seed = _cell_split_seed(args.random_seed, split_index)
        train_cells, test_cells = _split_cells(encoding.cell_ids, args.test_cell_fraction, split_seed)
        if train_cells.size == 0 or test_cells.size == 0:
            raise ValueError(
                f"cell split {split_index} produced train={train_cells.size}, test={test_cells.size}; "
                "need non-empty train and test cell sets"
            )
        train_encoding = encoding.select_cells(train_cells)
        joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))
        split_settings = {
            "cell_split_index": int(split_index),
            "cell_split_count": int(args.n_splits),
            "cell_split_seed": int(split_seed),
            "test_cell_fraction": float(args.test_cell_fraction),
            "train_cell_count": int(train_cells.size),
            "test_cell_count": int(test_cells.size),
            "train_cell_ids": _format_cell_ids(train_cells),
            "test_cell_ids": _format_cell_ids(test_cells),
        }

        for event_id in event_ids:
            event = session.ripple(int(event_id))
            event_window = SimpleNamespace(start=float(event.start), end=float(event.end))
            train_emissions = build_sorted_emissions_with_replay_calibration(
                session,
                train_encoding,
                event_window,
                emissions_cfg,
                calibration=sorted_calibration,
            )
            joint_emissions = build_sorted_emissions_with_replay_calibration(
                session,
                joint_encoding,
                event_window,
                emissions_cfg,
                calibration=sorted_calibration,
            )
            if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
                continue
            event_settings = {
                "event_start_s": float(event.start),
                "event_end_s": float(event.end),
                "event_duration_s": float(event.end - event.start),
            }
            for requested_model, model in models.items():
                start = time.perf_counter()
                try:
                    train_score, joint_score = _score_train_joint_model(
                        model,
                        train_emissions,
                        joint_emissions,
                        encoding.bin_centers,
                        occupancy_s=encoding.occupancy_s,
                    )
                    runtime_s = float(time.perf_counter() - start)
                    heldout = float(joint_score.log_likelihood - train_score.log_likelihood)
                    test_spikes = int(joint_emissions.n_spikes - train_emissions.n_spikes)
                    model_name = str(joint_score.model_name)
                    row: dict[str, object] = {
                        "status": "success",
                        "session": session.session_id,
                        "rat": session.rat,
                        "event_index": int(event_id),
                        **event_settings,
                        **split_settings,
                        "model": model_name,
                        "requested_model": requested_model,
                        "model_family": _family(model_name),
                        "heldout_log_likelihood": heldout,
                        "log_evidence": heldout,
                        "log_evidence_scope": "cell_split_heldout_conditional",
                        "heldout_log_likelihood_per_spike": _safe_ratio(heldout, test_spikes),
                        "heldout_bits_per_spike": _safe_ratio(heldout, test_spikes * np.log(2.0)),
                        "joint_log_likelihood": float(joint_score.log_likelihood),
                        "train_log_likelihood": float(train_score.log_likelihood),
                        "train_spikes": int(train_emissions.n_spikes),
                        "test_spikes": test_spikes,
                        "joint_spikes": int(joint_emissions.n_spikes),
                        "n_time": int(train_emissions.n_time),
                        "runtime_s": runtime_s,
                        "error": "",
                        **_run_settings(args),
                        **_session_mark_diagnostics(session),
                    }
                    metadata = getattr(joint_emissions, "metadata", {}) or {}
                    row.update({f"emission_{key}": value for key, value in metadata.items()})
                    row.update({f"diagnostic_{key}": value for key, value in joint_score.diagnostics.items()})
                    rows.append(row)
                    print(
                        "[cell-split-heldout] "
                        f"session={session.session_id} event={event_id} split={split_index + 1}/{args.n_splits} "
                        f"model={requested_model} heldout={heldout:.3f} elapsed={runtime_s:.2f}s",
                        flush=True,
                    )
                except Exception as exc:
                    runtime_s = float(time.perf_counter() - start)
                    rows.append(
                        {
                            "status": "failure",
                            "session": session.session_id,
                            "rat": session.rat,
                            "event_index": int(event_id),
                            **event_settings,
                            **split_settings,
                            "model": requested_model,
                            "requested_model": requested_model,
                            "model_family": _family(requested_model),
                            "heldout_log_likelihood": np.nan,
                            "log_evidence": np.nan,
                            "log_evidence_scope": "cell_split_heldout_conditional",
                            "heldout_log_likelihood_per_spike": np.nan,
                            "heldout_bits_per_spike": np.nan,
                            "joint_log_likelihood": np.nan,
                            "train_log_likelihood": np.nan,
                            "train_spikes": int(train_emissions.n_spikes),
                            "test_spikes": int(joint_emissions.n_spikes - train_emissions.n_spikes),
                            "joint_spikes": int(joint_emissions.n_spikes),
                            "n_time": int(train_emissions.n_time),
                            "runtime_s": runtime_s,
                            "error": f"{type(exc).__name__}: {exc}",
                            **_run_settings(args),
                        }
                    )
                    print(
                        "[cell-split-heldout] "
                        f"failed session={session.session_id} event={event_id} split={split_index + 1}/{args.n_splits} "
                        f"model={requested_model}: {exc}",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        raise
    return ensure_evidence_support_columns(pd.DataFrame(rows))


def cell_split_family_margin_decisions(
    frame: pd.DataFrame,
    *,
    required_models: tuple[str, ...] = DEFAULT_REQUIRED_MODELS,
    trajectory_models: tuple[str, ...] = DEFAULT_EXACT_TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Return held-out best exact trajectory versus nontrajectory decisions."""

    columns = [
        "session",
        "rat",
        "event_index",
        "cell_split_index",
        "cell_split_seed",
        "test_cell_count",
        "test_spikes",
        "n_time",
        "required_models_present",
        "required_models_total",
        "required_models_complete",
        "missing_required_models",
        "present_required_models",
        "margin_threshold",
        "best_trajectory_model",
        "best_trajectory_heldout_log_likelihood",
        "best_nontrajectory_model",
        "best_nontrajectory_heldout_log_likelihood",
        "trajectory_minus_nontrajectory_heldout_log_likelihood",
        "trajectory_minus_nontrajectory_heldout_bits_per_spike",
        "trajectory_raw_win",
        "trajectory_confident_claim",
        "nontrajectory_confident_claim",
        "margin_decision",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    scored = ensure_evidence_support_columns(frame)
    status_ok = scored["status"].eq("success") if "status" in scored else pd.Series(True, index=scored.index)
    comparable = scored["evidence_comparable"].fillna(False).astype(bool) if "evidence_comparable" in scored else pd.Series(True, index=scored.index)
    ok = scored[status_ok & comparable].copy()
    required = tuple(str(model) for model in required_models)
    required_set = set(required)
    trajectory_set = set(str(model) for model in trajectory_models)
    rows: list[dict[str, object]] = []
    group_cols = ["session", "event_index", "cell_split_index"]
    for key, group in ok.groupby(group_cols, sort=True):
        session, event_index, split_index = key
        core = group[group["model"].astype(str).isin(required_set)].dropna(subset=["heldout_log_likelihood"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        complete = not missing
        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        if trajectory.empty or nontrajectory.empty:
            best_trajectory_model = ""
            best_trajectory_heldout = np.nan
            best_nontrajectory_model = ""
            best_nontrajectory_heldout = np.nan
            margin = np.nan
            decision = "incomplete_core"
            raw_win = False
            trajectory_claim = False
            nontrajectory_claim = False
        else:
            best_trajectory = trajectory.sort_values("heldout_log_likelihood", ascending=False).iloc[0]
            best_nontrajectory = nontrajectory.sort_values("heldout_log_likelihood", ascending=False).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_trajectory_heldout = float(best_trajectory["heldout_log_likelihood"])
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_nontrajectory_heldout = float(best_nontrajectory["heldout_log_likelihood"])
            margin = best_trajectory_heldout - best_nontrajectory_heldout
            raw_win = bool(margin > 0.0)
            trajectory_claim = bool(complete and margin >= float(margin_threshold))
            nontrajectory_claim = bool(complete and margin <= -float(margin_threshold))
            if not complete:
                decision = "incomplete_core"
            elif trajectory_claim:
                decision = "trajectory"
            elif nontrajectory_claim:
                decision = "nontrajectory"
            else:
                decision = "ambiguous"
        test_spikes = _first_numeric_value(group, "test_spikes")
        rows.append(
            {
                "session": str(session),
                "rat": _first_text_value(group, "rat") or str(session).split("/")[0],
                "event_index": int(event_index),
                "cell_split_index": int(split_index),
                "cell_split_seed": _first_numeric_value(group, "cell_split_seed"),
                "test_cell_count": _first_numeric_value(group, "test_cell_count"),
                "test_spikes": test_spikes,
                "n_time": _first_numeric_value(group, "n_time"),
                "required_models_present": int(len(present)),
                "required_models_total": int(len(required)),
                "required_models_complete": bool(complete),
                "missing_required_models": " ".join(missing),
                "present_required_models": " ".join(present),
                "margin_threshold": float(margin_threshold),
                "best_trajectory_model": best_trajectory_model,
                "best_trajectory_heldout_log_likelihood": best_trajectory_heldout,
                "best_nontrajectory_model": best_nontrajectory_model,
                "best_nontrajectory_heldout_log_likelihood": best_nontrajectory_heldout,
                "trajectory_minus_nontrajectory_heldout_log_likelihood": margin,
                "trajectory_minus_nontrajectory_heldout_bits_per_spike": _safe_ratio(margin, test_spikes * np.log(2.0)),
                "trajectory_raw_win": raw_win,
                "trajectory_confident_claim": trajectory_claim,
                "nontrajectory_confident_claim": nontrajectory_claim,
                "margin_decision": decision,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def cell_split_family_margin_summary(
    decisions: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize held-out family-margin decisions."""

    base_columns = [
        *group_cols,
        "split_event_rows",
        "events",
        "cell_splits",
        "required_complete_rows",
        "incomplete_core_rows",
        "margin_threshold",
        "trajectory_raw_wins",
        "trajectory_raw_win_fraction",
        "trajectory_confident_claims",
        "trajectory_confident_claim_fraction",
        "nontrajectory_confident_claims",
        "ambiguous_rows",
        "mean_family_margin",
        "median_family_margin",
        "mean_family_margin_bits_per_spike",
        "median_family_margin_bits_per_spike",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=base_columns)
    grouped = decisions.groupby(list(group_cols), sort=True) if group_cols else (((), decisions),)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        margins = pd.to_numeric(group["trajectory_minus_nontrajectory_heldout_log_likelihood"], errors="coerce")
        bits = pd.to_numeric(group["trajectory_minus_nontrajectory_heldout_bits_per_spike"], errors="coerce")
        row.update(
            {
                "split_event_rows": int(len(group)),
                "events": int(group[["session", "event_index"]].drop_duplicates().shape[0]),
                "cell_splits": int(group["cell_split_index"].nunique()),
                "required_complete_rows": int(group["required_models_complete"].fillna(False).astype(bool).sum()),
                "incomplete_core_rows": int((group["margin_decision"] == "incomplete_core").sum()),
                "margin_threshold": float(pd.to_numeric(group["margin_threshold"], errors="coerce").dropna().iloc[0]),
                "trajectory_raw_wins": int(group["trajectory_raw_win"].fillna(False).astype(bool).sum()),
                "trajectory_raw_win_fraction": float(group["trajectory_raw_win"].fillna(False).astype(bool).mean()),
                "trajectory_confident_claims": int(group["trajectory_confident_claim"].fillna(False).astype(bool).sum()),
                "trajectory_confident_claim_fraction": float(group["trajectory_confident_claim"].fillna(False).astype(bool).mean()),
                "nontrajectory_confident_claims": int(group["nontrajectory_confident_claim"].fillna(False).astype(bool).sum()),
                "ambiguous_rows": int((group["margin_decision"] == "ambiguous").sum()),
                "mean_family_margin": float(margins.mean()),
                "median_family_margin": float(margins.median()),
                "mean_family_margin_bits_per_spike": float(bits.mean()),
                "median_family_margin_bits_per_spike": float(bits.median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=base_columns)


def cell_split_control_gate_summary(scores: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    failures = int((scores["status"].astype(str) != "success").sum()) if "status" in scores else 0
    _append_gate(rows, "failures_zero", failures == 0, failures, "cell-split held-out scoring has zero failed model rows")
    if decisions.empty:
        _append_gate(rows, "decisions_exist", False, 0, "held-out family-margin decisions exist")
        rows.append(
            {
                "gate": "overall",
                "passed": False,
                "observed": f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} gates passed",
                "criterion": "all cell-split held-out gates pass",
            }
        )
        return pd.DataFrame(rows)
    _append_gate(rows, "required_models_complete", bool(decisions["required_models_complete"].fillna(False).astype(bool).all()), int(decisions["required_models_complete"].fillna(False).astype(bool).sum()), "all split-event rows include the exact core required models")
    margins = pd.to_numeric(decisions["trajectory_minus_nontrajectory_heldout_log_likelihood"], errors="coerce")
    _append_gate(rows, "median_family_margin_positive", float(margins.median()) > 0.0, float(margins.median()), "median held-out trajectory-family margin > 0")
    _append_gate(rows, "majority_split_events_positive", float((margins > 0.0).mean()) > 0.5, float((margins > 0.0).mean()), "majority of split-event held-out margins > 0")
    rat_summary = cell_split_family_margin_summary(decisions, group_cols=("rat",))
    rat_medians = pd.to_numeric(rat_summary["median_family_margin"], errors="coerce") if not rat_summary.empty else pd.Series(dtype=float)
    _append_gate(rows, "per_rat_median_positive", bool(not rat_medians.empty and (rat_medians > 0.0).all()), "" if rat_medians.empty else float(rat_medians.min()), "per-rat median held-out family margin > 0")
    nontrajectory = int(decisions["nontrajectory_confident_claim"].fillna(False).astype(bool).sum())
    _append_gate(rows, "nontrajectory_claims_near_zero", nontrajectory <= max(1, math.ceil(0.05 * len(decisions))), nontrajectory, "false/nontrajectory confident claims remain near zero")
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in rows),
            "observed": f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} gates passed",
            "criterion": "all cell-split held-out gates pass",
        }
    )
    return pd.DataFrame(rows)


def aggregate_cell_split_heldout_scores(
    score_glob: str,
    outdir: Path,
    *,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    paths = [Path(path) for path in sorted(glob.glob(str(score_glob), recursive=True))]
    if not paths:
        raise FileNotFoundError(f"no cell-split held-out score files found for {score_glob!r}")
    scores = ensure_evidence_support_columns(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    outdir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(outdir / "cell_split_heldout_model_evidence.csv", index=False)
    decisions = cell_split_family_margin_decisions(scores, margin_threshold=margin_threshold)
    decisions.to_csv(outdir / "cell_split_heldout_family_margin_decisions.csv", index=False)
    cell_split_family_margin_summary(decisions).to_csv(outdir / "cell_split_heldout_family_margin_summary.csv", index=False)
    cell_split_family_margin_summary(decisions, group_cols=("rat",)).to_csv(outdir / "rat_cell_split_heldout_summary.csv", index=False)
    cell_split_control_gate_summary(scores, decisions).to_csv(outdir / "cell_split_control_gate_summary.csv", index=False)
    return scores


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-10")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--models", default=DEFAULT_CELL_SPLIT_HELDOUT_MODELS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--test-cell-fraction", type=float, default=DEFAULT_TEST_CELL_FRACTION)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--null-random-seed", type=int, default=1)
    _add_model_arguments(parser)
    parser.add_argument("--output", default="results/cell-split-heldout-control")
    parser.add_argument("--continue-on-error", action="store_true")


def _format_cell_ids(cell_ids: np.ndarray) -> str:
    return ",".join(str(int(cell_id)) for cell_id in np.asarray(cell_ids, dtype=int))


def _cell_split_seed(base_seed: int, split_index: int) -> int:
    return int(base_seed) + int(split_index)


def _first_numeric_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _first_text_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    return str(values.iloc[0]) if not values.empty else ""


def _safe_ratio(value: float, denominator: float) -> float:
    if not np.isfinite(value) or not np.isfinite(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(value) / float(denominator)


def _append_gate(rows: list[dict[str, object]], gate: str, passed: bool, observed: object, criterion: str) -> None:
    rows.append({"gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score")
    _add_scoring_arguments(score_parser)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--score-glob", required=True)
    aggregate_parser.add_argument("--output", default="results/cell-split-heldout-control")
    aggregate_parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    args = parser.parse_args()

    if args.command == "score":
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        scores = score_cell_split_heldout(args)
        if scores.empty:
            raise RuntimeError("No cell-split held-out scores were generated.")
        scores.to_csv(outdir / "cell_split_heldout_model_evidence.csv", index=False)
        aggregate_cell_split_heldout_scores(str(outdir / "cell_split_heldout_model_evidence.csv"), outdir)
        return 0
    aggregate_cell_split_heldout_scores(
        args.score_glob,
        Path(args.output),
        margin_threshold=args.margin_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
