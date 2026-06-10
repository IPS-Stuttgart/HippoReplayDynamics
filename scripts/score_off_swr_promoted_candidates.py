#!/usr/bin/env python3
"""Score promoted off-SWR candidates with an exact-core model set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_event_window_sensitivity import DEFAULT_MARGIN_THRESHOLD
from benchmark_model_evidence import _check_session, _postprocess_evidence_scores, _session_path
from benchmark_model_evidence_improved import _models
from hipporeplayimm.clusterless import ClusterlessStateSpaceReplayModel, fit_clusterless_mark_encoding
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, fit_place_field_encoding
from hipporeplayimm.result_improvement_extensions import ReplayEmissionCalibration
from spike_matched_event_window_null import (
    DEFAULT_MAX_NON_RUN_CANDIDATE_WINDOWS,
    FULL_CORE_REQUIRED_MODELS,
    _add_comparison_scope_argument,
    _add_model_arguments,
    _clusterless_mark_config,
    _parse_required_models,
    _score_one_window,
    _spike_count_and_active_cells,
    _window_position_summary,
    matched_null_family_margin_decisions,
)


PROMOTED_WINDOW_ROLE = "promoted_off_swr_candidate"
DEFAULT_VALIDATION_MODELS = " ".join(FULL_CORE_REQUIRED_MODELS)

VALIDATION_SCORE_COLUMNS = (
    "status",
    "session",
    "event_index",
    "window_role",
    "null_index",
    "model",
    "log_evidence",
)

SUMMARY_COLUMNS = (
    "comparison_scope",
    "candidate_filter",
    "selected_candidates",
    "scored_candidates",
    "required_complete_candidates",
    "trajectory_confident_claims",
    "nontrajectory_confident_claims",
    "ambiguous_candidates",
    "incomplete_candidates",
    "strong_exact_candidates",
    "extreme_exact_candidates",
    "mean_exact_family_margin",
    "median_exact_family_margin",
    "min_exact_family_margin",
    "max_exact_family_margin",
    "candidate_sessions",
    "candidate_rats",
    "validation_status",
    "paper_claim_guidance",
)

GROUP_SUMMARY_COLUMNS = (
    "comparison_scope",
    "candidate_filter",
    "rat",
    "group",
    "selected_candidates",
    "required_complete_candidates",
    "trajectory_confident_claims",
    "nontrajectory_confident_claims",
    "strong_exact_candidates",
    "extreme_exact_candidates",
    "median_exact_family_margin",
    "min_exact_family_margin",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(numeric) and numeric != 0.0)


def _parse_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part for part in str(value).replace(",", " ").split() if part)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _encoding_config_from_args(args: argparse.Namespace) -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=args.bin_size_cm,
        smoothing_sigma_bins=args.smoothing_sigma_bins,
        min_speed_cm_s=args.min_speed_cm_s,
        min_occupancy_s=args.min_occupancy_s,
        rate_floor_hz=args.rate_floor_hz,
    )


def select_candidate_windows(
    candidate_table: pd.DataFrame,
    *,
    candidate_filter: str,
    session_filter: tuple[str, ...] = (),
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """Return candidate rows selected for exact validation."""

    if candidate_table.empty:
        return pd.DataFrame()
    table = candidate_table.copy()
    if "session" not in table or "event_index" not in table or "null_index" not in table:
        raise ValueError("candidate table must include session, event_index, and null_index")
    if session_filter:
        table = table[table["session"].astype(str).isin(set(session_filter))].copy()
    if candidate_filter == "promotion-ready":
        if "passes_high_specificity_promotion_filter" not in table:
            raise ValueError("promotion-ready filter requires passes_high_specificity_promotion_filter")
        table = table[table["passes_high_specificity_promotion_filter"].map(_as_bool)].copy()
    elif candidate_filter == "strong-immobile":
        margin = pd.to_numeric(table.get("trajectory_family_margin", pd.Series(index=table.index)), errors="coerce")
        run_state = table.get("run_or_immobility_state", pd.Series("", index=table.index)).astype(str)
        distance = pd.to_numeric(table.get("distance_to_nearest_swr_s", pd.Series(index=table.index)), errors="coerce")
        table = table[(margin >= 50.0) & run_state.eq("immobile") & distance.notna() & (distance >= 1.0)].copy()
    elif candidate_filter == "all-high-specificity":
        if "high_specificity_label" in table:
            labels = table["high_specificity_label"].astype(str).str.strip()
            table = table[labels.ne("") & labels.ne("nan")].copy()
    elif candidate_filter == "all":
        table = table.copy()
    else:
        raise ValueError(f"unknown candidate_filter: {candidate_filter!r}")
    sort_cols = [column for column in ("candidate_rank", "session", "event_index", "null_index") if column in table]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="mergesort")
    if max_candidates is not None and int(max_candidates) >= 0:
        table = table.head(int(max_candidates)).copy()
    table = table.reset_index(drop=True)
    table["validation_candidate_index"] = np.arange(len(table), dtype=int)
    table["validation_window_role"] = PROMOTED_WINDOW_ROLE
    return table


def score_promoted_candidates(args: argparse.Namespace, candidates: pd.DataFrame) -> pd.DataFrame:
    """Score selected promoted candidates with the requested model set."""

    rows: list[dict[str, object]] = []
    if candidates.empty:
        return pd.DataFrame(columns=list(VALIDATION_SCORE_COLUMNS))
    for session_id, session_candidates in candidates.groupby("session", sort=True):
        session_dir = _session_path(args.dataset_root, str(session_id))
        _check_session(session_dir)
        session = load_replay_session(session_dir)
        encoding = fit_place_field_encoding(
            session,
            _encoding_config_from_args(args),
        )
        models = _models(args, session, encoding=encoding)
        has_clusterless = any(isinstance(model, ClusterlessStateSpaceReplayModel) for model in models.values())
        clusterless_encoding = fit_clusterless_mark_encoding(session, _clusterless_mark_config(args)) if has_clusterless else None
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
        spikes = session.excitatory_spikes()
        for _, candidate in session_candidates.iterrows():
            event_index = int(candidate["event_index"])
            start = float(candidate["window_start_s"])
            end = float(candidate["window_end_s"])
            event = session.ripple(event_index)
            real_count, real_active = _spike_count_and_active_cells(spikes, float(event.start), float(event.end))
            null_count, null_active = _spike_count_and_active_cells(spikes, start, end)
            duration = float(end - start)
            denominator = max(int(real_count), 1)
            window = {
                "window_role": PROMOTED_WINDOW_ROLE,
                "event_window_variant": "off_swr_promoted_candidate",
                "null_index": int(candidate["null_index"]),
                "matched_null_rank": int(candidate.get("matched_null_rank", int(candidate["null_index"]) + 1)),
                "template_event_index": event_index,
                "window_start_s": start,
                "window_end_s": end,
                "window_duration_s": duration,
                "real_event_start_s": float(event.start),
                "real_event_end_s": float(event.end),
                "real_event_duration_s": float(event.end - event.start),
                "real_n_spikes": int(real_count),
                "real_active_cell_count": int(real_active),
                "null_n_spikes": int(null_count),
                "null_active_cell_count": int(null_active),
                "n_spikes_delta": int(null_count) - int(real_count),
                "active_cell_count_delta": int(null_active) - int(real_active),
                "n_spikes_relative_delta": (int(null_count) - int(real_count)) / float(denominator),
                "off_swr": True,
                "restrict_to_run_times": False,
                **_window_position_summary(session.position, start, end),
            }
            _score_one_window(
                args,
                session,
                encoding,
                clusterless_encoding,
                models,
                emissions_cfg,
                sorted_calibration,
                event_id=event_index,
                window_index=int(candidate["validation_candidate_index"]),
                window=window,
                rows=rows,
            )
    scores = _postprocess_evidence_scores(pd.DataFrame(rows))
    return add_candidate_metadata(scores, candidates)


def add_candidate_metadata(scores: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or candidates.empty:
        return scores
    metadata_columns = [
        column
        for column in (
            "session",
            "event_index",
            "null_index",
            "validation_candidate_index",
            "candidate_rank",
            "candidate_specificity_label",
            "candidate_tier",
            "high_specificity_label",
            "trajectory_family_margin",
            "trajectory_confidence",
            "run_or_immobility_state",
            "animal_speed_mean",
            "distance_to_nearest_swr_s",
            "candidate_cluster_id",
        )
        if column in candidates.columns
    ]
    meta = candidates[metadata_columns].copy()
    meta["window_role"] = PROMOTED_WINDOW_ROLE
    out = scores.merge(meta, on=["session", "event_index", "window_role", "null_index"], how="left", suffixes=("", "_input"))
    return out


def validation_decisions(
    scores: pd.DataFrame,
    *,
    comparison_scope: str,
    required_models: tuple[str, ...] | None,
    margin_threshold: float,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    decisions = matched_null_family_margin_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    metadata = scores.drop_duplicates(["session", "event_index", "window_role", "null_index"])
    metadata_columns = [
        column
        for column in (
            "session",
            "event_index",
            "window_role",
            "null_index",
            "validation_candidate_index",
            "candidate_rank",
            "candidate_specificity_label",
            "candidate_tier",
            "high_specificity_label",
            "trajectory_family_margin",
            "trajectory_confidence",
            "run_or_immobility_state",
            "animal_speed_mean",
            "distance_to_nearest_swr_s",
            "candidate_cluster_id",
        )
        if column in metadata.columns
    ]
    if metadata_columns:
        decisions = decisions.merge(
            metadata[metadata_columns],
            on=["session", "event_index", "window_role", "null_index"],
            how="left",
        )
    return decisions


def validation_summary(decisions: pd.DataFrame, *, candidate_filter: str, comparison_scope: str) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(
            [
                {
                    "comparison_scope": comparison_scope,
                    "candidate_filter": candidate_filter,
                    "selected_candidates": 0,
                    "scored_candidates": 0,
                    "required_complete_candidates": 0,
                    "trajectory_confident_claims": 0,
                    "nontrajectory_confident_claims": 0,
                    "ambiguous_candidates": 0,
                    "incomplete_candidates": 0,
                    "strong_exact_candidates": 0,
                    "extreme_exact_candidates": 0,
                    "mean_exact_family_margin": np.nan,
                    "median_exact_family_margin": np.nan,
                    "min_exact_family_margin": np.nan,
                    "max_exact_family_margin": np.nan,
                    "candidate_sessions": 0,
                    "candidate_rats": 0,
                    "validation_status": "no_candidates_scored",
                    "paper_claim_guidance": "No promoted off-SWR candidates were available for exact-core validation.",
                }
            ],
            columns=list(SUMMARY_COLUMNS),
        )
    margins = pd.to_numeric(decisions["trajectory_minus_nontrajectory_log_evidence"], errors="coerce")
    complete = decisions["required_models_complete"].map(_as_bool)
    trajectory_claim = decisions["trajectory_confident_claim"].map(_as_bool)
    nontrajectory_claim = decisions["nontrajectory_confident_claim"].map(_as_bool)
    selected = int(len(decisions))
    required_complete = int(complete.sum())
    trajectory_count = int(trajectory_claim.sum())
    nontrajectory_count = int(nontrajectory_claim.sum())
    strong_count = int((margins >= 50.0).sum())
    extreme_count = int((margins >= 100.0).sum())
    if required_complete == 0:
        status = "exact_core_validation_incomplete"
        guidance = "Do not claim exact-core support; required model scores are incomplete."
    elif trajectory_count > 0 and nontrajectory_count == 0:
        status = "exact_core_supports_promoted_off_swr_candidates"
        guidance = "Paper-safe wording can describe exact-core support for a strict promoted off-SWR candidate subset."
    elif trajectory_count > 0:
        status = "mixed_exact_core_promoted_candidate_support"
        guidance = "Keep the result exploratory; exact-core support is mixed across promoted candidates."
    else:
        status = "exact_core_does_not_promote_candidates"
        guidance = "Do not make an off-SWR replay claim from this promoted candidate subset."
    row = {
        "comparison_scope": comparison_scope,
        "candidate_filter": candidate_filter,
        "selected_candidates": selected,
        "scored_candidates": selected,
        "required_complete_candidates": required_complete,
        "trajectory_confident_claims": trajectory_count,
        "nontrajectory_confident_claims": nontrajectory_count,
        "ambiguous_candidates": int((decisions["margin_decision"].astype(str) == "ambiguous").sum()),
        "incomplete_candidates": int((decisions["margin_decision"].astype(str) == "incomplete_core").sum()),
        "strong_exact_candidates": strong_count,
        "extreme_exact_candidates": extreme_count,
        "mean_exact_family_margin": float(margins.mean()) if margins.notna().any() else np.nan,
        "median_exact_family_margin": float(margins.median()) if margins.notna().any() else np.nan,
        "min_exact_family_margin": float(margins.min()) if margins.notna().any() else np.nan,
        "max_exact_family_margin": float(margins.max()) if margins.notna().any() else np.nan,
        "candidate_sessions": int(decisions["session"].nunique()),
        "candidate_rats": int(decisions["session"].map(_rat_from_session).nunique()),
        "validation_status": status,
        "paper_claim_guidance": guidance,
    }
    return pd.DataFrame([row], columns=list(SUMMARY_COLUMNS))


def validation_group_summary(
    decisions: pd.DataFrame,
    *,
    candidate_filter: str,
    comparison_scope: str,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=list(GROUP_SUMMARY_COLUMNS))
    frame = decisions.copy()
    frame["rat"] = frame["session"].map(_rat_from_session)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(list(group_cols), sort=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_values = {column: value for column, value in zip(group_cols, keys_tuple, strict=True)}
        margins = pd.to_numeric(group["trajectory_minus_nontrajectory_log_evidence"], errors="coerce")
        rows.append(
            {
                "comparison_scope": comparison_scope,
                "candidate_filter": candidate_filter,
                "rat": str(key_values.get("rat", "")),
                "group": "/".join(str(key_values[column]) for column in group_cols),
                "selected_candidates": int(len(group)),
                "required_complete_candidates": int(group["required_models_complete"].map(_as_bool).sum()),
                "trajectory_confident_claims": int(group["trajectory_confident_claim"].map(_as_bool).sum()),
                "nontrajectory_confident_claims": int(group["nontrajectory_confident_claim"].map(_as_bool).sum()),
                "strong_exact_candidates": int((margins >= 50.0).sum()),
                "extreme_exact_candidates": int((margins >= 100.0).sum()),
                "median_exact_family_margin": float(margins.median()) if margins.notna().any() else np.nan,
                "min_exact_family_margin": float(margins.min()) if margins.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=list(GROUP_SUMMARY_COLUMNS))


def validation_gate_summary(candidates: pd.DataFrame, scores: pd.DataFrame, decisions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(gate: str, passed: bool, observed: object, criterion: str, *, required: bool = True) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required),
            }
        )

    add("candidate_input_present", not candidates.empty, int(len(candidates)), "one or more candidate rows selected for validation")
    scored_windows = int(scores.drop_duplicates(["session", "event_index", "window_role", "null_index"]).shape[0]) if not scores.empty else 0
    add("selected_candidates_scored", scored_windows == len(candidates), f"{scored_windows}/{len(candidates)}", "every selected candidate has score rows")
    required_complete = int(decisions["required_models_complete"].map(_as_bool).sum()) if not decisions.empty else 0
    add("required_models_complete", decisions.empty or required_complete == len(decisions), f"{required_complete}/{len(decisions)}", "selected candidates have complete required model evidence")
    status = str(summary.iloc[0]["validation_status"]) if not summary.empty else "missing_summary"
    add(
        "exact_core_candidate_support_reported",
        status in {
            "exact_core_supports_promoted_off_swr_candidates",
            "mixed_exact_core_promoted_candidate_support",
            "exact_core_does_not_promote_candidates",
        },
        status,
        "validation status is reported from exact-core decisions",
        required=False,
    )
    required_rows = [row for row in rows if row["required_for_overall"]]
    add("overall", all(row["passed"] for row in required_rows), f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed", "all required validation infrastructure gates pass")
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_validation_outputs(
    *,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
    comparison_scope: str,
    candidate_filter: str,
    required_models: tuple[str, ...] | None,
    margin_threshold: float,
) -> dict[str, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    decisions = validation_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    summary = validation_summary(decisions, candidate_filter=candidate_filter, comparison_scope=comparison_scope)
    session_summary = validation_group_summary(
        decisions,
        candidate_filter=candidate_filter,
        comparison_scope=comparison_scope,
        group_cols=("rat", "session"),
    )
    rat_summary = validation_group_summary(
        decisions,
        candidate_filter=candidate_filter,
        comparison_scope=comparison_scope,
        group_cols=("rat",),
    )
    gates = validation_gate_summary(candidates, scores, decisions, summary)
    outputs = {
        "promoted_off_swr_candidate_exact_core_event_model_evidence.csv": scores,
        "promoted_off_swr_candidate_exact_core_decisions.csv": decisions,
        "promoted_off_swr_candidate_exact_core_summary.csv": summary,
        "promoted_off_swr_candidate_exact_core_session_summary.csv": session_summary,
        "promoted_off_swr_candidate_exact_core_rat_summary.csv": rat_summary,
        "promoted_off_swr_candidate_exact_core_gate_summary.csv": gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    manifest = {
        "comparison_scope": comparison_scope,
        "candidate_filter": candidate_filter,
        "required_models": list(required_models or FULL_CORE_REQUIRED_MODELS),
        "margin_threshold": float(margin_threshold),
        "selected_candidates": int(len(candidates)),
        "score_rows": int(len(scores)),
        "decision_rows": int(len(decisions)),
        "output_files": sorted(outputs),
    }
    (output / "promoted_off_swr_candidate_exact_core_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--output", default="results/off-swr-promoted-candidate-validation")
    parser.add_argument("--models", default=DEFAULT_VALIDATION_MODELS)
    parser.add_argument(
        "--candidate-filter",
        choices=("promotion-ready", "strong-immobile", "all-high-specificity", "all"),
        default="promotion-ready",
    )
    parser.add_argument("--session-filter", default="")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--nulls-per-event", type=int, default=0)
    parser.add_argument("--null-random-seed", type=int, default=1)
    parser.add_argument("--spike-count-tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--active-cell-tolerance", type=int)
    parser.add_argument("--null-candidate-step-s", type=float)
    parser.add_argument("--max-null-candidate-windows", type=int, default=DEFAULT_MAX_NON_RUN_CANDIDATE_WINDOWS)
    parser.add_argument("--swr-exclusion-padding-s", type=float, default=0.0)
    parser.add_argument("--allow-non-run-nulls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--required-models", default="")
    _add_comparison_scope_argument(parser)
    parser.set_defaults(comparison_scope="full-core")
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--continue-on-error", action="store_true")
    _add_model_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_table = pd.read_csv(args.candidate_table)
    candidates = select_candidate_windows(
        candidate_table,
        candidate_filter=args.candidate_filter,
        session_filter=_parse_names(args.session_filter),
        max_candidates=args.max_candidates,
    )
    scores = score_promoted_candidates(args, candidates)
    outputs = write_validation_outputs(
        candidates=candidates,
        scores=scores,
        output=Path(args.output),
        comparison_scope=args.comparison_scope,
        candidate_filter=args.candidate_filter,
        required_models=_parse_required_models(args.required_models),
        margin_threshold=args.margin_threshold,
    )
    print(outputs["promoted_off_swr_candidate_exact_core_summary.csv"].to_string(index=False))
    print(outputs["promoted_off_swr_candidate_exact_core_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
