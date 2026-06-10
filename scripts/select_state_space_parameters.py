#!/usr/bin/env python3
"""Select state-space replay parameters from evidence and recovery sweeps."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

import pandas as pd


BASE_PARAMETER_COLUMNS = [
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_candidate_top_k",
]
REQUIRED_PARAMETER_COLUMNS = [
    column
    for column in BASE_PARAMETER_COLUMNS
    if column != "state_space_momentum_candidate_top_k"
]
OPTIONAL_PARAMETER_DEFAULTS = {
    # Older recovery/evidence artifacts predate these support controls.  Default
    # them explicitly so historical rows remain loadable while new sweeps are
    # not accidentally pooled across different candidate-support mechanisms.
    "state_space_momentum_candidate_top_k": 128,
    "state_space_max_step_sigma": 4.0,
    "state_space_valid_occupancy_threshold_s": 0.0,
    "state_space_momentum_predicted_candidate_top_k": 8,
    "state_space_momentum_candidate_source": "emission",
    "state_space_displacement_radius_bins": 2,
    "state_space_displacement_position_sigma_cm": 0.0,
    "state_space_displacement_transition_sigma_cm_sqrt_s": 0.0,
    "state_space_displacement_prior_sigma_cm": 0.0,
    "time_bin_s": 0.003,
    "spike_rate_scale": 1.0,
    "emission_likelihood_temperature": 1.0,
    "emission_negative_binomial_overdispersion": 0.0,
}
INTEGER_PARAMETER_COLUMNS = {
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_displacement_radius_bins",
}
PARAMETER_COLUMNS = [
    *BASE_PARAMETER_COLUMNS,
    "state_space_max_step_sigma",
    "state_space_valid_occupancy_threshold_s",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_momentum_candidate_source",
    "state_space_displacement_radius_bins",
    "state_space_displacement_position_sigma_cm",
    "state_space_displacement_transition_sigma_cm_sqrt_s",
    "state_space_displacement_prior_sigma_cm",
    "time_bin_s",
    "spike_rate_scale",
    "emission_likelihood_temperature",
    "emission_negative_binomial_overdispersion",
]

SESSION_COLUMN_CANDIDATES = ["requested_session", "session"]

EVIDENCE_COUNT_COLUMNS = [
    "momentum_confidence_claim_events",
    "momentum_confidence_reference_events",
    "momentum_confidence_ambiguous_events",
    "momentum_beats_diffusion_log5_events",
    "momentum_ambiguous_vs_diffusion_log5_events",
    "momentum_beats_diffusion_events",
    "momentum_beats_imm_events",
]
EVIDENCE_MEAN_COLUMNS = [
    "mean_momentum_minus_diffusion_log_evidence",
    "mean_momentum_minus_imm_log_evidence",
]
EVIDENCE_MEDIAN_COLUMNS = [
    "median_momentum_minus_diffusion_log_evidence",
    "median_momentum_minus_imm_log_evidence",
]
CONFIDENCE_RECOVERY_COLUMNS = [
    "momentum_confidence_recovery_thresholded_binary_accuracy",
    "momentum_confidence_recovery_positive_claim_recall",
    "momentum_confidence_recovery_reference_specificity",
    "momentum_confidence_recovery_false_positive_claims",
    "momentum_confidence_recovery_false_negative_claims",
    "momentum_confidence_recovery_passes_threshold_gate",
    "momentum_confidence_recovery_selection_status",
    "momentum_confidence_recovery_threshold_scope",
]
RECOVERY_COUNT_COLUMNS = [
    "failures",
    "momentum_recovered_events",
    "overall_recovered_events",
    "overall_simulated_events",
    "momentum_certified_vs_exact_recovered_events",
    "overall_certified_vs_exact_recovered_events",
    "diffusion_certified_vs_exact_recovered_events",
    "exact_displacement_momentum_scored_events",
    "exact_displacement_momentum_beats_diffusion_events",
]
RECOVERY_ACCURACY_COLUMNS = [
    "overall_recovery_accuracy",
    "momentum_recovery_accuracy",
    "diffusion_recovery_accuracy",
    "overall_certified_vs_exact_recovery_accuracy",
    "momentum_certified_vs_exact_recovery_accuracy",
    "diffusion_certified_vs_exact_recovery_accuracy",
    "overall_oracle_candidate_recovery_accuracy",
    "momentum_oracle_candidate_recovery_accuracy",
    "diffusion_oracle_candidate_recovery_accuracy",
    "overall_oracle_candidate_certified_vs_exact_recovery_accuracy",
    "momentum_oracle_candidate_certified_vs_exact_recovery_accuracy",
    "diffusion_oracle_candidate_certified_vs_exact_recovery_accuracy",
    "exact_displacement_momentum_strict_best_fraction",
    "exact_displacement_momentum_beats_diffusion_fraction",
]


def select_parameters(
    evidence: str | Path,
    recovery: str | Path,
    *,
    output: str | Path,
    confidence_evidence: str | Path | None = None,
    min_momentum_recovery_accuracy: float = 0.5,
    min_overall_recovery_accuracy: float = 0.5,
    max_failures: int = 0,
    leave_one_session_out: bool = False,
    session_column: str = "requested_session",
    holdout_sessions: Sequence[str] | None = None,
    recovery_gate_metric: str = "auto",
    force_strict_recovery_gate: bool = False,
) -> dict[str, pd.DataFrame]:
    evidence_frame = _load_table(evidence, "state_space_evidence_sweep_config_ranked.csv")
    recovery_frame = _load_table(recovery, "simulation_recovery_sweep_config_ranked.csv")
    _inherit_single_evidence_parameter_defaults(recovery_frame, evidence_frame)
    if confidence_evidence is not None:
        evidence_frame = _merge_confidence_evidence(
            evidence_frame,
            _load_confidence_evidence(confidence_evidence),
        )

    tables = _select_from_frames(
        evidence_frame,
        recovery_frame,
        min_momentum_recovery_accuracy=min_momentum_recovery_accuracy,
        min_overall_recovery_accuracy=min_overall_recovery_accuracy,
        max_failures=max_failures,
        recovery_gate_metric=recovery_gate_metric,
        force_strict_recovery_gate=force_strict_recovery_gate,
    )
    ranked = tables["decision"]
    candidates = tables["candidates"]
    recommendation = tables["recommendation"]

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "state_space_parameter_decision_table.csv", index=False)
    candidates.to_csv(out_dir / "state_space_parameter_candidates.csv", index=False)
    recommendation.to_csv(out_dir / "state_space_parameter_recommendation.csv", index=False)
    _write_selected_parameter_files(recommendation, out_dir)

    loso_recommendations = pd.DataFrame()
    if leave_one_session_out:
        loso_recommendations = _select_leave_one_session_out(
            evidence_frame,
            recovery_frame,
            session_column=session_column,
            holdout_sessions=holdout_sessions,
            min_momentum_recovery_accuracy=min_momentum_recovery_accuracy,
            min_overall_recovery_accuracy=min_overall_recovery_accuracy,
            max_failures=max_failures,
            recovery_gate_metric=recovery_gate_metric,
            force_strict_recovery_gate=force_strict_recovery_gate,
        )
        loso_recommendations.to_csv(out_dir / "state_space_loso_parameter_recommendations.csv", index=False)
        _write_loso_parameter_files(loso_recommendations, out_dir)

    _write_selection_manifest(
        evidence=evidence,
        recovery=recovery,
        confidence_evidence=confidence_evidence,
        output_dir=out_dir,
        decision=ranked,
        candidates=candidates,
        recommendation=recommendation,
        min_momentum_recovery_accuracy=min_momentum_recovery_accuracy,
        min_overall_recovery_accuracy=min_overall_recovery_accuracy,
        max_failures=max_failures,
        leave_one_session_out=leave_one_session_out,
        recovery_gate_metric=recovery_gate_metric,
        session_column=session_column,
        holdout_sessions=holdout_sessions,
        force_strict_recovery_gate=force_strict_recovery_gate,
        loso_recommendations=loso_recommendations,
    )
    result = {
        "decision": ranked,
        "candidates": candidates,
        "recommendation": recommendation,
    }
    if leave_one_session_out:
        result["leave_one_session_out"] = loso_recommendations
    return result


def _select_from_frames(
    evidence_frame: pd.DataFrame,
    recovery_frame: pd.DataFrame,
    *,
    min_momentum_recovery_accuracy: float,
    min_overall_recovery_accuracy: float,
    max_failures: int,
    recovery_gate_metric: str,
    force_strict_recovery_gate: bool,
) -> dict[str, pd.DataFrame]:
    evidence_prepared = _aggregate_evidence(_prepare_evidence(evidence_frame))
    recovery_prepared = _aggregate_recovery(_prepare_recovery(recovery_frame))
    decision = _build_decision_table(
        evidence_prepared,
        recovery_prepared,
        min_momentum_recovery_accuracy=min_momentum_recovery_accuracy,
        min_overall_recovery_accuracy=min_overall_recovery_accuracy,
        max_failures=max_failures,
        recovery_gate_metric=recovery_gate_metric,
        force_strict_recovery_gate=force_strict_recovery_gate,
    )

    ranked = _rank_decision_table(decision)
    ranked.insert(0, "recommendation_rank", range(1, len(ranked) + 1))
    candidates = ranked[ranked["is_recommendable"]].copy()
    recommendation = candidates.head(1).copy()
    if recommendation.empty and not ranked.empty:
        recommendation = ranked.head(1).copy()
        recommendation["recommendation_note"] = "No configuration passed the recovery gate; showing the best available row."
    elif not recommendation.empty:
        recommendation["recommendation_note"] = "Top configuration passing the recovery gate."
    if not recommendation.empty and "recovery_gate_warning" in recommendation:
        warning = str(recommendation.iloc[0].get("recovery_gate_warning", "")).strip()
        if warning:
            recommendation["recommendation_note"] = (
                recommendation["recommendation_note"].astype(str) + " " + warning
            )
    return {
        "decision": ranked,
        "candidates": candidates,
        "recommendation": recommendation,
    }


def _build_decision_table(
    evidence_frame: pd.DataFrame,
    recovery_frame: pd.DataFrame,
    *,
    min_momentum_recovery_accuracy: float,
    min_overall_recovery_accuracy: float,
    max_failures: int,
    recovery_gate_metric: str,
    force_strict_recovery_gate: bool,
) -> pd.DataFrame:
    decision = evidence_frame.merge(
        recovery_frame,
        on=PARAMETER_COLUMNS,
        how="outer",
        suffixes=("_evidence", "_recovery"),
        indicator="source_match",
    )
    decision["has_evidence"] = decision["source_match"].isin(["both", "left_only"])
    decision["has_recovery"] = decision["source_match"].isin(["both", "right_only"])
    gate_columns = _recovery_gate_columns(decision, recovery_gate_metric)
    momentum_gate_col, overall_gate_col, diffusion_gate_col, resolved_gate_metric = gate_columns
    decision["recovery_gate_metric"] = resolved_gate_metric
    decision["gate_momentum_recovery_accuracy"] = decision.get(
        momentum_gate_col, pd.Series(float("nan"), index=decision.index)
    )
    decision["gate_overall_recovery_accuracy"] = decision.get(
        overall_gate_col, pd.Series(float("nan"), index=decision.index)
    )
    decision["gate_diffusion_recovery_accuracy"] = decision.get(
        diffusion_gate_col, pd.Series(float("nan"), index=decision.index)
    )
    decision["momentum_recovery_gate_column"] = momentum_gate_col
    decision["overall_recovery_gate_column"] = overall_gate_col
    decision["diffusion_recovery_gate_column"] = diffusion_gate_col
    decision["strict_momentum_recovery_accuracy"] = decision.get(
        "momentum_recovery_accuracy", pd.Series(float("nan"), index=decision.index)
    )
    decision["strict_overall_recovery_accuracy"] = decision.get(
        "overall_recovery_accuracy", pd.Series(float("nan"), index=decision.index)
    )
    decision["strict_diffusion_recovery_accuracy"] = decision.get(
        "diffusion_recovery_accuracy", pd.Series(float("nan"), index=decision.index)
    )
    decision["certified_vs_exact_momentum_recovery_accuracy"] = decision.get(
        "momentum_certified_vs_exact_recovery_accuracy",
        pd.Series(float("nan"), index=decision.index),
    )
    decision["certified_vs_exact_overall_recovery_accuracy"] = decision.get(
        "overall_certified_vs_exact_recovery_accuracy",
        pd.Series(float("nan"), index=decision.index),
    )
    decision["certified_vs_exact_diffusion_recovery_accuracy"] = decision.get(
        "diffusion_certified_vs_exact_recovery_accuracy",
        pd.Series(float("nan"), index=decision.index),
    )
    decision["exact_displacement_momentum_beats_diffusion_fraction"] = decision.get(
        "exact_displacement_momentum_beats_diffusion_fraction",
        pd.Series(float("nan"), index=decision.index),
    )
    decision["exact_displacement_momentum_strict_best_fraction"] = decision.get(
        "exact_displacement_momentum_strict_best_fraction",
        pd.Series(float("nan"), index=decision.index),
    )
    candidate_top_k = pd.to_numeric(
        decision.get(
            "state_space_momentum_candidate_top_k",
            pd.Series(0, index=decision.index),
        ),
        errors="coerce",
    ).fillna(0)
    momentum_best_model = decision.get(
        "momentum_most_common_best_model",
        pd.Series("", index=decision.index),
    ).fillna("").astype(str)
    exact_sparse_momentum_best = momentum_best_model.str.contains(
        "momentum-exact-sparse",
        regex=False,
    )
    certified_columns_available = _certified_recovery_columns_available(decision)
    decision["uses_candidate_pruned_momentum"] = (candidate_top_k > 0) & ~exact_sparse_momentum_best
    decision["certified_recovery_columns_available"] = bool(certified_columns_available)
    decision["recovery_gate_warning"] = ""
    strict_candidate_rows = (
        decision["uses_candidate_pruned_momentum"]
        & (decision["recovery_gate_metric"] == "strict")
    )
    decision.loc[
        strict_candidate_rows & ~decision["certified_recovery_columns_available"],
        "recovery_gate_warning",
    ] = "Candidate-pruned momentum/IMM recovery is being gated with strict exact-comparable recovery because certified-vs-exact columns are missing."
    decision.loc[
        strict_candidate_rows
        & decision["certified_recovery_columns_available"],
        "recovery_gate_warning",
    ] = "Strict gate was requested even though certified-vs-exact recovery columns are available for candidate-pruned momentum/IMM."
    decision["strict_candidate_recovery_gate_blocked"] = (
        strict_candidate_rows & (not force_strict_recovery_gate)
    )
    decision.loc[
        decision["strict_candidate_recovery_gate_blocked"],
        "recovery_gate_warning",
    ] = (
        "Candidate-pruned momentum/IMM rows are not eligible under a strict recovery gate. "
        "Use certified-vs-exact recovery columns, run full-grid/oracle-support recovery, "
        "or pass --force-strict-recovery-gate for exploratory legacy selection."
    )
    decision["passes_recovery_gate"] = (
        decision["has_recovery"]
        & ~decision["strict_candidate_recovery_gate_blocked"]
        & (decision.get("failures", pd.Series(0, index=decision.index)).fillna(0) <= max_failures)
        & (
            decision["gate_momentum_recovery_accuracy"].fillna(-1.0)
            >= min_momentum_recovery_accuracy
        )
        & (
            decision["gate_overall_recovery_accuracy"].fillna(-1.0)
            >= min_overall_recovery_accuracy
        )
    )
    decision["is_recommendable"] = decision["has_evidence"] & decision["passes_recovery_gate"]
    decision["recovery_gate"] = decision["passes_recovery_gate"].map({True: "pass", False: "fail"})
    decision.loc[~decision["has_recovery"], "recovery_gate"] = "missing-recovery"
    decision.loc[~decision["has_evidence"], "recovery_gate"] = "missing-evidence"
    decision.loc[
        decision["strict_candidate_recovery_gate_blocked"] & decision["has_recovery"] & decision["has_evidence"],
        "recovery_gate",
    ] = "strict-gate-blocked"
    return decision


def _recovery_gate_columns(frame: pd.DataFrame, metric: str) -> tuple[str, str, str, str]:
    """Return momentum/overall/diffusion recovery columns and the resolved metric."""

    normalized = metric.strip().lower().replace("_", "-")
    aliases = {
        "certified": "certified-vs-exact",
        "certified-vs-comparable": "certified-vs-exact",
        "lower-bound-certified": "certified-vs-exact",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"auto", "strict", "certified-vs-exact"}
    if normalized not in allowed:
        raise ValueError(f"recovery_gate_metric must be one of {sorted(allowed)}")

    certified = (
        "momentum_certified_vs_exact_recovery_accuracy",
        "overall_certified_vs_exact_recovery_accuracy",
        "diffusion_certified_vs_exact_recovery_accuracy",
    )
    strict = (
        "momentum_recovery_accuracy",
        "overall_recovery_accuracy",
        "diffusion_recovery_accuracy",
    )
    has_certified = (
        certified[0] in frame.columns
        and certified[1] in frame.columns
        and frame[certified[0]].notna().any()
        and frame[certified[1]].notna().any()
    )
    if normalized == "certified-vs-exact" and not has_certified:
        raise ValueError("certified-vs-exact recovery columns are absent from the recovery table")
    if normalized == "certified-vs-exact" or (normalized == "auto" and has_certified):
        return (*certified, "certified-vs-exact")
    return (*strict, "strict")


def _certified_recovery_columns_available(frame: pd.DataFrame) -> bool:
    required = (
        "momentum_certified_vs_exact_recovery_accuracy",
        "overall_certified_vs_exact_recovery_accuracy",
    )
    return all(
        column in frame.columns and frame[column].notna().any()
        for column in required
    )


def _select_leave_one_session_out(
    evidence_frame: pd.DataFrame,
    recovery_frame: pd.DataFrame,
    *,
    session_column: str,
    holdout_sessions: Sequence[str] | None,
    min_momentum_recovery_accuracy: float,
    min_overall_recovery_accuracy: float,
    max_failures: int,
    recovery_gate_metric: str,
    force_strict_recovery_gate: bool,
) -> pd.DataFrame:
    if session_column not in evidence_frame.columns:
        available = [col for col in SESSION_COLUMN_CANDIDATES if col in evidence_frame.columns]
        detail = f" Available session-like columns: {available}." if available else ""
        raise ValueError(f"Cannot run leave-one-session-out selection: {session_column!r} is absent from evidence.{detail}")

    holdouts = _resolve_holdout_sessions(evidence_frame[session_column], holdout_sessions)
    rows: list[dict[str, object]] = []
    for holdout in holdouts:
        evidence_sessions = evidence_frame[session_column].astype(str)
        train_mask = evidence_sessions != str(holdout)
        train_evidence = evidence_frame.loc[train_mask].copy()
        train_sessions = _join_unique(train_evidence[session_column]) if not train_evidence.empty else ""
        row_prefix = {
            "held_out_session": str(holdout),
            "train_session_count": int(train_evidence[session_column].nunique()) if not train_evidence.empty else 0,
            "train_sessions": train_sessions,
            "selection_scope": f"train_without_{holdout}",
        }
        if train_evidence.empty:
            rows.append(
                {
                    **row_prefix,
                    "recommendation_note": "No training sessions were available after holding out this session.",
                }
            )
            continue

        train_recovery = recovery_frame.copy()
        if session_column in train_recovery.columns:
            recovery_sessions = train_recovery[session_column].astype(str)
            filtered_recovery = train_recovery.loc[recovery_sessions != str(holdout)].copy()
            if not filtered_recovery.empty:
                train_recovery = filtered_recovery

        tables = _select_from_frames(
            train_evidence,
            train_recovery,
            min_momentum_recovery_accuracy=min_momentum_recovery_accuracy,
            min_overall_recovery_accuracy=min_overall_recovery_accuracy,
            max_failures=max_failures,
            recovery_gate_metric=recovery_gate_metric,
            force_strict_recovery_gate=force_strict_recovery_gate,
        )
        recommendation = tables["recommendation"]
        if recommendation.empty:
            rows.append({**row_prefix, "recommendation_note": "No recommendation was available for this fold."})
            continue
        rows.append({**row_prefix, **recommendation.iloc[0].to_dict()})
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    preferred_order = [
        "held_out_session",
        "train_session_count",
        "train_sessions",
        "selection_scope",
        "recommendation_rank",
        *PARAMETER_COLUMNS,
        "is_recommendable",
        "recovery_gate",
        "recommendation_note",
    ]
    ordered = [col for col in preferred_order if col in result.columns]
    return result[ordered + [col for col in result.columns if col not in ordered]]


def _load_table(path: str | Path, default_name: str) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = _find_ranked_table(path, default_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    frame = pd.read_csv(path)
    if "time_bin_s" not in frame.columns and "time_bin_ms" in frame.columns:
        frame["time_bin_s"] = pd.to_numeric(frame["time_bin_ms"], errors="raise") / 1000.0
    missing = set(REQUIRED_PARAMETER_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required parameter columns: {sorted(missing)}")
    missing_parameter_columns = sorted(set(PARAMETER_COLUMNS) - set(frame.columns))
    for column, default in OPTIONAL_PARAMETER_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
    frame.attrs["source_path"] = str(path)
    frame.attrs["missing_parameter_columns"] = missing_parameter_columns
    return frame


def _find_ranked_table(root: Path, default_name: str) -> Path:
    candidates = [default_name]
    if default_name == "simulation_recovery_sweep_config_ranked.csv":
        candidates.extend(
            [
                "simulation_recovery_emission_calibration_config_ranked.csv",
                "simulation_recovery_sweep_seed_replicated_ranked.csv",
                "simulation_recovery_emission_calibration_seed_replicated_ranked.csv",
            ]
        )
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return root / default_name


def _inherit_single_evidence_parameter_defaults(
    recovery: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    """Fill missing recovery parameter columns from single-valued evidence axes."""

    missing = set(recovery.attrs.get("missing_parameter_columns", []))
    for column in sorted(missing.intersection(PARAMETER_COLUMNS)):
        if column not in evidence.columns:
            continue
        values = evidence[column].dropna().unique()
        if len(values) == 1:
            recovery[column] = values[0]


def _load_confidence_evidence(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        evidence_path = path / "momentum_confidence_threshold_evidence_by_stratum.csv"
        selection_path = path / "momentum_confidence_threshold_selection.csv"
        if not evidence_path.exists():
            raise FileNotFoundError(f"{evidence_path} does not exist")
        frame = pd.read_csv(evidence_path)
        if selection_path.exists():
            frame = _merge_confidence_recovery_selection(frame, pd.read_csv(selection_path))
        return frame
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    frame = pd.read_csv(path)
    return frame


def _merge_confidence_evidence(evidence: pd.DataFrame, confidence: pd.DataFrame) -> pd.DataFrame:
    evidence, confidence, keys = _confidence_merge_inputs(evidence, confidence)
    columns = {
        "margin_threshold": "momentum_confidence_threshold",
        "positive_model_claims": "momentum_confidence_claim_events",
        "reference_model_claims": "momentum_confidence_reference_events",
        "ambiguous_events": "momentum_confidence_ambiguous_events",
        "positive_claim_fraction": "momentum_confidence_claim_event_fraction",
        "momentum_confidence_recovery_thresholded_binary_accuracy": (
            "momentum_confidence_recovery_thresholded_binary_accuracy"
        ),
        "momentum_confidence_recovery_positive_claim_recall": (
            "momentum_confidence_recovery_positive_claim_recall"
        ),
        "momentum_confidence_recovery_reference_specificity": (
            "momentum_confidence_recovery_reference_specificity"
        ),
        "momentum_confidence_recovery_false_positive_claims": (
            "momentum_confidence_recovery_false_positive_claims"
        ),
        "momentum_confidence_recovery_false_negative_claims": (
            "momentum_confidence_recovery_false_negative_claims"
        ),
        "momentum_confidence_recovery_passes_threshold_gate": (
            "momentum_confidence_recovery_passes_threshold_gate"
        ),
        "momentum_confidence_recovery_selection_status": (
            "momentum_confidence_recovery_selection_status"
        ),
        "momentum_confidence_recovery_threshold_scope": (
            "momentum_confidence_recovery_threshold_scope"
        ),
    }
    available = {source: target for source, target in columns.items() if source in confidence.columns}
    if "positive_model_claims" not in available:
        raise ValueError("confidence evidence must include positive_model_claims")
    confidence_subset = confidence[list(dict.fromkeys([*keys, *available]))].rename(columns=available)
    if not keys:
        merged = evidence.copy()
        row = confidence_subset.iloc[0]
        for column in confidence_subset.columns:
            merged[column] = row[column]
        return merged
    if confidence_subset.duplicated(keys).any():
        duplicates = confidence_subset.loc[confidence_subset.duplicated(keys, keep=False), keys].drop_duplicates()
        raise ValueError(f"confidence evidence has duplicate merge keys: {duplicates.to_dict(orient='records')}")
    merged = evidence.merge(confidence_subset, on=keys, how="left", validate="many_to_one")
    missing = merged["momentum_confidence_claim_events"].isna()
    if missing.any():
        missing_keys = merged.loc[missing, keys].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"confidence evidence is missing rows for merge keys: {missing_keys}")
    return merged


def _merge_confidence_recovery_selection(confidence: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    selection_columns = {
        "thresholded_binary_accuracy": "momentum_confidence_recovery_thresholded_binary_accuracy",
        "positive_claim_recall": "momentum_confidence_recovery_positive_claim_recall",
        "reference_specificity": "momentum_confidence_recovery_reference_specificity",
        "false_positive_claims": "momentum_confidence_recovery_false_positive_claims",
        "false_negative_claims": "momentum_confidence_recovery_false_negative_claims",
        "passes_threshold_gate": "momentum_confidence_recovery_passes_threshold_gate",
        "selection_status": "momentum_confidence_recovery_selection_status",
        "threshold_scope": "momentum_confidence_recovery_threshold_scope",
    }
    available = {source: target for source, target in selection_columns.items() if source in selection.columns}
    if not available:
        return confidence
    confidence, selection, keys = _confidence_merge_inputs(confidence, selection)
    subset = selection[list(dict.fromkeys([*keys, *available]))].rename(columns=available)
    if not keys:
        out = confidence.copy()
        row = subset.iloc[0]
        for column in subset.columns:
            out[column] = row[column]
        return out
    if subset.duplicated(keys).any():
        duplicates = subset.loc[subset.duplicated(keys, keep=False), keys].drop_duplicates()
        raise ValueError(f"confidence threshold selection has duplicate merge keys: {duplicates.to_dict(orient='records')}")
    return confidence.merge(subset, on=keys, how="left", validate="many_to_one")


def _confidence_merge_inputs(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keys = _confidence_merge_keys(left, right)
    left = left.copy()
    right = right.copy()
    if keys and keys != ["matrix_id"]:
        _normalize_parameter_columns(left)
        _normalize_parameter_columns(right)
    return left, right, keys


def _confidence_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "matrix_id" in left.columns and "matrix_id" in right.columns:
        return ["matrix_id"]
    keys = [column for column in PARAMETER_COLUMNS if column in left.columns and column in right.columns]
    if keys:
        return keys
    if len(right) == 1:
        return []
    raise ValueError("calibrated confidence evidence must share matrix_id or parameter columns")


def _prepare_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _normalize_parameter_columns(frame)
    if "events" in frame and "momentum_confidence_claim_events" in frame:
        events = frame["events"].replace(0, pd.NA)
        frame["momentum_confidence_claim_event_fraction"] = (
            frame["momentum_confidence_claim_events"] / events
        )
    if "events" in frame and "momentum_beats_diffusion_log5_events" in frame:
        events = frame["events"].replace(0, pd.NA)
        frame["momentum_beats_diffusion_log5_event_fraction"] = (
            frame["momentum_beats_diffusion_log5_events"] / events
        )
    if "events" in frame and "momentum_beats_diffusion_events" in frame:
        events = frame["events"].replace(0, pd.NA)
        frame["momentum_beats_diffusion_event_fraction"] = frame["momentum_beats_diffusion_events"] / events
    keep = PARAMETER_COLUMNS + _present_session_columns(frame) + [
        col
        for col in [
            "matrix_id",
            "events",
            "momentum_confidence_threshold",
            "momentum_confidence_claim_events",
            "momentum_confidence_claim_event_fraction",
            "momentum_confidence_reference_events",
            "momentum_confidence_ambiguous_events",
            "momentum_beats_diffusion_log5_events",
            "momentum_beats_diffusion_log5_event_fraction",
            "momentum_ambiguous_vs_diffusion_log5_events",
            "momentum_beats_diffusion_events",
            "momentum_beats_diffusion_event_fraction",
            "mean_momentum_minus_diffusion_log_evidence",
            "median_momentum_minus_diffusion_log_evidence",
            "mean_momentum_minus_imm_log_evidence",
            "median_momentum_minus_imm_log_evidence",
            *CONFIDENCE_RECOVERY_COLUMNS,
        ]
        if col in frame.columns
    ]
    return frame[keep].rename(columns={"matrix_id": "evidence_matrix_id"})


def _prepare_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    _normalize_parameter_columns(frame)
    keep = PARAMETER_COLUMNS + _present_session_columns(frame) + [
        col
        for col in [
            "matrix_id",
            "failures",
            "overall_recovery_accuracy",
            "momentum_recovery_accuracy",
            "diffusion_recovery_accuracy",
            "momentum_recovered_events",
            "overall_recovered_events",
            "overall_simulated_events",
            "overall_certified_vs_exact_recovered_events",
            "momentum_certified_vs_exact_recovered_events",
            "diffusion_certified_vs_exact_recovered_events",
            "momentum_most_common_best_model",
            "overall_certified_vs_exact_recovery_accuracy",
            "momentum_certified_vs_exact_recovery_accuracy",
            "diffusion_certified_vs_exact_recovery_accuracy",
            "overall_oracle_candidate_recovery_accuracy",
            "momentum_oracle_candidate_recovery_accuracy",
            "diffusion_oracle_candidate_recovery_accuracy",
            "overall_oracle_candidate_certified_vs_exact_recovery_accuracy",
            "momentum_oracle_candidate_certified_vs_exact_recovery_accuracy",
            "diffusion_oracle_candidate_certified_vs_exact_recovery_accuracy",
            "exact_displacement_momentum_strict_best_fraction",
            "exact_displacement_momentum_beats_diffusion_fraction",
        ]
        if col in frame.columns
    ]
    return frame[keep].rename(columns={"matrix_id": "recovery_matrix_id"})


def _normalize_parameter_columns(frame: pd.DataFrame) -> None:
    for col in PARAMETER_COLUMNS:
        if col not in frame.columns and col in OPTIONAL_PARAMETER_DEFAULTS:
            frame[col] = OPTIONAL_PARAMETER_DEFAULTS[col]
        if col in OPTIONAL_PARAMETER_DEFAULTS:
            frame[col] = frame[col].where(frame[col].notna(), OPTIONAL_PARAMETER_DEFAULTS[col])
        if col in INTEGER_PARAMETER_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("int64")
        elif col == "state_space_momentum_candidate_source":
            frame[col] = frame[col].map(_normalize_candidate_source)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="raise").round(8)


def _present_session_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in SESSION_COLUMN_CANDIDATES if col in frame.columns]


def _normalize_candidate_source(value: object) -> str:
    if pd.isna(value):
        return str(OPTIONAL_PARAMETER_DEFAULTS["state_space_momentum_candidate_source"])
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "": "emission",
        "none": "emission",
        "null": "emission",
        "nan": "emission",
        "likelihood": "emission",
        "log-likelihood": "emission",
        "train-posterior": "posterior",
        "diffusion-posterior": "posterior",
        "first-order-posterior": "posterior",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"emission", "posterior"}:
        raise ValueError("state_space_momentum_candidate_source must be 'emission' or 'posterior'")
    return normalized


def _aggregate_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not frame.duplicated(PARAMETER_COLUMNS).any():
        return frame

    rows: list[dict[str, object]] = []
    for _, group in frame.groupby(PARAMETER_COLUMNS, sort=False, dropna=False):
        row: dict[str, object] = {col: group.iloc[0][col] for col in PARAMETER_COLUMNS}
        if "evidence_matrix_id" in group:
            row["evidence_matrix_id"] = _join_unique(group["evidence_matrix_id"])
        for col in _present_session_columns(group):
            row[f"evidence_{col}s"] = _join_unique(group[col])
        if "events" in group:
            row["events"] = int(pd.to_numeric(group["events"], errors="coerce").fillna(0).sum())
        for col in EVIDENCE_COUNT_COLUMNS:
            if col in group:
                row[col] = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
        if "events" in row and row["events"]:
            if "momentum_confidence_claim_events" in row:
                row["momentum_confidence_claim_event_fraction"] = (
                    row["momentum_confidence_claim_events"] / row["events"]
                )
            if "momentum_beats_diffusion_log5_events" in row:
                row["momentum_beats_diffusion_log5_event_fraction"] = (
                    row["momentum_beats_diffusion_log5_events"] / row["events"]
                )
            if "momentum_beats_diffusion_events" in row:
                row["momentum_beats_diffusion_event_fraction"] = row["momentum_beats_diffusion_events"] / row["events"]
        else:
            if "momentum_confidence_claim_event_fraction" in group:
                row["momentum_confidence_claim_event_fraction"] = _weighted_average(
                    group["momentum_confidence_claim_event_fraction"]
                )
            if "momentum_beats_diffusion_log5_event_fraction" in group:
                row["momentum_beats_diffusion_log5_event_fraction"] = _weighted_average(
                    group["momentum_beats_diffusion_log5_event_fraction"]
                )
            if "momentum_beats_diffusion_event_fraction" in group:
                row["momentum_beats_diffusion_event_fraction"] = _weighted_average(
                    group["momentum_beats_diffusion_event_fraction"]
                )
        weights = group["events"] if "events" in group else None
        for col in EVIDENCE_MEAN_COLUMNS:
            if col in group:
                row[col] = _weighted_average(group[col], weights)
        for col in EVIDENCE_MEDIAN_COLUMNS:
            if col in group:
                row[col] = _median(group[col])
        for col in CONFIDENCE_RECOVERY_COLUMNS:
            if col in group:
                row[col] = _unique_or_first(group[col])
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not frame.duplicated(PARAMETER_COLUMNS).any():
        return frame

    rows: list[dict[str, object]] = []
    for _, group in frame.groupby(PARAMETER_COLUMNS, sort=False, dropna=False):
        row: dict[str, object] = {col: group.iloc[0][col] for col in PARAMETER_COLUMNS}
        if "recovery_matrix_id" in group:
            row["recovery_matrix_id"] = _join_unique(group["recovery_matrix_id"])
        for col in _present_session_columns(group):
            row[f"recovery_{col}s"] = _join_unique(group[col])
        for col in RECOVERY_COUNT_COLUMNS:
            if col in group:
                row[col] = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
        weights = group["overall_simulated_events"] if "overall_simulated_events" in group else None
        for col in RECOVERY_ACCURACY_COLUMNS:
            if col in group:
                row[col] = _weighted_average(group[col], weights)
        if "momentum_most_common_best_model" in group:
            modes = group["momentum_most_common_best_model"].dropna().mode()
            row["momentum_most_common_best_model"] = None if modes.empty else modes.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _join_unique(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values.dropna() if str(value)})
    return ";".join(unique)


def _unique_or_first(values: pd.Series) -> object:
    non_missing = values.dropna()
    if non_missing.empty:
        return pd.NA
    unique = non_missing.unique()
    return unique[0]


def _weighted_average(values: pd.Series, weights: pd.Series | None = None) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    if not valid.any():
        return float("nan")
    if weights is not None:
        numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        positive = valid & (numeric_weights > 0)
        total_weight = numeric_weights[positive].sum()
        if total_weight > 0:
            return float((numeric[positive] * numeric_weights[positive]).sum() / total_weight)
    return float(numeric[valid].mean())


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def _resolve_holdout_sessions(values: pd.Series, requested: Sequence[str] | None) -> list[str]:
    available = [str(value) for value in values.dropna().unique()]
    if requested:
        requested_set = [str(value) for value in requested]
        missing = sorted(set(requested_set) - set(available))
        if missing:
            raise ValueError(f"Requested hold-out sessions are absent from evidence: {missing}")
        return requested_set
    return sorted(available)


def _rank_decision_table(frame: pd.DataFrame) -> pd.DataFrame:
    sortable = frame.copy()
    sort_columns = [
        "is_recommendable",
        "gate_momentum_recovery_accuracy",
        "gate_overall_recovery_accuracy",
        "gate_diffusion_recovery_accuracy",
        "momentum_recovery_accuracy",
        "overall_recovery_accuracy",
        "diffusion_recovery_accuracy",
        "momentum_confidence_recovery_passes_threshold_gate",
        "momentum_confidence_recovery_false_positive_claims",
        "momentum_confidence_recovery_thresholded_binary_accuracy",
        "momentum_confidence_recovery_reference_specificity",
        "momentum_confidence_recovery_positive_claim_recall",
        "momentum_confidence_recovery_false_negative_claims",
        "momentum_confidence_claim_event_fraction",
        "momentum_beats_diffusion_log5_event_fraction",
        "momentum_beats_diffusion_event_fraction",
        "median_momentum_minus_diffusion_log_evidence",
        "mean_momentum_minus_diffusion_log_evidence",
        "failures",
    ]
    ascending_by_column = {
        "is_recommendable": False,
        "gate_momentum_recovery_accuracy": False,
        "gate_overall_recovery_accuracy": False,
        "gate_diffusion_recovery_accuracy": False,
        "momentum_recovery_accuracy": False,
        "overall_recovery_accuracy": False,
        "diffusion_recovery_accuracy": False,
        "momentum_confidence_recovery_passes_threshold_gate": False,
        "momentum_confidence_recovery_false_positive_claims": True,
        "momentum_confidence_recovery_thresholded_binary_accuracy": False,
        "momentum_confidence_recovery_reference_specificity": False,
        "momentum_confidence_recovery_positive_claim_recall": False,
        "momentum_confidence_recovery_false_negative_claims": True,
        "momentum_confidence_claim_event_fraction": False,
        "momentum_beats_diffusion_log5_event_fraction": False,
        "momentum_beats_diffusion_event_fraction": False,
        "median_momentum_minus_diffusion_log_evidence": False,
        "mean_momentum_minus_diffusion_log_evidence": False,
        "failures": True,
    }
    present = [col for col in sort_columns if col in sortable.columns]
    sort_key_columns: list[str] = []
    for col in present:
        sort_col = f"__sort_{col}"
        sort_key_columns.append(sort_col)
        fill_value = float("inf") if ascending_by_column[col] else float("-inf")
        sortable[sort_col] = sortable[col].fillna(fill_value)
    ranked = sortable.sort_values(
        sort_key_columns,
        ascending=[ascending_by_column[col] for col in present],
        kind="stable",
    )
    return ranked.drop(columns=sort_key_columns)


def _write_selected_parameter_files(recommendation: pd.DataFrame, out_dir: Path) -> None:
    yaml_path = out_dir / "state_space_selected_workflow_inputs.yml"
    cli_path = out_dir / "state_space_selected_cli_args.txt"
    if recommendation.empty:
        yaml_path.write_text("# No state-space parameter recommendation was available.\n", encoding="utf-8")
        cli_path.write_text("# No state-space parameter recommendation was available.\n", encoding="utf-8")
        return

    row = recommendation.iloc[0]
    values = {col: _format_parameter_value(row[col]) for col in PARAMETER_COLUMNS}
    yaml_lines = [
        "# Selected state-space workflow inputs generated by select_state_space_parameters.py\n",
        *[f"{col}: {value}\n" for col, value in values.items()],
    ]
    yaml_path.write_text("".join(yaml_lines), encoding="utf-8")

    args = [f"--{col.replace('_', '-')} {value}" for col, value in values.items()]
    continuation = " \\" + "\n  "
    cli_text = continuation.join(args) + "\n"
    cli_path.write_text(cli_text, encoding="utf-8")


def _write_loso_parameter_files(recommendations: pd.DataFrame, out_dir: Path) -> None:
    yaml_path = out_dir / "state_space_loso_selected_workflow_inputs.yml"
    cli_path = out_dir / "state_space_loso_selected_cli_args.txt"
    if recommendations.empty:
        yaml_path.write_text("# No leave-one-session-out recommendations were available.\n", encoding="utf-8")
        cli_path.write_text("# No leave-one-session-out recommendations were available.\n", encoding="utf-8")
        return

    yaml_lines = [
        "# Leave-one-session-out state-space workflow inputs generated by select_state_space_parameters.py\n"
    ]
    cli_blocks: list[str] = []
    for _, row in recommendations.iterrows():
        if any(col not in row or pd.isna(row[col]) for col in PARAMETER_COLUMNS):
            continue
        held_out = str(row["held_out_session"])
        values = {col: _format_parameter_value(row[col]) for col in PARAMETER_COLUMNS}
        yaml_lines.append(f"{json.dumps(held_out)}:\n")
        yaml_lines.extend(f"  {col}: {value}\n" for col, value in values.items())
        args = [f"--{col.replace('_', '-')} {value}" for col, value in values.items()]
        cli_blocks.append(f"# held_out_session={held_out}\n" + (" \\\n  ".join(args)) + "\n")
    yaml_path.write_text("".join(yaml_lines), encoding="utf-8")
    cli_path.write_text("\n".join(cli_blocks), encoding="utf-8")


def _write_selection_manifest(
    *,
    evidence: str | Path,
    recovery: str | Path,
    confidence_evidence: str | Path | None,
    output_dir: Path,
    decision: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendation: pd.DataFrame,
    min_momentum_recovery_accuracy: float,
    min_overall_recovery_accuracy: float,
    max_failures: int,
    leave_one_session_out: bool = False,
    recovery_gate_metric: str = "auto",
    session_column: str | None = None,
    holdout_sessions: Sequence[str] | None = None,
    force_strict_recovery_gate: bool = False,
    loso_recommendations: pd.DataFrame | None = None,
) -> None:
    selected_row = None if recommendation.empty else _json_ready(recommendation.iloc[0].to_dict())
    manifest = {
        "schema_version": 1,
        "input_paths": {
            "evidence": str(evidence),
            "recovery": str(recovery),
            "confidence_evidence": None if confidence_evidence is None else str(confidence_evidence),
        },
        "recovery_gate": {
            "requested_metric": recovery_gate_metric,
            "resolved_metric": None
            if decision.empty or "recovery_gate_metric" not in decision.columns
            else str(decision["recovery_gate_metric"].dropna().iloc[0]),
            "min_momentum_recovery_accuracy": min_momentum_recovery_accuracy,
            "min_overall_recovery_accuracy": min_overall_recovery_accuracy,
            "max_failures": max_failures,
            "momentum_column": None if recommendation.empty else selected_row.get("momentum_recovery_gate_column"),
            "overall_column": None if recommendation.empty else selected_row.get("overall_recovery_gate_column"),
            "force_strict_recovery_gate": bool(force_strict_recovery_gate),
            "strict_candidate_rows_blocked": int(
                _bool_series(
                    decision.get(
                        "strict_candidate_recovery_gate_blocked",
                        pd.Series(False, index=decision.index),
                    )
                ).sum()
            ),
        },
        "parameter_columns": PARAMETER_COLUMNS,
        "row_counts": {
            "decision_rows": int(len(decision)),
            "candidate_rows": int(len(candidates)),
            "recommendation_rows": int(len(recommendation)),
        },
        "selected_parameters": None if selected_row is None else {col: selected_row[col] for col in PARAMETER_COLUMNS},
        "recommendation": selected_row,
    }
    if leave_one_session_out:
        manifest["leave_one_session_out"] = {
            "enabled": True,
            "session_column": session_column,
            "requested_holdout_sessions": None if holdout_sessions is None else [str(value) for value in holdout_sessions],
            "folds": 0 if loso_recommendations is None else int(len(loso_recommendations)),
            "output_files": [
                "state_space_loso_parameter_recommendations.csv",
                "state_space_loso_selected_workflow_inputs.yml",
                "state_space_loso_selected_cli_args.txt",
            ],
        }
    path = output_dir / "state_space_parameter_selection_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, bool):
        return bool(scalar)
    if isinstance(scalar, float):
        if not math.isfinite(scalar):
            return None
        return float(scalar)
    if isinstance(scalar, int):
        return int(scalar)
    return scalar


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)


def _format_parameter_value(value: object) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return f"{value:.8g}"
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, float):
        if scalar.is_integer():
            return f"{scalar:.1f}"
        return f"{scalar:.8g}"
    return str(scalar)


def _parse_holdout_sessions(value: str) -> list[str] | None:
    sessions = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    return sessions or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join state-space evidence and simulation-recovery sweeps into a parameter decision table."
    )
    parser.add_argument("--evidence", required=True, help="Evidence sweep summary directory or ranked CSV")
    parser.add_argument("--recovery", required=True, help="Simulation-recovery sweep summary directory or ranked CSV")
    parser.add_argument(
        "--confidence-evidence",
        help=(
            "Optional calibrated confidence-threshold evidence summary from "
            "select_momentum_confidence_threshold.py. When provided, ranking "
            "prefers the calibrated confident momentum-claim fraction."
        ),
    )
    parser.add_argument("--output", default="results/state-space-parameter-selection")
    parser.add_argument("--min-momentum-recovery-accuracy", type=float, default=0.5)
    parser.add_argument("--min-overall-recovery-accuracy", type=float, default=0.5)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument(
        "--leave-one-session-out",
        action="store_true",
        help="Also select parameters separately for each held-out session using only the remaining sessions.",
    )
    parser.add_argument(
        "--session-column",
        default="requested_session",
        help="Evidence column used to define leave-one-session-out folds.",
    )
    parser.add_argument(
        "--holdout-sessions",
        default="",
        help="Optional comma- or whitespace-separated subset of sessions to hold out.",
    )
    parser.add_argument(
        "--recovery-gate-metric",
        choices=("auto", "strict", "certified-vs-exact"),
        default="auto",
        help="Recovery columns used for the gate; auto prefers certified-vs-exact columns when present.",
    )
    parser.add_argument(
        "--force-strict-recovery-gate",
        action="store_true",
        help=(
            "Allow candidate-pruned momentum/IMM configurations to pass a strict exact-comparable recovery gate. "
            "This is intended for exploratory legacy sweeps; final selections should prefer certified-vs-exact recovery."
        ),
    )
    args = parser.parse_args()

    tables = select_parameters(
        args.evidence,
        args.recovery,
        output=args.output,
        confidence_evidence=args.confidence_evidence,
        min_momentum_recovery_accuracy=args.min_momentum_recovery_accuracy,
        min_overall_recovery_accuracy=args.min_overall_recovery_accuracy,
        max_failures=args.max_failures,
        leave_one_session_out=args.leave_one_session_out,
        session_column=args.session_column,
        holdout_sessions=_parse_holdout_sessions(args.holdout_sessions),
        recovery_gate_metric=args.recovery_gate_metric,
        force_strict_recovery_gate=args.force_strict_recovery_gate,
    )
    print("Top parameter rows:")
    print(tables["decision"].head(10).to_string(index=False))
    print("\nRecommendation:")
    print(tables["recommendation"].to_string(index=False))
    if args.leave_one_session_out:
        print("\nLeave-one-session-out recommendations:")
        print(tables["leave_one_session_out"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
