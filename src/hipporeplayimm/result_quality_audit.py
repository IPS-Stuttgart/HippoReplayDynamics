"""Post-hoc result-quality audit helpers for replay model-evidence tables.

This module deliberately avoids rerunning replay scoring.  It turns the
diagnostic hooks already present in the project into one reproducible audit pass:

* evidence-margin and model-disagreement summaries;
* window-sensitivity summaries when window variants are present;
* common-support comparison when a second score table is provided;
* observation-calibration selection from a validation/synthetic-recovery sweep;
* null-control recommendations;
* cell/session/rat influence diagnostics;
* provenance warnings.

The companion ``scripts/audit_model_evidence_results.py`` exposes these helpers
from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .advanced_result_diagnostics import (
    add_evidence_margin_columns,
    adversarial_synthetic_case_specs,
    common_support_audit,
    evidence_margin_table,
    hierarchical_summary,
    leave_one_group_influence,
    model_disagreement_events,
    provenance_audit,
    rat_from_session,
    summarize_window_sensitivity,
)
from .evidence_reporting import ensure_evidence_support_columns
from .result_improvements import add_candidate_support_quality_columns

_EVENT_GROUP_BASE_COLUMNS = ("session", "event_index")
_EVENT_GROUP_SCOPE_COLUMNS = (
    "window_role",
    "window_index",
    "event_window_variant",
    "window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "benchmark_random_seed",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
    "benchmark_event_subset_base_seed",
    "benchmark_test_cell_fraction",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
)
_WINDOW_VARIANT_SCOPE_COLUMNS = (
    "window_role",
    "window_index",
    "event_window_variant",
    "window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
)


@dataclass(frozen=True)
class ObservationCalibrationSelectionConfig:
    """Selection gates for observation-model calibration sweeps."""

    max_behavior_error_cm: float | None = None
    min_recovery_accuracy: float | None = None
    forbid_real_evidence_selected: bool = True
    top_k: int = 10


def event_group_columns(scores: pd.DataFrame) -> list[str]:
    """Return columns identifying one independent model-comparison unit."""

    columns = [column for column in _EVENT_GROUP_BASE_COLUMNS if column in scores.columns]
    for optional in _EVENT_GROUP_SCOPE_COLUMNS:
        if optional in scores.columns and optional not in columns:
            columns.append(optional)
    return columns


def null_control_catalog() -> pd.DataFrame:
    """Return a compact catalog of null controls worth running."""

    rows = [
        {
            "null_control": "spatial-bin permutation",
            "purpose": "Tests whether model evidence depends on spatially coherent place fields.",
            "implementation_hint": "Use improved evidence --null-shuffles, or compare a separately shuffled score table.",
        },
        {
            "null_control": "circular spike-time shift",
            "purpose": "Tests whether replay evidence survives session-preserving spike-time shifts.",
            "implementation_hint": "Use result_improvements.circular_shift_spikes_session before scoring.",
        },
        {
            "null_control": "cell-identity shuffle",
            "purpose": "Tests sensitivity to cell identity/place-field assignment.",
            "implementation_hint": "Use result_improvements.shuffle_cell_identities_session before scoring.",
        },
        {
            "null_control": "clusterless mark-feature shuffle",
            "purpose": "Tests whether clusterless evidence is carried by real mark structure.",
            "implementation_hint": "Use result_improvements.shuffle_mark_features_session before clusterless scoring.",
        },
        {
            "null_control": "wrong-environment map",
            "purpose": "Tests current-map evidence against an intentionally wrong spatial map.",
            "implementation_hint": "Run the same events with a wrong-map encoding and pass both CSVs to common/wrong-map diagnostics.",
        },
        {
            "null_control": "well-label shuffle",
            "purpose": "Tests endpoint/goal claims against randomized behavioral labels.",
            "implementation_hint": "Use result_improvements.shuffle_well_labels for ground-truth comparison tables.",
        },
    ]
    return pd.DataFrame(rows)


def select_observation_calibration(
    summary: pd.DataFrame,
    config: ObservationCalibrationSelectionConfig | None = None,
) -> pd.DataFrame:
    """Select observation-calibration rows from a validation/recovery summary.

    The function is intentionally schema-tolerant.  It recognizes common behavior
    error columns such as ``median_posterior_mean_error_cm`` and recovery columns
    such as ``recovery_accuracy`` or ``simulation_recovery_accuracy``.  Rows are
    gated before ranking; if no gate columns are present, the function still
    returns a sorted table with an explicit ``selection_gate_passed`` column.
    """

    config = ObservationCalibrationSelectionConfig() if config is None else config
    if summary.empty:
        return pd.DataFrame()
    frame = summary.copy()
    behavior_col = _first_existing(
        frame,
        (
            "median_posterior_mean_error_cm",
            "median_map_error_cm",
            "mean_posterior_mean_error_cm",
            "mean_map_error_cm",
            "behavior_error_cm",
            "position_error_cm",
        ),
    )
    recovery_col = _first_existing(
        frame,
        (
            "simulation_recovery_accuracy",
            "recovery_accuracy",
            "momentum_recovery_accuracy",
            "mean_recovery_accuracy",
            "synthetic_recovery_accuracy",
        ),
    )
    gate = pd.Series(True, index=frame.index)
    if behavior_col is not None and config.max_behavior_error_cm is not None:
        gate &= _numeric(frame[behavior_col]) <= float(config.max_behavior_error_cm)
    if recovery_col is not None and config.min_recovery_accuracy is not None:
        gate &= _numeric(frame[recovery_col]) >= float(config.min_recovery_accuracy)
    if config.forbid_real_evidence_selected:
        for column in ("selection_used_real_evidence", "used_real_evidence", "real_evidence_selected"):
            if column in frame.columns:
                gate &= ~_bool_series(frame[column])
    for column in ("selection_passed_recovery_gate", "passed_recovery_gate"):
        if column in frame.columns:
            gate &= _bool_series(frame[column])

    frame["selection_gate_passed"] = gate
    candidates = frame[gate].copy()
    if candidates.empty:
        candidates = frame.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if behavior_col is not None:
        candidates[f"_selection_behavior_error__{behavior_col}"] = _numeric(candidates[behavior_col])
        sort_columns.append(f"_selection_behavior_error__{behavior_col}")
        ascending.append(True)
    if recovery_col is not None:
        candidates[f"_selection_recovery__{recovery_col}"] = _numeric(candidates[recovery_col])
        sort_columns.append(f"_selection_recovery__{recovery_col}")
        ascending.append(False)
    if "candidate_support_quality_good" in candidates.columns:
        candidates["_selection_candidate_support_good"] = _bool_series(
            candidates["candidate_support_quality_good"]
        )
        sort_columns.append("_selection_candidate_support_good")
        ascending.append(False)
    if sort_columns:
        candidates = candidates.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    candidates = candidates.reset_index(drop=True)
    candidates["selection_rank"] = np.arange(1, len(candidates) + 1, dtype=int)
    drop_private = [column for column in candidates.columns if column.startswith("_selection_")]
    return candidates.drop(columns=drop_private).head(max(1, int(config.top_k)))


def write_result_quality_audit(
    scores: pd.DataFrame,
    output_dir: str | Path,
    *,
    common_support_scores: pd.DataFrame | None = None,
    observation_sweep_summary: pd.DataFrame | None = None,
    observation_selection_config: ObservationCalibrationSelectionConfig | None = None,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """Write a full result-quality audit directory and return the dashboard path."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scores.empty:
        raise ValueError("scores must not be empty")

    score_table = _score_table_with_log_evidence_alias(scores)
    group_cols = event_group_columns(score_table)
    scores_with_support = ensure_evidence_support_columns(score_table)
    scores_with_quality = add_candidate_support_quality_columns(scores_with_support)
    scores_with_margins = add_evidence_margin_columns(scores_with_quality, group_cols=group_cols or ("model",))
    margins = evidence_margin_table(scores_with_quality, group_cols=group_cols or ("model",))
    disagreements = model_disagreement_events(scores_with_margins, group_cols=group_cols or ("model",))
    hierarchy = _hierarchical_summary_if_available(scores_with_margins)
    window_summary = _window_summary_if_available(scores_with_margins, group_cols=group_cols)
    influence = _influence_summary(scores_with_margins)
    candidate_summary = _candidate_support_summary(scores_with_margins)
    nulls = null_control_catalog()
    synthetic_cases = adversarial_synthetic_case_specs()
    provenance_frame = provenance_audit(scores_with_margins, provenance)

    scores_with_margins.to_csv(out / "event_model_evidence_with_quality.csv", index=False)
    margins.to_csv(out / "evidence_margins.csv", index=False)
    disagreements.to_csv(out / "model_disagreement_events.csv", index=False)
    hierarchy.to_csv(out / "hierarchical_summary.csv", index=False)
    window_summary.to_csv(out / "window_sensitivity_summary.csv", index=False)
    influence.to_csv(out / "influence_summary.csv", index=False)
    candidate_summary.to_csv(out / "candidate_support_summary.csv", index=False)
    nulls.to_csv(out / "null_control_catalog.csv", index=False)
    synthetic_cases.to_csv(out / "adversarial_synthetic_case_specs.csv", index=False)
    provenance_frame.to_csv(out / "provenance_audit.csv", index=False)

    if common_support_scores is not None and not common_support_scores.empty:
        common = common_support_audit(scores_with_margins, _score_table_with_log_evidence_alias(common_support_scores))
        common.to_csv(out / "common_support_audit.csv", index=False)
    else:
        common = pd.DataFrame()

    if observation_sweep_summary is not None and not observation_sweep_summary.empty:
        selected = select_observation_calibration(
            observation_sweep_summary,
            observation_selection_config,
        )
        selected.to_csv(out / "selected_observation_calibrations.csv", index=False)
    else:
        selected = pd.DataFrame()

    dashboard = out / "result_quality_audit.md"
    dashboard.write_text(
        _dashboard_markdown(
            scores_with_margins,
            margins,
            candidate_summary,
            window_summary,
            common,
            selected,
            provenance_frame,
        ),
        encoding="utf-8",
    )
    return dashboard


def _score_table_with_log_evidence_alias(scores: pd.DataFrame) -> pd.DataFrame:
    """Return a score table with the canonical evidence column present.

    Some held-out predictive-control outputs store the model score only as
    ``heldout_log_likelihood``.  The audit helpers downstream of this function
    compare rows through the canonical ``log_evidence`` column, so copy the
    held-out score into that canonical column when it is otherwise absent.
    """

    if "log_evidence" in scores.columns or "heldout_log_likelihood" not in scores.columns:
        return scores.copy()
    out = scores.copy()
    out["log_evidence"] = out["heldout_log_likelihood"]
    return out


def _first_existing(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)


def _hierarchical_summary_if_available(scores: pd.DataFrame) -> pd.DataFrame:
    for value_col in ("relative_log_evidence", "log_evidence", "heldout_log_likelihood"):
        if value_col in scores.columns:
            return hierarchical_summary(scores, value_col=value_col)
    return pd.DataFrame()


def _window_summary_if_available(scores: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if "event_window_variant" in scores.columns:
        return summarize_window_sensitivity(
            scores,
            group_cols=_window_sensitivity_group_cols(group_cols, "event_window_variant"),
            variant_col="event_window_variant",
        )
    if "window_variant" in scores.columns:
        return summarize_window_sensitivity(
            scores,
            group_cols=_window_sensitivity_group_cols(group_cols, "window_variant"),
            variant_col="window_variant",
        )
    if "window_index" not in scores.columns:
        return pd.DataFrame()
    tmp = scores.copy()
    tmp["event_window_variant"] = "window_" + tmp["window_index"].astype(str)
    return summarize_window_sensitivity(
        tmp,
        group_cols=_window_sensitivity_group_cols(group_cols, "event_window_variant"),
        variant_col="event_window_variant",
    )


def _window_sensitivity_group_cols(group_cols: Sequence[str], variant_col: str) -> list[str]:
    """Drop window-variant identifiers before comparing variants of one event."""

    window_scope = set(_WINDOW_VARIANT_SCOPE_COLUMNS)
    window_scope.add(str(variant_col))
    return [column for column in group_cols if column not in window_scope]


def _influence_summary(scores: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "full_mean",
        "leave_one_mean",
        "left_out_group_col",
        "left_out_group",
        "influence_delta",
    ]
    value_col = _first_existing(scores, ("relative_log_evidence", "log_evidence", "heldout_log_likelihood"))
    frames: list[pd.DataFrame] = []
    if value_col is not None and "session" in scores.columns:
        frames.append(leave_one_group_influence(scores, group_col="session", value_col=value_col))
        rat_scores = scores.copy()
        rat_scores["rat"] = rat_scores["session"].map(rat_from_session)
        frames.append(leave_one_group_influence(rat_scores, group_col="rat", value_col=value_col))
    if not frames:
        return pd.DataFrame(columns=columns)
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=columns)
    out = pd.concat(nonempty, ignore_index=True)
    return out if not out.empty else pd.DataFrame(columns=columns)


def _candidate_support_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if "candidate_support_quality" not in scores.columns:
        return pd.DataFrame()
    group_columns = [column for column in ("model", "evidence_support", "candidate_support_quality") if column in scores.columns]
    if not group_columns:
        return pd.DataFrame()
    return (
        scores.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows"], ascending=False)
    )


def _dashboard_markdown(
    scores: pd.DataFrame,
    margins: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    common: pd.DataFrame,
    selected_calibrations: pd.DataFrame,
    provenance: pd.DataFrame,
) -> str:
    lines = [
        "# Replay model-evidence result-quality audit",
        "",
        f"Rows: {len(scores)}",
        f"Events: {_event_count(scores)}",
        "",
        "## Evidence margins",
        _value_counts_text(margins, "evidence_margin_category"),
        "",
        "## Candidate support",
        _small_frame_text(candidate_summary),
        "",
        "## Window sensitivity",
        _small_frame_text(window_summary),
        "",
        "## Common-support audit",
        _small_frame_text(common),
        "",
        "## Selected observation calibrations",
        _small_frame_text(selected_calibrations),
        "",
        "## Provenance audit",
        _small_frame_text(provenance),
    ]
    return "\n".join(lines) + "\n"


def _event_count(scores: pd.DataFrame) -> int | str:
    group_columns = event_group_columns(scores)
    if group_columns:
        return int(scores[group_columns].drop_duplicates().shape[0])
    return "unknown"


def _value_counts_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "No rows available."
    values = frame[column].fillna("missing").astype(str).value_counts()
    return "\n".join(f"- {key}: {int(value)}" for key, value in values.items())


def _small_frame_text(frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows available."
    return frame.head(max_rows).to_string(index=False)
