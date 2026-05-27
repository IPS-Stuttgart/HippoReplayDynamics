"""Advanced diagnostics for replay model-evidence result tables.

This module is intentionally independent of the benchmark entry points.  It
adds post-hoc checks that make model-evidence results harder to overinterpret:
wrong-map deltas, calibrated evidence margins, hierarchical summaries,
influence diagnostics, place-field quality filters, event-window plans,
posterior predictive checks, mark-drift summaries, provenance checks, and model
-disagreement mining.

The functions operate on ordinary ``pandas.DataFrame`` objects wherever
possible, so they can be used on existing ``event_model_evidence.csv`` files
without rerunning a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln


EVIDENCE_MARGIN_CATEGORIES: tuple[tuple[str, float], ...] = (
    ("tie", 1.0),
    ("weak", 3.0),
    ("strong", 10.0),
    ("decisive", np.inf),
)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Hyperparameter-selection provenance for a result table."""

    parameter_source: str = "unknown"
    selection_run_id: str = ""
    selection_metric: str = ""
    selection_passed_recovery_gate: bool | None = None
    selection_used_real_evidence: bool | None = None
    notes: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


def rat_from_session(session: object) -> str:
    """Return a rat identifier from a session label like ``Rat1/Open1``."""

    text = str(session)
    return text.split("/", 1)[0].split("\\", 1)[0]


def _successful_rows(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    if "status" in scores.columns:
        return scores[scores["status"].astype(str).eq("success")].copy()
    return scores.copy()


def _comparable_rows(scores: pd.DataFrame) -> pd.DataFrame:
    ok = _successful_rows(scores)
    if ok.empty:
        return ok
    if "evidence_comparable" in ok.columns:
        mask = ok["evidence_comparable"].fillna(False).astype(bool)
        ok = ok[mask].copy()
    return ok


def classify_evidence_margin(delta_log_evidence: float) -> str:
    """Classify an evidence margin into tie/weak/strong/decisive buckets."""

    value = float(delta_log_evidence)
    if not np.isfinite(value):
        return "missing"
    for label, upper in EVIDENCE_MARGIN_CATEGORIES:
        if value <= upper:
            return label
    return "decisive"


def evidence_margin_table(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("session", "event_index"),
    evidence_col: str = "log_evidence",
    model_col: str = "model",
) -> pd.DataFrame:
    """Return one calibrated evidence-margin row per event.

    Only successful exact-comparable rows are used when an ``evidence_comparable``
    column is present.  The returned ``evidence_margin_to_second_best`` is the
    log-evidence difference between the best and second-best exact-comparable
    models for the event.
    """

    ok = _comparable_rows(scores)
    if ok.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "best_model_by_evidence",
                "second_best_model_by_evidence",
                "best_log_evidence",
                "second_best_log_evidence",
                "evidence_margin_to_second_best",
                "evidence_margin_category",
                "models_compared",
            ]
        )
    missing = [column for column in (*group_cols, evidence_col, model_col) if column not in ok.columns]
    if missing:
        raise KeyError(f"scores is missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group = group.dropna(subset=[evidence_col]).sort_values(evidence_col, ascending=False)
        if group.empty:
            continue
        best = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        best_value = float(best[evidence_col])
        second_value = float(second[evidence_col]) if second is not None else np.nan
        margin = best_value - second_value if second is not None else np.inf
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "best_model_by_evidence": str(best[model_col]),
                "second_best_model_by_evidence": "" if second is None else str(second[model_col]),
                "best_log_evidence": best_value,
                "second_best_log_evidence": second_value,
                "evidence_margin_to_second_best": float(margin),
                "evidence_margin_category": classify_evidence_margin(margin),
                "models_compared": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_evidence_margin_columns(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("session", "event_index"),
) -> pd.DataFrame:
    """Merge event-level evidence-margin diagnostics back into score rows."""

    if scores.empty:
        return scores.copy()
    margins = evidence_margin_table(scores, group_cols=group_cols)
    if margins.empty:
        out = scores.copy()
        out["evidence_margin_to_second_best"] = np.nan
        out["evidence_margin_category"] = "missing"
        return out
    return scores.merge(margins, on=list(group_cols), how="left")


def paired_model_margin_decisions(
    scores: pd.DataFrame,
    *,
    positive_model: str,
    reference_model: str,
    margin_threshold: float = 0.0,
    group_cols: Sequence[str] = ("session", "event_index"),
    evidence_col: str = "log_evidence",
    model_col: str = "model",
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    """Classify paired model wins using a symmetric log-evidence margin.

    A positive-model claim is emitted only when
    ``logZ(positive_model) - logZ(reference_model) >= margin_threshold``.
    Reference claims use the symmetric negative threshold; rows between the
    two thresholds are intentionally labelled ``ambiguous``.  When a true-model
    label is present, the table also marks whether the margin-gated binary
    claim matches the positive-vs-reference synthetic family.
    """

    threshold = float(margin_threshold)
    if threshold < 0.0:
        raise ValueError("margin_threshold must be non-negative")
    ok = _comparable_rows(scores)
    columns = [
        *group_cols,
        "positive_model",
        "reference_model",
        "positive_log_evidence",
        "reference_log_evidence",
        "positive_minus_reference_log_evidence",
        "margin_threshold",
        "margin_decision",
        "positive_model_claimed",
    ]
    if true_model_col:
        columns.extend(
            [
                true_model_col,
                "positive_true_label",
                "true_is_positive",
                "margin_binary_correct",
            ]
        )
    if ok.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in (*group_cols, evidence_col, model_col) if column not in ok.columns]
    if true_model_col and true_model_col not in ok.columns:
        missing.append(true_model_col)
    if missing:
        raise KeyError(f"scores is missing required columns: {missing}")

    positive_label = positive_true_label or _model_family_label(positive_model)
    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        paired = group[group[model_col].astype(str).isin([positive_model, reference_model])]
        pivot = paired.dropna(subset=[evidence_col]).drop_duplicates(model_col, keep="last")
        by_model = pivot.set_index(model_col)
        if positive_model not in by_model.index or reference_model not in by_model.index:
            continue
        positive_value = float(by_model.loc[positive_model, evidence_col])
        reference_value = float(by_model.loc[reference_model, evidence_col])
        delta = positive_value - reference_value
        if delta >= threshold:
            decision = positive_model
            positive_claimed = True
        elif delta <= -threshold:
            decision = reference_model
            positive_claimed = False
        else:
            decision = "ambiguous"
            positive_claimed = False
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "positive_model": positive_model,
                "reference_model": reference_model,
                "positive_log_evidence": positive_value,
                "reference_log_evidence": reference_value,
                "positive_minus_reference_log_evidence": float(delta),
                "margin_threshold": threshold,
                "margin_decision": decision,
                "positive_model_claimed": bool(positive_claimed),
            }
        )
        if true_model_col:
            true_label = _unique_text_value(group[true_model_col])
            true_is_positive = _model_family_label(true_label) == _model_family_label(positive_label)
            row.update(
                {
                    true_model_col: true_label,
                    "positive_true_label": positive_label,
                    "true_is_positive": bool(true_is_positive),
                    "margin_binary_correct": bool(positive_claimed) == bool(true_is_positive),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def paired_model_margin_summary(
    decisions: pd.DataFrame,
    *,
    true_model_col: str | None = None,
) -> pd.DataFrame:
    """Summarize a paired margin-decision table."""

    if decisions.empty:
        return pd.DataFrame(
            [
                {
                    "events": 0,
                    "positive_model_claims": 0,
                    "reference_model_claims": 0,
                    "ambiguous_events": 0,
                    "positive_claim_fraction": np.nan,
                    "mean_positive_minus_reference_log_evidence": np.nan,
                    "median_positive_minus_reference_log_evidence": np.nan,
                }
            ]
        )
    out: dict[str, object] = {
        "events": int(len(decisions)),
        "positive_model": str(decisions["positive_model"].dropna().iloc[0]),
        "reference_model": str(decisions["reference_model"].dropna().iloc[0]),
        "margin_threshold": float(decisions["margin_threshold"].dropna().iloc[0]),
        "positive_model_claims": int(decisions["positive_model_claimed"].fillna(False).astype(bool).sum()),
        "reference_model_claims": int((decisions["margin_decision"] == decisions["reference_model"]).sum()),
        "ambiguous_events": int((decisions["margin_decision"] == "ambiguous").sum()),
        "positive_claim_fraction": float(decisions["positive_model_claimed"].fillna(False).astype(bool).mean()),
        "mean_positive_minus_reference_log_evidence": float(
            decisions["positive_minus_reference_log_evidence"].mean()
        ),
        "median_positive_minus_reference_log_evidence": float(
            decisions["positive_minus_reference_log_evidence"].median()
        ),
    }
    if true_model_col and true_model_col in decisions:
        correct = decisions["margin_binary_correct"].fillna(False).astype(bool)
        true_positive = decisions["true_is_positive"].fillna(False).astype(bool)
        claims = decisions["positive_model_claimed"].fillna(False).astype(bool)
        out.update(
            {
                "thresholded_binary_accuracy": float(correct.mean()),
                "positive_true_events": int(true_positive.sum()),
                "reference_true_events": int((~true_positive).sum()),
                "positive_true_claimed_events": int((claims & true_positive).sum()),
                "reference_true_rejected_events": int((~claims & ~true_positive).sum()),
                "positive_claim_recall": _safe_ratio(int((claims & true_positive).sum()), int(true_positive.sum())),
                "reference_specificity": _safe_ratio(int((~claims & ~true_positive).sum()), int((~true_positive).sum())),
                "false_positive_claims": int((claims & ~true_positive).sum()),
                "false_negative_claims": int((~claims & true_positive).sum()),
            }
        )
    return pd.DataFrame([out])


def _unique_text_value(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique()]
    if not unique:
        return ""
    return unique[0]


def _model_family_label(value: str) -> str:
    text = str(value).lower()
    if "momentum" in text:
        return "momentum"
    if "diffusion" in text:
        return "diffusion"
    return text


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def wrong_map_delta_summary(
    current_map_scores: pd.DataFrame,
    wrong_map_scores: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("session", "event_index", "model"),
    evidence_col: str = "log_evidence",
) -> pd.DataFrame:
    """Compare current-map evidence against wrong-environment-map evidence."""

    left = _successful_rows(current_map_scores)
    right = _successful_rows(wrong_map_scores)
    missing_left = [column for column in (*key_cols, evidence_col) if column not in left.columns]
    missing_right = [column for column in (*key_cols, evidence_col) if column not in right.columns]
    if missing_left:
        raise KeyError(f"current_map_scores is missing required columns: {missing_left}")
    if missing_right:
        raise KeyError(f"wrong_map_scores is missing required columns: {missing_right}")
    left_cols = list(key_cols) + [evidence_col]
    right_cols = list(key_cols) + [evidence_col]
    merged = left[left_cols].merge(
        right[right_cols],
        on=list(key_cols),
        how="inner",
        suffixes=("_current_map", "_wrong_map"),
    )
    if merged.empty:
        return merged
    merged["delta_vs_wrong_environment_map"] = (
        merged[f"{evidence_col}_current_map"] - merged[f"{evidence_col}_wrong_map"]
    )
    event_keys = [column for column in key_cols if column != "model"]
    best_wrong = (
        merged.sort_values(f"{evidence_col}_wrong_map", ascending=False)
        .drop_duplicates(event_keys, keep="first")
        [event_keys + ["model"]]
        .rename(columns={"model": "wrong_map_best_model"})
    )
    return merged.merge(best_wrong, on=event_keys, how="left")


def place_field_quality_from_arrays(
    rates_hz: np.ndarray,
    occupancy_s: np.ndarray,
    cell_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute simple place-field quality metrics from rate and occupancy arrays."""

    rates = np.asarray(rates_hz, dtype=float)
    occupancy = np.asarray(occupancy_s, dtype=float)
    if rates.ndim != 2:
        raise ValueError("rates_hz must have shape (n_cells, n_bins)")
    if occupancy.ndim != 1 or occupancy.shape[0] != rates.shape[1]:
        raise ValueError("occupancy_s must have one value per spatial bin")
    ids = np.arange(rates.shape[0], dtype=int) if cell_ids is None else np.asarray(cell_ids, dtype=int)
    if ids.shape[0] != rates.shape[0]:
        raise ValueError("cell_ids must have one entry per cell")

    occ_total = float(np.sum(occupancy))
    if occ_total <= 0.0:
        occ_prob = np.full(occupancy.shape, 1.0 / max(len(occupancy), 1), dtype=float)
    else:
        occ_prob = occupancy / occ_total
    rows: list[dict[str, object]] = []
    for row_index, cell_id in enumerate(ids):
        lam = np.maximum(rates[row_index], np.finfo(float).tiny)
        mean_rate = float(np.sum(occ_prob * lam))
        peak_rate = float(np.max(lam))
        info = float(np.sum(occ_prob * (lam / mean_rate) * np.log2(lam / mean_rate))) if mean_rate > 0 else np.nan
        sparsity_den = float(np.sum(occ_prob * lam * lam))
        sparsity = float(mean_rate * mean_rate / sparsity_den) if sparsity_den > 0 else np.nan
        active_bins = int(np.sum(lam > mean_rate))
        rows.append(
            {
                "cell_id": int(cell_id),
                "spatial_information_bits_per_spike": info,
                "peak_rate_hz": peak_rate,
                "mean_rate_hz": mean_rate,
                "field_sparsity": sparsity,
                "active_bins_above_mean": active_bins,
            }
        )
    return pd.DataFrame(rows)


def place_field_quality(encoding: object) -> pd.DataFrame:
    """Compute place-field quality metrics from an ``EncodingModel``-like object."""

    return place_field_quality_from_arrays(
        np.asarray(getattr(encoding, "rates_hz"), dtype=float),
        np.asarray(getattr(encoding, "occupancy_s"), dtype=float),
        getattr(encoding, "cell_ids", None),
    )


def stable_cell_ids(
    quality: pd.DataFrame,
    *,
    min_spatial_information_bits: float = 0.25,
    min_peak_rate_hz: float = 1.0,
    max_field_sparsity: float | None = None,
) -> np.ndarray:
    """Return cell IDs passing simple spatial-quality thresholds."""

    required = {"cell_id", "spatial_information_bits_per_spike", "peak_rate_hz"}
    missing = required - set(quality.columns)
    if missing:
        raise KeyError(f"quality is missing required columns: {sorted(missing)}")
    mask = quality["spatial_information_bits_per_spike"].fillna(-np.inf) >= min_spatial_information_bits
    mask &= quality["peak_rate_hz"].fillna(0.0) >= min_peak_rate_hz
    if max_field_sparsity is not None and "field_sparsity" in quality:
        mask &= quality["field_sparsity"].fillna(np.inf) <= float(max_field_sparsity)
    return quality.loc[mask, "cell_id"].to_numpy(dtype=int)


def drop_one_cell_influence(
    event_scores: pd.DataFrame,
    *,
    cell_column: str = "train_cell_ids",
    evidence_col: str = "log_evidence",
) -> pd.DataFrame:
    """Estimate influence of cells using score rows containing comma-separated cell IDs.

    This is a conservative table-level diagnostic.  It does not rerun scoring; it
    identifies cells disproportionately represented in high- or low-evidence rows.
    """

    if event_scores.empty or cell_column not in event_scores.columns or evidence_col not in event_scores.columns:
        return pd.DataFrame(columns=["cell_id", "rows", "mean_log_evidence", "delta_vs_global_mean"])
    global_mean = float(event_scores[evidence_col].dropna().mean())
    rows = []
    for _, row in event_scores.dropna(subset=[evidence_col]).iterrows():
        text = str(row.get(cell_column, ""))
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                cell_id = int(token)
            except ValueError:
                continue
            rows.append({"cell_id": cell_id, evidence_col: float(row[evidence_col])})
    if not rows:
        return pd.DataFrame(columns=["cell_id", "rows", "mean_log_evidence", "delta_vs_global_mean"])
    frame = pd.DataFrame(rows)
    out = frame.groupby("cell_id", as_index=False).agg(rows=(evidence_col, "count"), mean_log_evidence=(evidence_col, "mean"))
    out["delta_vs_global_mean"] = out["mean_log_evidence"] - global_mean
    return out.sort_values("delta_vs_global_mean", ascending=False)


def event_window_variants(
    events: pd.DataFrame,
    *,
    start_col: str = "start",
    end_col: str = "end",
    event_id_col: str = "event_index",
    paddings_s: Sequence[float] = (0.0, 0.01, 0.02),
    min_duration_s: float = 0.003,
) -> pd.DataFrame:
    """Create a plan of padded and center-only replay windows for sensitivity checks."""

    missing = [column for column in (start_col, end_col, event_id_col) if column not in events.columns]
    if missing:
        raise KeyError(f"events is missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        start = float(event[start_col])
        end = float(event[end_col])
        duration = max(end - start, min_duration_s)
        center = 0.5 * (start + end)
        for padding in paddings_s:
            pad = float(padding)
            rows.append(
                {
                    event_id_col: int(event[event_id_col]),
                    "window_variant": f"pad_{pad:.3f}s",
                    "window_start": max(start - pad, 0.0),
                    "window_end": end + pad,
                    "padding_s": pad,
                }
            )
        half = max(0.25 * duration, 0.5 * min_duration_s)
        rows.append(
            {
                event_id_col: int(event[event_id_col]),
                "window_variant": "center_half_duration",
                "window_start": max(center - half, 0.0),
                "window_end": center + half,
                "padding_s": np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_window_sensitivity(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("session", "event_index"),
    variant_col: str = "window_variant",
    evidence_col: str = "log_evidence",
) -> pd.DataFrame:
    """Summarize how stable model evidence is across replay-window variants."""

    if scores.empty or variant_col not in scores.columns:
        return pd.DataFrame()
    ok = _successful_rows(scores)
    keys = list(group_cols) + ["model"]
    summary = ok.groupby(keys, as_index=False).agg(
        window_variants=(variant_col, "nunique"),
        evidence_window_mean=(evidence_col, "mean"),
        evidence_window_sd=(evidence_col, "std"),
        evidence_window_min=(evidence_col, "min"),
        evidence_window_max=(evidence_col, "max"),
    )
    summary["evidence_window_range"] = summary["evidence_window_max"] - summary["evidence_window_min"]
    return summary


def posterior_predictive_count_checks(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    *,
    variance_counts: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare observed counts with posterior-predictive expected counts.

    ``observed_counts`` and ``expected_counts`` should have shape
    ``(n_time, n_cells)``.  ``variance_counts`` defaults to Poisson variance.
    """

    observed = np.asarray(observed_counts, dtype=float)
    expected = np.asarray(expected_counts, dtype=float)
    if observed.shape != expected.shape or observed.ndim != 2:
        raise ValueError("observed_counts and expected_counts must have shape (n_time, n_cells)")
    variance = np.maximum(expected, np.finfo(float).eps) if variance_counts is None else np.asarray(variance_counts, dtype=float)
    if variance.shape != observed.shape:
        raise ValueError("variance_counts must match observed_counts")
    residual = observed - expected
    z = residual / np.sqrt(np.maximum(variance, np.finfo(float).eps))
    rows = [
        {
            "predictive_check": "total_spike_count",
            "observed": float(observed.sum()),
            "expected": float(expected.sum()),
            "z_score": float(residual.sum() / np.sqrt(np.maximum(variance.sum(), np.finfo(float).eps))),
        },
        {
            "predictive_check": "silent_bin_fraction",
            "observed": float(np.mean(observed.sum(axis=1) == 0.0)),
            "expected": float(np.mean(np.exp(-np.maximum(expected.sum(axis=1), 0.0)))),
            "z_score": np.nan,
        },
        {
            "predictive_check": "mean_abs_cell_z",
            "observed": float(np.mean(np.abs(z))),
            "expected": 0.0,
            "z_score": np.nan,
        },
        {
            "predictive_check": "max_abs_cell_z",
            "observed": float(np.max(np.abs(z))),
            "expected": 0.0,
            "z_score": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def posterior_predictive_poisson_log_score(observed_counts: np.ndarray, expected_counts: np.ndarray) -> float:
    """Return a Poisson posterior-predictive log score for observed counts."""

    observed = np.asarray(observed_counts, dtype=float)
    expected = np.maximum(np.asarray(expected_counts, dtype=float), np.finfo(float).tiny)
    if observed.shape != expected.shape:
        raise ValueError("observed_counts and expected_counts must have matching shapes")
    return float(np.sum(observed * np.log(expected) - expected - gammaln(observed + 1.0)))


def hierarchical_summary(
    scores: pd.DataFrame,
    *,
    value_col: str = "relative_log_evidence",
    model_col: str = "model",
) -> pd.DataFrame:
    """Report model summaries at event, session, and rat levels."""

    ok = _successful_rows(scores)
    if ok.empty or value_col not in ok.columns:
        return pd.DataFrame()
    frame = ok.dropna(subset=[value_col]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["rat"] = frame["session"].map(rat_from_session) if "session" in frame else "unknown"
    event_summary = frame.groupby(model_col, as_index=False).agg(event_rows=(value_col, "count"), event_mean=(value_col, "mean"))
    session_means = frame.groupby([model_col, "session"], as_index=False)[value_col].mean()
    rat_means = frame.groupby([model_col, "rat"], as_index=False)[value_col].mean()
    session_summary = session_means.groupby(model_col, as_index=False).agg(
        sessions=("session", "nunique"),
        session_mean=(value_col, "mean"),
        session_sd=(value_col, "std"),
    )
    rat_summary = rat_means.groupby(model_col, as_index=False).agg(
        rats=("rat", "nunique"),
        rat_mean=(value_col, "mean"),
        rat_sd=(value_col, "std"),
    )
    return event_summary.merge(session_summary, on=model_col, how="outer").merge(rat_summary, on=model_col, how="outer")


def hierarchical_bootstrap(
    scores: pd.DataFrame,
    *,
    model: str,
    value_col: str = "relative_log_evidence",
    level: str = "session",
    n_bootstrap: int = 1000,
    random_seed: int = 1,
) -> dict[str, float | str | int]:
    """Bootstrap a model-level mean by event, session, or rat."""

    ok = _successful_rows(scores)
    frame = ok[ok["model"].astype(str).eq(str(model))].dropna(subset=[value_col]).copy()
    if frame.empty:
        return {"model": model, "level": level, "n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if level == "event":
        units = np.arange(len(frame))
        values_by_unit = {int(i): np.asarray([v], dtype=float) for i, v in enumerate(frame[value_col].to_numpy(float))}
    elif level == "session":
        values_by_unit = {str(k): g[value_col].to_numpy(float) for k, g in frame.groupby("session", sort=False)}
        units = np.asarray(list(values_by_unit), dtype=object)
    elif level == "rat":
        frame["rat"] = frame["session"].map(rat_from_session)
        values_by_unit = {str(k): g[value_col].to_numpy(float) for k, g in frame.groupby("rat", sort=False)}
        units = np.asarray(list(values_by_unit), dtype=object)
    else:
        raise ValueError("level must be one of: event, session, rat")
    rng = np.random.default_rng(random_seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled_values = np.concatenate([values_by_unit[unit] for unit in sampled_units])
        means[idx] = float(np.mean(sampled_values))
    return {
        "model": model,
        "level": level,
        "n": int(len(units)),
        "mean": float(np.mean(frame[value_col])),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def leave_one_group_influence(
    scores: pd.DataFrame,
    *,
    group_col: str = "session",
    value_col: str = "relative_log_evidence",
    model_col: str = "model",
) -> pd.DataFrame:
    """Compute leave-one-event/session/rat influence on model means."""

    ok = _successful_rows(scores)
    if ok.empty or value_col not in ok.columns or group_col not in ok.columns:
        return pd.DataFrame()
    frame = ok.dropna(subset=[value_col]).copy()
    baseline = frame.groupby(model_col)[value_col].mean().rename("full_mean").reset_index()
    rows = []
    for group_value in sorted(frame[group_col].dropna().unique(), key=str):
        keep = frame[frame[group_col] != group_value]
        if keep.empty:
            continue
        leave = keep.groupby(model_col)[value_col].mean().rename("leave_one_mean").reset_index()
        merged = baseline.merge(leave, on=model_col, how="left")
        merged["left_out_group_col"] = group_col
        merged["left_out_group"] = str(group_value)
        merged["influence_delta"] = merged["leave_one_mean"] - merged["full_mean"]
        rows.extend(merged.to_dict("records"))
    return pd.DataFrame(rows)


def common_support_from_emissions(
    log_likelihood: np.ndarray,
    *,
    top_k: int = 128,
    extra_candidate_sets: Sequence[Sequence[int]] | None = None,
) -> list[np.ndarray]:
    """Build a common candidate support from top emissions plus optional extras."""

    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must have shape (n_time, n_bins)")
    extras = extra_candidate_sets or ()
    output: list[np.ndarray] = []
    for time_index, row in enumerate(values):
        k = min(max(int(top_k), 1), row.shape[0])
        selected = np.argpartition(row, -k)[-k:]
        support = set(int(index) for index in selected)
        if time_index < len(extras):
            support.update(int(index) for index in extras[time_index])
        output.append(np.asarray(sorted(support), dtype=int))
    return output


def common_support_audit(
    native_scores: pd.DataFrame,
    common_support_scores: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("session", "event_index", "model"),
    evidence_col: str = "log_evidence",
) -> pd.DataFrame:
    """Compare native-support evidence to common-support diagnostic evidence."""

    native = _successful_rows(native_scores)
    common = _successful_rows(common_support_scores)
    merged = native[list(key_cols) + [evidence_col]].merge(
        common[list(key_cols) + [evidence_col]],
        on=list(key_cols),
        how="inner",
        suffixes=("_native", "_common_support"),
    )
    if merged.empty:
        return merged
    merged["common_support_delta"] = merged[f"{evidence_col}_common_support"] - merged[f"{evidence_col}_native"]
    return merged


def mark_drift_diagnostics(
    mark_times: np.ndarray,
    marks: np.ndarray,
    *,
    n_blocks: int = 4,
) -> pd.DataFrame:
    """Summarize clusterless mark drift across time blocks."""

    times = np.asarray(mark_times, dtype=float)
    values = np.asarray(marks, dtype=float)
    if values.ndim != 2 or times.ndim != 1 or times.shape[0] != values.shape[0]:
        raise ValueError("mark_times must be length n_marks and marks must have shape (n_marks, n_features)")
    finite = np.isfinite(times) & np.all(np.isfinite(values), axis=1)
    times = times[finite]
    values = values[finite]
    if values.size == 0:
        return pd.DataFrame()
    order = np.argsort(times)
    chunks = [chunk for chunk in np.array_split(order, max(int(n_blocks), 1)) if chunk.size]
    global_mean = np.mean(values, axis=0)
    rows = []
    first_mean = None
    for block_index, chunk in enumerate(chunks):
        block_values = values[chunk]
        block_mean = np.mean(block_values, axis=0)
        if first_mean is None:
            first_mean = block_mean
        rows.append(
            {
                "mark_time_block": int(block_index),
                "marks": int(block_values.shape[0]),
                "start_time": float(np.min(times[chunk])),
                "end_time": float(np.max(times[chunk])),
                "mark_mean_distance_from_global": float(np.linalg.norm(block_mean - global_mean)),
                "mark_mean_distance_from_first_block": float(np.linalg.norm(block_mean - first_mean)),
                "mark_mean": ",".join(f"{float(x):.6g}" for x in block_mean),
                "mark_variance": ",".join(f"{float(x):.6g}" for x in np.var(block_values, axis=0)),
            }
        )
    return pd.DataFrame(rows)


def context_conditioning_table(
    ground_truth: pd.DataFrame,
    *,
    session_col: str = "session",
    event_col: str = "event_index",
) -> pd.DataFrame:
    """Create pre/post behavioral context features from a ground-truth proxy table."""

    if ground_truth.empty:
        return pd.DataFrame()
    out = ground_truth.copy()
    if "well_id" in out.columns:
        out["next_well_context"] = out["well_id"].astype(str)
    elif "target_well" in out.columns:
        out["next_well_context"] = out["target_well"].astype(str)
    else:
        out["next_well_context"] = "unknown"
    if "time_to_arrival_s" in out.columns:
        out["context_confidence"] = 1.0 / (1.0 + out["time_to_arrival_s"].astype(float).clip(lower=0.0))
    else:
        out["context_confidence"] = np.nan
    keep_cols = [column for column in (session_col, event_col, "next_well_context", "context_confidence") if column in out.columns]
    return out[keep_cols].copy()


def model_disagreement_events(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("session", "event_index"),
    probability_col: str = "model_probability",
) -> pd.DataFrame:
    """Identify events where model families or observation models disagree."""

    if scores.empty:
        return pd.DataFrame()
    margins = evidence_margin_table(scores, group_cols=group_cols)
    ok = _successful_rows(scores)
    rows = []
    for key, group in ok.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        models = set(group["model"].astype(str)) if "model" in group else set()
        best = ""
        if "is_best_model" in group:
            winners = group[group["is_best_model"].fillna(False).astype(bool)]
            if not winners.empty:
                best = str(winners.iloc[0].get("model", ""))
        if not best and "log_evidence" in group:
            best = str(group.sort_values("log_evidence", ascending=False).iloc[0].get("model", ""))
        probability_entropy = np.nan
        if probability_col in group:
            p = group[probability_col].dropna().to_numpy(float)
            p = p[p > 0.0]
            if p.size:
                p = p / p.sum()
                probability_entropy = float(-np.sum(p * np.log(p)))
        row.update(
            {
                "best_model": best,
                "has_sorted_spike": any(model.startswith("sorted-spike") for model in models),
                "has_clusterless": any(model.startswith("clusterless") for model in models),
                "has_goal": any("goal" in model for model in models),
                "has_reverse": any("reverse" in model for model in models),
                "model_probability_entropy": probability_entropy,
                "models_scored": int(len(models)),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not margins.empty:
        out = out.merge(margins, on=list(group_cols), how="left")
    if "evidence_margin_category" in out:
        out["is_low_margin_disagreement"] = out["evidence_margin_category"].isin(["tie", "weak"])
    return out


def adversarial_synthetic_case_specs() -> pd.DataFrame:
    """Return a catalog of adversarial synthetic cases worth adding to recovery tests."""

    cases = [
        ("stationary_high_gain", "Stationary replay with high event gain; tests emission calibration."),
        ("fragmented_goal_like_endpoint", "Fragmented path whose final bin is near a goal; tests endpoint overinterpretation."),
        ("reverse_replay", "Continuous path traversed in reverse; tests direction hypotheses."),
        ("two_packet_replay", "Two separated replay packets inside one ripple; tests segmentation."),
        ("wrong_map_replay", "Spikes generated from a different environment map; tests map controls."),
        ("single_cell_dominated", "Evidence dominated by one high-rate cell; tests cell influence filters."),
        ("overdispersed_population", "Gamma-Poisson counts with population gain; tests overdispersion."),
        ("correlated_assembly", "Low-rank assembly coactivation added to place-field spikes."),
        ("clusterless_mark_drift", "Replay marks shifted relative to run-period marks; tests clusterless drift."),
        ("low_spike_short_event", "Sparse two-bin event; tests reliability thresholds."),
    ]
    return pd.DataFrame(cases, columns=["synthetic_case", "purpose"])


def provenance_audit(scores: pd.DataFrame, provenance: ProvenanceRecord | Mapping[str, object] | None = None) -> pd.DataFrame:
    """Return warnings about hyperparameter-selection provenance."""

    if provenance is None:
        record = ProvenanceRecord()
    elif isinstance(provenance, ProvenanceRecord):
        record = provenance
    else:
        record = ProvenanceRecord(**{key: provenance.get(key) for key in ProvenanceRecord.__dataclass_fields__})
    rows = [asdict(record)]
    warnings: list[str] = []
    if record.parameter_source in {"unknown", "manual", "real_selected"}:
        warnings.append("parameter_source is not synthetic/default pre-registered")
    if record.selection_used_real_evidence is True:
        warnings.append("selection_used_real_evidence=True; final evidence may be selection-biased")
    if record.selection_passed_recovery_gate is False:
        warnings.append("selection_passed_recovery_gate=False")
    if scores is not None and not scores.empty:
        settings_cols = [column for column in scores.columns if column.endswith("_run_id") or column.endswith("_source")]
        rows[0]["score_table_provenance_columns"] = ",".join(settings_cols)
    rows[0]["provenance_warnings"] = "; ".join(warnings)
    return pd.DataFrame(rows)


def write_dashboard(
    scores: pd.DataFrame,
    output_dir: str | Path,
    *,
    provenance: ProvenanceRecord | Mapping[str, object] | None = None,
) -> Path:
    """Write a compact Markdown dashboard and diagnostic CSVs for a run."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_with_margins = add_evidence_margin_columns(scores)
    margins = evidence_margin_table(scores)
    hierarchy = hierarchical_summary(scores_with_margins)
    disagreements = model_disagreement_events(scores_with_margins)
    prov = provenance_audit(scores, provenance)
    synthetic = adversarial_synthetic_case_specs()

    scores_with_margins.to_csv(out / "event_scores_with_margins.csv", index=False)
    margins.to_csv(out / "evidence_margins.csv", index=False)
    hierarchy.to_csv(out / "hierarchical_summary.csv", index=False)
    disagreements.to_csv(out / "model_disagreement_events.csv", index=False)
    prov.to_csv(out / "provenance_audit.csv", index=False)
    synthetic.to_csv(out / "adversarial_synthetic_case_specs.csv", index=False)

    lines = [
        "# Replay model-evidence diagnostic dashboard",
        "",
        f"Rows: {len(scores)}",
        f"Events: {scores[['session', 'event_index']].drop_duplicates().shape[0] if {'session', 'event_index'} <= set(scores.columns) else 'unknown'}",
        "",
        "## Evidence margin categories",
    ]
    if not margins.empty:
        lines.append(margins["evidence_margin_category"].value_counts().rename_axis("category").reset_index(name="events").to_markdown(index=False))
    else:
        lines.append("No exact-comparable evidence margins available.")
    lines.extend(["", "## Hierarchical summary", hierarchy.head(20).to_markdown(index=False) if not hierarchy.empty else "No hierarchical summary available."])
    lines.extend(["", "## Provenance audit", prov.to_markdown(index=False)])
    lines.extend(["", "## Suggested adversarial synthetic cases", synthetic.to_markdown(index=False)])
    path = out / "diagnostic_dashboard.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
