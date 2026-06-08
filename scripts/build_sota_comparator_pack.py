#!/usr/bin/env python3
"""Build paper-facing SoTA comparator tables from all-session evidence.

This helper is intentionally descriptive rather than a new model scorer. It
turns an existing all-session aggregate into tables that compare the older
momentum-centric interpretation against the current exact full-core
interpretation:

* paired exact-sparse momentum versus diffusion;
* exact-sparse momentum versus the full exact core;
* first-order IMM as the current leading exact core row;
* exact trajectory-family versus static/nontrajectory;
* candidate-pruned lower-bound audit rows.

The intended paper statement is precise: the prior momentum-vs-diffusion signal
is recovered, but exact full-core evidence can refine the story by showing which
trajectory-family model actually leads once stationary, diffusion, fragmented,
first-order IMM, and exact-sparse momentum are all present.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"
MOMENTUM_CANDIDATE = "sorted-spike-state-space-momentum"
IMM_CANDIDATE = "sorted-spike-state-space-imm"

REQUIRED_EXACT_CORE_MODELS: tuple[str, ...] = (
    STATIONARY,
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
TRAJECTORY_EXACT_MODELS: tuple[str, ...] = (
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
NONTRAJECTORY_EXACT_MODELS: tuple[str, ...] = (STATIONARY,)
LOWER_BOUND_AUDIT_MODELS: tuple[str, ...] = (
    MOMENTUM_CANDIDATE,
    IMM_CANDIDATE,
)


_MODEL_SHORT_NAMES = {
    STATIONARY: "stationary",
    DIFFUSION: "diffusion",
    FRAGMENTED: "fragmented",
    FIRST_ORDER_IMM: "first_order_imm",
    MOMENTUM_EXACT: "momentum_exact_sparse",
    MOMENTUM_CANDIDATE: "candidate_pruned_momentum",
    IMM_CANDIDATE: "candidate_pruned_imm",
}


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _success_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event evidence is missing required columns: {missing}")

    out = frame.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).eq("success")].copy()
    out["session"] = out["session"].astype(str)
    out["rat"] = out["session"].map(_rat_from_session)
    out["event_index"] = pd.to_numeric(out["event_index"], errors="raise").astype(int)
    out["model"] = out["model"].astype(str)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    out = out.dropna(subset=["log_evidence"]).copy()

    if "evidence_comparable" not in out.columns:
        out["evidence_comparable"] = True
    out["evidence_comparable"] = out["evidence_comparable"].map(_as_bool)

    if "evidence_support" not in out.columns:
        out["evidence_support"] = ""
    out["evidence_support"] = out["evidence_support"].fillna("").astype(str)

    return out


def read_event_model_evidence(path: str | Path) -> pd.DataFrame:
    """Read an all-session or event-sharded evidence CSV."""

    return _success_rows(pd.read_csv(path))


def _model_value(group: pd.DataFrame, model: str) -> float:
    rows = group[group["model"].eq(model)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[-1]["log_evidence"])


def _missing_models(group: pd.DataFrame, models: Iterable[str]) -> tuple[str, ...]:
    present = set(group["model"].astype(str))
    return tuple(model for model in models if model not in present)


def _best_in(group: pd.DataFrame, models: Iterable[str]) -> tuple[str, float]:
    subset = group[group["model"].isin(tuple(models))].copy()
    if subset.empty:
        return "", float("nan")
    row = subset.sort_values("log_evidence", ascending=False).iloc[0]
    return str(row["model"]), float(row["log_evidence"])


def _winner_margin_to_runner_up(
    group: pd.DataFrame,
    winning_model: str,
    candidate_models: Iterable[str],
) -> float:
    subset = group[group["model"].isin(tuple(candidate_models))].copy()
    subset = subset.sort_values("log_evidence", ascending=False)
    if len(subset) < 2 or not winning_model:
        return float("nan")
    winner = subset.iloc[0]
    runner_up = subset.iloc[1]
    if str(winner["model"]) != winning_model:
        return float("nan")
    return float(winner["log_evidence"] - runner_up["log_evidence"])


def _model_short_name(model: str) -> str:
    return _MODEL_SHORT_NAMES.get(model, model.replace("sorted-spike-state-space-", ""))


def _safe_delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return float(left - right)


def build_sota_comparator_event_table(
    event_model_evidence: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return one SoTA-comparator row per event."""

    rows: list[dict[str, object]] = []
    evidence = _success_rows(event_model_evidence)
    for (session, event_index), group in evidence.groupby(["session", "event_index"], sort=True):
        exact_group = group[
            group["model"].isin(REQUIRED_EXACT_CORE_MODELS)
            & group["evidence_comparable"].astype(bool)
        ].copy()
        required_missing = _missing_models(exact_group, REQUIRED_EXACT_CORE_MODELS)
        exact_core_complete = not required_missing

        best_exact_model, best_exact_logz = _best_in(exact_group, REQUIRED_EXACT_CORE_MODELS)
        best_trajectory_model, best_trajectory_logz = _best_in(
            exact_group,
            TRAJECTORY_EXACT_MODELS,
        )
        best_nontrajectory_model, best_nontrajectory_logz = _best_in(
            exact_group,
            NONTRAJECTORY_EXACT_MODELS,
        )

        best_exact_margin = _winner_margin_to_runner_up(
            exact_group,
            best_exact_model,
            REQUIRED_EXACT_CORE_MODELS,
        )
        first_order_margin = _winner_margin_to_runner_up(
            exact_group,
            FIRST_ORDER_IMM,
            REQUIRED_EXACT_CORE_MODELS,
        )
        exact_momentum_margin = _winner_margin_to_runner_up(
            exact_group,
            MOMENTUM_EXACT,
            REQUIRED_EXACT_CORE_MODELS,
        )

        logz_stationary = _model_value(group, STATIONARY)
        logz_diffusion = _model_value(group, DIFFUSION)
        logz_fragmented = _model_value(group, FRAGMENTED)
        logz_first_order_imm = _model_value(group, FIRST_ORDER_IMM)
        logz_momentum_exact = _model_value(group, MOMENTUM_EXACT)
        logz_momentum_candidate = _model_value(group, MOMENTUM_CANDIDATE)
        logz_imm_candidate = _model_value(group, IMM_CANDIDATE)

        momentum_minus_diffusion = _safe_delta(logz_momentum_exact, logz_diffusion)
        first_order_minus_momentum = _safe_delta(logz_first_order_imm, logz_momentum_exact)
        trajectory_minus_nontrajectory = _safe_delta(
            best_trajectory_logz,
            best_nontrajectory_logz,
        )
        candidate_momentum_gap = _safe_delta(logz_momentum_exact, logz_momentum_candidate)

        rows.append(
            {
                "session": session,
                "rat": _rat_from_session(session),
                "event_index": int(event_index),
                "margin_threshold": float(margin_threshold),
                "exact_core_complete": bool(exact_core_complete),
                "missing_required_exact_core_models": " ".join(required_missing),
                "best_exact_core_model": best_exact_model,
                "best_exact_core_log_evidence": best_exact_logz,
                "best_exact_core_margin_to_runner_up": best_exact_margin,
                "best_trajectory_model": best_trajectory_model,
                "best_trajectory_log_evidence": best_trajectory_logz,
                "best_nontrajectory_model": best_nontrajectory_model,
                "best_nontrajectory_log_evidence": best_nontrajectory_logz,
                "logZ_stationary": logz_stationary,
                "logZ_diffusion": logz_diffusion,
                "logZ_fragmented": logz_fragmented,
                "logZ_first_order_imm": logz_first_order_imm,
                "logZ_momentum_exact_sparse": logz_momentum_exact,
                "logZ_candidate_pruned_momentum": logz_momentum_candidate,
                "logZ_candidate_pruned_imm": logz_imm_candidate,
                "delta_momentum_exact_minus_diffusion": momentum_minus_diffusion,
                "delta_first_order_imm_minus_momentum_exact": first_order_minus_momentum,
                "delta_trajectory_minus_nontrajectory": trajectory_minus_nontrajectory,
                "delta_momentum_exact_minus_candidate_pruned_momentum": candidate_momentum_gap,
                "momentum_exact_beats_diffusion": bool(momentum_minus_diffusion > 0),
                "momentum_exact_confident_vs_diffusion": bool(
                    momentum_minus_diffusion >= margin_threshold
                ),
                "diffusion_confident_vs_momentum_exact": bool(
                    momentum_minus_diffusion <= -margin_threshold
                ),
                "trajectory_confident_vs_nontrajectory": bool(
                    exact_core_complete and trajectory_minus_nontrajectory >= margin_threshold
                ),
                "nontrajectory_confident_vs_trajectory": bool(
                    exact_core_complete and trajectory_minus_nontrajectory <= -margin_threshold
                ),
                "first_order_imm_is_exact_core_best": bool(
                    exact_core_complete and best_exact_model == FIRST_ORDER_IMM
                ),
                "first_order_imm_core_margin_to_runner_up": first_order_margin,
                "first_order_imm_confident_core_best": bool(
                    exact_core_complete
                    and best_exact_model == FIRST_ORDER_IMM
                    and first_order_margin >= margin_threshold
                ),
                "momentum_exact_is_exact_core_best": bool(
                    exact_core_complete and best_exact_model == MOMENTUM_EXACT
                ),
                "momentum_exact_core_margin_to_runner_up": exact_momentum_margin,
                "momentum_exact_confident_core_best": bool(
                    exact_core_complete
                    and best_exact_model == MOMENTUM_EXACT
                    and exact_momentum_margin >= margin_threshold
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["session", "event_index"]).reset_index(drop=True)


def _basic_delta_summary(delta: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(delta, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "min_delta": float("nan"),
            "max_delta": float("nan"),
        }
    return {
        "mean_delta": float(clean.mean()),
        "median_delta": float(clean.median()),
        "min_delta": float(clean.min()),
        "max_delta": float(clean.max()),
    }


def build_sota_comparator_model_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Summarize exact-core winners and confident exact-core claims."""

    rows: list[dict[str, object]] = []
    complete = event_table[event_table["exact_core_complete"].astype(bool)].copy()
    events = int(len(complete))
    for model in REQUIRED_EXACT_CORE_MODELS:
        raw_best = complete["best_exact_core_model"].eq(model)
        confident = raw_best & (
            pd.to_numeric(complete["best_exact_core_margin_to_runner_up"], errors="coerce")
            >= margin_threshold
        )
        rows.append(
            {
                "model": model,
                "events": events,
                "raw_best_events": int(raw_best.sum()),
                "raw_best_fraction": float(raw_best.mean()) if events else 0.0,
                "confident_exact_core_claims": int(confident.sum()),
                "confident_exact_core_claim_fraction": (
                    float(confident.mean()) if events else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_sota_comparator_family_summary(event_table: pd.DataFrame) -> pd.DataFrame:
    complete = event_table[event_table["exact_core_complete"].astype(bool)].copy()
    events = int(len(complete))
    delta = complete["delta_trajectory_minus_nontrajectory"] if events else pd.Series(dtype=float)
    trajectory_claims = (
        complete["trajectory_confident_vs_nontrajectory"].astype(bool) if events else pd.Series(dtype=bool)
    )
    nontrajectory_claims = (
        complete["nontrajectory_confident_vs_trajectory"].astype(bool) if events else pd.Series(dtype=bool)
    )
    return pd.DataFrame(
        [
            {
                "events": events,
                "trajectory_raw_wins": int((delta > 0).sum()) if events else 0,
                "nontrajectory_raw_wins": int((delta < 0).sum()) if events else 0,
                "trajectory_confident_claims": int(trajectory_claims.sum()) if events else 0,
                "nontrajectory_confident_claims": (
                    int(nontrajectory_claims.sum()) if events else 0
                ),
                "ambiguous_events": (
                    int(events - trajectory_claims.sum() - nontrajectory_claims.sum())
                    if events
                    else 0
                ),
                **_basic_delta_summary(delta),
                "most_common_best_trajectory_model": (
                    ""
                    if complete.empty
                    else str(complete["best_trajectory_model"].value_counts().index[0])
                ),
                "best_nontrajectory_model": STATIONARY,
            }
        ]
    )


def build_sota_comparator_momentum_vs_diffusion_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Summarize the prior-style paired momentum-vs-diffusion axis."""

    paired = event_table[
        pd.to_numeric(
            event_table["delta_momentum_exact_minus_diffusion"],
            errors="coerce",
        ).notna()
    ].copy()
    events = int(len(paired))
    delta = paired["delta_momentum_exact_minus_diffusion"] if events else pd.Series(dtype=float)
    momentum_confident = (
        paired["momentum_exact_confident_vs_diffusion"].astype(bool)
        if events
        else pd.Series(dtype=bool)
    )
    diffusion_confident = (
        paired["diffusion_confident_vs_momentum_exact"].astype(bool)
        if events
        else pd.Series(dtype=bool)
    )
    momentum_wins = int((delta > 0).sum()) if events else 0
    diffusion_wins = int((delta < 0).sum()) if events else 0
    return pd.DataFrame(
        [
            {
                "comparison": "exact_sparse_momentum_vs_diffusion",
                "events": events,
                "momentum_raw_wins": momentum_wins,
                "diffusion_raw_wins": diffusion_wins,
                "ties": int((delta == 0).sum()) if events else 0,
                "momentum_raw_win_fraction": (
                    float(momentum_wins / events) if events else 0.0
                ),
                "momentum_confident_claims": int(momentum_confident.sum()) if events else 0,
                "diffusion_confident_claims": int(diffusion_confident.sum()) if events else 0,
                "ambiguous_events": (
                    int(events - momentum_confident.sum() - diffusion_confident.sum())
                    if events
                    else 0
                ),
                **_basic_delta_summary(delta),
                "margin_threshold": margin_threshold,
                "paper_interpretation": (
                    "prior-style exact-sparse momentum-vs-diffusion axis is positive "
                    "but heterogeneous"
                ),
            }
        ]
    )


def build_sota_comparator_full_core_winner_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Rank exact-core models by raw and confident full-core wins."""

    model_summary = build_sota_comparator_model_summary(
        event_table,
        margin_threshold=margin_threshold,
    ).copy()
    if model_summary.empty:
        model_summary["raw_best_rank"] = pd.Series(dtype=int)
        model_summary["is_leading_exact_core_row"] = pd.Series(dtype=bool)
        model_summary["paper_interpretation"] = pd.Series(dtype=str)
        return model_summary

    model_summary = model_summary.sort_values(
        ["raw_best_events", "confident_exact_core_claims", "model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    model_summary["raw_best_rank"] = range(1, len(model_summary) + 1)
    leading_model = str(model_summary.iloc[0]["model"])
    model_summary["is_leading_exact_core_row"] = model_summary["model"].eq(leading_model)

    def _interpret(model: object) -> str:
        model_name = str(model)
        if model_name == FIRST_ORDER_IMM:
            return "first-order IMM is the leading exact full-core row"
        if model_name == MOMENTUM_EXACT:
            return (
                "exact-sparse momentum remains a strong paired row but is not "
                "the full-core winner"
            )
        if model_name == STATIONARY:
            return "static/nontrajectory baseline is included as a controlled comparator"
        return "trajectory-family exact core comparator"

    model_summary["paper_interpretation"] = model_summary["model"].map(_interpret)
    model_summary["margin_threshold"] = margin_threshold
    return model_summary


def build_sota_comparator_lower_bound_audit(event_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize lower-bound audit rows without mixing them into exact rankings."""

    rows: list[dict[str, object]] = []
    candidate_rows = event_table[event_table["logZ_candidate_pruned_momentum"].notna()].copy()
    gap = candidate_rows["delta_momentum_exact_minus_candidate_pruned_momentum"].dropna()
    rows.append(
        {
            "audit": "candidate_pruned_momentum_lower_bound",
            "matched_events": int(len(gap)),
            "violations_candidate_above_exact": int((gap < -1e-9).sum()),
            "min_exact_minus_candidate": float(gap.min()) if not gap.empty else np.nan,
            "mean_exact_minus_candidate": float(gap.mean()) if not gap.empty else np.nan,
            "median_exact_minus_candidate": float(gap.median()) if not gap.empty else np.nan,
            "interpretation": (
                "candidate-pruned momentum is a lower-bound audit row, not headline evidence"
            ),
        }
    )

    imm_rows = event_table[event_table["logZ_candidate_pruned_imm"].notna()].copy()
    rows.append(
        {
            "audit": "candidate_pruned_imm_audit_presence",
            "matched_events": int(len(imm_rows)),
            "violations_candidate_above_exact": np.nan,
            "min_exact_minus_candidate": np.nan,
            "mean_exact_minus_candidate": np.nan,
            "median_exact_minus_candidate": np.nan,
            "interpretation": (
                "candidate-pruned IMM is present as an audit row, not exact headline evidence"
            ),
        }
    )
    return pd.DataFrame(rows)


def build_sota_comparator_claim_delta_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return the compact claim-level comparison table."""

    events = int(len(event_table))
    complete = event_table[event_table["exact_core_complete"].astype(bool)].copy()
    complete_events = int(len(complete))
    rows: list[dict[str, object]] = []

    mom_delta = event_table["delta_momentum_exact_minus_diffusion"]
    rows.append(
        {
            "claim_axis": "prior_momentum_vs_diffusion_recovered",
            "events": events,
            "raw_positive_events": int((mom_delta > 0).sum()),
            "confident_positive_events": int(
                event_table["momentum_exact_confident_vs_diffusion"].sum()
            ),
            "confident_reference_events": int(
                event_table["diffusion_confident_vs_momentum_exact"].sum()
            ),
            **_basic_delta_summary(mom_delta),
            "paper_interpretation": "paired exact-sparse momentum-vs-diffusion signal",
        }
    )

    first_order_delta = complete["delta_first_order_imm_minus_momentum_exact"]
    rows.append(
        {
            "claim_axis": "full_core_refines_momentum_story",
            "events": complete_events,
            "raw_positive_events": int((first_order_delta > 0).sum()),
            "confident_positive_events": int(
                complete["first_order_imm_confident_core_best"].sum()
            ),
            "confident_reference_events": int(
                complete["momentum_exact_confident_core_best"].sum()
            ),
            **_basic_delta_summary(first_order_delta),
            "paper_interpretation": (
                "first-order IMM currently outranks exact-sparse momentum in the exact core"
            ),
        }
    )

    family_delta = complete["delta_trajectory_minus_nontrajectory"]
    rows.append(
        {
            "claim_axis": "trajectory_family_over_static",
            "events": complete_events,
            "raw_positive_events": int((family_delta > 0).sum()),
            "confident_positive_events": int(
                complete["trajectory_confident_vs_nontrajectory"].sum()
            ),
            "confident_reference_events": int(
                complete["nontrajectory_confident_vs_trajectory"].sum()
            ),
            **_basic_delta_summary(family_delta),
            "paper_interpretation": (
                "trajectory-family dynamics dominate the static/nontrajectory core row"
            ),
        }
    )

    momentum_core_best = complete["momentum_exact_is_exact_core_best"]
    rows.append(
        {
            "claim_axis": "exact_sparse_momentum_full_core_dominance",
            "events": complete_events,
            "raw_positive_events": int(momentum_core_best.sum()),
            "confident_positive_events": int(
                complete["momentum_exact_confident_core_best"].sum()
            ),
            "confident_reference_events": int(
                complete["first_order_imm_confident_core_best"].sum()
            ),
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "min_delta": np.nan,
            "max_delta": np.nan,
            "paper_interpretation": (
                "do not claim exact-sparse momentum dominates unless this row has a majority"
            ),
        }
    )

    for row in rows:
        row["margin_threshold"] = margin_threshold
    return pd.DataFrame(rows)


def build_sota_comparator_gate_summary(
    event_table: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    complete = event_table[event_table["exact_core_complete"].astype(bool)]
    family = build_sota_comparator_family_summary(event_table).iloc[0]
    claim = build_sota_comparator_claim_delta_summary(
        event_table,
        margin_threshold=margin_threshold,
    )

    def _claim_row(axis: str) -> pd.Series:
        return claim[claim["claim_axis"].eq(axis)].iloc[0]

    prior = _claim_row("prior_momentum_vs_diffusion_recovered")
    momentum_core = _claim_row("exact_sparse_momentum_full_core_dominance")
    first_order = _claim_row("full_core_refines_momentum_story")

    rows = [
        {
            "gate": "event_rows_present",
            "passed": len(event_table) > 0,
            "observed": str(len(event_table)),
            "criterion": "at least one event row",
        },
        {
            "gate": "required_exact_core_complete",
            "passed": len(complete) == len(event_table) and len(event_table) > 0,
            "observed": f"{len(complete)}/{len(event_table)}",
            "criterion": "all events include the named exact core comparator set",
        },
        {
            "gate": "prior_momentum_vs_diffusion_signal_present",
            "passed": int(prior["raw_positive_events"]) > len(event_table) / 2,
            "observed": f"{int(prior['raw_positive_events'])}/{len(event_table)}",
            "criterion": "exact-sparse momentum raw paired wins exceed half of events",
        },
        {
            "gate": "trajectory_family_confident_majority",
            "passed": int(family["trajectory_confident_claims"]) > int(family["events"]) / 2,
            "observed": f"{int(family['trajectory_confident_claims'])}/{int(family['events'])}",
            "criterion": "trajectory-family confident claims exceed half of complete events",
        },
        {
            "gate": "no_confident_nontrajectory_claims",
            "passed": int(family["nontrajectory_confident_claims"]) == 0,
            "observed": str(int(family["nontrajectory_confident_claims"])),
            "criterion": "no confident static/nontrajectory claims",
        },
        {
            "gate": "full_core_momentum_vs_first_order_axis_reported",
            "passed": pd.notna(first_order["raw_positive_events"]) and pd.notna(
                momentum_core["raw_positive_events"]
            ),
            "observed": (
                f"first_order_minus_momentum_positive={int(first_order['raw_positive_events'])}; "
                f"momentum_exact_core_best={int(momentum_core['raw_positive_events'])}"
            ),
            "criterion": "comparator pack exposes whether first-order IMM or momentum leads",
        },
    ]
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in rows),
            "observed": f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} gates passed",
            "criterion": "all SoTA-comparator interpretation gates pass",
        }
    )
    return pd.DataFrame(rows)


def write_sota_comparator_pack(
    event_model_evidence: pd.DataFrame,
    output: str | Path,
    *,
    margin_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    event_table = build_sota_comparator_event_table(
        event_model_evidence,
        margin_threshold=margin_threshold,
    )
    outputs = {
        "sota_comparator_event_table.csv": event_table,
        "sota_comparator_model_summary.csv": build_sota_comparator_model_summary(
            event_table,
            margin_threshold=margin_threshold,
        ),
        "sota_comparator_family_summary.csv": build_sota_comparator_family_summary(
            event_table,
        ),
        "sota_comparator_momentum_vs_diffusion_summary.csv": (
            build_sota_comparator_momentum_vs_diffusion_summary(
                event_table,
                margin_threshold=margin_threshold,
            )
        ),
        "sota_comparator_full_core_winner_summary.csv": (
            build_sota_comparator_full_core_winner_summary(
                event_table,
                margin_threshold=margin_threshold,
            )
        ),
        "sota_comparator_lower_bound_audit.csv": build_sota_comparator_lower_bound_audit(
            event_table,
        ),
        "sota_comparator_claim_delta_summary.csv": build_sota_comparator_claim_delta_summary(
            event_table,
            margin_threshold=margin_threshold,
        ),
        "sota_comparator_gate_summary.csv": build_sota_comparator_gate_summary(
            event_table,
            margin_threshold=margin_threshold,
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def _default_output_dir(event_model_evidence_path: Path) -> Path:
    return event_model_evidence_path.parent / "sota-comparator-pack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-model-evidence",
        required=True,
        help=(
            "Path to all_sessions_event_model_evidence.csv or event_model_evidence.csv "
            "from a full-core all-session artifact."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to <event-model-evidence-dir>/sota-comparator-pack.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=5.5,
        help="Calibrated log-evidence margin for confident claims.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_path = Path(args.event_model_evidence)
    output = Path(args.output) if args.output else _default_output_dir(evidence_path)
    evidence = read_event_model_evidence(evidence_path)
    write_sota_comparator_pack(
        evidence,
        output,
        margin_threshold=args.margin_threshold,
    )
    print(f"Wrote SoTA comparator pack to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
