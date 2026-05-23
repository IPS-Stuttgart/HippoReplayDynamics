#!/usr/bin/env python3
"""Paper-facing effect summaries for state-space replay benchmarks.

The benchmark can produce very strong event-level effects while still mixing
three evidence regimes: exact full-grid rows, candidate-pruned lower bounds, and
failed/missing rows.  This script writes analysis tables that keep those regimes
separate and summarize the two claims that are currently most defensible:

* trajectory dynamics versus non-trajectory baselines; and
* momentum versus diffusion, including lower-bound-certified wins.

The intended input is an ``event_model_evidence.csv`` artifact directory or a
CSV with the same columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)

_EVENT_KEY_CANDIDATES = (
    "session",
    "event_index",
    "window_index",
    "benchmark_cell_split_index",
    "event_window_variant",
)
_TRAJECTORY_MODELS = {
    "diffusion",
    "fragmented",
    "jump",
    "momentum",
    "momentum-reverse",
    "momentum-bidirectional",
    "displacement-momentum",
    "displacement-imm",
    "imm",
    "first-order-imm",
    "goal",
    "goal-reverse",
    "goal-bidirectional",
}
_NONTRAJECTORY_MODELS = {"random", "stationary", "stationary-gaussian"}
_DEFAULT_INPUT_NAMES = (
    "event_model_evidence.csv",
    "state_space_marginalized_event_model_evidence.csv",
    "simulation_recovery_event_scores.csv",
)


def summarize_paper_effects(
    input_path: str | Path,
    output: str | Path,
    *,
    bootstrap_replicates: int = 5000,
    random_seed: int = 1,
    exact_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Write paper-facing event, session, and summary effect tables."""

    scores = load_event_scores(input_path, exact_only=exact_only)
    event_effects = event_effect_table(scores)
    session_effects = session_effect_table(event_effects)
    summary = summary_table(
        event_effects,
        session_effects,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
        exact_only=exact_only,
    )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_effects.to_csv(out_dir / "paper_state_space_event_effects.csv", index=False)
    session_effects.to_csv(out_dir / "paper_state_space_session_effects.csv", index=False)
    summary.to_csv(out_dir / "paper_state_space_effect_summary.csv", index=False)
    (out_dir / "paper_state_space_claims.md").write_text(
        claims_markdown(summary, session_effects, exact_only=exact_only),
        encoding="utf-8",
    )
    return {
        "event_effects": event_effects,
        "session_effects": session_effects,
        "summary": summary,
    }


def load_event_scores(input_path: str | Path, *, exact_only: bool = False) -> pd.DataFrame:
    """Load and normalize an event-model-evidence table."""

    path = _resolve_input_path(input_path)
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if "status" not in frame.columns:
        frame["status"] = "success"
    frame = ensure_evidence_support_columns(frame)
    frame = frame[frame["status"].astype(str).eq("success")].copy()
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    frame = frame[np.isfinite(frame["log_evidence"].to_numpy(float))].copy()
    if exact_only:
        frame = frame[frame["evidence_comparable"].fillna(False).astype(bool)].copy()
    frame["canonical_model"] = frame["model"].map(canonical_model_name)
    frame["canonical_model_family"] = frame["canonical_model"].map(model_family)
    return frame.reset_index(drop=True)


def event_effect_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Return one row per event with trajectory and momentum/diffusion effects."""

    if scores.empty:
        return pd.DataFrame(columns=[*_event_key_columns(scores), "events_in_input"])

    rows: list[dict[str, object]] = []
    group_columns = _event_key_columns(scores)
    for event_key, group in scores.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(event_key, tuple):
            event_key = (event_key,)
        row: dict[str, object] = {
            column: value for column, value in zip(group_columns, event_key, strict=True)
        }
        row.update(_event_counts(group))
        row.update(_trajectory_effects(group))
        row.update(_momentum_diffusion_effects(group))
        rows.append(row)
    return pd.DataFrame(rows)


def session_effect_table(event_effects: pd.DataFrame) -> pd.DataFrame:
    """Summarize effect sizes separately by recording session."""

    columns = [
        "session",
        "events",
        "trajectory_strict_reference_events",
        "trajectory_strict_wins",
        "trajectory_strict_win_fraction",
        "trajectory_certified_reference_events",
        "trajectory_certified_wins",
        "trajectory_certified_win_fraction",
        "momentum_diffusion_paired_events",
        "momentum_reported_wins",
        "momentum_reported_win_fraction",
        "momentum_certified_wins",
        "momentum_certified_win_fraction",
        "momentum_certified_losses",
        "momentum_inconclusive_lower_bound_nonwins",
        "mean_momentum_minus_diffusion_log_evidence",
        "median_momentum_minus_diffusion_log_evidence",
        "momentum_lower_bound_events",
        "momentum_exact_events",
        "session_interpretation",
    ]
    if event_effects.empty or "session" not in event_effects.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for session, group in event_effects.groupby("session", sort=True, dropna=False):
        strict = _bool_series(group, "trajectory_strict_win")
        certified = _bool_series(group, "trajectory_certified_win")
        reported = _bool_series(group, "momentum_beats_diffusion_reported")
        certified_momentum = _bool_series(group, "momentum_beats_diffusion_certified")
        certified_loss = _bool_series(group, "diffusion_beats_momentum_certified")
        deltas = pd.to_numeric(
            group.get("momentum_minus_diffusion_log_evidence", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        mean_delta = float(deltas.mean()) if not deltas.empty else float("nan")
        reported_fraction = _mean_or_nan(reported)
        row = {
            "session": session,
            "events": int(len(group)),
            "trajectory_strict_reference_events": int(strict.notna().sum()),
            "trajectory_strict_wins": _true_count(strict),
            "trajectory_strict_win_fraction": _mean_or_nan(strict),
            "trajectory_certified_reference_events": int(certified.notna().sum()),
            "trajectory_certified_wins": _true_count(certified),
            "trajectory_certified_win_fraction": _mean_or_nan(certified),
            "momentum_diffusion_paired_events": int(deltas.shape[0]),
            "momentum_reported_wins": _true_count(reported),
            "momentum_reported_win_fraction": reported_fraction,
            "momentum_certified_wins": _true_count(certified_momentum),
            "momentum_certified_win_fraction": _mean_or_nan(certified_momentum),
            "momentum_certified_losses": _true_count(certified_loss),
            "momentum_inconclusive_lower_bound_nonwins": int(
                group.get(
                    "momentum_vs_diffusion_certification",
                    pd.Series(dtype=object),
                )
                .astype(str)
                .eq("momentum_lower_bound_not_above_exact_diffusion")
                .sum()
            ),
            "mean_momentum_minus_diffusion_log_evidence": mean_delta,
            "median_momentum_minus_diffusion_log_evidence": float(deltas.median()) if not deltas.empty else float("nan"),
            "momentum_lower_bound_events": int(
                group.get("momentum_evidence_support", pd.Series(dtype=object))
                .astype(str)
                .eq(TRUNCATED_EVIDENCE_SUPPORT)
                .sum()
            ),
            "momentum_exact_events": int(
                group.get("momentum_evidence_support", pd.Series(dtype=object))
                .astype(str)
                .eq(EXACT_EVIDENCE_SUPPORT)
                .sum()
            ),
        }
        row["session_interpretation"] = _session_interpretation(mean_delta, reported_fraction)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def summary_table(
    event_effects: pd.DataFrame,
    session_effects: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    random_seed: int,
    exact_only: bool,
) -> pd.DataFrame:
    """Return a one-row paper-effect summary with CIs and sign tests."""

    row: dict[str, object] = {
        "events": int(len(event_effects)),
        "sessions": int(event_effects["session"].nunique()) if "session" in event_effects else 0,
        "evidence_policy": "exact_only" if exact_only else "exact_plus_lower_bound_certification",
        "bootstrap_replicates": int(bootstrap_replicates),
        "random_seed": int(random_seed),
    }
    row.update(
        _boolean_effect_summary(
            event_effects,
            "trajectory_strict_win",
            "trajectory_strict",
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed,
        )
    )
    row.update(
        _boolean_effect_summary(
            event_effects,
            "trajectory_certified_win",
            "trajectory_certified",
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed + 1,
        )
    )
    row.update(
        _boolean_effect_summary(
            event_effects,
            "momentum_beats_diffusion_reported",
            "momentum_vs_diffusion_reported",
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed + 2,
        )
    )
    row.update(
        _boolean_effect_summary(
            event_effects,
            "momentum_beats_diffusion_certified",
            "momentum_vs_diffusion_certified",
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed + 3,
        )
    )
    row.update(_momentum_delta_summary(event_effects, bootstrap_replicates, random_seed + 4))

    certified_loss = _bool_series(event_effects, "diffusion_beats_momentum_certified")
    row["momentum_vs_diffusion_certified_losses"] = _true_count(certified_loss)
    row["momentum_vs_diffusion_inconclusive_lower_bound_nonwins"] = int(
        event_effects.get("momentum_vs_diffusion_certification", pd.Series(dtype=object))
        .astype(str)
        .eq("momentum_lower_bound_not_above_exact_diffusion")
        .sum()
    )
    if not session_effects.empty:
        row["sessions_with_positive_mean_momentum_delta"] = int(
            (session_effects["mean_momentum_minus_diffusion_log_evidence"] > 0.0).sum()
        )
        row["sessions_with_negative_mean_momentum_delta"] = int(
            (session_effects["mean_momentum_minus_diffusion_log_evidence"] < 0.0).sum()
        )
        row["sessions_momentum_dominant"] = int(
            session_effects["session_interpretation"].eq("momentum-dominant").sum()
        )
        row["sessions_diffusion_dominant"] = int(
            session_effects["session_interpretation"].eq("diffusion-dominant").sum()
        )
    else:
        row["sessions_with_positive_mean_momentum_delta"] = 0
        row["sessions_with_negative_mean_momentum_delta"] = 0
        row["sessions_momentum_dominant"] = 0
        row["sessions_diffusion_dominant"] = 0
    return pd.DataFrame([row])


def canonical_model_name(model: object) -> str:
    """Map implementation-specific model names to dynamics labels."""

    name = str(model).strip().lower().replace("_", "-")
    for prefix in (
        "sorted-spike-state-space-",
        "clusterless-state-space-",
        "state-space-",
    ):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    if name.endswith("-marginalized"):
        name = name.removesuffix("-marginalized")
    if name in {"velocity", "finite-displacement-momentum"}:
        return "displacement-momentum"
    if name in {"finite-displacement-imm", "displacement-imm"}:
        return "displacement-imm"
    if name == "jump":
        return "fragmented"
    return name


def model_family(canonical_model: object) -> str:
    name = str(canonical_model).strip().lower()
    if name in _TRAJECTORY_MODELS:
        return "trajectory"
    if name in _NONTRAJECTORY_MODELS:
        return "nontrajectory"
    return "other"


def claims_markdown(summary: pd.DataFrame, session_effects: pd.DataFrame, *, exact_only: bool) -> str:
    """Render a compact, caveated claims block for reports/manuscripts."""

    if summary.empty:
        return "# State-space replay effect report\n\nNo summary rows were available.\n"
    row = summary.iloc[0]
    policy = "exact comparable evidence only" if exact_only else "exact evidence plus lower-bound-certified wins"
    lines = [
        "# State-space replay effect report",
        "",
        f"Evidence policy: **{policy}**.",
        "",
        "## Main effects",
        "",
        _claim_fraction(
            row,
            label="Trajectory models versus nontrajectory baselines",
            prefix="trajectory_certified",
        ),
        _claim_fraction(
            row,
            label="Momentum lower-bound/exact wins versus diffusion",
            prefix="momentum_vs_diffusion_certified",
        ),
        _claim_delta(row),
        "",
        "## Interpretation guardrails",
        "",
        "Momentum and IMM candidate-supported rows can be truncated full-grid lower bounds. "
        "The finite-displacement momentum row is exact over its declared augmented state space. "
        "A candidate lower-bound row is counted as a certified win only when it exceeds the exact comparator. "
        "A lower bound below an exact comparator is marked inconclusive rather than as a certified loss.",
        "",
        "## Session heterogeneity",
        "",
        f"Sessions with positive mean momentum-minus-diffusion delta: {int(row.get('sessions_with_positive_mean_momentum_delta', 0))}. "
        f"Sessions with negative mean delta: {int(row.get('sessions_with_negative_mean_momentum_delta', 0))}.",
    ]
    if not session_effects.empty:
        compact = session_effects[
            [
                "session",
                "events",
                "momentum_reported_win_fraction",
                "mean_momentum_minus_diffusion_log_evidence",
                "session_interpretation",
            ]
        ].copy()
        lines.extend(["", _markdown_table(compact)])
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    rows = [[_markdown_cell(value) for value in record] for record in frame.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _markdown_cell(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return _fmt_number(value)
    return str(value).replace("|", "\\|")


def _resolve_input_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    if path.is_dir():
        for name in _DEFAULT_INPUT_NAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{path} does not contain any of: {', '.join(_DEFAULT_INPUT_NAMES)}"
        )
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def _event_key_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _EVENT_KEY_CANDIDATES if column in frame.columns]


def _event_counts(group: pd.DataFrame) -> dict[str, object]:
    return {
        "successful_model_rows": int(len(group)),
        "exact_model_rows": int(group["evidence_comparable"].fillna(False).astype(bool).sum()),
        "truncated_lower_bound_rows": int(group["evidence_support"].astype(str).eq(TRUNCATED_EVIDENCE_SUPPORT).sum()),
        "models_scored": int(group["canonical_model"].nunique()),
    }


def _trajectory_effects(group: pd.DataFrame) -> dict[str, object]:
    exact = group[group["evidence_comparable"].fillna(False).astype(bool)].copy()
    exact_trajectory = exact[exact["canonical_model_family"].eq("trajectory")]
    exact_nontrajectory = exact[exact["canonical_model_family"].eq("nontrajectory")]
    any_trajectory = group[group["canonical_model_family"].eq("trajectory")]
    lower_bound_trajectory = any_trajectory[any_trajectory["evidence_support"].astype(str).eq(TRUNCATED_EVIDENCE_SUPPORT)]

    strict_best = _best_row(exact)
    best_exact_trajectory = _best_row(exact_trajectory)
    best_exact_nontrajectory = _best_row(exact_nontrajectory)
    best_lower_bound_trajectory = _best_row(lower_bound_trajectory)

    out: dict[str, object] = {
        "strict_best_model": _row_value(strict_best, "model"),
        "strict_best_canonical_model": _row_value(strict_best, "canonical_model"),
        "strict_best_model_family": _row_value(strict_best, "canonical_model_family"),
        "best_exact_trajectory_model": _row_value(best_exact_trajectory, "model"),
        "best_exact_trajectory_log_evidence": _row_float(best_exact_trajectory, "log_evidence"),
        "best_exact_nontrajectory_model": _row_value(best_exact_nontrajectory, "model"),
        "best_exact_nontrajectory_log_evidence": _row_float(best_exact_nontrajectory, "log_evidence"),
        "best_lower_bound_trajectory_model": _row_value(best_lower_bound_trajectory, "model"),
        "best_lower_bound_trajectory_log_evidence": _row_float(best_lower_bound_trajectory, "log_evidence"),
        "trajectory_strict_win": np.nan,
        "trajectory_certified_win": np.nan,
        "trajectory_certification_reason": "no_exact_nontrajectory_reference",
    }

    if best_exact_trajectory is not None and best_exact_nontrajectory is not None:
        strict_delta = _row_float(best_exact_trajectory, "log_evidence") - _row_float(best_exact_nontrajectory, "log_evidence")
        out["trajectory_minus_nontrajectory_exact_log_evidence"] = strict_delta
        out["trajectory_strict_win"] = bool(strict_delta > 0.0)
    else:
        out["trajectory_minus_nontrajectory_exact_log_evidence"] = np.nan

    if best_exact_nontrajectory is None:
        return out

    nontrajectory_log = _row_float(best_exact_nontrajectory, "log_evidence")
    exact_certified = (
        best_exact_trajectory is not None
        and _row_float(best_exact_trajectory, "log_evidence") > nontrajectory_log
    )
    lower_bound_certified = (
        best_lower_bound_trajectory is not None
        and _row_float(best_lower_bound_trajectory, "log_evidence") > nontrajectory_log
    )
    if exact_certified:
        out["trajectory_certified_win"] = True
        out["trajectory_certification_reason"] = "exact_trajectory_beats_exact_nontrajectory"
    elif lower_bound_certified:
        out["trajectory_certified_win"] = True
        out["trajectory_certification_reason"] = "trajectory_lower_bound_beats_exact_nontrajectory"
    else:
        out["trajectory_certified_win"] = False
        out["trajectory_certification_reason"] = "no_trajectory_evidence_above_exact_nontrajectory"
    return out


def _momentum_diffusion_effects(group: pd.DataFrame) -> dict[str, object]:
    candidate_momentum = _best_row(group[group["canonical_model"].eq("momentum")])
    exact_displacement_momentum = _best_row(group[group["canonical_model"].eq("displacement-momentum")])
    momentum = _best_row(group[group["canonical_model"].isin(["momentum", "displacement-momentum"])])
    diffusion = _best_row(group[group["canonical_model"].eq("diffusion")])
    out: dict[str, object] = {
        "momentum_model": _row_value(momentum, "model"),
        "momentum_family_canonical_model": _row_value(momentum, "canonical_model"),
        "momentum_log_evidence": _row_float(momentum, "log_evidence"),
        "momentum_evidence_support": _row_value(momentum, "evidence_support"),
        "momentum_evidence_comparable": _row_bool(momentum, "evidence_comparable"),
        "candidate_momentum_model": _row_value(candidate_momentum, "model"),
        "candidate_momentum_log_evidence": _row_float(candidate_momentum, "log_evidence"),
        "candidate_momentum_evidence_support": _row_value(candidate_momentum, "evidence_support"),
        "exact_displacement_momentum_model": _row_value(exact_displacement_momentum, "model"),
        "exact_displacement_momentum_log_evidence": _row_float(exact_displacement_momentum, "log_evidence"),
        "exact_displacement_momentum_evidence_support": _row_value(exact_displacement_momentum, "evidence_support"),
        "exact_displacement_momentum_evidence_comparable": _row_bool(exact_displacement_momentum, "evidence_comparable"),
        "diffusion_model": _row_value(diffusion, "model"),
        "diffusion_log_evidence": _row_float(diffusion, "log_evidence"),
        "diffusion_evidence_support": _row_value(diffusion, "evidence_support"),
        "diffusion_evidence_comparable": _row_bool(diffusion, "evidence_comparable"),
        "momentum_minus_diffusion_log_evidence": np.nan,
        "exact_displacement_momentum_minus_diffusion_log_evidence": np.nan,
        "momentum_beats_diffusion_reported": np.nan,
        "momentum_beats_diffusion_certified": np.nan,
        "diffusion_beats_momentum_certified": np.nan,
        "exact_displacement_momentum_beats_diffusion": np.nan,
        "momentum_vs_diffusion_certification": "missing_pair",
    }
    if momentum is None or diffusion is None:
        return out

    delta = _row_float(momentum, "log_evidence") - _row_float(diffusion, "log_evidence")
    momentum_support = str(momentum.get("evidence_support", ""))
    diffusion_comparable = bool(diffusion.get("evidence_comparable", False))
    momentum_comparable = bool(momentum.get("evidence_comparable", False))
    momentum_is_lower_bound = momentum_support == TRUNCATED_EVIDENCE_SUPPORT
    out["momentum_minus_diffusion_log_evidence"] = float(delta)
    out["momentum_beats_diffusion_reported"] = bool(delta > 0.0)
    if exact_displacement_momentum is not None:
        exact_delta = _row_float(exact_displacement_momentum, "log_evidence") - _row_float(diffusion, "log_evidence")
        exact_comparable = bool(exact_displacement_momentum.get("evidence_comparable", False)) and diffusion_comparable
        out["exact_displacement_momentum_minus_diffusion_log_evidence"] = float(exact_delta)
        out["exact_displacement_momentum_beats_diffusion"] = bool(exact_comparable and exact_delta > 0.0)

    if momentum_comparable and diffusion_comparable:
        out["momentum_beats_diffusion_certified"] = bool(delta > 0.0)
        out["diffusion_beats_momentum_certified"] = bool(delta < 0.0)
        out["momentum_vs_diffusion_certification"] = "exact_pair"
    elif momentum_is_lower_bound and diffusion_comparable and delta > 0.0:
        out["momentum_beats_diffusion_certified"] = True
        out["diffusion_beats_momentum_certified"] = False
        out["momentum_vs_diffusion_certification"] = "momentum_lower_bound_beats_exact_diffusion"
    elif momentum_is_lower_bound and diffusion_comparable:
        out["momentum_beats_diffusion_certified"] = False
        out["diffusion_beats_momentum_certified"] = False
        out["momentum_vs_diffusion_certification"] = "momentum_lower_bound_not_above_exact_diffusion"
    else:
        out["momentum_vs_diffusion_certification"] = "noncomparable_pair"
    return out


def _best_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    values = frame["log_evidence"].to_numpy(float)
    return frame.iloc[int(np.nanargmax(values))]


def _row_value(row: pd.Series | None, column: str) -> object:
    return "" if row is None or column not in row else row[column]


def _row_float(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return float("nan")
    return float(row[column])


def _row_bool(row: pd.Series | None, column: str) -> bool:
    if row is None or column not in row:
        return False
    return bool(row[column])


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=object)
    return frame[column].map(_coerce_optional_bool)


def _coerce_optional_bool(value: object) -> bool | float:
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "nan", "none"}:
            return np.nan
        if value in {"true", "1", "yes"}:
            return True
        if value in {"false", "0", "no"}:
            return False
    return bool(value)


def _true_count(values: pd.Series) -> int:
    valid = values.dropna()
    return int(valid.astype(bool).sum()) if not valid.empty else 0


def _mean_or_nan(values: pd.Series) -> float:
    valid = values.dropna()
    return float(valid.mean()) if not valid.empty else float("nan")


def _boolean_effect_summary(
    event_effects: pd.DataFrame,
    column: str,
    prefix: str,
    *,
    bootstrap_replicates: int,
    random_seed: int,
) -> dict[str, object]:
    values = _bool_series(event_effects, column)
    valid = values.dropna().astype(bool)
    n = int(valid.shape[0])
    wins = int(valid.sum()) if n else 0
    fraction = wins / n if n else float("nan")
    low, high = _wilson_interval(wins, n)
    boot_low, boot_high = _cluster_bootstrap_ci(
        event_effects,
        lambda sample: _mean_or_nan(_bool_series(sample, column)),
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
    )
    p_value = float(binomtest(wins, n, p=0.5).pvalue) if n else float("nan")
    return {
        f"{prefix}_events": n,
        f"{prefix}_wins": wins,
        f"{prefix}_win_fraction": float(fraction),
        f"{prefix}_wilson95_low": low,
        f"{prefix}_wilson95_high": high,
        f"{prefix}_session_bootstrap95_low": boot_low,
        f"{prefix}_session_bootstrap95_high": boot_high,
        f"{prefix}_binomial_p_value_vs_0_5": p_value,
    }


def _momentum_delta_summary(
    event_effects: pd.DataFrame,
    bootstrap_replicates: int,
    random_seed: int,
) -> dict[str, object]:
    deltas = pd.to_numeric(
        event_effects.get("momentum_minus_diffusion_log_evidence", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    boot_low, boot_high = _cluster_bootstrap_ci(
        event_effects,
        lambda sample: float(
            pd.to_numeric(
                sample.get("momentum_minus_diffusion_log_evidence", pd.Series(dtype=float)),
                errors="coerce",
            ).mean()
        ),
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
    )
    return {
        "momentum_diffusion_paired_events": int(deltas.shape[0]),
        "mean_momentum_minus_diffusion_log_evidence": float(deltas.mean()) if not deltas.empty else float("nan"),
        "median_momentum_minus_diffusion_log_evidence": float(deltas.median()) if not deltas.empty else float("nan"),
        "mean_momentum_minus_diffusion_session_bootstrap95_low": boot_low,
        "mean_momentum_minus_diffusion_session_bootstrap95_high": boot_high,
    }


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half_width = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _cluster_bootstrap_ci(
    event_effects: pd.DataFrame,
    reducer: Callable[[pd.DataFrame], float],
    *,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[float, float]:
    if bootstrap_replicates <= 0 or event_effects.empty or "session" not in event_effects:
        return float("nan"), float("nan")
    sessions = np.asarray(sorted(event_effects["session"].dropna().astype(str).unique()), dtype=object)
    if sessions.size == 0:
        return float("nan"), float("nan")
    by_session = {
        session: event_effects[event_effects["session"].astype(str).eq(session)]
        for session in sessions
    }
    rng = np.random.default_rng(random_seed)
    values: list[float] = []
    for _ in range(int(bootstrap_replicates)):
        sampled = rng.choice(sessions, size=sessions.size, replace=True)
        sample = pd.concat([by_session[str(session)] for session in sampled], ignore_index=True)
        value = float(reducer(sample))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def _session_interpretation(mean_delta: float, reported_fraction: float) -> str:
    if not np.isfinite(mean_delta) or not np.isfinite(reported_fraction):
        return "insufficient-paired-events"
    if mean_delta > 0.0 and reported_fraction >= 0.5:
        return "momentum-dominant"
    if mean_delta < 0.0 and reported_fraction <= 0.5:
        return "diffusion-dominant"
    return "mixed"


def _claim_fraction(row: pd.Series, *, label: str, prefix: str) -> str:
    events = int(row.get(f"{prefix}_events", 0))
    wins = int(row.get(f"{prefix}_wins", 0))
    fraction = row.get(f"{prefix}_win_fraction", float("nan"))
    low = row.get(f"{prefix}_session_bootstrap95_low", float("nan"))
    high = row.get(f"{prefix}_session_bootstrap95_high", float("nan"))
    p_value = row.get(f"{prefix}_binomial_p_value_vs_0_5", float("nan"))
    return (
        f"- {label}: {wins}/{events} events "
        f"({ _fmt_fraction(fraction) }; session-bootstrap 95% CI "
        f"{ _fmt_fraction(low) }–{ _fmt_fraction(high) }; binomial p={_fmt_p(p_value)})."
    )


def _claim_delta(row: pd.Series) -> str:
    events = int(row.get("momentum_diffusion_paired_events", 0))
    mean_delta = row.get("mean_momentum_minus_diffusion_log_evidence", float("nan"))
    low = row.get("mean_momentum_minus_diffusion_session_bootstrap95_low", float("nan"))
    high = row.get("mean_momentum_minus_diffusion_session_bootstrap95_high", float("nan"))
    median_delta = row.get("median_momentum_minus_diffusion_log_evidence", float("nan"))
    return (
        f"- Momentum-minus-diffusion log-evidence delta: mean {_fmt_number(mean_delta)} "
        f"over {events} paired events (session-bootstrap 95% CI {_fmt_number(low)}–{_fmt_number(high)}; "
        f"median {_fmt_number(median_delta)})."
    )


def _fmt_fraction(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def _fmt_number(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.3g}"


def _fmt_p(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create paper-facing state-space replay effect and heterogeneity tables."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="event_model_evidence.csv or an artifact directory containing it.",
    )
    parser.add_argument("--output", default="results/paper-state-space-effects")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Drop all non-exact rows before summarizing; disables lower-bound certification.",
    )
    args = parser.parse_args()

    tables = summarize_paper_effects(
        args.input,
        args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
        exact_only=args.exact_only,
    )
    print("Summary:")
    print(tables["summary"].to_string(index=False))
    print("\nSession effects:")
    print(tables["session_effects"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
