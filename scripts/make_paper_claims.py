#!/usr/bin/env python3
"""Generate paper-audit tables from HippoReplayIMM benchmark score CSVs.

The benchmark intentionally separates exact full-grid evidences from candidate-
pruned lower bounds.  This script turns that distinction into paper-ready claim
artifacts for the most defensible current story: paired model effects, session
heterogeneity, and certified lower-bound wins where a pruned model's reported
lower bound already exceeds an exact baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

try:
    from hipporeplayimm.evidence_reporting import (
        EXACT_EVIDENCE_SUPPORT,
        EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS,
        TRUNCATED_EVIDENCE_SUPPORT,
        ensure_evidence_support_columns,
    )
except ModuleNotFoundError:  # pragma: no cover - helpful when run before editable install.
    EXACT_EVIDENCE_SUPPORT = "exact_full_grid"
    TRUNCATED_EVIDENCE_SUPPORT = "truncated_full_grid"
    EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS = (
        "diagnostic_candidate_evidence_support",
        "diagnostic_state_space_momentum_evidence_support",
        "diagnostic_state_space_imm_evidence_support",
        "diagnostic_state_space_sparse_momentum_evidence_support",
        "diagnostic_state_space_trajectory_imm_evidence_support",
        "diagnostic_state_space_displacement_momentum_evidence_support",
        "diagnostic_goal_state_space_evidence_support",
        "diagnostic_pyrecest_evidence_support",
    )

    def ensure_evidence_support_columns(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[no-redef]
        out = df.copy()
        if "evidence_support" not in out:
            support_cols = [
                col
                for col in EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS
                if col in out.columns
            ]
            if support_cols:
                out["evidence_support"] = out[support_cols].bfill(axis=1).iloc[:, 0].fillna(EXACT_EVIDENCE_SUPPORT)
            else:
                out["evidence_support"] = EXACT_EVIDENCE_SUPPORT
        status_ok = out["status"].eq("success") if "status" in out else pd.Series(True, index=out.index)
        out["evidence_comparable"] = status_ok & out["evidence_support"].astype(str).eq(EXACT_EVIDENCE_SUPPORT)
        return out


DEFAULT_PRIMARY_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
DEFAULT_BASELINE_MODEL = "sorted-spike-state-space-diffusion"
DEFAULT_VALUE_COLUMN = "heldout_log_likelihood"
OPTIONAL_IDENTITY_COLUMNS = (
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_random_seed",
    "requested_session",
    "cell_split_index",
    "random_seed",
)
SCORE_FILENAMES = (
    "event_scores.csv",
    "benchmark_event_scores.csv",
    "model_evidence_event_scores.csv",
    "scores.csv",
)
EVIDENCE_SUPPORT_METADATA_COLUMNS = (
    "evidence_support",
    "evidence_comparable",
    *EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS,
)


@dataclass(frozen=True)
class PaperClaimConfig:
    """Configuration for paired claim extraction."""

    primary_model: str = DEFAULT_PRIMARY_MODEL
    baseline_model: str = DEFAULT_BASELINE_MODEL
    value_column: str = DEFAULT_VALUE_COLUMN
    n_bootstrap: int = 5000
    random_seed: int = 1
    require_evidence_support: bool = True


@dataclass
class PaperClaimTables:
    """Output tables for a paper-claim audit."""

    event_deltas: pd.DataFrame
    session_summary: pd.DataFrame
    rat_summary: pd.DataFrame
    leave_one_rat_out_summary: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, object]

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_deltas.to_csv(out_dir / "paper_claim_event_deltas.csv", index=False)
        self.session_summary.to_csv(out_dir / "paper_claim_session_summary.csv", index=False)
        self.rat_summary.to_csv(out_dir / "paper_claim_rat_summary.csv", index=False)
        self.leave_one_rat_out_summary.to_csv(out_dir / "paper_claim_leave_one_rat_out_summary.csv", index=False)
        self.summary.to_csv(out_dir / "paper_claim_summary.csv", index=False)
        (out_dir / "paper_claims.md").write_text(render_claim_markdown(self), encoding="utf-8")
        (out_dir / "paper_claim_manifest.json").write_text(
            json.dumps(_json_ready(self.manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_score_tables(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load one or more score CSVs or directories containing score CSVs."""

    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = _resolve_score_path(Path(raw_path))
        frame = pd.read_csv(path)
        frame["source_score_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one score table is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_paper_claim_tables(
    scores: pd.DataFrame,
    config: PaperClaimConfig | None = None,
) -> PaperClaimTables:
    """Build paired model-effect and session-heterogeneity tables."""

    config = PaperClaimConfig() if config is None else config
    if scores.empty:
        raise ValueError("score table is empty")
    missing = [col for col in ("session", "event_index", "model", config.value_column) if col not in scores]
    if missing:
        raise KeyError(f"score table is missing required columns: {missing}")
    if config.require_evidence_support:
        require_evidence_support_metadata(scores)

    frame = ensure_evidence_support_columns(scores)
    primary_model = resolve_model_spec(frame["model"], config.primary_model)
    baseline_model = resolve_model_spec(frame["model"], config.baseline_model)
    event_deltas = paired_event_deltas(
        frame,
        primary_model=primary_model,
        baseline_model=baseline_model,
        value_column=config.value_column,
    )
    session_summary = summarize_sessions(event_deltas)
    rat_summary = summarize_rats(event_deltas)
    leave_one_rat_out_summary = summarize_leave_one_rat_out(event_deltas)
    summary = summarize_overall(
        event_deltas,
        primary_model=primary_model,
        baseline_model=baseline_model,
        value_column=config.value_column,
        n_bootstrap=config.n_bootstrap,
        random_seed=config.random_seed,
    )
    manifest = {
        "schema_version": 1,
        "primary_model_requested": config.primary_model,
        "baseline_model_requested": config.baseline_model,
        "primary_model_resolved": primary_model,
        "baseline_model_resolved": baseline_model,
        "value_column": config.value_column,
        "n_input_rows": int(len(scores)),
        "n_paired_events": int(len(event_deltas)),
        "n_rats": int(rat_summary["group"].nunique()) if not rat_summary.empty else 0,
        "identity_columns": identity_columns(frame),
        "n_bootstrap": int(config.n_bootstrap),
        "random_seed": int(config.random_seed),
    }
    return PaperClaimTables(
        event_deltas=event_deltas,
        session_summary=session_summary,
        rat_summary=rat_summary,
        leave_one_rat_out_summary=leave_one_rat_out_summary,
        summary=summary,
        manifest=manifest,
    )


def require_evidence_support_metadata(scores: pd.DataFrame) -> None:
    """Require exact/lower-bound support metadata before making headline claims."""

    present = [column for column in EVIDENCE_SUPPORT_METADATA_COLUMNS if column in scores.columns]
    if not present:
        raise KeyError(
            "score table does not contain evidence-support metadata; rerun the benchmark with support-aware scoring "
            "or pass --allow-missing-evidence-support for exploratory legacy tables"
        )


def paired_event_deltas(
    frame: pd.DataFrame,
    *,
    primary_model: str,
    baseline_model: str,
    value_column: str = DEFAULT_VALUE_COLUMN,
) -> pd.DataFrame:
    """Return one paired primary-minus-baseline row per benchmark event."""

    ids = identity_columns(frame)
    subset = frame[frame["model"].astype(str).isin([primary_model, baseline_model])].copy()
    if subset.empty:
        raise ValueError("none of the requested models were found in the score table")
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=[value_column])
    grouped = (
        subset.groupby([*ids, "model"], sort=False, dropna=False)
        .agg(
            score=(value_column, "mean"),
            evidence_support=("evidence_support", _first_nonempty),
            evidence_comparable=("evidence_comparable", _all_bool),
            status=("status", _first_nonempty) if "status" in subset else (value_column, lambda _: "success"),
        )
        .reset_index()
    )
    values = grouped.pivot_table(index=ids, columns="model", values="score", aggfunc="first")
    support = grouped.pivot_table(index=ids, columns="model", values="evidence_support", aggfunc="first")
    comparable = grouped.pivot_table(index=ids, columns="model", values="evidence_comparable", aggfunc="first")
    required = [primary_model, baseline_model]
    missing_models = [name for name in required if name not in values.columns]
    if missing_models:
        raise ValueError(f"requested model(s) not present after aggregation: {missing_models}")
    paired = values.dropna(subset=required).copy()
    if paired.empty:
        raise ValueError("no paired events contain both requested models")
    out = paired.reset_index()
    out["primary_model"] = primary_model
    out["baseline_model"] = baseline_model
    out["primary_value"] = paired[primary_model].to_numpy(float)
    out["baseline_value"] = paired[baseline_model].to_numpy(float)
    out["delta_primary_minus_baseline"] = out["primary_value"] - out["baseline_value"]
    out["primary_evidence_support"] = _lookup_pivot_column(support, paired.index, primary_model, EXACT_EVIDENCE_SUPPORT)
    out["baseline_evidence_support"] = _lookup_pivot_column(support, paired.index, baseline_model, EXACT_EVIDENCE_SUPPORT)
    out["primary_evidence_comparable"] = _coerce_bool_series(
        _lookup_pivot_column(comparable, paired.index, primary_model, True),
    )
    out["baseline_evidence_comparable"] = _coerce_bool_series(
        _lookup_pivot_column(comparable, paired.index, baseline_model, True),
    )
    out["both_evidence_comparable"] = out["primary_evidence_comparable"] & out["baseline_evidence_comparable"]
    out["primary_is_truncated_lower_bound"] = out["primary_evidence_support"].astype(str).eq(TRUNCATED_EVIDENCE_SUPPORT)
    out["baseline_is_exact"] = out["baseline_evidence_support"].astype(str).eq(EXACT_EVIDENCE_SUPPORT)
    delta = out["delta_primary_minus_baseline"]
    out["apparent_primary_win"] = delta > 0.0
    out["apparent_baseline_win"] = delta < 0.0
    out["apparent_tie"] = delta == 0.0
    out["strict_primary_win"] = out["both_evidence_comparable"] & (delta > 0.0)
    out["strict_baseline_win"] = out["both_evidence_comparable"] & (delta < 0.0)
    out["certified_primary_win"] = (
        (out["both_evidence_comparable"] | (out["primary_is_truncated_lower_bound"] & out["baseline_is_exact"]))
        & (delta > 0.0)
    )
    out["nondecisive_due_to_primary_lower_bound"] = (
        out["primary_is_truncated_lower_bound"] & out["baseline_is_exact"] & (delta <= 0.0)
    )
    out["claim_category"] = np.select(
        [
            out["strict_primary_win"],
            out["certified_primary_win"],
            out["strict_baseline_win"],
            out["nondecisive_due_to_primary_lower_bound"],
        ],
        [
            "strict_primary_win",
            "lower_bound_certified_primary_win",
            "strict_baseline_win",
            "nondecisive_primary_lower_bound",
        ],
        default="nondecisive_or_noncomparable",
    )
    ordered = [
        *ids,
        "primary_model",
        "baseline_model",
        "primary_value",
        "baseline_value",
        "delta_primary_minus_baseline",
        "primary_evidence_support",
        "baseline_evidence_support",
        "primary_evidence_comparable",
        "baseline_evidence_comparable",
        "both_evidence_comparable",
        "primary_is_truncated_lower_bound",
        "baseline_is_exact",
        "apparent_primary_win",
        "apparent_baseline_win",
        "apparent_tie",
        "strict_primary_win",
        "strict_baseline_win",
        "certified_primary_win",
        "nondecisive_due_to_primary_lower_bound",
        "claim_category",
    ]
    return out[ordered].sort_values(ids).reset_index(drop=True)


def summarize_sessions(event_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize apparent and certified effects by session."""

    if event_deltas.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for session, group in event_deltas.groupby("session", sort=True):
        rows.append(_summary_row(str(session), group))
    return pd.DataFrame(rows)


def summarize_rats(event_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize apparent and certified effects by rat, preserving session nesting."""

    if event_deltas.empty or "session" not in event_deltas:
        return pd.DataFrame()
    frame = event_deltas.copy()
    frame["rat"] = frame["session"].map(_rat_from_session)
    rows = [_summary_row(str(rat), group) for rat, group in frame.groupby("rat", sort=True)]
    return pd.DataFrame(rows)


def summarize_leave_one_rat_out(event_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize sensitivity to excluding one rat at a time."""

    if event_deltas.empty or "session" not in event_deltas:
        return pd.DataFrame()
    frame = event_deltas.copy()
    frame["rat"] = frame["session"].map(_rat_from_session)
    rows: list[dict[str, object]] = []
    for rat in sorted(frame["rat"].dropna().astype(str).unique()):
        subset = frame[frame["rat"].astype(str) != rat]
        row = _summary_row(f"leave_out_{rat}", subset)
        row["left_out_rat"] = rat
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(
    event_deltas: pd.DataFrame,
    *,
    primary_model: str,
    baseline_model: str,
    value_column: str,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Return a one-row overall paper-claim summary."""

    row = _summary_row("overall", event_deltas)
    row.update(
        {
            "primary_model": primary_model,
            "baseline_model": baseline_model,
            "value_column": value_column,
            "apparent_sign_test_p_value": _sign_test_p_value(
                int(event_deltas["apparent_primary_win"].sum()),
                int(event_deltas["apparent_baseline_win"].sum()),
            ),
            "mean_delta_bootstrap_ci_low": _hierarchical_bootstrap_mean_ci(
                event_deltas,
                values_column="delta_primary_minus_baseline",
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )[0],
            "mean_delta_bootstrap_ci_high": _hierarchical_bootstrap_mean_ci(
                event_deltas,
                values_column="delta_primary_minus_baseline",
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )[1],
        }
    )
    return pd.DataFrame([row])


def render_claim_markdown(tables: PaperClaimTables) -> str:
    """Render a compact Markdown narrative for the paper supplement."""

    if tables.summary.empty:
        return "# HippoReplayIMM paper-claim audit\n\nNo paired events were available.\n"
    row = tables.summary.iloc[0]
    session_rows = tables.session_summary.sort_values(
        "apparent_primary_win_fraction",
        ascending=False,
    )
    rat_rows = tables.rat_summary.sort_values("group") if not tables.rat_summary.empty else pd.DataFrame()
    best_session = "n/a" if session_rows.empty else str(session_rows.iloc[0]["group"])
    worst_session = "n/a" if session_rows.empty else str(session_rows.iloc[-1]["group"])
    return "\n".join(
        [
            "# HippoReplayIMM paper-claim audit",
            "",
            f"Primary model: `{row['primary_model']}`",
            f"Baseline model: `{row['baseline_model']}`",
            f"Paired event units: {int(row['paired_events'])} across {int(row['sessions'])} sessions.",
            "",
            "## Main paired-effect claim",
            "",
            (
                f"The apparent primary-minus-baseline mean {row['value_column']} delta is "
                f"{float(row['mean_delta_primary_minus_baseline']):.4g} "
                f"(hierarchical bootstrap 95% CI "
                f"[{float(row['mean_delta_bootstrap_ci_low']):.4g}, "
                f"{float(row['mean_delta_bootstrap_ci_high']):.4g}])."
            ),
            (
                f"The primary model has apparent wins on {int(row['apparent_primary_wins'])}/"
                f"{int(row['paired_events'])} paired events "
                f"({float(row['apparent_primary_win_fraction']):.1%}); two-sided sign-test "
                f"p={float(row['apparent_sign_test_p_value']):.4g}."
            ),
            "",
            "## Lower-bound-safe claim",
            "",
            (
                f"The primary model has {int(row['certified_primary_wins'])} certified wins. "
                "For a truncated primary row against an exact baseline, this counts only events where the "
                "reported lower bound already exceeds the exact baseline evidence."
            ),
            (
                f"There are {int(row['nondecisive_primary_lower_bound_events'])} paired events where a "
                "truncated primary lower bound does not exceed the baseline, so those events should not be "
                "reported as strict baseline wins."
            ),
            "",
            "## Session heterogeneity",
            "",
            f"The strongest apparent primary-win session is `{best_session}`; the weakest is `{worst_session}`.",
            "Use `paper_claim_session_summary.csv` for a table suitable for the paper or supplement.",
            "",
            "## Rat-level summary",
            "",
            f"Rat-level rows available: {0 if rat_rows.empty else int(len(rat_rows))}. Use `paper_claim_rat_summary.csv` for nested reporting.",
            "",
            "## Leave-one-rat-out sensitivity",
            "",
            (
                f"Leave-one-rat-out rows available: {len(tables.leave_one_rat_out_summary)}. "
                "Use `paper_claim_leave_one_rat_out_summary.csv` for rat-influence reporting."
            ),
            "",
        ]
    )


def resolve_model_spec(models: Iterable[object], requested: str) -> str:
    """Resolve a short model alias such as 'momentum' to one concrete score name."""

    available = [str(model) for model in pd.Series(list(models)).dropna().unique()]
    if requested in available:
        return requested
    requested_norm = str(requested).strip().lower().replace("_", "-")
    exact_aliases = [
        requested_norm,
        f"sorted-spike-state-space-{requested_norm}",
        f"state-space-{requested_norm}",
        f"clusterless-state-space-{requested_norm}",
    ]
    for alias in exact_aliases:
        matches = [model for model in available if model.lower() == alias]
        if len(matches) == 1:
            return matches[0]
    suffix_matches = [model for model in available if model.lower().endswith(f"-{requested_norm}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(f"model alias {requested!r} is ambiguous: {suffix_matches}")
    if requested_norm.endswith("-exact-sparse"):
        legacy_requested = requested_norm.removesuffix("-exact-sparse")
        legacy_aliases = [
            legacy_requested,
            f"sorted-spike-state-space-{legacy_requested}",
            f"state-space-{legacy_requested}",
            f"clusterless-state-space-{legacy_requested}",
        ]
        for alias in legacy_aliases:
            matches = [model for model in available if model.lower() == alias]
            if len(matches) == 1:
                return matches[0]
    raise ValueError(f"model {requested!r} not found; available models include: {sorted(available)[:20]}")


def identity_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["session", "event_index"]
    columns.extend(col for col in OPTIONAL_IDENTITY_COLUMNS if col in frame.columns)
    return columns


def _summary_row(label: str, group: pd.DataFrame) -> dict[str, object]:
    delta = group["delta_primary_minus_baseline"].to_numpy(float)
    apparent_primary_wins = int(group["apparent_primary_win"].sum())
    apparent_baseline_wins = int(group["apparent_baseline_win"].sum())
    certified_primary_wins = int(group["certified_primary_win"].sum())
    strict_baseline_wins = int(group["strict_baseline_win"].sum())
    paired_events = int(len(group))
    return {
        "group": label,
        "sessions": int(group["session"].nunique()) if "session" in group else 0,
        "paired_events": paired_events,
        "apparent_primary_wins": apparent_primary_wins,
        "apparent_baseline_wins": apparent_baseline_wins,
        "apparent_ties": int(group["apparent_tie"].sum()),
        "apparent_primary_win_fraction": _safe_fraction(apparent_primary_wins, paired_events),
        "certified_primary_wins": certified_primary_wins,
        "certified_primary_win_fraction": _safe_fraction(certified_primary_wins, paired_events),
        "strict_baseline_wins": strict_baseline_wins,
        "strict_baseline_win_fraction": _safe_fraction(strict_baseline_wins, paired_events),
        "nondecisive_primary_lower_bound_events": int(group["nondecisive_due_to_primary_lower_bound"].sum()),
        "mean_delta_primary_minus_baseline": float(np.mean(delta)) if delta.size else float("nan"),
        "median_delta_primary_minus_baseline": float(np.median(delta)) if delta.size else float("nan"),
        "sum_delta_primary_minus_baseline": float(np.sum(delta)) if delta.size else float("nan"),
    }


def _rat_from_session(session: object) -> str:
    text = str(session)
    return text.split("/", 1)[0] if "/" in text else text


def _hierarchical_bootstrap_mean_ci(
    frame: pd.DataFrame,
    *,
    values_column: str,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[float, float]:
    if frame.empty or values_column not in frame:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_seed)
    groups = [group[values_column].dropna().to_numpy(float) for _, group in frame.groupby("session", sort=False)]
    groups = [group for group in groups if group.size]
    if not groups:
        return (float("nan"), float("nan"))
    draws = np.empty(max(1, int(n_bootstrap)), dtype=float)
    for idx in range(draws.size):
        sampled_group_indices = rng.choice(len(groups), size=len(groups), replace=True)
        sampled = []
        for group_index in sampled_group_indices:
            values = groups[int(group_index)]
            sampled.append(rng.choice(values, size=values.size, replace=True))
        merged = np.concatenate(sampled)
        draws[idx] = float(np.mean(merged))
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def _sign_test_p_value(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n <= 0:
        return float("nan")
    return float(binomtest(int(wins), n=n, p=0.5, alternative="two-sided").pvalue)


def _lookup_pivot_column(pivot: pd.DataFrame, index: pd.Index, column: str, default: object) -> pd.Series:
    if column not in pivot.columns:
        return pd.Series(default, index=range(len(index)))
    return pivot.reindex(index)[column].reset_index(drop=True).fillna(default)


def _first_nonempty(values: pd.Series) -> object:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text:
            return value
    return ""


def _all_bool(values: pd.Series) -> bool:
    if values.empty:
        return False
    return bool(_coerce_bool_series(values).all())


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "", "nan", "none"}:
        return False
    return default


def _coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    return values.map(lambda value: _coerce_bool(value, default=default)).astype(bool)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float("nan") if denominator <= 0 else float(numerator) / float(denominator)


def _resolve_score_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    for name in SCORE_FILENAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    csvs = sorted(path.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(
        f"Could not infer score CSV inside {path}; expected one of {SCORE_FILENAMES}"
    )


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        scalar = value.item()  # NumPy scalar
    except AttributeError:
        scalar = value
    if isinstance(scalar, float):
        return None if not math.isfinite(scalar) else float(scalar)
    if isinstance(scalar, (int, bool, str)) or scalar is None:
        return scalar
    return str(scalar)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready paired model-effect and session-heterogeneity claim tables."
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="One or more benchmark score CSVs, or directories containing event_scores.csv.",
    )
    parser.add_argument("--output", required=True, help="Output directory for claim tables.")
    parser.add_argument("--primary-model", default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--value-column", default=DEFAULT_VALUE_COLUMN)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--allow-missing-evidence-support",
        action="store_true",
        help="Allow legacy score tables without exact/lower-bound support metadata. Not recommended for final claims.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scores = load_score_tables(args.scores)
    tables = build_paper_claim_tables(
        scores,
        PaperClaimConfig(
            primary_model=args.primary_model,
            baseline_model=args.baseline_model,
            value_column=args.value_column,
            n_bootstrap=args.n_bootstrap,
            random_seed=args.random_seed,
            require_evidence_support=not args.allow_missing_evidence_support,
        ),
    )
    tables.write(args.output)
    print(tables.summary.to_string(index=False))
    print(f"\nWrote paper-claim audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
