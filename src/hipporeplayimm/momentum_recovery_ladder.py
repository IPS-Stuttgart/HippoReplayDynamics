"""Momentum-recovery ladder utilities.

The ladder is a compact diagnostic for the otherwise ambiguous statement
"synthetic momentum did not recover".  It evaluates the same synthetic world
under increasingly realistic scoring support:

1. exact/full-grid pairwise momentum;
2. exact finite-displacement/velocity momentum;
3. candidate-pruned pairwise momentum with oracle true-path support; and
4. native candidate-pruned pairwise momentum.

Rows from these tiers separate implementation or generative/scoring mismatch,
finite-surrogate misspecification, candidate-support loss, and genuinely weak
evidence.  The module is pure table logic plus small configuration builders; the
CLI script performs the expensive session-level runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .evidence_reporting import _coerce_bool_series
from .simulation_recovery import (
    SimulationRecoveryConfig,
    SimulationRecoveryResult,
    certified_vs_exact_event_recovery,
    run_session_simulation_recovery,
)
from .state_space import StateSpaceDecoderConfig


PAIRWISE_MOMENTUM_MODEL = "sorted-spike-state-space-momentum"
FINITE_VELOCITY_MODEL = "sorted-spike-state-space-velocity-momentum"
FINITE_DISPLACEMENT_MODEL = "sorted-spike-state-space-displacement-momentum"
DIFFUSION_MODEL = "sorted-spike-state-space-diffusion"
FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"
STATIONARY_MODEL = "sorted-spike-state-space-stationary"
IMM_MODEL = "sorted-spike-state-space-imm"


LADDER_TIER_ORDER = (
    "full_grid_pairwise_momentum",
    "exact_finite_velocity_momentum",
    "oracle_candidate_pairwise_momentum",
    "native_candidate_pairwise_momentum",
)


DEFAULT_LADDER_MODELS = (
    STATIONARY_MODEL,
    DIFFUSION_MODEL,
    FRAGMENTED_MODEL,
    PAIRWISE_MOMENTUM_MODEL,
    FINITE_VELOCITY_MODEL,
    IMM_MODEL,
)


@dataclass(frozen=True)
class MomentumRecoveryLadderTier:
    """One tier of the momentum-recovery ladder."""

    name: str
    expected_model: str
    scoring_models: tuple[str, ...]
    state_space: StateSpaceDecoderConfig
    oracle_candidate_support: bool = False
    description: str = ""


@dataclass
class MomentumRecoveryLadderResult:
    """Output tables for a full ladder run."""

    event_scores: pd.DataFrame
    tier_summary: pd.DataFrame
    interpretation: pd.DataFrame
    tier_event_recovery: pd.DataFrame
    settings: pd.DataFrame

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_scores.to_csv(out_dir / "momentum_recovery_ladder_event_scores.csv", index=False)
        self.tier_summary.to_csv(out_dir / "momentum_recovery_ladder_tier_summary.csv", index=False)
        self.interpretation.to_csv(out_dir / "momentum_recovery_ladder_interpretation.csv", index=False)
        self.tier_event_recovery.to_csv(out_dir / "momentum_recovery_ladder_event_recovery.csv", index=False)
        self.settings.to_csv(out_dir / "momentum_recovery_ladder_settings.csv", index=False)
        (out_dir / "momentum_recovery_ladder.md").write_text(
            render_momentum_recovery_ladder_markdown(self),
            encoding="utf-8",
        )


def default_ladder_tiers(
    base_state_space: StateSpaceDecoderConfig | None = None,
    *,
    native_candidate_top_k: int = 128,
    native_predicted_candidate_top_k: int = 8,
    finite_displacement_radius_bins: int = 2,
) -> tuple[MomentumRecoveryLadderTier, ...]:
    """Return the default four-tier momentum-recovery ladder.

    The first tier forces pairwise momentum/IMM to full spatial support.  The
    second tier uses the exact finite-displacement implementation as a
    comparable finite-velocity surrogate.  The last two tiers keep native
    candidate pruning but toggle oracle true-path support for synthetic events.
    """

    base = StateSpaceDecoderConfig() if base_state_space is None else base_state_space
    full_grid = replace(
        base,
        momentum_candidate_top_k=0,
        momentum_candidate_mass_threshold=None,
        momentum_candidate_min_k=1,
        momentum_candidate_max_k=0,
        momentum_predicted_candidate_top_k=0,
    )
    finite_velocity = replace(
        full_grid,
        displacement_radius_bins=int(finite_displacement_radius_bins),
    )
    native = replace(
        base,
        momentum_candidate_top_k=int(native_candidate_top_k),
        momentum_predicted_candidate_top_k=int(native_predicted_candidate_top_k),
    )
    return (
        MomentumRecoveryLadderTier(
            name="full_grid_pairwise_momentum",
            expected_model=PAIRWISE_MOMENTUM_MODEL,
            scoring_models=(STATIONARY_MODEL, DIFFUSION_MODEL, FRAGMENTED_MODEL, PAIRWISE_MOMENTUM_MODEL, IMM_MODEL),
            state_space=full_grid,
            oracle_candidate_support=False,
            description="Pairwise momentum/IMM with full spatial candidate support; exact but expensive.",
        ),
        MomentumRecoveryLadderTier(
            name="exact_finite_velocity_momentum",
            expected_model=FINITE_VELOCITY_MODEL,
            scoring_models=(STATIONARY_MODEL, DIFFUSION_MODEL, FRAGMENTED_MODEL, FINITE_VELOCITY_MODEL),
            state_space=finite_velocity,
            oracle_candidate_support=False,
            description="Exact finite-displacement/velocity surrogate; comparable over its declared state space.",
        ),
        MomentumRecoveryLadderTier(
            name="oracle_candidate_pairwise_momentum",
            expected_model=PAIRWISE_MOMENTUM_MODEL,
            scoring_models=(STATIONARY_MODEL, DIFFUSION_MODEL, FRAGMENTED_MODEL, PAIRWISE_MOMENTUM_MODEL, IMM_MODEL),
            state_space=native,
            oracle_candidate_support=True,
            description="Native candidate support augmented with the synthetic true path.",
        ),
        MomentumRecoveryLadderTier(
            name="native_candidate_pairwise_momentum",
            expected_model=PAIRWISE_MOMENTUM_MODEL,
            scoring_models=(STATIONARY_MODEL, DIFFUSION_MODEL, FRAGMENTED_MODEL, PAIRWISE_MOMENTUM_MODEL, IMM_MODEL),
            state_space=native,
            oracle_candidate_support=False,
            description="Native train-only candidate support used by the replay benchmark.",
        ),
    )


def run_momentum_recovery_ladder(
    dataset_root: str | Path,
    session_id: str,
    base_config: SimulationRecoveryConfig,
    *,
    tiers: Sequence[MomentumRecoveryLadderTier] | None = None,
    output: str | Path | None = None,
) -> MomentumRecoveryLadderResult:
    """Run all ladder tiers for one session and return combined tables."""

    tiers = default_ladder_tiers(base_config.state_space) if tiers is None else tuple(tiers)
    event_frames: list[pd.DataFrame] = []
    settings_rows: list[dict[str, object]] = []
    out_dir = None if output is None else Path(output)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for tier_index, tier in enumerate(tiers):
        config = replace(
            base_config,
            true_models=("momentum",),
            scoring_models=tuple(tier.scoring_models),
            state_space=tier.state_space,
            oracle_candidate_support=bool(tier.oracle_candidate_support),
        )
        tier_output = None if out_dir is None else out_dir / tier.name
        result = run_session_simulation_recovery(dataset_root, session_id, config)
        if tier_output is not None:
            result.write(tier_output)
        frame = _annotate_tier_scores(
            result.event_scores,
            tier=tier,
            tier_index=tier_index,
        )
        event_frames.append(frame)
        settings_rows.append(_tier_settings_row(tier, tier_index, result))

    event_scores = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    tier_event_recovery = ladder_event_recovery(event_scores)
    tier_summary = summarize_ladder_tiers(tier_event_recovery)
    interpretation = interpret_ladder_summary(tier_summary)
    return MomentumRecoveryLadderResult(
        event_scores=event_scores,
        tier_summary=tier_summary,
        interpretation=interpretation,
        tier_event_recovery=tier_event_recovery,
        settings=pd.DataFrame(settings_rows),
    )


def _annotate_tier_scores(
    scores: pd.DataFrame,
    *,
    tier: MomentumRecoveryLadderTier,
    tier_index: int,
) -> pd.DataFrame:
    frame = scores.copy()
    frame["ladder_tier"] = str(tier.name)
    frame["ladder_tier_index"] = int(tier_index)
    frame["ladder_expected_model"] = str(tier.expected_model)
    frame["ladder_oracle_candidate_support"] = bool(tier.oracle_candidate_support)
    frame["ladder_description"] = str(tier.description)
    if "expected_model" in frame.columns:
        frame["base_expected_model"] = frame["expected_model"].astype(str)
    frame["expected_model"] = str(tier.expected_model)
    return frame


def _tier_settings_row(
    tier: MomentumRecoveryLadderTier,
    tier_index: int,
    result: SimulationRecoveryResult,
) -> dict[str, object]:
    settings = dict(result.settings)
    return {
        "ladder_tier": tier.name,
        "ladder_tier_index": int(tier_index),
        "expected_model": tier.expected_model,
        "scoring_models": ",".join(tier.scoring_models),
        "oracle_candidate_support": bool(tier.oracle_candidate_support),
        "description": tier.description,
        "n_cells": settings.get("n_cells", np.nan),
        "n_position_bins": settings.get("n_position_bins", np.nan),
        "time_bin_s": settings.get("time_bin_s", np.nan),
        "events_per_model": settings.get("events_per_model", np.nan),
        "random_seed": settings.get("random_seed", np.nan),
    }


def ladder_event_recovery(event_scores: pd.DataFrame) -> pd.DataFrame:
    """Return certified-vs-exact event recovery with ladder annotations."""

    if event_scores.empty:
        return pd.DataFrame()
    required = {"ladder_tier", "ladder_tier_index", "expected_model"}
    missing = sorted(required - set(event_scores.columns))
    if missing:
        raise KeyError(f"ladder score table is missing required columns: {missing}")

    rows: list[pd.DataFrame] = []
    for (tier, tier_index), group in event_scores.groupby(
        ["ladder_tier", "ladder_tier_index"],
        sort=False,
        dropna=False,
    ):
        events = certified_vs_exact_event_recovery(group)
        if events.empty:
            continue
        first = group.iloc[0]
        events.insert(0, "ladder_tier", str(tier))
        events.insert(1, "ladder_tier_index", int(tier_index))
        events["ladder_expected_model"] = str(first.get("ladder_expected_model", first.get("expected_model", "")))
        events["oracle_candidate_support"] = bool(first.get("ladder_oracle_candidate_support", False))
        events["ladder_description"] = str(first.get("ladder_description", ""))
        rows.append(events)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False).sort_values(
        ["ladder_tier_index", "session", "event_index"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_ladder_tiers(tier_event_recovery: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-level ladder recovery by tier."""

    if tier_event_recovery.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (tier, tier_index), group in tier_event_recovery.groupby(
        ["ladder_tier", "ladder_tier_index"],
        sort=True,
        dropna=False,
    ):
        recovered = _coerce_bool_series(
            group["certified_vs_exact_recovered_expected_model"]
        )
        margin = pd.to_numeric(group["expected_minus_best_comparable_log_evidence"], errors="coerce")
        oracle = _coerce_bool_series(
            group.get("oracle_candidate_support", pd.Series([False], index=group.index))
        )
        rows.append(
            {
                "ladder_tier": str(tier),
                "ladder_tier_index": int(tier_index),
                "expected_model": _first_nonempty(group.get("ladder_expected_model", group.get("expected_model"))),
                "oracle_candidate_support": bool(oracle.iloc[0]),
                "description": _first_nonempty(group.get("ladder_description", pd.Series([""]))),
                "momentum_events": int(group["event_index"].nunique()),
                "certified_or_strict_recovered_events": int(recovered.sum()),
                "certified_or_strict_recovery_fraction": _fraction(recovered.sum(), group["event_index"].nunique()),
                "mean_expected_minus_best_comparable_log_evidence": _mean(margin),
                "median_expected_minus_best_comparable_log_evidence": _median(margin),
                "events_without_comparable_exact_reference": int(
                    group["certified_vs_exact_reason"].astype(str).eq("no_comparable_exact_reference").sum()
                ),
                **_reason_counts(group["certified_vs_exact_reason"]),
            }
        )
    return pd.DataFrame(rows).sort_values("ladder_tier_index").reset_index(drop=True)


def interpret_ladder_summary(tier_summary: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row diagnostic interpretation of the ladder outcome."""

    if tier_summary.empty:
        return pd.DataFrame(
            [
                {
                    "diagnosis": "no_ladder_events",
                    "recommended_next_step": "Run the ladder on at least one session with true_model=momentum.",
                }
            ]
        )

    recovery = {
        str(row["ladder_tier"]): float(row.get("certified_or_strict_recovery_fraction", np.nan))
        for _, row in tier_summary.iterrows()
    }
    full_grid = _recovered(recovery.get("full_grid_pairwise_momentum"))
    finite = _recovered(recovery.get("exact_finite_velocity_momentum"))
    oracle = _recovered(recovery.get("oracle_candidate_pairwise_momentum"))
    native = _recovered(recovery.get("native_candidate_pairwise_momentum"))

    if not full_grid:
        diagnosis = "full_grid_pairwise_momentum_fails"
        next_step = (
            "Check the synthetic momentum generator, pairwise momentum transition density, "
            "and scoring/generative parameter match before tuning real replay evidence."
        )
    elif not finite:
        diagnosis = "finite_velocity_surrogate_mismatch"
        next_step = (
            "Increase the displacement lattice radius or tune finite-velocity noise; use "
            "full-grid pairwise momentum as the implementation reference."
        )
    elif not oracle:
        diagnosis = "candidate_pairwise_scoring_or_lower_bound_issue"
        next_step = (
            "Inspect oracle-support event scores; the true path is present but the candidate "
            "pairwise lower bound is still not certified against exact baselines."
        )
    elif not native:
        diagnosis = "native_candidate_support_loss"
        next_step = (
            "Improve train-only candidate support: posterior source, adaptive mass thresholds, "
            "or momentum-predicted augmentation."
        )
    else:
        diagnosis = "momentum_recovery_supported"
        next_step = (
            "Use exact finite-velocity/displacement evidence for headline model comparison and "
            "report candidate-pruned pairwise momentum/IMM as lower-bound diagnostics."
        )

    return pd.DataFrame(
        [
            {
                "diagnosis": diagnosis,
                "recommended_next_step": next_step,
                "full_grid_pairwise_recovery_fraction": recovery.get("full_grid_pairwise_momentum", np.nan),
                "exact_finite_velocity_recovery_fraction": recovery.get("exact_finite_velocity_momentum", np.nan),
                "oracle_candidate_recovery_fraction": recovery.get("oracle_candidate_pairwise_momentum", np.nan),
                "native_candidate_recovery_fraction": recovery.get("native_candidate_pairwise_momentum", np.nan),
            }
        ]
    )


def render_momentum_recovery_ladder_markdown(result: MomentumRecoveryLadderResult) -> str:
    """Render a compact Markdown summary for the paper pack."""

    lines = ["# Momentum-recovery ladder", ""]
    if result.tier_summary.empty:
        lines.append("No ladder tiers produced event recovery rows.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| tier | recovered / events | recovery fraction | mean margin |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in result.tier_summary.iterrows():
        lines.append(
            "| {tier} | {recovered} / {events} | {frac:.3f} | {margin:.3f} |".format(
                tier=row["ladder_tier"],
                recovered=int(row["certified_or_strict_recovered_events"]),
                events=int(row["momentum_events"]),
                frac=float(row["certified_or_strict_recovery_fraction"]),
                margin=float(row["mean_expected_minus_best_comparable_log_evidence"]),
            )
        )
    if not result.interpretation.empty:
        interp = result.interpretation.iloc[0]
        lines.extend(
            [
                "",
                f"**Diagnosis:** {interp['diagnosis']}",
                "",
                f"**Recommended next step:** {interp['recommended_next_step']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _recovered(value: object, *, threshold: float = 0.50) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > threshold)


def _reason_counts(values: Iterable[object]) -> dict[str, int]:
    counts = pd.Series([str(value) for value in values]).value_counts()
    return {f"reason_{label}_events": int(count) for label, count in counts.items()}


def _first_nonempty(values: pd.Series | None) -> str:
    if values is None:
        return ""
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text:
            return text
    return ""


def _fraction(numerator: object, denominator: object) -> float:
    denom = int(denominator)
    return float("nan") if denom <= 0 else float(numerator) / float(denom)


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float("nan") if numeric.empty else float(numeric.mean())


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float("nan") if numeric.empty else float(numeric.median())
