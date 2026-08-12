#!/usr/bin/env python3
"""Test wall-distance speed only in independently decoded ordered Tanni events.

This analysis deliberately separates decoding from dynamical-model comparison.
Each 20 ms population spike vector is decoded independently with a uniform
prior over valid spatial bins. A conservative linear-sequence statistic is then
tested by whole-bin temporal permutation. Momentum, IMM, fragmented, and
stationary labels are joined only after the replay gate and are descriptive.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from hipporeplayimm.data import RippleEvent
from hipporeplayimm.encoding import EmissionConfig, build_emissions
from hipporeplayimm.tanni2022 import posterior_from_log_likelihood, read_tanni_position

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import build_script_provenance  # noqa: E402
from analyze_tanni2022_wall_distance_replay import (  # noqa: E402
    association_summary,
    fit_decoder_encoding,
    make_replay_session,
    wall_quartile_summary,
)


EVENT_KEYS = ["animal", "session", "event_index"]
MODEL_NAMES = ["stationary", "diffusion", "fragmented", "first-order-imm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--decode-bin-s", type=float, default=0.020)
    parser.add_argument("--spatial-bin-size-cm", type=float, default=8.0)
    parser.add_argument("--min-spatial-extent-cm", type=float, default=32.0)
    parser.add_argument("--n-shuffles", type=int, default=9_999)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--source-overlap-gap-s", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20220716)
    parser.add_argument("--max-events-per-session", type=int)
    return parser.parse_args()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""

    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return output
    order = finite[np.argsort(values[finite], kind="stable")]
    ranked = values[order] * finite.size / np.arange(1, finite.size + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(adjusted, 1.0)
    return output


def permutation_bank(n_time_bins: int, n_shuffles: int, seed: int) -> np.ndarray:
    """Create deterministic whole-bin permutations for one event length."""

    if n_time_bins < 2 or n_shuffles < 1:
        raise ValueError("n_time_bins must be >=2 and n_shuffles must be >=1")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(n_time_bins)]))
    return np.argsort(rng.random((int(n_shuffles), int(n_time_bins))), axis=1)


def _absolute_rank_correlation(values: np.ndarray) -> float:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    time = np.arange(ranks.size, dtype=float)
    ranks -= ranks.mean()
    time -= time.mean()
    denominator = float(np.linalg.norm(ranks) * np.linalg.norm(time))
    return abs(float(np.dot(time, ranks) / denominator)) if denominator > 0.0 else 0.0


def independent_sequence_metrics(
    posterior: np.ndarray,
    bin_centers: np.ndarray,
    permutations: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Measure monotonic linear progression without a temporal transition prior."""

    probabilities = np.asarray(posterior, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    if probabilities.ndim != 2 or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("posterior must be time x space and bin_centers must be space x 2")
    if probabilities.shape[1] != centers.shape[0] or permutations.shape[1] != probabilities.shape[0]:
        raise ValueError("posterior, bin centers, and permutations do not align")
    means = probabilities @ centers
    centered = means - means.mean(axis=0, keepdims=True)
    scatter = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection = centered @ axis
    total_variance = float(np.maximum(eigenvalues, 0.0).sum())
    explained = float(np.max(eigenvalues) / total_variance) if total_variance > 0.0 else 0.0
    score = _absolute_rank_correlation(projection)

    projection_ranks = rankdata(projection, method="average")
    projection_ranks -= projection_ranks.mean()
    time = np.arange(projection.size, dtype=float)
    time -= time.mean()
    denominator = float(np.linalg.norm(projection_ranks) * np.linalg.norm(time))
    if denominator > 0.0:
        null_scores = np.abs(projection_ranks[permutations] @ time / denominator)
    else:
        null_scores = np.zeros(permutations.shape[0], dtype=float)
    empirical_p = float((1 + np.count_nonzero(null_scores >= score - 1e-12)) / (1 + null_scores.size))
    steps = np.linalg.norm(np.diff(means, axis=0), axis=1)
    path_length = float(steps.sum())
    net_displacement = float(np.linalg.norm(means[-1] - means[0]))
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=1)
    spread = np.sqrt(
        np.sum(
            probabilities
            * np.sum((centers[None, :, :] - means[:, None, :]) ** 2, axis=2),
            axis=1,
        )
    )
    metrics: dict[str, float | int] = {
        "n_time_bins": int(probabilities.shape[0]),
        "linear_order_score_abs_spearman": score,
        "shuffle_p95_linear_order_score": float(np.quantile(null_scores, 0.95)),
        "shuffle_empirical_p": empirical_p,
        "principal_axis_extent_cm": float(np.ptp(projection)),
        "principal_axis_explained_variance": explained,
        "posterior_path_length_cm": path_length,
        "posterior_net_displacement_cm": net_displacement,
        "posterior_path_efficiency": net_displacement / path_length if path_length > 0.0 else 0.0,
        "median_posterior_entropy": float(np.median(entropy)),
        "median_posterior_spread_cm": float(np.median(spread)),
        "n_shuffles": int(null_scores.size),
    }
    bins = pd.DataFrame(
        {
            "time_bin_index": np.arange(probabilities.shape[0]),
            "posterior_mean_x_cm": means[:, 0],
            "posterior_mean_y_cm": means[:, 1],
            "principal_axis_position_cm": projection,
            "posterior_entropy": entropy,
            "posterior_spread_cm": spread,
        }
    )
    return metrics, bins


def assign_source_groups(events: pd.DataFrame, overlap_gap_s: float) -> pd.Series:
    """Group overlapping fixed windows within animal/session."""

    output = pd.Series(index=events.index, dtype="Int64")
    next_group = 0
    for _, frame in events.groupby(["animal", "session"], sort=True):
        current_end = -np.inf
        current_group = -1
        for index, row in frame.sort_values("window_start_time_s").iterrows():
            if float(row["window_start_time_s"]) > current_end + float(overlap_gap_s):
                current_group = next_group
                next_group += 1
                current_end = float(row["window_end_time_s"])
            else:
                current_end = max(current_end, float(row["window_end_time_s"]))
            output.loc[index] = current_group
    return output.astype(int)


def finalize_replay_gate(
    events: pd.DataFrame,
    *,
    min_spatial_extent_cm: float,
    fdr_alpha: float,
    source_overlap_gap_s: float,
) -> pd.DataFrame:
    """Apply an order-independent extent screen, BH correction, and de-duplication."""

    output = events.copy()
    output["source_event_group"] = assign_source_groups(output, source_overlap_gap_s)
    output["source_group_representative"] = False
    representatives = (
        output.sort_values(
            ["peak_ripple_z", "n_spikes", "n_active_cells", "event_index"],
            ascending=[False, False, False, True],
        )
        .groupby("source_event_group", sort=False)
        .head(1)
        .index
    )
    output.loc[representatives, "source_group_representative"] = True
    output["spatially_extended"] = output["principal_axis_extent_cm"].ge(float(min_spatial_extent_cm))
    output["shuffle_bh_q"] = 1.0
    family = (
        output["source_group_representative"]
        & output["spatially_extended"]
        & output["shuffle_empirical_p"].notna()
    )
    output.loc[family, "shuffle_bh_q"] = benjamini_hochberg(output.loc[family, "shuffle_empirical_p"].to_numpy())
    output["uniform_prior_ordered_nominal_p05"] = family & output["shuffle_empirical_p"].le(0.05)
    output["uniform_prior_ordered_replay"] = family & output["shuffle_bh_q"].le(float(fdr_alpha))
    output["uniform_prior_ordered_replay_deduplicated"] = output["uniform_prior_ordered_replay"]
    output["uniform_prior_ordered_nominal_p05_deduplicated"] = output["uniform_prior_ordered_nominal_p05"]
    return output


def sequence_enrichment_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Describe excess nominal order tests among pre-evidence source representatives."""

    rows: list[dict[str, object]] = []
    scopes = [("all_animals", events)] + [
        (str(animal), frame) for animal, frame in events.groupby("animal", sort=True)
    ]
    for scope, frame in scopes:
        family = frame.loc[frame["source_group_representative"] & frame["spatially_extended"]]
        observed = int(family["uniform_prior_ordered_nominal_p05"].sum())
        tested = int(len(family))
        expected = 0.05 * tested
        rows.append(
            {
                "scope": scope,
                "tested_source_events": tested,
                "nominal_p05_events": observed,
                "expected_at_uniform_null": expected,
                "excess_events": observed - expected,
                "observed_over_expected": observed / expected if expected > 0.0 else np.nan,
                "binomial_enrichment_p_descriptive": (
                    float(binomtest(observed, tested, 0.05, alternative="greater").pvalue)
                    if tested > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def subset_segments(segments: pd.DataFrame, events: pd.DataFrame, selection_column: str) -> pd.DataFrame:
    selected = events.loc[events[selection_column], EVENT_KEYS]
    return segments.merge(selected, on=EVENT_KEYS, how="inner", validate="many_to_one")


def model_label_summary(events: pd.DataFrame, decisions: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = events.loc[events["uniform_prior_ordered_nominal_p05_deduplicated"]].copy()
    if decisions is None or decisions.empty:
        return selected, pd.DataFrame(
            [
                {"scope": scope, "events": int(events[selection].sum()), "model_scored_events": 0}
                for scope, selection in (
                    ("fdr_independent_replay", "uniform_prior_ordered_replay_deduplicated"),
                    ("nominal_p05_sensitivity", "uniform_prior_ordered_nominal_p05_deduplicated"),
                )
            ]
        )
    columns = [column for column in [*EVENT_KEYS, *MODEL_NAMES, "best_model", "ordered_trajectory_confident", "imm_confident_over_fragmented"] if column in decisions]
    joined = selected.merge(decisions[columns], on=EVENT_KEYS, how="left", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for scope, selection in (
        ("fdr_independent_replay", "uniform_prior_ordered_replay_deduplicated"),
        ("nominal_p05_sensitivity", "uniform_prior_ordered_nominal_p05_deduplicated"),
    ):
        scope_events = joined.loc[joined[selection]]
        scored = scope_events.loc[scope_events["best_model"].notna()]
        ordered_confident = pd.to_numeric(
            scored.get("ordered_trajectory_confident", pd.Series(index=scored.index, dtype=float)),
            errors="coerce",
        ).eq(1)
        imm_confident = pd.to_numeric(
            scored.get("imm_confident_over_fragmented", pd.Series(index=scored.index, dtype=float)),
            errors="coerce",
        ).eq(1)
        rows.append(
            {
                "scope": scope,
                "events": int(events[selection].sum()),
                "model_scored_events": int(len(scored)),
                **{f"best_{model}": int((scored["best_model"] == model).sum()) for model in MODEL_NAMES},
                "ordered_trajectory_confident": int(ordered_confident.sum()),
                "imm_confident_over_fragmented": int(imm_confident.sum()),
            }
        )
    return joined, pd.DataFrame(rows)


def _association_row(associations: pd.DataFrame, metric: str, scope: str = "animal_balanced") -> pd.Series:
    rows = associations.loc[(associations["metric"] == metric) & (associations["scope"] == scope)]
    return rows.iloc[0] if len(rows) else pd.Series(dtype=object)


def _format_signed(value: object) -> str:
    number = float(value)
    return f"{number:+.3f}" if np.isfinite(number) else "not estimable"


def build_gate_summary(
    events: pd.DataFrame,
    associations: pd.DataFrame,
    *,
    n_expected_events: int,
    n_shuffles: int,
) -> tuple[pd.DataFrame, str]:
    selected = events.loc[events["uniform_prior_ordered_replay_deduplicated"]]
    selected_associations = associations.loc[associations["event_scope"] == "uniform_prior_ordered_replay"]
    physical = _association_row(selected_associations, "physical_speed_cm_s")
    code = _association_row(selected_associations, "code_speed_sqrt_hz_per_s")
    null = _association_row(selected_associations, "synthetic_decoded_wall_physical_speed_cm_s")
    physical_adjusted = float(physical.get("quality_adjusted_partial_r", np.nan))
    physical_ci_low = float(physical.get("ci95_low", np.nan))
    physical_ci_high = float(physical.get("ci95_high", np.nan))
    physical_raw = float(physical.get("raw_spearman_r", np.nan))
    null_low = float(null.get("ci95_low", np.nan))
    null_high = float(null.get("ci95_high", np.nan))
    outside_null = bool(np.isfinite(physical_raw) and np.isfinite(null_low) and (physical_raw < null_low or physical_raw > null_high))
    adjusted_nonzero = bool(np.isfinite(physical_ci_low) and (physical_ci_low > 0.0 or physical_ci_high < 0.0))
    technical = [
        ("all_selected_candidates_decoded", len(events) == n_expected_events and len(events) > 0, f"{len(events)}/{n_expected_events}"),
        ("uniform_independent_prior_recorded", events["decoder_prior"].eq("uniform_over_valid_bins_independent_each_time_bin").all(), "no temporal transition prior"),
        ("shuffle_counts_complete", events["n_shuffles"].eq(n_shuffles).all(), f"{int(events['n_shuffles'].eq(n_shuffles).sum())}/{len(events)}"),
        ("all_five_animals_decoded", events["animal"].nunique() == 5, f"{events['animal'].nunique()}/5"),
    ]
    scientific = [
        ("independent_ordered_replay_exists", len(selected) > 0, f"{len(selected)} de-duplicated events"),
        ("independent_ordered_replay_multi_animal", selected["animal"].nunique() >= 3, f"{selected['animal'].nunique()} animals"),
        (
            "nominal_p05_sensitivity_available",
            int(events["uniform_prior_ordered_nominal_p05_deduplicated"].sum()) > 0,
            f"{int(events['uniform_prior_ordered_nominal_p05_deduplicated'].sum())} de-duplicated events; exploratory only",
        ),
        (
            "physical_speed_adjusted_effect_excludes_zero",
            adjusted_nonzero,
            f"rho={_format_signed(physical_adjusted)}; CI=[{_format_signed(physical_ci_low)},{_format_signed(physical_ci_high)}]",
        ),
        (
            "physical_speed_effect_outside_constant_speed_decoder_null",
            outside_null,
            f"observed={_format_signed(physical_raw)}; null_CI=[{_format_signed(null_low)},{_format_signed(null_high)}]",
        ),
    ]
    technical_pass = all(value for _, value, _ in technical)
    speed_supported = technical_pass and len(selected) > 0 and selected["animal"].nunique() >= 3 and adjusted_nonzero and outside_null
    if not technical_pass:
        verdict = "technical_incomplete"
    elif len(selected) == 0:
        verdict = "no_independently_ordered_replay_detected"
    elif not speed_supported:
        verdict = "independent_replay_present_but_wall_speed_mechanism_not_supported"
    else:
        verdict = "wall_speed_effect_survives_independent_replay_gate"
    rows = [
        {"gate": name, "passed": bool(passed), "observed": observed}
        for name, passed, observed in [*technical, *scientific]
    ]
    rows.extend(
        [
            {
                "gate": "code_speed_wall_association_descriptive",
                "passed": None,
                "observed": (
                    f"raw={_format_signed(code.get('raw_spearman_r', np.nan))}; "
                    f"adjusted={_format_signed(code.get('quality_adjusted_partial_r', np.nan))}"
                ),
            },
            {"gate": "overall_technical", "passed": technical_pass, "observed": f"{sum(value for _, value, _ in technical)}/{len(technical)}"},
            {"gate": "biological_wall_speed_supported", "passed": speed_supported, "observed": verdict},
        ]
    )
    return pd.DataFrame(rows), verdict


def make_figure(events: pd.DataFrame, quartiles: pd.DataFrame, model_summary: pd.DataFrame, output_path: Path) -> None:
    selected = events.loc[events["uniform_prior_ordered_replay_deduplicated"]]
    nominal = events.loc[events["uniform_prior_ordered_nominal_p05_deduplicated"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].scatter(events["principal_axis_extent_cm"], events["linear_order_score_abs_spearman"], s=8, alpha=0.25, color="#7d8792")
    axes[0, 0].scatter(selected["principal_axis_extent_cm"], selected["linear_order_score_abs_spearman"], s=20, color="#b8323d", label="FDR-significant, de-duplicated")
    axes[0, 0].set(xlabel="Principal-axis extent (cm)", ylabel="|Spearman(time, represented position)|", title="Uniform-prior sequence gate")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].hist(events["shuffle_bh_q"], bins=np.linspace(0, 1, 31), color="#7d8792")
    axes[0, 1].set(xlabel="Whole-bin shuffle BH q", ylabel="Events", title="Order-shuffle significance")
    plot_scope = "uniform_prior_ordered_replay" if len(selected) >= 20 else "nominal_p05_sensitivity"
    selected_quartiles = quartiles.loc[
        (quartiles["event_scope"] == plot_scope)
        & (quartiles["aggregation_level"] == "animal_balanced_median")
    ]
    order = ["Q1_nearest", "Q2", "Q3", "Q4_farthest"]
    for axis, metric, title, ylabel in [
        (axes[1, 0], "physical_speed_cm_s", "Decoded physical speed", "cm/s"),
        (axes[1, 1], "code_speed_sqrt_hz_per_s", "Population-code speed", "sqrt(Hz)/s"),
    ]:
        values = selected_quartiles.set_index("wall_quartile")[metric].reindex(order)
        axis.plot(np.arange(4), values, marker="o", color="#b8323d", linewidth=2)
        axis.set_xticks(np.arange(4), ["Q1\nnear", "Q2", "Q3", "Q4\nfar"])
        axis.set(ylabel=ylabel, title=title)
        axis.grid(axis="y", color="#d8dde2", linewidth=0.7)
    title_suffix = (
        f"FDR subset n={len(selected)}"
        if plot_scope == "uniform_prior_ordered_replay"
        else f"nominal-p<.05 sensitivity n={len(nominal)}; FDR subset n={len(selected)}"
    )
    fig.suptitle(f"Tanni: independent uniform-prior replay gate ({title_suffix})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    events: pd.DataFrame,
    associations: pd.DataFrame,
    model_summary: pd.DataFrame,
    enrichment: pd.DataFrame,
    gates: pd.DataFrame,
    verdict: str,
    *,
    min_spatial_extent_cm: float,
    n_shuffles: int,
    fdr_alpha: float,
) -> None:
    selected = events.loc[events["uniform_prior_ordered_replay_deduplicated"]]
    selected_assoc = associations.loc[associations["event_scope"] == "uniform_prior_ordered_replay"]
    nominal = events.loc[events["uniform_prior_ordered_nominal_p05_deduplicated"]]
    nominal_assoc = associations.loc[associations["event_scope"] == "nominal_p05_sensitivity"]
    physical = _association_row(selected_assoc, "physical_speed_cm_s")
    code = _association_row(selected_assoc, "code_speed_sqrt_hz_per_s")
    nominal_physical = _association_row(nominal_assoc, "physical_speed_cm_s")
    nominal_code = _association_row(nominal_assoc, "code_speed_sqrt_hz_per_s")
    overlap_rows = model_summary.loc[model_summary["scope"] == "fdr_independent_replay"] if len(model_summary) else pd.DataFrame()
    overlap = overlap_rows.iloc[0] if len(overlap_rows) else pd.Series(dtype=object)
    nominal_overlap_rows = model_summary.loc[model_summary["scope"] == "nominal_p05_sensitivity"] if len(model_summary) else pd.DataFrame()
    nominal_overlap = nominal_overlap_rows.iloc[0] if len(nominal_overlap_rows) else pd.Series(dtype=object)
    enrichment_all = enrichment.loc[enrichment["scope"] == "all_animals"].iloc[0]
    lines = [
        "# Tanni uniform-prior replay-speed test",
        "",
        f"**Verdict:** `{verdict}`.",
        "",
        "## Method boundary",
        "",
        "Events were decoded independently in 20 ms bins with a uniform prior over valid spatial bins. No HMM, momentum transition, diffusion transition, or temporal smoother entered the decoded path or replay-selection statistic.",
        "",
        f"A conservative linear ordered sequence requires at least {min_spatial_extent_cm:g} cm principal-axis extent and an absolute Spearman time-position score significant after {n_shuffles:,} whole-population-bin order shuffles and BH correction at q <= {fdr_alpha:g} across all spatially extended candidates. Model labels are joined only after this gate.",
        "",
        "## Independent replay gate",
        "",
        f"- Ripple-positive, immobile, spike-supported candidates decoded: {len(events)}",
        f"- Pre-evidence one-per-source representatives: {int(events['source_group_representative'].sum())}",
        f"- Spatially extended source representatives entering the order-test family: {int((events['spatially_extended'] & events['source_group_representative']).sum())}",
        f"- Ordered events at nominal permutation p <= 0.05: {int(events['uniform_prior_ordered_nominal_p05'].sum())}",
        f"- Expected nominal events under a uniform null: {float(enrichment_all['expected_at_uniform_null']):.1f}; observed/expected: {float(enrichment_all['observed_over_expected']):.2f}x; descriptive binomial p={float(enrichment_all['binomial_enrichment_p_descriptive']):.3g}",
        f"- FDR-significant ordered events before source-window de-duplication: {int(events['uniform_prior_ordered_replay'].sum())}",
        f"- FDR-significant one-per-source events: {len(selected)}",
        f"- Animals represented: {selected['animal'].nunique()}",
        f"- Nominal-p<.05 one-per-source sensitivity set: {len(nominal)} events across {nominal['animal'].nunique()} animals",
        "",
        "## Wall-distance result in independently selected events",
        "",
        f"- Physical speed: raw animal-median rho {_format_signed(physical.get('raw_spearman_r', np.nan))}; quality-adjusted rho {_format_signed(physical.get('quality_adjusted_partial_r', np.nan))}; 95% animal bootstrap CI [{_format_signed(physical.get('ci95_low', np.nan))}, {_format_signed(physical.get('ci95_high', np.nan))}].",
        f"- Population-code speed: raw rho {_format_signed(code.get('raw_spearman_r', np.nan))}; quality-adjusted rho {_format_signed(code.get('quality_adjusted_partial_r', np.nan))}.",
        "",
        "The confirmatory FDR subset is too small for correlation inference. For comparison with the conventional per-event p<.05 practice, the explicitly exploratory nominal subset gives:",
        "",
        f"- Physical speed: raw animal-median rho {_format_signed(nominal_physical.get('raw_spearman_r', np.nan))}; quality-adjusted rho {_format_signed(nominal_physical.get('quality_adjusted_partial_r', np.nan))}; 95% animal bootstrap CI [{_format_signed(nominal_physical.get('ci95_low', np.nan))}, {_format_signed(nominal_physical.get('ci95_high', np.nan))}].",
        f"- Population-code speed: raw rho {_format_signed(nominal_code.get('raw_spearman_r', np.nan))}; quality-adjusted rho {_format_signed(nominal_code.get('quality_adjusted_partial_r', np.nan))}.",
        "",
        "## Secondary model characterization",
        "",
        f"- Independently selected events overlapping the pre-existing model-scored subset: {int(overlap.get('model_scored_events', 0))}/{len(selected)}.",
        f"- Nominal sensitivity events overlapping the model-scored subset: {int(nominal_overlap.get('model_scored_events', 0))}/{len(nominal)}.",
        "- Momentum/IMM/fragmented/stationary labels did not define the decoded paths or select events; their counts are descriptive only.",
        "",
        "A failure of the wall-speed gate does not imply replay is absent. It means the proposed wall-dependent propagation mechanism is not established after an independent traditional sequence criterion and decoder controls.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(evidence_dir / "tanni2022_session_manifest.csv")
    ripple_events = pd.read_csv(evidence_dir / "tanni2022_ripple_candidates.csv")
    expected_events = ripple_events.loc[ripple_events["selected_for_decoding"]].copy()
    segments = pd.read_csv(evidence_dir / "tanni2022_replay_speed_segments.csv")
    decoder_samples = pd.read_csv(evidence_dir / "tanni2022_decoder_qc_samples.csv")
    synthetic = pd.read_csv(evidence_dir / "tanni2022_synthetic_constant_speed_null.csv")

    event_rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    banks: dict[int, np.ndarray] = {}
    for manifest_row in manifest.itertuples(index=False):
        session_events = expected_events.loc[
            (expected_events["animal"] == manifest_row.animal)
            & (expected_events["session"] == manifest_row.session)
        ].sort_values("event_index")
        if args.max_events_per_session is not None:
            session_events = session_events.head(args.max_events_per_session)
        if session_events.empty:
            continue
        position = read_tanni_position(Path(manifest_row.nwb_path))
        session = make_replay_session(Path(manifest_row.nwb_path), position)
        encoding, _ = fit_decoder_encoding(
            session,
            position,
            bin_size_cm=args.spatial_bin_size_cm,
            smoothing_sigma_bins=1.5,
            running_speed_cm_s=10.0,
            min_running_spikes=30,
            max_mean_rate_hz=4.0,
            min_peak_rate_hz=2.0,
            min_split_half_stability=0.25,
        )
        selected_session = replace(session, excitatory_neurons=encoding.cell_ids)
        valid_bins = encoding.occupancy_s >= 0.05
        for event in session_events.itertuples(index=False):
            ripple = RippleEvent(
                start=float(event.window_start_time_s),
                end=float(event.window_end_time_s),
                peak=float(event.peak_time_s),
                raw_power=float(event.peak_ripple_z),
                z_power_session=float(event.peak_ripple_z),
                z_power_epoch=float(event.peak_ripple_z),
            )
            emissions = build_emissions(selected_session, encoding, ripple, EmissionConfig(time_bin_s=args.decode_bin_s))
            log_likelihood = emissions.log_likelihood.copy()
            log_likelihood[:, ~valid_bins] = -np.inf
            posterior = posterior_from_log_likelihood(log_likelihood)
            if posterior.shape[0] not in banks:
                banks[posterior.shape[0]] = permutation_bank(
                    posterior.shape[0],
                    args.n_shuffles,
                    args.seed,
                )
            bank = banks[posterior.shape[0]]
            metrics, bins = independent_sequence_metrics(posterior, encoding.bin_centers, bank)
            common = {
                "animal": str(event.animal),
                "session": str(event.session),
                "event_index": int(event.event_index),
                "window_start_time_s": float(event.window_start_time_s),
                "window_end_time_s": float(event.window_end_time_s),
                "peak_time_s": float(event.peak_time_s),
                "peak_ripple_z": float(event.peak_ripple_z),
                "n_spikes": int(event.n_spikes),
                "n_active_cells": int(event.n_active_cells),
                "decoder_prior": "uniform_over_valid_bins_independent_each_time_bin",
                "path_estimator": "independent_emission_posterior_mean",
            }
            event_rows.append(common | metrics)
            for key, value in reversed(list(common.items())):
                bins.insert(0, key, value)
            bins["time_s"] = emissions.times
            bin_frames.append(bins)
        print(f"{manifest_row.animal}: decoded {len(session_events)} candidates", flush=True)

    events = pd.DataFrame(event_rows)
    events = finalize_replay_gate(
        events,
        min_spatial_extent_cm=args.min_spatial_extent_cm,
        fdr_alpha=args.fdr_alpha,
        source_overlap_gap_s=args.source_overlap_gap_s,
    )
    enrichment = sequence_enrichment_summary(events)
    posterior_bins = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    posterior_bins = posterior_bins.merge(
        events[
            [
                *EVENT_KEYS,
                "spatially_extended",
                "shuffle_bh_q",
                "uniform_prior_ordered_nominal_p05",
                "uniform_prior_ordered_nominal_p05_deduplicated",
                "uniform_prior_ordered_replay",
                "uniform_prior_ordered_replay_deduplicated",
            ]
        ],
        on=EVENT_KEYS,
        how="left",
        validate="many_to_one",
    )
    selected_segments = subset_segments(segments, events, "uniform_prior_ordered_replay_deduplicated")
    nominal_segments = subset_segments(
        segments,
        events,
        "uniform_prior_ordered_nominal_p05_deduplicated",
    )
    broad_associations = association_summary(
        segments,
        synthetic,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    broad_associations.insert(0, "event_scope", "all_ripple_candidates")
    broad_quartiles = wall_quartile_summary(segments, decoder_samples)
    broad_quartiles.insert(0, "event_scope", "all_ripple_candidates")
    if selected_segments.empty:
        selected_associations = pd.DataFrame(columns=broad_associations.columns)
        selected_quartiles = pd.DataFrame(columns=broad_quartiles.columns)
    else:
        selected_associations = association_summary(
            selected_segments,
            synthetic,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed + 1,
        )
        selected_associations.insert(0, "event_scope", "uniform_prior_ordered_replay")
        selected_quartiles = wall_quartile_summary(selected_segments, decoder_samples)
        selected_quartiles.insert(0, "event_scope", "uniform_prior_ordered_replay")
    if nominal_segments.empty:
        nominal_associations = pd.DataFrame(columns=broad_associations.columns)
        nominal_quartiles = pd.DataFrame(columns=broad_quartiles.columns)
    else:
        nominal_associations = association_summary(
            nominal_segments,
            synthetic,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed + 2,
        )
        nominal_associations.insert(0, "event_scope", "nominal_p05_sensitivity")
        nominal_quartiles = wall_quartile_summary(nominal_segments, decoder_samples)
        nominal_quartiles.insert(0, "event_scope", "nominal_p05_sensitivity")
    associations = pd.concat(
        [broad_associations, selected_associations, nominal_associations],
        ignore_index=True,
    )
    quartiles = pd.concat(
        [broad_quartiles, selected_quartiles, nominal_quartiles],
        ignore_index=True,
    )

    decisions = None
    if args.model_dir is not None:
        decisions = pd.read_csv(args.model_dir.resolve() / "tanni2022_wall_balanced_model_decisions.csv")
    event_models, model_summary = model_label_summary(events, decisions)
    expected_count = int(len(expected_events)) if args.max_events_per_session is None else int(len(events))
    gates, verdict = build_gate_summary(
        events,
        associations,
        n_expected_events=expected_count,
        n_shuffles=args.n_shuffles,
    )

    events.to_csv(output_dir / "tanni2022_uniform_prior_replay_events.csv", index=False)
    posterior_bins.to_csv(output_dir / "tanni2022_uniform_prior_posterior_bins.csv", index=False)
    selected_segments.to_csv(output_dir / "tanni2022_uniform_prior_replay_speed_segments.csv", index=False)
    nominal_segments.to_csv(
        output_dir / "tanni2022_uniform_prior_nominal_p05_speed_segments.csv",
        index=False,
    )
    associations.to_csv(output_dir / "tanni2022_uniform_prior_replay_wall_associations.csv", index=False)
    quartiles.to_csv(output_dir / "tanni2022_uniform_prior_replay_wall_quartiles.csv", index=False)
    event_models.to_csv(output_dir / "tanni2022_uniform_prior_replay_model_labels.csv", index=False)
    model_summary.to_csv(output_dir / "tanni2022_uniform_prior_replay_model_summary.csv", index=False)
    enrichment.to_csv(
        output_dir / "tanni2022_uniform_prior_sequence_enrichment.csv",
        index=False,
    )
    gates.to_csv(output_dir / "tanni2022_uniform_prior_replay_gate_summary.csv", index=False)
    make_figure(events, quartiles, model_summary, output_dir / "tanni2022_uniform_prior_replay_speed_figure.png")
    write_report(
        output_dir / "tanni2022_uniform_prior_replay_speed_report.md",
        events,
        associations,
        model_summary,
        enrichment,
        gates,
        verdict,
        min_spatial_extent_cm=args.min_spatial_extent_cm,
        n_shuffles=args.n_shuffles,
        fdr_alpha=args.fdr_alpha,
    )
    provenance = build_script_provenance(
        input_paths={
            "session_manifest": evidence_dir / "tanni2022_session_manifest.csv",
            "ripple_candidates": evidence_dir / "tanni2022_ripple_candidates.csv",
            "segments": evidence_dir / "tanni2022_replay_speed_segments.csv",
            "decoder_samples": evidence_dir / "tanni2022_decoder_qc_samples.csv",
            "synthetic_null": evidence_dir / "tanni2022_synthetic_constant_speed_null.csv",
            **(
                {"model_decisions": args.model_dir.resolve() / "tanni2022_wall_balanced_model_decisions.csv"}
                if args.model_dir is not None
                else {}
            ),
        }
    )
    payload = {
        "analysis": "tanni2022_uniform_prior_replay_speed",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "decoder_prior": "uniform_over_valid_bins_independent_each_time_bin",
        "temporal_transition_model_used_for_decoding_or_selection": False,
        "model_labels_role": "secondary_descriptive_only",
        "verdict": verdict,
        "provenance": provenance,
    }
    (output_dir / "tanni2022_uniform_prior_replay_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
