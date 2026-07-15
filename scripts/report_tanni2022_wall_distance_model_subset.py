#!/usr/bin/env python3
"""Write a non-rescoring stop-gate report for the Tanni exact-model subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-subset-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.model_subset_dir.resolve()
    evidence = pd.read_csv(output_dir / "tanni2022_wall_balanced_model_evidence.csv")
    decisions = pd.read_csv(output_dir / "tanni2022_wall_balanced_model_decisions.csv")
    associations = pd.read_csv(output_dir / "tanni2022_wall_distance_model_subset_associations.csv")
    required_models = {"stationary", "diffusion", "fragmented", "first-order-imm"}
    models_per_event = evidence.groupby(["animal", "session", "event_index"])["model"].agg(set)
    complete = bool(models_per_event.map(lambda values: required_models <= values).all())
    ordered = decisions.loc[decisions["ordered_trajectory_confident"]]
    ordered_animals = int(ordered["animal"].nunique())
    ordered_speed_rows = associations.loc[
        (associations["model_subset"] == "ordered_trajectory_confident")
        & (associations["metric"] == "physical_speed_cm_s")
        & (associations["scope"] == "animal_balanced")
    ]
    speed_estimable = bool(
        not ordered_speed_rows.empty
        and np.isfinite(ordered_speed_rows.iloc[0]["raw_spearman_r"])
        and ordered.shape[0] >= 20
        and ordered_animals >= 4
    )
    gates = pd.DataFrame(
        [
            {"gate": "selected_model_events_present", "passed": decisions.shape[0] > 0, "observed": int(decisions.shape[0]), "criterion": "> 0"},
            {"gate": "required_models_complete", "passed": complete, "observed": f"{models_per_event.shape[0]}/{decisions.shape[0]}", "criterion": "four exact rows per event"},
            {"gate": "five_animals_represented", "passed": decisions["animal"].nunique() == 5, "observed": int(decisions["animal"].nunique()), "criterion": "5"},
            {"gate": "ordered_trajectory_events_sufficient", "passed": ordered.shape[0] >= 20, "observed": int(ordered.shape[0]), "criterion": ">= 20"},
            {"gate": "ordered_trajectory_animals_sufficient", "passed": ordered_animals >= 4, "observed": ordered_animals, "criterion": ">= 4"},
            {"gate": "ordered_wall_speed_association_estimable", "passed": speed_estimable, "observed": "finite" if speed_estimable else "not_estimable", "criterion": "finite animal-balanced association with >=20 events across >=4 animals"},
            {
                "gate": "biological_wall_speed_claim_supported",
                "passed": False,
                "observed": "no_robust_broad_event_effect_and_too_few_ordered_events",
                "criterion": "broad effect clears decoder null and ordered subset is estimable",
            },
        ]
    )
    gates.to_csv(output_dir / "tanni2022_wall_distance_model_subset_gate_summary.csv", index=False)
    best_counts = decisions["best_model"].value_counts().to_dict()
    lines = [
        "# Tanni exact-model wall-distance stop gate",
        "",
        "This is a non-rescoring report for the pre-model, wall-balanced subset.",
        "",
        f"- Events scored: {decisions.shape[0]}",
        f"- Best stationary / diffusion / fragmented / first-order IMM: {best_counts.get('stationary', 0)} / {best_counts.get('diffusion', 0)} / {best_counts.get('fragmented', 0)} / {best_counts.get('first-order-imm', 0)}",
        f"- Confident ordered trajectory events: {ordered.shape[0]} across {ordered_animals}/5 animals",
        f"- IMM confidently above fragmented: {int(decisions['imm_confident_over_fragmented'].sum())}",
        f"- Ordered-vs-static/fragmented ambiguous: {int(decisions['ambiguous'].sum())}",
        "",
        "## Decision",
        "",
        "The ordered trajectory subset is too small for a defensible wall-distance speed test. The larger ripple-associated analysis is technically valid but its apparent physical-speed effect lies inside the constant-speed decoder null and is estimator-dependent. Therefore this dataset does not currently support a biological replay-speed-versus-wall-distance claim.",
        "",
        "The 53 IMM-over-fragmented margins are not a rescue: many occur when stationary remains competitive or best. The relevant clean gate is ordered dynamics over both stationary and fragmented, which only 5 events pass.",
    ]
    (output_dir / "tanni2022_wall_distance_model_subset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
