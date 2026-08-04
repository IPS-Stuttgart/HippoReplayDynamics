#!/usr/bin/env python3
"""Freeze a non-overlapping, high-information Tanni 2022 replay cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402


SELECTION_OUTPUT = "tanni2022_clean_imm_holdout_selection.csv"
BY_ANIMAL_OUTPUT = "tanni2022_clean_imm_holdout_selection_by_animal.csv"
GATE_OUTPUT = "tanni2022_clean_imm_holdout_selection_gate_summary.csv"
MANIFEST_OUTPUT = "tanni2022_clean_imm_holdout_selection_manifest.json"

KEYS = ["animal", "session", "event_index"]
PRE_EVIDENCE_RANK_COLUMNS = ["n_active_cells", "n_spikes", "peak_ripple_z", "event_index"]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def select_holdout_events(
    ripple_candidates: pd.DataFrame,
    prior_selection: pd.DataFrame,
    *,
    events_per_animal: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        *KEYS,
        "window_start_time_s",
        "window_end_time_s",
        "peak_time_s",
        "peak_ripple_z",
        "n_spikes",
        "n_active_cells",
        "immobile",
        "spike_supported",
        "selected_for_decoding",
        "event_definition",
    }
    missing = required.difference(ripple_candidates.columns)
    if missing:
        raise ValueError(f"ripple candidates are missing columns: {sorted(missing)}")
    prior_missing = set(KEYS).difference(prior_selection.columns)
    if prior_missing:
        raise ValueError(f"prior selection is missing columns: {sorted(prior_missing)}")
    if events_per_animal <= 0:
        raise ValueError("events_per_animal must be positive")

    prior_keys = {
        (str(row.animal), str(row.session), int(row.event_index))
        for row in prior_selection.itertuples(index=False)
    }
    candidates = ripple_candidates.copy()
    candidates["immobile"] = _as_bool(candidates["immobile"])
    candidates["spike_supported"] = _as_bool(candidates["spike_supported"])
    candidates["selected_for_decoding"] = _as_bool(candidates["selected_for_decoding"])
    candidates["excluded_prior_model_event"] = [
        (str(row.animal), str(row.session), int(row.event_index)) in prior_keys
        for row in candidates.itertuples(index=False)
    ]
    candidates["pre_evidence_qc_valid"] = (
        candidates["immobile"]
        & candidates["spike_supported"]
        & candidates["selected_for_decoding"]
    )
    eligible = candidates[
        candidates["pre_evidence_qc_valid"]
        & ~candidates["excluded_prior_model_event"]
    ].copy()
    eligible = eligible.sort_values(
        ["animal", *PRE_EVIDENCE_RANK_COLUMNS],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    selected = eligible.groupby("animal", sort=True, group_keys=False).head(events_per_animal).copy()
    selected["selection_rank_within_animal"] = selected.groupby("animal", sort=True).cumcount() + 1
    selected["selection_rule"] = (
        "pre_evidence_qc_then_n_active_cells_n_spikes_peak_ripple_z"
    )
    selected["selection_score_name"] = "lexicographic_active_cells_spikes_ripple_z"
    selected["selection_tier"] = "clean_imm_high_information_holdout_diagnostic"
    selected = selected.sort_values(["animal", "selection_rank_within_animal"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    animals = sorted(candidates["animal"].astype(str).unique())
    for animal in animals:
        animal_candidates = candidates[candidates["animal"].astype(str).eq(animal)]
        animal_eligible = eligible[eligible["animal"].astype(str).eq(animal)]
        animal_selected = selected[selected["animal"].astype(str).eq(animal)]
        rows.append(
            {
                "animal": animal,
                "candidate_events": int(len(animal_candidates)),
                "pre_evidence_qc_events": int(animal_candidates["pre_evidence_qc_valid"].sum()),
                "prior_model_events_excluded": int(animal_candidates["excluded_prior_model_event"].sum()),
                "eligible_holdout_events": int(len(animal_eligible)),
                "selected_events": int(len(animal_selected)),
                "minimum_selected_active_cells": int(animal_selected["n_active_cells"].min()) if len(animal_selected) else 0,
                "median_selected_active_cells": float(animal_selected["n_active_cells"].median()) if len(animal_selected) else float("nan"),
                "minimum_selected_spikes": int(animal_selected["n_spikes"].min()) if len(animal_selected) else 0,
                "median_selected_spikes": float(animal_selected["n_spikes"].median()) if len(animal_selected) else float("nan"),
            }
        )
    return selected, pd.DataFrame(rows)


def selection_gates(
    selected: pd.DataFrame,
    by_animal: pd.DataFrame,
    *,
    events_per_animal: int,
    expected_animals: int,
) -> pd.DataFrame:
    expected_events = int(events_per_animal * expected_animals)
    keys_unique = not selected.duplicated(KEYS).any() if not selected.empty else False
    pre_evidence_columns_only = not any(
        token in column.lower()
        for column in PRE_EVIDENCE_RANK_COLUMNS
        for token in ("evidence", "logz", "best_model", "posterior")
    )
    checks = [
        ("expected_animals_present", selected["animal"].nunique() == expected_animals if not selected.empty else False, f"{selected['animal'].nunique() if not selected.empty else 0}/{expected_animals}"),
        ("balanced_event_target_complete", len(selected) == expected_events and expected_events > 0, f"{len(selected)}/{expected_events}"),
        ("all_animals_reach_target", bool(len(by_animal) == expected_animals and by_animal["selected_events"].eq(events_per_animal).all()), f"target={events_per_animal}"),
        ("selection_keys_unique", bool(keys_unique), f"duplicates={int(selected.duplicated(KEYS).sum()) if not selected.empty else 0}"),
        ("prior_model_overlap_zero", bool(not selected.empty and ~selected["excluded_prior_model_event"].any()), f"overlap={int(selected['excluded_prior_model_event'].sum()) if not selected.empty else 0}"),
        ("all_selected_immobile", bool(not selected.empty and selected["immobile"].all()), f"{int(selected['immobile'].sum()) if not selected.empty else 0}/{len(selected)}"),
        ("all_selected_spike_supported", bool(not selected.empty and selected["spike_supported"].all()), f"{int(selected['spike_supported'].sum()) if not selected.empty else 0}/{len(selected)}"),
        ("uses_pre_evidence_rank_fields_only", bool(pre_evidence_columns_only), ",".join(PRE_EVIDENCE_RANK_COLUMNS)),
    ]
    overall = all(passed for _, passed, _ in checks)
    checks.append(("overall_technical", overall, "selection frozen before holdout scoring"))
    return pd.DataFrame(
        {"gate": gate, "passed": bool(passed), "detail": detail}
        for gate, passed, detail in checks
    )


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    ripple_candidates_path = Path(args.ripple_candidates).resolve()
    prior_selection_path = Path(args.prior_model_selection).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(ripple_candidates_path)
    prior = pd.read_csv(prior_selection_path)
    selected, by_animal = select_holdout_events(
        candidates,
        prior,
        events_per_animal=args.events_per_animal,
    )
    gates = selection_gates(
        selected,
        by_animal,
        events_per_animal=args.events_per_animal,
        expected_animals=args.expected_animals,
    )
    selected.to_csv(output_dir / SELECTION_OUTPUT, index=False)
    by_animal.to_csv(output_dir / BY_ANIMAL_OUTPUT, index=False)
    gates.to_csv(output_dir / GATE_OUTPUT, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Tanni_et_al_2022_large_2d",
        "selection_tier": "clean_imm_high_information_holdout_diagnostic",
        "selection_is_pre_evidence": True,
        "selection_was_pilot_informed": True,
        "events_per_animal": int(args.events_per_animal),
        "expected_animals": int(args.expected_animals),
        "rank_columns": PRE_EVIDENCE_RANK_COLUMNS,
        "selected_events": int(len(selected)),
        **build_script_provenance(
            input_paths={
                "ripple_candidates": ripple_candidates_path,
                "prior_model_selection": prior_selection_path,
            }
        ),
    }
    (output_dir / MANIFEST_OUTPUT).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"selection": selected, "by_animal": by_animal, "gates": gates}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ripple-candidates", required=True)
    parser.add_argument("--prior-model-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--events-per-animal", type=int, default=50)
    parser.add_argument("--expected-animals", type=int, default=5)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
