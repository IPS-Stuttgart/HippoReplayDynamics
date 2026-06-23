#!/usr/bin/env python3
"""Freeze pre-evidence SleepPOST pilot event subsets for Olafsdottir2016."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SELECTION_OUTPUT = "olafsdottir_sleeppost_pilot_event_selection.csv"
ANIMAL_OUTPUT = "olafsdottir_sleeppost_pilot_event_selection_by_animal.csv"
GATE_OUTPUT = "olafsdottir_sleeppost_pilot_event_selection_gate_summary.csv"
SUMMARY_OUTPUT = "olafsdottir_sleeppost_pilot_event_selection_summary.md"

REQUIRED_EVENT_COLUMNS = {
    "animal",
    "date",
    "session",
    "event_id",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_mua_rate_hz",
    "peak_mua_rate_hz",
    "mean_speed_cm_s",
    "event_detection_score",
    "candidate_tier",
    "event_qc_status",
    "event_qc_reason",
}
REQUIRED_DECODER_COLUMNS = {
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "decoder_status",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
}

TIER_TARGETS = {
    "pilot_20_balanced": 2,
    "pilot_50_balanced": 5,
    "pilot_100_balanced": 10,
}

SELECTION_COLUMNS = [
    "selection_tier",
    "selection_seed",
    "tier_target_events_per_pair",
    "selection_rule_version",
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "session",
    "event_id",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_mua_rate_hz",
    "peak_mua_rate_hz",
    "mean_speed_cm_s",
    "event_detection_score",
    "candidate_tier",
    "event_qc_status",
    "event_qc_reason",
    "decoder_status",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
    "eligible_event_count_for_pair",
    "tier_rank_within_pair",
    "pre_evidence_random_score",
]


def run_pilot_event_selection(
    *,
    candidate_events_csv: str | Path,
    decoder_qc_csv: str | Path,
    output_dir: str | Path,
    seed: int = 27011374643,
    immobility_speed_threshold_cm_s: float = 5.0,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 500.0,
    min_event_spikes: int = 5,
    min_event_active_units: int = 3,
    pilot20_events_per_pair: int = 2,
    pilot50_events_per_pair: int = 5,
    pilot100_events_per_pair: int = 10,
    min_pilot20_animals_fraction: float = 0.80,
) -> dict[str, pd.DataFrame]:
    events = load_candidate_events(candidate_events_csv)
    decoder = load_decoder_qc(decoder_qc_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tier_targets = {
        "pilot_20_balanced": int(pilot20_events_per_pair),
        "pilot_50_balanced": int(pilot50_events_per_pair),
        "pilot_100_balanced": int(pilot100_events_per_pair),
    }
    decoder_pass = decoder[decoder["decoder_status"].astype(str).eq("pass")].copy()
    eligible = eligible_events(
        events,
        decoder_pass,
        seed=seed,
        immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        min_event_spikes=min_event_spikes,
        min_event_active_units=min_event_active_units,
    )
    selection = build_selection(eligible, tier_targets=tier_targets, seed=seed)
    animals = summarize_by_animal(selection, eligible, decoder_pass, tier_targets=tier_targets)
    gates = gate_summary(
        events=events,
        decoder_pass=decoder_pass,
        eligible=eligible,
        selection=selection,
        tier_targets=tier_targets,
        min_pilot20_animals_fraction=min_pilot20_animals_fraction,
    )

    selection.to_csv(out / SELECTION_OUTPUT, index=False)
    animals.to_csv(out / ANIMAL_OUTPUT, index=False)
    gates.to_csv(out / GATE_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(
        build_markdown_summary(
            selection,
            animals,
            gates,
            seed=seed,
            immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_event_spikes=min_event_spikes,
            min_event_active_units=min_event_active_units,
            tier_targets=tier_targets,
        ),
        encoding="utf-8",
    )
    return {
        "selection": selection,
        "animals": animals,
        "gates": gates,
        "eligible": eligible,
    }


def load_candidate_events(path: str | Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    missing = sorted(REQUIRED_EVENT_COLUMNS.difference(events.columns))
    if missing:
        raise ValueError(f"candidate event CSV is missing required columns: {missing}")
    return events


def load_decoder_qc(path: str | Path) -> pd.DataFrame:
    decoder = pd.read_csv(path)
    missing = sorted(REQUIRED_DECODER_COLUMNS.difference(decoder.columns))
    if missing:
        raise ValueError(f"decoder QC CSV is missing required columns: {missing}")
    return decoder


def eligible_events(
    events: pd.DataFrame,
    decoder_pass: pd.DataFrame,
    *,
    seed: int,
    immobility_speed_threshold_cm_s: float,
    min_duration_ms: float,
    max_duration_ms: float,
    min_event_spikes: int,
    min_event_active_units: int,
) -> pd.DataFrame:
    if events.empty or decoder_pass.empty:
        return pd.DataFrame(columns=eligible_columns())
    prepared_events = events.copy()
    prepared_events["animal"] = prepared_events["animal"].astype(str).str.upper()
    prepared_events["date"] = prepared_events["date"].astype(str)
    prepared_events["session"] = prepared_events["session"].astype(str)
    prepared_decoder = decoder_pass.copy()
    prepared_decoder["animal"] = prepared_decoder["animal"].astype(str).str.upper()
    prepared_decoder["date"] = prepared_decoder["date"].astype(str)
    prepared_decoder["sleeppost_session"] = prepared_decoder["sleeppost_session"].astype(str)
    joined = prepared_events.merge(
        prepared_decoder[
            [
                "animal",
                "date",
                "track1_session",
                "sleeppost_session",
                "decoder_status",
                "posterior_mean_error_cm_median",
                "map_error_cm_median",
                "posterior_coverage_fraction",
            ]
        ],
        left_on=["animal", "date", "session"],
        right_on=["animal", "date", "sleeppost_session"],
        how="inner",
    )
    keep = (
        joined["event_qc_status"].astype(str).eq("pass")
        & (pd.to_numeric(joined["mean_speed_cm_s"], errors="coerce") <= float(immobility_speed_threshold_cm_s))
        & (pd.to_numeric(joined["duration_ms"], errors="coerce") >= float(min_duration_ms))
        & (pd.to_numeric(joined["duration_ms"], errors="coerce") <= float(max_duration_ms))
        & (pd.to_numeric(joined["n_spikes"], errors="coerce") >= int(min_event_spikes))
        & (pd.to_numeric(joined["n_active_units"], errors="coerce") >= int(min_event_active_units))
    )
    eligible = joined[keep].copy()
    if eligible.empty:
        return pd.DataFrame(columns=eligible_columns())
    eligible["pre_evidence_random_score"] = [
        stable_random_score(seed, row.animal, row.date, row.session, row.event_id)
        for row in eligible.itertuples(index=False)
    ]
    eligible = eligible.sort_values(
        ["animal", "date", "sleeppost_session", "pre_evidence_random_score", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    pair_counts = eligible.groupby(["animal", "date", "sleeppost_session"]).size().rename("eligible_event_count_for_pair")
    eligible = eligible.merge(pair_counts.reset_index(), on=["animal", "date", "sleeppost_session"], how="left")
    eligible["tier_rank_within_pair"] = eligible.groupby(["animal", "date", "sleeppost_session"]).cumcount() + 1
    return eligible[eligible_columns()]


def build_selection(eligible: pd.DataFrame, *, tier_targets: dict[str, int], seed: int) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    rows: list[pd.DataFrame] = []
    for tier, target in tier_targets.items():
        selected = select_n_per_pair(eligible, target)
        selected = selected.copy()
        selected.insert(0, "selection_tier", tier)
        selected.insert(1, "selection_seed", int(seed))
        selected.insert(2, "tier_target_events_per_pair", int(target))
        selected.insert(3, "selection_rule_version", "pre_evidence_v1")
        rows.append(selected)
    all_events = eligible.copy()
    all_events["tier_rank_within_pair"] = all_events.groupby(["animal", "date", "sleeppost_session"]).cumcount() + 1
    all_events.insert(0, "selection_tier", "all_immobile_qc_valid")
    all_events.insert(1, "selection_seed", int(seed))
    all_events.insert(2, "tier_target_events_per_pair", np.nan)
    all_events.insert(3, "selection_rule_version", "pre_evidence_v1")
    rows.append(all_events)
    return pd.concat(rows, ignore_index=True)[SELECTION_COLUMNS]


def select_n_per_pair(eligible: pd.DataFrame, target: int) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for _pair, group in eligible.groupby(["animal", "date", "sleeppost_session"], sort=True):
        take = group.head(int(target)).copy()
        take["tier_rank_within_pair"] = np.arange(1, len(take) + 1, dtype=int)
        selected_parts.append(take)
    if not selected_parts:
        return pd.DataFrame(columns=eligible_columns())
    return pd.concat(selected_parts, ignore_index=True)


def summarize_by_animal(
    selection: pd.DataFrame,
    eligible: pd.DataFrame,
    decoder_pass: pd.DataFrame,
    *,
    tier_targets: dict[str, int],
) -> pd.DataFrame:
    columns = [
        "animal",
        "decoder_pass_pairs",
        "eligible_events",
        "eligible_pairs",
        "min_eligible_events_per_pair",
        "pilot_20_balanced_events",
        "pilot_50_balanced_events",
        "pilot_100_balanced_events",
        "all_immobile_qc_valid_events",
    ]
    animals = sorted(set(decoder_pass["animal"].astype(str).str.upper()) | set(eligible["animal"].astype(str).str.upper()))
    rows: list[dict[str, object]] = []
    for animal in animals:
        animal_decoder = decoder_pass[decoder_pass["animal"].astype(str).str.upper().eq(animal)]
        animal_eligible = eligible[eligible["animal"].astype(str).str.upper().eq(animal)]
        animal_selection = selection[selection["animal"].astype(str).str.upper().eq(animal)] if not selection.empty else pd.DataFrame(columns=SELECTION_COLUMNS)
        pair_counts = animal_eligible.groupby(["date", "sleeppost_session"]).size() if not animal_eligible.empty else pd.Series(dtype=int)
        rows.append(
            {
                "animal": animal,
                "decoder_pass_pairs": int(len(animal_decoder)),
                "eligible_events": int(len(animal_eligible)),
                "eligible_pairs": int(pair_counts.shape[0]),
                "min_eligible_events_per_pair": int(pair_counts.min()) if not pair_counts.empty else 0,
                "pilot_20_balanced_events": int(tier_count(animal_selection, "pilot_20_balanced")),
                "pilot_50_balanced_events": int(tier_count(animal_selection, "pilot_50_balanced")),
                "pilot_100_balanced_events": int(tier_count(animal_selection, "pilot_100_balanced")),
                "all_immobile_qc_valid_events": int(tier_count(animal_selection, "all_immobile_qc_valid")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def gate_summary(
    *,
    events: pd.DataFrame,
    decoder_pass: pd.DataFrame,
    eligible: pd.DataFrame,
    selection: pd.DataFrame,
    tier_targets: dict[str, int],
    min_pilot20_animals_fraction: float,
) -> pd.DataFrame:
    pass_pairs = decoder_pass_pairs(decoder_pass)
    eligible_pairs = event_pairs(eligible)
    selected_20 = selection[selection["selection_tier"].astype(str).eq("pilot_20_balanced")] if not selection.empty else pd.DataFrame(columns=SELECTION_COLUMNS)
    pilot20_pairs = event_pairs(selected_20)
    input_animals = set(decoder_pass["animal"].astype(str).str.upper()) if not decoder_pass.empty else set()
    pilot20_animals = set(selected_20["animal"].astype(str).str.upper()) if not selected_20.empty else set()
    expected_20 = int(tier_targets["pilot_20_balanced"]) * len(pass_pairs)
    expected_50 = int(tier_targets["pilot_50_balanced"]) * len(pass_pairs)
    expected_100 = int(tier_targets["pilot_100_balanced"]) * len(pass_pairs)
    gates = [
        _gate(
            "candidate_events_loaded",
            len(events) > 0,
            f"candidate_events={len(events)}",
            "SleepPOST candidate event table is non-empty",
            "Pilot selection requires candidate windows from pre-evidence event QC.",
        ),
        _gate(
            "decoder_pass_pairs_present",
            len(pass_pairs) > 0,
            f"decoder_pass_pairs={len(pass_pairs)}",
            "at least one Track1/SleepPOST pair passed decoder QC",
            "The selector only freezes events for decoder-ready pairs.",
        ),
        _gate(
            "eligible_events_for_decoder_pairs",
            bool(pass_pairs) and pass_pairs.issubset(eligible_pairs),
            f"eligible_pairs={len(eligible_pairs)}; decoder_pass_pairs={len(pass_pairs)}",
            "every decoder-pass pair has at least one immobile QC-valid event",
            "Pre-evidence filters should not silently drop complete sessions.",
        ),
        _gate(
            "pilot_20_balanced_complete",
            expected_20 > 0 and tier_count(selection, "pilot_20_balanced") == expected_20,
            f"selected={tier_count(selection, 'pilot_20_balanced')}; expected={expected_20}",
            f"{tier_targets['pilot_20_balanced']} events per decoder-pass pair",
            "The first evidence smoke should use the frozen 20-event balanced pilot when 10 pairs are present.",
        ),
        _gate(
            "pilot_50_balanced_available",
            expected_50 > 0 and tier_count(selection, "pilot_50_balanced") == expected_50,
            f"selected={tier_count(selection, 'pilot_50_balanced')}; expected={expected_50}",
            f"{tier_targets['pilot_50_balanced']} events per decoder-pass pair",
            "The next scale-up tier should already be frozen if enough pre-evidence candidates exist.",
        ),
        _gate(
            "pilot_100_balanced_available",
            expected_100 > 0 and tier_count(selection, "pilot_100_balanced") == expected_100,
            f"selected={tier_count(selection, 'pilot_100_balanced')}; expected={expected_100}",
            f"{tier_targets['pilot_100_balanced']} events per decoder-pass pair",
            "The 100-event scale-up tier should already be frozen if enough pre-evidence candidates exist.",
        ),
        _gate(
            "pilot_20_spans_decoder_animals",
            bool(input_animals) and len(pilot20_animals) / len(input_animals) >= float(min_pilot20_animals_fraction),
            f"pilot20_animals={len(pilot20_animals)}; decoder_animals={len(input_animals)}",
            f"pilot_20 covers at least {float(min_pilot20_animals_fraction):g} of decoder-pass animals",
            "The first pilot should not collapse to one animal if decoder-ready candidates are broader.",
        ),
        _gate(
            "pilot_20_spans_decoder_pairs",
            bool(pass_pairs) and pass_pairs == pilot20_pairs,
            f"pilot20_pairs={len(pilot20_pairs)}; decoder_pass_pairs={len(pass_pairs)}",
            "pilot_20 has selected events for every decoder-pass pair",
            "The first pilot should be session-balanced.",
        ),
        _gate(
            "selection_is_pre_evidence_only",
            True,
            "uses event QC, immobility, duration, spike/MUA fields, decoder-pass membership, session balance, animal balance, seed",
            "no replay model evidence or logZ scores are used",
            "This freezes events before any 1D model-evidence scoring.",
        ),
    ]
    return pd.DataFrame(gates)


def build_markdown_summary(
    selection: pd.DataFrame,
    animals: pd.DataFrame,
    gates: pd.DataFrame,
    *,
    seed: int,
    immobility_speed_threshold_cm_s: float,
    min_duration_ms: float,
    max_duration_ms: float,
    min_event_spikes: int,
    min_event_active_units: int,
    tier_targets: dict[str, int],
) -> str:
    gate_passes = int(gates["passed"].map(_as_bool).sum()) if not gates.empty else 0
    lines = [
        "# Olafsdottir SleepPOST Pilot Event Selection Summary",
        "",
        "This freezes pre-evidence pilot subsets only. It does not use replay model scores and does not run 1D model-evidence scoring.",
        "",
        "## Selection Rule",
        "",
        _markdown_table(
            ["Parameter", "Value"],
            [
                ("selection_rule_version", "pre_evidence_v1"),
                ("seed", seed),
                ("immobility_speed_threshold_cm_s", immobility_speed_threshold_cm_s),
                ("min_duration_ms", min_duration_ms),
                ("max_duration_ms", max_duration_ms),
                ("min_event_spikes", min_event_spikes),
                ("min_event_active_units", min_event_active_units),
                ("pilot_20_balanced events per pair", tier_targets["pilot_20_balanced"]),
                ("pilot_50_balanced events per pair", tier_targets["pilot_50_balanced"]),
                ("pilot_100_balanced events per pair", tier_targets["pilot_100_balanced"]),
            ],
        ),
        "",
        "## Overview",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ("pilot_20_balanced events", tier_count(selection, "pilot_20_balanced")),
                ("pilot_50_balanced events", tier_count(selection, "pilot_50_balanced")),
                ("pilot_100_balanced events", tier_count(selection, "pilot_100_balanced")),
                ("all_immobile_qc_valid events", tier_count(selection, "all_immobile_qc_valid")),
                ("Selection gates passed", f"{gate_passes}/{len(gates)}"),
            ],
        ),
        "",
        "## Gate Summary",
        "",
        _markdown_table(["Gate", "Status", "Value"], gates[["gate", "status", "value"]].itertuples(index=False, name=None)),
        "",
        "## Animal Summary",
        "",
        _markdown_table(
            ["Animal", "Decoder pairs", "Eligible events", "Pilot 20", "Pilot 50", "Pilot 100"],
            animal_summary_rows(animals),
        ),
        "",
    ]
    return "\n".join(lines)


def eligible_columns() -> list[str]:
    return [
        "animal",
        "date",
        "track1_session",
        "sleeppost_session",
        "session",
        "event_id",
        "start_time_s",
        "end_time_s",
        "duration_ms",
        "n_spikes",
        "n_active_units",
        "mean_mua_rate_hz",
        "peak_mua_rate_hz",
        "mean_speed_cm_s",
        "event_detection_score",
        "candidate_tier",
        "event_qc_status",
        "event_qc_reason",
        "decoder_status",
        "posterior_mean_error_cm_median",
        "map_error_cm_median",
        "posterior_coverage_fraction",
        "eligible_event_count_for_pair",
        "tier_rank_within_pair",
        "pre_evidence_random_score",
    ]


def decoder_pass_pairs(decoder_pass: pd.DataFrame) -> set[tuple[str, str, str]]:
    if decoder_pass.empty:
        return set()
    return {
        (str(row.animal).upper(), str(row.date), str(row.sleeppost_session))
        for row in decoder_pass.itertuples(index=False)
    }


def event_pairs(events: pd.DataFrame) -> set[tuple[str, str, str]]:
    if events.empty:
        return set()
    session_column = "sleeppost_session" if "sleeppost_session" in events.columns else "session"
    return {
        (str(row.animal).upper(), str(row.date), str(getattr(row, session_column)))
        for row in events.itertuples(index=False)
    }


def tier_count(selection: pd.DataFrame, tier: str) -> int:
    if selection.empty or "selection_tier" not in selection:
        return 0
    return int(selection["selection_tier"].astype(str).eq(tier).sum())


def stable_random_score(seed: int, animal: object, date: object, session: object, event_id: object) -> float:
    key = f"{int(seed)}|{animal}|{date}|{session}|{event_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return float(value / 2**64)


def _gate(gate: str, passed: bool, value: str, requirement: str, note: str) -> dict[str, object]:
    return {
        "gate": gate,
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "value": value,
        "requirement": requirement,
        "note": note,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def animal_summary_rows(animals: pd.DataFrame) -> list[tuple[object, ...]]:
    if animals.empty:
        return []
    return [
        (
            row.animal,
            int(row.decoder_pass_pairs),
            int(row.eligible_events),
            int(row.pilot_20_balanced_events),
            int(row.pilot_50_balanced_events),
            int(row.pilot_100_balanced_events),
        )
        for row in animals.itertuples(index=False)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-events", type=Path, required=True)
    parser.add_argument("--decoder-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/olafsdottir-sleeppost-pilot-events"))
    parser.add_argument("--seed", type=int, default=27011374643)
    parser.add_argument("--immobility-speed-threshold-cm-s", type=float, default=5.0)
    parser.add_argument("--min-duration-ms", type=float, default=20.0)
    parser.add_argument("--max-duration-ms", type=float, default=500.0)
    parser.add_argument("--min-event-spikes", type=int, default=5)
    parser.add_argument("--min-event-active-units", type=int, default=3)
    parser.add_argument("--pilot20-events-per-pair", type=int, default=2)
    parser.add_argument("--pilot50-events-per-pair", type=int, default=5)
    parser.add_argument("--pilot100-events-per-pair", type=int, default=10)
    parser.add_argument("--min-pilot20-animals-fraction", type=float, default=0.80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = run_pilot_event_selection(
        candidate_events_csv=args.candidate_events,
        decoder_qc_csv=args.decoder_qc,
        output_dir=args.output_dir,
        seed=args.seed,
        immobility_speed_threshold_cm_s=args.immobility_speed_threshold_cm_s,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        min_event_spikes=args.min_event_spikes,
        min_event_active_units=args.min_event_active_units,
        pilot20_events_per_pair=args.pilot20_events_per_pair,
        pilot50_events_per_pair=args.pilot50_events_per_pair,
        pilot100_events_per_pair=args.pilot100_events_per_pair,
        min_pilot20_animals_fraction=args.min_pilot20_animals_fraction,
    )
    print(tables["animals"].to_string(index=False))
    print()
    print(tables["gates"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
