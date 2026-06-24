#!/usr/bin/env python3
"""Summarize Olafsdottir2016 Track1/SleepPOST manifest usability."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd


PAIR_OUTPUT = "olafsdottir_track_sleep_pairs.csv"
SUMMARY_OUTPUT = "olafsdottir_dataset_qc_summary.md"
REQUIRED_MANIFEST_COLUMNS = {
    "animal",
    "date",
    "session_type",
    "session_name",
    "session_path",
    "has_pos",
    "n_cut_files",
    "n_egf_files",
    "hippocampal_tetrodes",
    "mec_tetrodes",
}
PAIR_COLUMNS = [
    "animal",
    "date",
    "track_session",
    "sleepPOST_session",
    "track_has_pos",
    "track_n_cut_files",
    "sleep_has_egf",
    "sleep_n_cut_files",
    "hippocampal_tetrodes",
    "mec_tetrodes",
    "r2142_reversal_applied",
    "usable_pair",
    "exclusion_reason",
]
_TRUE_BOOL_STRINGS = {"true", "t", "1", "1.0", "yes", "y", "on"}
_FALSE_BOOL_STRINGS = {"", "false", "f", "0", "0.0", "no", "n", "off", "nan", "none", "null", "<na>"}


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    return manifest


def build_track_sleep_pairs(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if manifest.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    frame = manifest.copy()
    frame["animal"] = frame["animal"].astype(str).str.upper()
    for (animal, date), group in frame.groupby(["animal", "date"], sort=True):
        tracks = _session_rows(group, "track1")
        sleeps = _session_rows(group, "sleepPOST")
        track = tracks.iloc[0] if len(tracks) == 1 else pd.Series(dtype=object)
        sleep = sleeps.iloc[0] if len(sleeps) == 1 else pd.Series(dtype=object)

        hpc_tetrodes = _select_tetrode_string(track, sleep, "hippocampal_tetrodes")
        mec_tetrodes = _select_tetrode_string(track, sleep, "mec_tetrodes")
        hpc = _parse_tetrodes(hpc_tetrodes)
        mec = _parse_tetrodes(mec_tetrodes)
        r2142_reversal = animal == "R2142" and hpc == tuple(range(1, 9)) and mec == tuple(range(9, 17))

        reasons = _pair_exclusion_reasons(
            animal=animal,
            tracks=tracks,
            sleeps=sleeps,
            track=track,
            sleep=sleep,
            hpc=hpc,
            mec=mec,
            r2142_reversal=r2142_reversal,
        )
        rows.append(
            {
                "animal": animal,
                "date": str(date),
                "track_session": _session_names(tracks),
                "sleepPOST_session": _session_names(sleeps),
                "track_has_pos": _row_bool(track, "has_pos") if len(tracks) == 1 else False,
                "track_n_cut_files": _row_int(track, "n_cut_files") if len(tracks) == 1 else 0,
                "sleep_has_egf": _row_int(sleep, "n_egf_files") > 0 if len(sleeps) == 1 else False,
                "sleep_n_cut_files": _row_int(sleep, "n_cut_files") if len(sleeps) == 1 else 0,
                "hippocampal_tetrodes": hpc_tetrodes,
                "mec_tetrodes": mec_tetrodes,
                "r2142_reversal_applied": bool(r2142_reversal),
                "usable_pair": not reasons,
                "exclusion_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def write_qc_outputs(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    manifest = load_manifest(manifest_path)
    pairs = build_track_sleep_pairs(manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_path = out / PAIR_OUTPUT
    summary_path = out / SUMMARY_OUTPUT
    pairs.to_csv(pair_path, index=False)
    summary_path.write_text(build_markdown_summary(manifest, pairs, manifest_path=Path(manifest_path)), encoding="utf-8")
    return {"pairs": pair_path, "summary": summary_path}


def build_markdown_summary(manifest: pd.DataFrame, pairs: pd.DataFrame, *, manifest_path: Path | None = None) -> str:
    animals = sorted(manifest["animal"].dropna().astype(str).str.upper().unique()) if "animal" in manifest else []
    usable = pairs[pairs["usable_pair"].map(_as_bool)] if "usable_pair" in pairs else pd.DataFrame(columns=pairs.columns)
    usable_animals = sorted(usable["animal"].dropna().astype(str).str.upper().unique()) if not usable.empty else []
    r2142_rows = manifest[manifest["animal"].astype(str).str.upper().eq("R2142")] if "animal" in manifest else pd.DataFrame()
    r2142_status = _r2142_status(r2142_rows, pairs)

    overview_rows = [
        ("Manifest", "" if manifest_path is None else str(manifest_path)),
        ("Animals", len(animals)),
        ("Session rows", len(manifest)),
        ("Track1 sessions", int(manifest["session_type"].astype(str).eq("track1").sum())),
        ("SleepPOST sessions", int(manifest["session_type"].astype(str).eq("sleepPOST").sum())),
        ("Track1/SleepPOST date groups", len(pairs)),
        ("Usable pairs", int(pairs["usable_pair"].map(_as_bool).sum()) if not pairs.empty else 0),
        ("Usable-pair animals", len(usable_animals)),
        ("Pairs with Track1 position", _count_true(pairs, "track_has_pos")),
        ("Pairs with Track1 cut files", _count_positive(pairs, "track_n_cut_files")),
        ("Pairs with SleepPOST EGF", _count_true(pairs, "sleep_has_egf")),
        ("Pairs with SleepPOST cut files", _count_positive(pairs, "sleep_n_cut_files")),
        ("Pairs with hippocampal tetrode metadata", _count_nonempty(pairs, "hippocampal_tetrodes")),
        ("R2142 reversal check", r2142_status),
    ]
    recommendation = _recommendation(usable_pairs=len(usable), usable_animals=len(usable_animals))
    lines = [
        "# Olafsdottir2016 Dataset QC Summary",
        "",
        "This is a dataset-ingestion checkpoint only. It does not support a biological 1D-vs-2D comparison by itself.",
        "",
        "## Overview",
        "",
        _markdown_table(["Metric", "Value"], overview_rows),
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Exclusion Reasons",
        "",
        _exclusion_markdown(pairs),
        "",
    ]
    return "\n".join(lines)


def _pair_exclusion_reasons(
    *,
    animal: str,
    tracks: pd.DataFrame,
    sleeps: pd.DataFrame,
    track: pd.Series,
    sleep: pd.Series,
    hpc: tuple[int, ...],
    mec: tuple[int, ...],
    r2142_reversal: bool,
) -> list[str]:
    reasons: list[str] = []
    if len(tracks) == 0:
        reasons.append("no_track1")
    elif len(tracks) > 1:
        reasons.append("multiple_track1")
    if len(sleeps) == 0:
        reasons.append("no_sleepPOST")
    elif len(sleeps) > 1:
        reasons.append("multiple_sleepPOST")

    if len(tracks) == 1:
        if not _row_bool(track, "has_pos"):
            reasons.append("track_missing_pos")
        if _row_int(track, "n_cut_files") <= 0:
            reasons.append("track_missing_cut")
    if len(sleeps) == 1:
        if _row_int(sleep, "n_egf_files") <= 0:
            reasons.append("sleep_missing_egf")
        if _row_int(sleep, "n_cut_files") <= 0:
            reasons.append("sleep_missing_cut")
    if not hpc:
        reasons.append("missing_hippocampal_tetrodes")
    if len(tracks) == 1 and len(sleeps) == 1:
        if _row_str(track, "hippocampal_tetrodes") != _row_str(sleep, "hippocampal_tetrodes"):
            reasons.append("hippocampal_tetrodes_mismatch")
        if _row_str(track, "mec_tetrodes") != _row_str(sleep, "mec_tetrodes"):
            reasons.append("mec_tetrodes_mismatch")
    if animal == "R2142" and not r2142_reversal:
        reasons.append("r2142_reversal_not_applied")
    if animal != "R2142" and hpc and mec and (hpc != tuple(range(9, 17)) or mec != tuple(range(1, 9))):
        reasons.append("standard_tetrode_mapping_unexpected")
    return reasons


def _session_rows(group: pd.DataFrame, session_type: str) -> pd.DataFrame:
    return group[group["session_type"].astype(str).eq(session_type)].sort_values("session_name")


def _session_names(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    return ";".join(rows["session_name"].astype(str))


def _select_tetrode_string(track: pd.Series, sleep: pd.Series, column: str) -> str:
    track_value = _row_str(track, column)
    if track_value:
        return track_value
    return _row_str(sleep, column)


def _parse_tetrodes(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            return ()
    return tuple(sorted(dict.fromkeys(values)))


def _row_str(row: pd.Series, column: str) -> str:
    if row.empty or column not in row or pd.isna(row[column]):
        return ""
    return str(row[column]).strip()


def _row_bool(row: pd.Series, column: str) -> bool:
    return _as_bool(_row_str(row, column))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    text = str(value).strip().lower()
    if text in _TRUE_BOOL_STRINGS:
        return True
    if text in _FALSE_BOOL_STRINGS:
        return False
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return bool(pd.notna(numeric) and float(numeric) == 1.0)


def _row_int(row: pd.Series, column: str) -> int:
    if row.empty or column not in row:
        return 0
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return int(value) if pd.notna(value) else 0


def _count_true(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].map(_as_bool).sum()) if column in frame else 0


def _count_positive(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int((values > 0).sum())


def _count_nonempty(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].fillna("").astype(str).str.strip().ne("").sum())


def _r2142_status(r2142_rows: pd.DataFrame, pairs: pd.DataFrame) -> str:
    if r2142_rows.empty:
        return "not_present"
    hpc_ok = _tetrode_column_matches(r2142_rows, "hippocampal_tetrodes", tuple(range(1, 9)))
    mec_ok = _tetrode_column_matches(r2142_rows, "mec_tetrodes", tuple(range(9, 17)))
    pair_ok = True
    if not pairs.empty and "animal" in pairs:
        r2142_pairs = pairs[pairs["animal"].astype(str).str.upper().eq("R2142")]
        pair_ok = bool(r2142_pairs["r2142_reversal_applied"].map(_as_bool).all()) if not r2142_pairs.empty else True
    return "pass" if hpc_ok and mec_ok and pair_ok else "fail"


def _tetrode_column_matches(rows: pd.DataFrame, column: str, expected: tuple[int, ...]) -> bool:
    if column not in rows:
        return False
    parsed = rows[column].map(_parse_tetrodes)
    return bool(parsed.map(lambda value: value == expected).all())


def _recommendation(*, usable_pairs: int, usable_animals: int) -> str:
    if usable_pairs > 0 and usable_animals > 1:
        return "Proceed to multi-session linearization and event-detection QC before any evidence scaling."
    if usable_pairs > 0:
        return "Keep the analysis as a feasibility smoke until usable Track1/SleepPOST pairs span more than one animal."
    return "Do not run evidence yet; no usable Track1/SleepPOST pair passed the manifest-level gates."


def _exclusion_markdown(pairs: pd.DataFrame) -> str:
    if pairs.empty or "exclusion_reason" not in pairs:
        return "No Track1/SleepPOST date groups were found."
    counter: Counter[str] = Counter()
    for raw in pairs["exclusion_reason"].fillna("").astype(str):
        for reason in raw.split(";"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    if not counter:
        return "No manifest-level exclusions."
    return _markdown_table(["Exclusion reason", "Pairs"], sorted(counter.items()))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_qc_outputs(args.manifest, args.output_dir)
    print(f"Wrote {paths['pairs']}")
    print(f"Wrote {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
