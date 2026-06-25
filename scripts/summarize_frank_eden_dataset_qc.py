#!/usr/bin/env python3
"""Summarize Frank/Eden replay trajectory dataset availability and layout.

This is an ingestion/QC checkpoint for the Denovellis et al. eLife / Frank-Eden
replay-trajectory data ecosystem. It does not score replay events. The script is
intended to work with local exports from the Dryad repository, CRCNS hc-6 style
Frank-lab data, or a clone/export of the replay_trajectory_paper repository.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DATASET_NAME = "denovellis2021_frank_eden"
DRYAD_DOI = "10.7272/Q61N7ZC3"
CRCNS_DATASET = "hc-6"
REPLAY_TRAJECTORY_PAPER_REPO = "https://github.com/Eden-Kramer-Lab/replay_trajectory_paper"
REPLAY_TRAJECTORY_CLASSIFICATION_REPO = "https://github.com/Eden-Kramer-Lab/replay_trajectory_classification"

KNOWN_FRANK_FILE_TYPES = {
    "task": "task_metadata",
    "tetinfo": "tetrode_metadata",
    "cellinfo": "cell_metadata",
    "spikes": "sorted_spikes",
    "marks": "clusterless_marks",
    "pos": "position",
    "rawpos": "raw_position",
    "linpos": "linearized_position",
    "ripples": "ripple_events",
    "eeg": "lfp_eeg",
    "egf": "lfp_egf",
    "lfp": "lfp",
    "dio": "digital_io",
    "theta": "theta",
    "ca1gamma": "ca1_gamma",
}

INVENTORY_COLUMNS = [
    "relative_path",
    "file_name",
    "extension",
    "size_bytes",
    "animal",
    "day",
    "frank_file_type",
    "semantic_role",
    "is_mat_file",
    "is_source_data_csv",
]

DAY_QC_COLUMNS = [
    "animal",
    "day",
    "n_files",
    "has_task",
    "has_position",
    "has_linearized_position",
    "has_sorted_spikes",
    "has_clusterless_marks",
    "has_ripples",
    "has_lfp",
    "has_tetrode_metadata",
    "has_cell_metadata",
    "candidate_replay_day",
    "candidate_clusterless_day",
    "missing_for_sorted_replay",
    "missing_for_clusterless_replay",
]

SOURCE_DATA_COLUMNS = [
    "relative_path",
    "file_name",
    "figure",
    "source_data_index",
    "size_bytes",
]

GATE_COLUMNS = ["gate", "passed", "observed", "criterion"]


def build_file_inventory(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Frank/Eden dataset root does not exist: {root}")
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        parsed = _parse_frank_file_name(path.name)
        source = _parse_source_data_name(path.name)
        file_type = parsed["frank_file_type"]
        rows.append(
            {
                "relative_path": relative,
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "animal": parsed["animal"],
                "day": parsed["day"],
                "frank_file_type": file_type,
                "semantic_role": KNOWN_FRANK_FILE_TYPES.get(str(file_type).lower(), ""),
                "is_mat_file": path.suffix.lower() == ".mat",
                "is_source_data_csv": bool(source),
            }
        )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def build_source_data_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if inventory.empty:
        return pd.DataFrame(columns=SOURCE_DATA_COLUMNS)
    for _, row in inventory.iterrows():
        parsed = _parse_source_data_name(str(row["file_name"]))
        if not parsed:
            continue
        rows.append(
            {
                "relative_path": row["relative_path"],
                "file_name": row["file_name"],
                "figure": parsed["figure"],
                "source_data_index": parsed["source_data_index"],
                "size_bytes": row["size_bytes"],
            }
        )
    return pd.DataFrame(rows, columns=SOURCE_DATA_COLUMNS)


def build_session_day_qc(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty or "animal" not in inventory:
        return pd.DataFrame(columns=DAY_QC_COLUMNS)
    parsed = inventory[
        inventory["animal"].astype(str).ne("")
        & inventory["day"].astype(str).ne("")
        & inventory["frank_file_type"].astype(str).ne("")
    ].copy()
    if parsed.empty:
        return pd.DataFrame(columns=DAY_QC_COLUMNS)

    rows: list[dict[str, object]] = []
    for (animal, day), group in parsed.groupby(["animal", "day"], sort=True):
        types = {str(value).lower() for value in group["frank_file_type"].dropna() if str(value)}
        has_task = "task" in types
        has_position = bool(types.intersection({"pos", "rawpos", "linpos"}))
        has_linearized_position = "linpos" in types
        has_sorted_spikes = "spikes" in types
        has_clusterless_marks = "marks" in types
        has_ripples = "ripples" in types
        has_lfp = bool(types.intersection({"eeg", "egf", "lfp"}))
        has_tetrode_metadata = "tetinfo" in types
        has_cell_metadata = "cellinfo" in types
        missing_sorted = _missing(
            {
                "task": has_task,
                "position": has_position,
                "spikes": has_sorted_spikes,
                "ripples": has_ripples,
            }
        )
        missing_clusterless = _missing(
            {
                "task": has_task,
                "position": has_position,
                "marks": has_clusterless_marks,
                "ripples": has_ripples,
            }
        )
        rows.append(
            {
                "animal": animal,
                "day": day,
                "n_files": int(len(group)),
                "has_task": has_task,
                "has_position": has_position,
                "has_linearized_position": has_linearized_position,
                "has_sorted_spikes": has_sorted_spikes,
                "has_clusterless_marks": has_clusterless_marks,
                "has_ripples": has_ripples,
                "has_lfp": has_lfp,
                "has_tetrode_metadata": has_tetrode_metadata,
                "has_cell_metadata": has_cell_metadata,
                "candidate_replay_day": not missing_sorted,
                "candidate_clusterless_day": not missing_clusterless,
                "missing_for_sorted_replay": ";".join(missing_sorted),
                "missing_for_clusterless_replay": ";".join(missing_clusterless),
            }
        )
    return pd.DataFrame(rows, columns=DAY_QC_COLUMNS)


def build_gate_summary(inventory: pd.DataFrame, day_qc: pd.DataFrame, source_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str) -> None:
        rows.append({"gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion})

    n_files = int(len(inventory))
    n_mat = int(inventory["is_mat_file"].sum()) if not inventory.empty else 0
    n_days = int(len(day_qc))
    n_candidate_sorted = int(day_qc["candidate_replay_day"].sum()) if not day_qc.empty else 0
    n_candidate_clusterless = int(day_qc["candidate_clusterless_day"].sum()) if not day_qc.empty else 0
    n_animals = int(day_qc["animal"].nunique()) if not day_qc.empty else 0

    add("files_present", n_files > 0, n_files, "dataset root contains files")
    add("frank_mat_files_present", n_mat > 0, n_mat, "Frank-lab .mat files are present")
    add("parsed_day_rows_present", n_days > 0, n_days, "at least one animal/day can be parsed")
    add(
        "sorted_replay_candidate_days_present",
        n_candidate_sorted > 0,
        n_candidate_sorted,
        "at least one day has task, position, spikes, and ripples",
    )
    add(
        "clusterless_candidate_days_reported",
        True,
        n_candidate_clusterless,
        "clusterless-ready days are counted without requiring marks",
    )
    add("animal_coverage_reported", True, n_animals, "animal coverage is counted")
    add("source_data_csvs_reported", True, len(source_data), "eLife source-data CSVs are counted when present")
    rows.append(
        {
            "gate": "overall",
            "passed": all(row["passed"] for row in rows[:4]),
            "observed": f"{sum(row['passed'] for row in rows[:4])}/4 required gates passed",
            "criterion": "files, Frank-lab .mat files, parsed days, and sorted replay candidate days exist",
        }
    )
    return pd.DataFrame(rows, columns=GATE_COLUMNS)


def build_markdown_summary(
    *,
    dataset_root: str | Path,
    inventory: pd.DataFrame,
    day_qc: pd.DataFrame,
    source_data: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    candidate_days = int(day_qc["candidate_replay_day"].sum()) if not day_qc.empty else 0
    clusterless_days = int(day_qc["candidate_clusterless_day"].sum()) if not day_qc.empty else 0
    animals = int(day_qc["animal"].nunique()) if not day_qc.empty else 0
    lines = [
        "# Frank/Eden Denovellis2021 Dataset QC Summary",
        "",
        "This is an ingestion checkpoint only. It does not score replay evidence or claim replication.",
        "",
        "## Sources",
        "",
        _markdown_table(
            ["Item", "Value"],
            [
                ("Dataset", DATASET_NAME),
                ("Dryad DOI", DRYAD_DOI),
                ("CRCNS legacy dataset", CRCNS_DATASET),
                ("Paper code", REPLAY_TRAJECTORY_PAPER_REPO),
                ("Decoder package", REPLAY_TRAJECTORY_CLASSIFICATION_REPO),
                ("Local root", str(dataset_root)),
            ],
        ),
        "",
        "## Local Inventory",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ("Files", len(inventory)),
                (".mat files", int(inventory["is_mat_file"].sum()) if not inventory.empty else 0),
                ("Parsed animal/day rows", len(day_qc)),
                ("Animals with parsed days", animals),
                ("Sorted-spike replay candidate days", candidate_days),
                ("Clusterless/mark candidate days", clusterless_days),
                ("eLife source-data CSV files", len(source_data)),
            ],
        ),
        "",
        "## Gate Summary",
        "",
        _markdown_table(["Gate", "Passed", "Observed", "Criterion"], gates.itertuples(index=False, name=None)),
        "",
        "## Interpretation",
        "",
        _recommendation(candidate_days=candidate_days, clusterless_days=clusterless_days),
        "",
    ]
    return "\n".join(lines)


def write_qc_outputs(dataset_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory = build_file_inventory(dataset_root)
    source_data = build_source_data_inventory(inventory)
    day_qc = build_session_day_qc(inventory)
    gates = build_gate_summary(inventory, day_qc, source_data)

    paths = {
        "inventory": out / "frank_eden_file_inventory.csv",
        "day_qc": out / "frank_eden_session_day_qc.csv",
        "source_data": out / "frank_eden_source_data_inventory.csv",
        "gates": out / "frank_eden_dataset_qc_gate_summary.csv",
        "summary": out / "frank_eden_dataset_qc_summary.md",
    }
    inventory.to_csv(paths["inventory"], index=False)
    day_qc.to_csv(paths["day_qc"], index=False)
    source_data.to_csv(paths["source_data"], index=False)
    gates.to_csv(paths["gates"], index=False)
    paths["summary"].write_text(
        build_markdown_summary(
            dataset_root=dataset_root,
            inventory=inventory,
            day_qc=day_qc,
            source_data=source_data,
            gates=gates,
        ),
        encoding="utf-8",
    )
    return paths


def _parse_frank_file_name(file_name: str) -> dict[str, str]:
    stem = Path(file_name).stem
    if Path(file_name).suffix.lower() != ".mat":
        return {"animal": "", "day": "", "frank_file_type": ""}

    type_group = "|".join(sorted(KNOWN_FRANK_FILE_TYPES, key=len, reverse=True))
    patterns = [
        rf"^(?P<animal>[A-Za-z]{{2,8}})(?P<file_type>{type_group})(?P<day>\d{{1,3}})?$",
        rf"^(?P<animal>[A-Za-z]{{2,8}})(?P<day>\d{{1,3}})(?P<file_type>{type_group})$",
        rf"^(?P<animal>[A-Za-z]{{2,8}})[_-]?(?P<file_type>{type_group})[_-]?(?P<day>\d{{1,3}})?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if not match:
            continue
        animal = match.group("animal").lower()
        day = match.groupdict().get("day") or ""
        file_type = match.group("file_type").lower()
        return {"animal": animal, "day": day, "frank_file_type": file_type}
    return {"animal": "", "day": "", "frank_file_type": ""}


def _parse_source_data_name(file_name: str) -> dict[str, str] | None:
    match = re.match(r"^elife-64505-fig(?P<figure>\d+)-data(?P<index>\d+).*[.]csv$", file_name, flags=re.IGNORECASE)
    if not match:
        return None
    return {"figure": match.group("figure"), "source_data_index": match.group("index")}


def _missing(flags: dict[str, bool]) -> list[str]:
    return [name for name, present in flags.items() if not present]


def _recommendation(*, candidate_days: int, clusterless_days: int) -> str:
    if candidate_days <= 0:
        return (
            "No sorted-spike replay candidate days were detected. Check whether the local export uses "
            "Frank-lab file names with task/position/spikes/ripples .mat files before attempting evidence scoring."
        )
    if clusterless_days > 0:
        return (
            "The local export contains sorted-spike replay candidate days and at least one day with clusterless marks. "
            "Next step: build a day/session-level decoding QC that reads task, position, spikes/marks, and ripples."
        )
    return (
        "The local export contains sorted-spike replay candidate days but no parsed clusterless mark-ready days. "
        "Next step: build sorted-spike decoding QC before any HippoReplayIMM evidence comparison."
    )


def _markdown_table(headers: list[str], rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Local Dryad/CRCNS/replay_trajectory_paper export root")
    parser.add_argument("--output-dir", required=True, help="Directory for Frank/Eden QC outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = write_qc_outputs(args.dataset_root, args.output_dir)
    print("Wrote Frank/Eden dataset QC outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
