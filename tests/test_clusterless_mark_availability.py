from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from clusterless_mark_availability import (  # noqa: E402
    audit_clusterless_mark_availability,
    parse_sessions,
    write_outputs,
)


def test_parse_sessions_accepts_commas_and_whitespace():
    assert parse_sessions("Rat1/Open1, Rat1/Open2\nRat2/Open1") == [
        "Rat1/Open1",
        "Rat1/Open2",
        "Rat2/Open1",
    ]


def test_clusterless_mark_audit_distinguishes_marks_absent_and_missing_sessions(tmp_path):
    dataset = tmp_path / "DataSetFromPfeifferFoster"
    marked = dataset / "Rat1" / "Open1"
    unmarked = dataset / "Rat1" / "Open2"
    marked.mkdir(parents=True)
    unmarked.mkdir(parents=True)
    (marked / "spike_marks.csv").write_text(
        "time,mark_1,mark_2,waveform_peak\n0.1,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    (unmarked / "spike_times.csv").write_text("time,unit_id\n0.1,7\n", encoding="utf-8")

    rows = audit_clusterless_mark_availability(
        dataset,
        sessions=["Rat1/Open1", "Rat1/Open2", "Rat2/Open1"],
        max_files_per_session=100,
    )
    by_session = {row.session: row for row in rows}
    assert by_session["Rat1/Open1"].status == "marks_detected"
    assert by_session["Rat1/Open1"].has_clusterless_marks
    assert by_session["Rat1/Open2"].status == "no_marks_detected"
    assert not by_session["Rat1/Open2"].has_clusterless_marks
    assert by_session["Rat2/Open1"].status == "session_missing"
    assert by_session["Rat1/Open1"].mark_path_hits >= 1
    assert by_session["Rat1/Open1"].mark_key_hits >= 1

    out = tmp_path / "audit"
    write_outputs(rows, out, dataset_root=dataset, max_files_per_session=100)
    availability = _read_csv(out / "clusterless_mark_availability.csv")
    gates = _read_csv(out / "clusterless_mark_gate_summary.csv")
    manifest = json.loads((out / "clusterless_mark_availability_manifest.json").read_text())

    assert len(availability) == 3
    gate_by_name = {row["gate"]: row for row in gates}
    assert gate_by_name["any_session_has_clusterless_marks"]["passed"] == "True"
    assert gate_by_name["all_sessions_have_clusterless_marks"]["passed"] == "False"
    assert gate_by_name["overall_clusterless_goal3_ready"]["observed"] == "partial_mark_coverage"
    assert manifest["sessions_with_marks"] == ["Rat1/Open1"]
    assert manifest["sessions_without_marks"] == ["Rat1/Open2"]
    assert manifest["missing_sessions"] == ["Rat2/Open1"]


def test_clusterless_mark_audit_reports_blocked_when_sessions_have_only_sorted_spikes(tmp_path):
    dataset = tmp_path / "DataSetFromPfeifferFoster"
    session = dataset / "Rat1" / "Open1"
    session.mkdir(parents=True)
    (session / "sorted_spikes.csv").write_text("time,unit_id\n0.1,1\n", encoding="utf-8")
    rows = audit_clusterless_mark_availability(
        dataset,
        sessions=["Rat1/Open1"],
        max_files_per_session=100,
    )
    out = tmp_path / "audit"
    write_outputs(rows, out, dataset_root=dataset, max_files_per_session=100)
    gates = _read_csv(out / "clusterless_mark_gate_summary.csv")
    gate_by_name = {row["gate"]: row for row in gates}
    assert gate_by_name["any_session_has_clusterless_marks"]["passed"] == "False"
    assert gate_by_name["overall_clusterless_goal3_ready"]["observed"] == "blocked_no_marks_detected"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
