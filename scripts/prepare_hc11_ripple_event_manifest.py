#!/usr/bin/env python3
"""Build an auditable hc-11 native/generated ripple-event manifest.

Published ripple tables are preferred. Sessions without one may use an LFP-
detected table only when the detector has passed overlap validation against at
least one published table and that session has valid PRE and POST NREM events.
The scorer receives only rows that pass; the QC table retains every session.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
import score_hc11_pre_post_learning_evidence as learning  # noqa: E402
import score_hc11_webshare_native_ripple_evidence as hc11  # noqa: E402


MANIFEST_OUTPUT = "hc11_ripple_event_manifest.csv"
QC_OUTPUT = "hc11_ripple_event_manifest_qc.csv"
GATE_OUTPUT = "hc11_ripple_event_manifest_gate_summary.csv"
PROVENANCE_OUTPUT = "hc11_ripple_event_manifest_provenance.json"
SUMMARY_OUTPUT = "hc11_ripple_event_manifest_summary.md"


def load_ripple_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if "ripplesNREM" in handle:
            group = handle["ripplesNREM"]
        elif "ripples" in handle:
            group = handle["ripples"]
        else:
            raise ValueError(f"{path}: missing ripplesNREM or ripples group")
        times = np.asarray(group["times"], dtype=float)
        peaks = (
            np.asarray(group["peaks"], dtype=float).ravel()
            if "peaks" in group
            else np.array([], dtype=float)
        )
    if times.ndim != 2:
        raise ValueError(f"{path}: ripplesNREM/times must be two-dimensional")
    if times.shape[0] == 2:
        times = times.T
    elif times.shape[1] != 2:
        raise ValueError(f"{path}: ripplesNREM/times must have two columns")
    if peaks.size == 0:
        peaks = np.mean(times, axis=1)
    if len(peaks) != len(times):
        raise ValueError(f"{path}: ripple peaks and times have different lengths")
    return times, peaks


def event_phase_counts(session_dir: Path, event_path: Path) -> dict[str, object]:
    times, peaks = load_ripple_intervals(event_path)
    finite = np.isfinite(times).all(axis=1) & np.isfinite(peaks)
    valid_duration = times[:, 1] > times[:, 0]
    phases = learning.phase_intervals(session_dir)
    nrem = learning.nrem_intervals(session_dir)
    in_nrem = hc11.times_in_intervals(peaks, nrem)
    return {
        "events": int(len(times)),
        "finite_events": int(np.sum(finite)),
        "valid_duration_events": int(np.sum(valid_duration)),
        "nrem_events": int(np.sum(in_nrem)),
        "pre_events": int(np.sum(in_nrem & hc11.times_in_intervals(peaks, phases["PRE"]))),
        "post_events": int(np.sum(in_nrem & hc11.times_in_intervals(peaks, phases["POST"]))),
    }


def detector_validation_status(
    validation_qc_paths: list[Path],
    *,
    min_precision: float,
    min_recall: float,
) -> tuple[bool, dict[str, object]]:
    frames = [pd.read_csv(path) for path in validation_qc_paths]
    validation = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    required = {
        "overlap_precision",
        "overlap_recall",
        "native_validation_available",
        "ripple_channel_source",
    }
    usable = (
        validation[validation["native_validation_available"].astype(bool)].copy()
        if not validation.empty and required.issubset(validation.columns)
        else pd.DataFrame()
    )
    passing_rows = usable[
        usable["overlap_precision"].ge(float(min_precision))
        & usable["overlap_recall"].ge(float(min_recall))
    ]
    passed = bool(not passing_rows.empty)
    return passed, {
        "validation_sessions": int(len(usable)),
        "minimum_overlap_precision": (
            float(usable["overlap_precision"].min()) if not usable.empty else np.nan
        ),
        "minimum_overlap_recall": (
            float(usable["overlap_recall"].min()) if not usable.empty else np.nan
        ),
        "required_overlap_precision": float(min_precision),
        "required_overlap_recall": float(min_recall),
        "validated_channel_sources": sorted(
            passing_rows["ripple_channel_source"].dropna().astype(str).unique()
        ),
    }


def generated_paths(detector_root: Path, session: str) -> tuple[Path, Path]:
    session_root = detector_root / session
    manifest_path = session_root / "hc11_lfp_ripple_detection_manifest.json"
    event_path = session_root / f"{session}.ripplesNREM.generated.event.mat"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        recorded = metadata.get("output_mat")
        if recorded:
            event_path = Path(recorded).resolve()
    return event_path, session_root / "hc11_lfp_ripple_detection_qc.csv"


def build_event_manifest(
    dataset_root: Path,
    detector_root: Path,
    validation_qc_paths: list[Path],
    *,
    min_validation_precision: float,
    min_validation_recall: float,
    min_events_per_phase: int,
    expected_sessions: int,
    expected_animals: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    method_passed, validation = detector_validation_status(
        validation_qc_paths,
        min_precision=min_validation_precision,
        min_recall=min_validation_recall,
    )
    rows: list[dict[str, object]] = []
    sessions = sorted(path for path in Path(dataset_root).glob("*/*") if path.is_dir())
    for session_dir in sessions:
        animal = session_dir.parent.name
        session = session_dir.name
        native_all_path = session_dir / f"{session}.ripplesALL.event.mat"
        native_nrem_path = session_dir / f"{session}.ripplesNREM.event.mat"
        native_path = native_all_path if native_all_path.exists() else native_nrem_path
        if native_path.exists():
            event_path = native_path
            event_source = (
                "published_all_intersect_current_nrem"
                if native_path == native_all_path
                else "published_native_nrem"
            )
            generated_qc_available = True
            detector_qc_path: Path | None = None
            method_ok = True
            session_channel_source = "published_native"
        else:
            event_path, detector_qc_path = generated_paths(Path(detector_root), session)
            event_source = "lfp_detected_method_validated"
            generated_qc_available = detector_qc_path.exists()
            method_ok = method_passed
            session_channel_source = ""
        phase_metrics: dict[str, object] = {
            "events": 0,
            "finite_events": 0,
            "valid_duration_events": 0,
            "nrem_events": 0,
            "pre_events": 0,
            "post_events": 0,
        }
        event_error = ""
        if event_path.exists():
            try:
                phase_metrics = event_phase_counts(session_dir, event_path)
            except Exception as exc:
                event_error = f"{type(exc).__name__}: {exc}"
        session_detector_qc_passed = bool(native_path.exists())
        if detector_qc_path is not None and detector_qc_path.exists():
            detector_qc = pd.read_csv(detector_qc_path)
            session_channel_source = (
                str(detector_qc.iloc[0]["ripple_channel_source"])
                if len(detector_qc) == 1 and "ripple_channel_source" in detector_qc
                else ""
            )
            session_detector_qc_passed = bool(
                len(detector_qc) == 1
                and "detected_events" in detector_qc
                and int(detector_qc.iloc[0]["detected_events"]) == int(phase_metrics["events"])
            )
            method_ok = bool(
                method_passed
                and session_channel_source in validation["validated_channel_sources"]
            )
        all_events_valid = bool(
            phase_metrics["events"] > 0
            and phase_metrics["finite_events"] == phase_metrics["events"]
            and phase_metrics["valid_duration_events"] == phase_metrics["events"]
        )
        nrem_restriction_passed = bool(
            native_path.exists()
            or phase_metrics["nrem_events"] == phase_metrics["events"]
        )
        phase_coverage = bool(
            phase_metrics["pre_events"] >= int(min_events_per_phase)
            and phase_metrics["post_events"] >= int(min_events_per_phase)
        )
        passed = bool(
            event_path.exists()
            and session_detector_qc_passed
            and method_ok
            and all_events_valid
            and nrem_restriction_passed
            and phase_coverage
        )
        rows.append(
            {
                "animal": animal,
                "session": session,
                "ripple_event_path": str(event_path.resolve()),
                "event_source": event_source,
                "ripple_channel_source": session_channel_source,
                "native_event_table": bool(native_path.exists()),
                "detector_method_validation_passed": bool(method_ok),
                "session_detector_qc_available": bool(generated_qc_available),
                "session_detector_qc_passed": bool(session_detector_qc_passed),
                **phase_metrics,
                "all_events_valid": all_events_valid,
                "nrem_restriction_passed": nrem_restriction_passed,
                "phase_coverage_passed": phase_coverage,
                "detector_qc_passed": passed,
                "failure_reason": event_error,
            }
        )
    qc = pd.DataFrame(rows)
    manifest = qc[qc["detector_qc_passed"]].loc[
        :, ["animal", "session", "ripple_event_path", "event_source", "detector_qc_passed"]
    ].copy()
    passed_qc = qc[qc["detector_qc_passed"]] if not qc.empty else qc
    animals = int(passed_qc["animal"].nunique()) if not passed_qc.empty else 0
    checks = [
        ("expected_sessions_discovered", len(qc) == int(expected_sessions), f"sessions={len(qc)}/{expected_sessions}"),
        ("all_sessions_pass_event_qc", bool(len(qc) > 0 and qc["detector_qc_passed"].all()), f"passed={int(qc['detector_qc_passed'].sum())}/{len(qc)}"),
        ("expected_animals_represented", animals == int(expected_animals), f"animals={animals}/{expected_animals}"),
        ("generated_method_validated", bool(method_passed), json.dumps(validation, sort_keys=True)),
        ("pre_post_phase_coverage_complete", bool(len(qc) > 0 and qc["phase_coverage_passed"].all()), f"passed={int(qc['phase_coverage_passed'].sum())}/{len(qc)}"),
    ]
    checks.append(("overall", all(value for _, value, _ in checks), "all event-manifest gates required"))
    gates = pd.DataFrame(checks, columns=["gate", "passed", "detail"])
    return manifest, qc, gates, validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--detector-output-root", required=True)
    parser.add_argument("--validation-qc", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-validation-overlap-precision", type=float, default=0.70)
    parser.add_argument("--min-validation-overlap-recall", type=float, default=0.80)
    parser.add_argument("--min-events-per-phase", type=int, default=20)
    parser.add_argument("--expected-sessions", type=int, default=8)
    parser.add_argument("--expected-animals", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_paths = [Path(value).resolve() for value in args.validation_qc]
    manifest, qc, gates, validation = build_event_manifest(
        Path(args.dataset_root).resolve(),
        Path(args.detector_output_root).resolve(),
        validation_paths,
        min_validation_precision=args.min_validation_overlap_precision,
        min_validation_recall=args.min_validation_overlap_recall,
        min_events_per_phase=args.min_events_per_phase,
        expected_sessions=args.expected_sessions,
        expected_animals=args.expected_animals,
    )
    manifest.to_csv(output_dir / MANIFEST_OUTPUT, index=False)
    qc.to_csv(output_dir / QC_OUTPUT, index=False)
    gates.to_csv(output_dir / GATE_OUTPUT, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
        "outputs": [MANIFEST_OUTPUT, QC_OUTPUT, GATE_OUTPUT],
        **build_script_provenance(
            input_paths={
                "dataset_root": Path(args.dataset_root).resolve(),
                **{f"validation_qc_{index}": path for index, path in enumerate(validation_paths)},
            },
            cwd=ROOT,
            argv=sys.argv,
        ),
    }
    (output_dir / PROVENANCE_OUTPUT).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    overall = bool(gates.loc[gates["gate"].eq("overall"), "passed"].iloc[0])
    (output_dir / SUMMARY_OUTPUT).write_text(
        "\n".join(
            [
                "# hc-11 ripple-event manifest",
                "",
                f"Overall event-manifest status: **{'pass' if overall else 'fail'}**.",
                "",
                "```text",
                qc.to_string(index=False),
                "```",
                "",
                "Generated LFP events enter scoring only after native-detector overlap validation and per-session PRE/POST coverage checks.",
                "",
            ]
        )
    )
    print(f"Prepared {len(manifest)}/{len(qc)} validated session rows in {output_dir}")


if __name__ == "__main__":
    main()
