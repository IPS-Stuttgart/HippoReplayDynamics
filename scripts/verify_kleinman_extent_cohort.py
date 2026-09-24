#!/usr/bin/env python3
"""Verify every cohort eligibility decision and sample each session's spike fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from audit_kleinman_extent_cohort import cohort_tables, session_eligible
from calibrate_kleinman_integrated_extent import condition_summary, screens
from verify_kleinman_integrated_extent import reference_counts, reference_fit, reference_library


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert file_sha256(output / name) == digest, name
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(Path(path)) == manifest["input_file_sha256"][key], key
    sessions = pd.read_csv(output / "kleinman_extent_cohort_sessions.csv")
    assert len(sessions) == 135 and sessions.attempted.sum() == 127
    old_qc = pd.read_csv(manifest["input_file_paths"]["run_qc"])
    expected = set(zip(old_qc.animal, old_qc.session, strict=True))
    assert set(zip(sessions.animal, sessions.session, strict=True)) == expected
    checked, n_events, ids, failures = 0, 0, set(), []
    for row in sessions.itertuples():
        if row.status != "scored":
            assert not row.extent_eligible
            if row.run_pass:
                failures.append(row.animal + "/" + row.session)
            continue
        directory = output / "sessions" / row.animal / row.session
        frame = pd.read_csv(directory / "estimates.csv")
        assert len(frame) == 2880 and not frame.event_index.duplicated().any()
        assert set(frame.animal) == {row.animal} and set(frame.session) == {row.session}
        assert not ids.intersection(frame.event_index)
        ids.update(frame.event_index)
        n_events += len(frame)
        sizes = frame.groupby(["side", "extent", "profile", "duration_s", "expected_spikes"]).size()
        assert len(sizes) == 180 and sizes.eq(16).all()
        recomputed = screens(frame)
        recomputed.insert(1, "session", row.session)
        assert session_eligible(recomputed) == row.extent_eligible
        pd.testing.assert_frame_equal(pd.read_csv(directory / "screens.csv"), recomputed, check_dtype=False)
        pd.testing.assert_frame_equal(pd.read_csv(directory / "condition_summary.csv"), condition_summary(frame), check_dtype=False, rtol=1e-9, atol=1e-9)
        model = dict(np.load(directory / "known_map.npz"))
        chosen = np.random.default_rng(1931).choice(frame.event_index, 16, replace=False)
        for duration, sub in frame.groupby("duration_s"):
            q, meta = reference_library(model, duration)
            for event in sub.loc[sub.event_index.isin(chosen)].itertuples():
                reference = reference_fit(reference_counts(model, event), q, meta)
                for name, value in reference.items():
                    np.testing.assert_allclose(getattr(event, name), value, atol=1e-8, rtol=1e-9, err_msg=f"{row.animal}/{row.session}/{event.event_index}/{name}")
                checked += 1
        print(json.dumps({"verified_session": row.animal + "/" + row.session, "independent_events": checked}), flush=True)
    for label, frame in zip(["by_animal", "by_condition", "gates"], cohort_tables(sessions), strict=True):
        pd.testing.assert_frame_equal(pd.read_csv(output / ("kleinman_extent_cohort_" + label + ".csv")), frame, check_dtype=False)
    result = {
        "status": "passed",
        "producer_commit": manifest["code_commit"],
        "source_sessions": 135,
        "scored_sessions": int(sessions.status.eq("scored").sum()),
        "technical_failures": failures,
        "synthetic_events": n_events,
        "independent_events_recomputed": checked,
        "result_manifest_sha256": file_sha256(output / "manifest.json"),
        "verifier_sha256": file_sha256(Path(__file__)),
        "scope": "All session/bank accounting, input/output hashes and eligibility decisions; independent scalar likelihood and cross-fitting on 16 events per scored session. Not independent refitting of raw RUN maps or of all event estimates.",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    verify(parser.parse_args().output_dir)
