"""Audit release identities and held-out RUN encoding before replay scoring."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipporeplayimm.tirole_two_track import file_sha256, fit_maps, load_session, run_crossvalidation


def duplicate_sessions(files):
    groups = {}
    for row in files:
        if row["kind"] in {"clusters", "position"}:
            groups.setdefault((row["kind"], row["sha256"]), []).append(row["session"])
    duplicates = [sorted(set(v)) for v in groups.values() if len(set(v)) > 1]
    return {name for group in duplicates for name in group}, duplicates


def session_gate(row, cv):
    reasons = []
    if row["duplicate_identity"]:
        reasons.append("duplicate_raw_identity_quarantined")
    if row["n_tracks"] != 2:
        reasons.append("track_identity_requires_external_resolution")
    if row["n_common_units"] < 20:
        reasons.append("fewer_than_20_common_RUN_units")
    if row["minimum_track_occupancy_fraction"] < 0.8:
        reasons.append("poor_RUN_occupancy")
    complete = len(cv) == 10 and all(r["status"] == "complete" and r["n_scored_windows"] >= 20 for r in cv)
    if not complete:
        reasons.append("incomplete_RUN_crossvalidation")
    elif any(
        not np.isfinite(r.get("context_accuracy", np.nan))
        or r["context_accuracy"] < 0.8
        or not np.isfinite(r.get("median_position_error_cm", np.nan))
        or r["median_position_error_cm"] > 35
        or r["valid_bin_fraction"] < 0.8
        for r in cv
    ):
        reasons.append("RUN_decoder_not_calibrated")
    if row["pre_rest_s"] < 300 or row["post_rest_s"] < 300:
        reasons.append("insufficient_PRE_or_POST_rest")
    return reasons


def run(root, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    pages = json.loads((root / "release_file_metadata.json").read_text())
    expected = {r["path"]: r for p in pages for r in p["_embedded"]["stash:files"]}
    files = []
    for path in sorted(root.glob("*_extracted_clusters.mat")):
        name = path.name.removesuffix("_extracted_clusters.mat")
        for kind in ("clusters", "position"):
            p = root / f"{name}_extracted_{kind}.mat"
            digest = file_sha256(p)
            row = expected[p.name]
            if digest != row["digest"] or p.stat().st_size != int(row["size"]):
                raise ValueError("release checksum mismatch: " + p.name)
            files.append({"session": name, "kind": kind, "path": str(p.resolve()), "sha256": digest, "size": p.stat().st_size})
    if not files:
        raise ValueError("no spike/position pairs found")
    duplicates, groups = duplicate_sessions(files)
    sessions, cells, crossval = [], [], []
    for name in sorted({f["session"] for f in files}):
        data = load_session(root, name)
        maps = fit_maps(data)
        epoch_times = [data.times[np.isfinite(x)] for x in data.positions]
        dt = float(np.median(np.diff(data.times)))
        pre = (data.times < min(t.min() for t in epoch_times)) & data.sleepbox & (data.speed <= 5)
        post = (data.times > max(t.max() for t in epoch_times)) & data.sleepbox & (data.speed <= 5)
        row = {
            "session": name,
            "animal": name.split("_")[0],
            "duplicate_identity": name in duplicates,
            "n_tracks": data.n_tracks,
            "n_units": len(data.unit_ids),
            "n_spikes": len(data.spike_times),
            "position_start_s": data.times[0],
            "position_end_s": data.times[-1],
            "position_dt_s": dt,
            "n_spikes_outside_position": int((data.spike_samples < 0).sum()),
            "n_common_units": len(maps["common_units"]),
            "n_union_units": len(maps["union_units"]),
            "minimum_track_occupancy_fraction": float(maps["valid_bins"].mean(axis=1).min()),
            "pre_rest_s": pre.sum() * dt,
            "post_rest_s": post.sum() * dt,
            "track_intervals_json": json.dumps([[float(t.min()), float(t.max())] for t in epoch_times]),
            "anatomical_provenance": "supplied_single_units; no_per_unit_region_or_waveform_metadata",
            "ripple_file_present": (root / f"{name}_extracted_CSC.mat").exists(),
        }
        cv = run_crossvalidation(data) if data.n_tracks == 2 and name not in duplicates else []
        reasons = session_gate(row, cv)
        row["preflight_passed"] = not reasons
        row["exclusion_reason"] = ";".join(reasons)
        sessions.append(row)
        crossval.extend(dict(session=name, animal=row["animal"], **r) for r in cv)
        for k in range(data.n_tracks):
            for u, uid in enumerate(data.unit_ids):
                cells.append(
                    {
                        "session": name,
                        "animal": row["animal"],
                        "track": k + 1,
                        "unit_id": uid,
                        "original_unit_id": data.original_ids[u],
                        "mean_track_epoch_rate_hz": maps["mean_rate_hz"][u],
                        "running_spikes": int(maps["spike_counts"][k, u].sum()),
                        "smooth_peak_rate_hz": maps["rates"][k, u].max(),
                        "raw_peak_rate_hz": maps["raw_rates"][k, u].max(),
                        "spatial_information": maps["information"][k, u],
                        "unit_passed": maps["passing_by_track"][k, u],
                        "common_unit_passed": u in maps["common_units"],
                    }
                )
        np.savez_compressed(output / f"{name}_RUN_maps.npz", **maps, unit_ids=data.unit_ids)
        print(name, "eligible" if not reasons else row["exclusion_reason"], "common_units", row["n_common_units"], flush=True)
    tables = {"session_qc": sessions, "unit_qc": cells, "RUN_crossvalidation": crossval, "input_file_manifest": files}
    for suffix, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"two_track_{suffix}.csv", index=False)
    repo = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    passed = [r for r in sessions if r["preflight_passed"]]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command_line": sys.argv,
        "dataset_root": str(root.resolve()),
        "source_doi": "10.5061/dryad.ksn02v76h",
        "source_version": 238435,
        "duplicate_identity_groups": groups,
        "scientific_replay_scores_computed": False,
        "input_files": files,
        "code_sha256": {str(p.relative_to(repo)): file_sha256(p) for p in [Path(__file__), repo / "src/hipporeplayimm/tirole_two_track.py"]},
        "eligible_sessions": [r["session"] for r in passed],
        "eligible_animals": sorted({r["animal"] for r in passed}),
        "output_sha256": {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "two_track_preflight_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    text = [
        "# Two-track preflight",
        "",
        f"Release sessions: {len(sessions)}; eligible: {len(passed)}; eligible animals: {len(manifest['eligible_animals'])}.",
        "",
        "No replay result has been calculated. RUN context decoding is not a validation of latent replay truth.",
        "",
        "Defaults fixed before RUN validation: 10 cm bins; 5 < speed < 50 cm/s; source peak/info/mean-rate unit QC on both tracks;",
        "at least 20 common units; occupancy >=0.2 s in >=80% of bins; five time-block folds with 1 s guard; 250 ms RUN decoding;",
        "each track/fold >=20 test windows, context accuracy >=0.8, median conditional-position error <=35 cm.",
        "",
        "Decoder maps AND unit inclusion are fitted inside each RUN training fold. Position error is conditional on the true track;",
        "track classification is evaluated separately. No waveform/anatomy filter is invented from unlabelled tetrode numbers.",
        "",
        "PRE and POST are remote immobile rest epochs, not sleep-stage labels. Source-aligned does not mean exact author reproduction.",
        "",
        "## Session decisions",
        "",
    ]
    text.extend(f"- {r['session']}: {r['n_common_units']} common units; {'pass' if r['preflight_passed'] else r['exclusion_reason']}" for r in sessions)
    (output / "two_track_preflight.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir)
