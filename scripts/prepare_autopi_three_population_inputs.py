#!/usr/bin/env python3
"""Extract native AutoPI CA1 foraging/rest inputs without decoding or pickle I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


EVENT_COLUMNS = ["event_index", "start_s", "end_s", "peak_s", "n_spikes", "n_active_units", "peak_mua_z"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def foraging_rest_pair(names, intervals):
    intervals = np.asarray(intervals, float)
    if (intervals.shape != (len(names), 2) or not np.isfinite(intervals).all()
            or (intervals[:, 1] <= intervals[:, 0]).any()
            or (intervals[1:, 0] < intervals[:-1, 1] - 1e-8).any()):
        raise ValueError("invalid native epoch clock")
    matches = [i for i, name in enumerate(names) if name.strip().lower() == "circ80"]
    if not matches:
        raise ValueError("no native circ80 epoch")
    i = matches[0]
    if i + 1 == len(names) or names[i+1].strip().lower() != "rest":
        raise ValueError("first circ80 not immediately followed by native rest")
    return i, intervals[i].copy(), intervals[i+1].copy()


def select_ca1_units(info, groups, shanks, regions, session):
    for frame, key in ((info, "cluster_id"), (groups, "cluster_id"), (shanks, "id")):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"missing/duplicate native unit identity: {key}")
    merged = info.merge(groups[["cluster_id", "group"]], on="cluster_id", how="outer",
                        suffixes=("_info", "_curated"), validate="one_to_one", indicator=True)
    good = merged.group_curated.eq("good") | merged.group_info.eq("good")
    if (merged.loc[good, "_merge"].ne("both").any()
            or merged.loc[good, "group_info"].ne(merged.loc[good, "group_curated"]).any()):
        raise ValueError("inconsistent author-good cluster metadata")
    merged = merged.loc[good].drop(columns="_merge").copy()
    if (merged.cluster_id < 0).any() or (merged.cluster_id != np.floor(merged.cluster_id)).any():
        raise ValueError("invalid native cluster IDs")
    merged["id"] = [f"{session}_{int(i)}" for i in merged.cluster_id]
    merged = merged.merge(shanks, on="id", how="left", validate="one_to_one")
    if merged.shank.isna().any() or (merged.shank != np.floor(merged.shank)).any():
        raise ValueError("missing/noninteger native shank identity")
    shank = merged.shank.to_numpy(int)
    if (shank < 1).any() or (shank > len(regions)).any():
        raise ValueError("native shank outside desel regions")
    if "sh" in merged and not np.array_equal(merged.sh.to_numpy(float) + 1, shank):
        raise ValueError("cluster_info and shank_neuron electrode assignments differ")
    merged["brain_region"] = [regions[i-1].strip().lower() for i in shank]
    merged["included_ca1"] = merged.brain_region.eq("ca1")
    merged["cell_type"] = "unknown_author_good_sorted_unit"
    return merged.sort_values("cluster_id").reset_index(drop=True)


def detect_rest_candidates(spikes, cell_ids, interval):
    start, end = np.asarray(interval, float)
    if not np.isfinite([start, end]).all() or end <= start or len(cell_ids) == 0:
        raise ValueError("invalid rest interval or empty unit universe")
    spikes = np.asarray(spikes, float).reshape(-1, 2)
    if not np.isfinite(spikes).all() or not np.isin(spikes[:, 1], cell_ids).all():
        raise ValueError("invalid spike clock or unmapped cluster")
    local = spikes[(spikes[:, 0] >= start) & (spikes[:, 0] < end)]
    # Only full 1-ms bins define the detection clock; no final partial-bin inflation.
    n_bins = int(np.floor((end-start) / .001 + 1e-8))
    if n_bins < 2:
        raise ValueError("rest interval too short")
    valid_end = start + .001 * n_bins
    local = local[local[:, 0] < valid_end]
    bins = np.floor((local[:, 0] - start) / .001).astype(int)
    density = gaussian_filter1d(np.bincount(bins, minlength=n_bins).astype(float), 10, mode="reflect") / .001
    mean, sd = float(density.mean()), float(density.std())
    summary = dict(rest_start_s=float(start), rest_end_s=float(end), n_rest_spikes=len(local),
                   detector_mean_hz=mean, detector_sd_hz=sd, detected_mean_crossing_bursts=0,
                   peak_threshold_bursts=0, boundary_clipped_bursts=0)
    if sd == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS), summary
    above = density > mean
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    ends = np.flatnonzero(above & ~np.r_[above[1:], False]) + 1
    summary["detected_mean_crossing_bursts"] = len(starts)
    rows = []
    for left, right in zip(starts, ends, strict=True):
        peak = left + int(np.argmax(density[left:right]))
        if density[peak] <= mean + 3*sd:
            continue
        summary["peak_threshold_bursts"] += 1
        if left == 0 or right == n_bins:
            summary["boundary_clipped_bursts"] += 1
            continue
        a, b = start + left*.001, start + right*.001
        if b-a < .05 - 1e-9 or b-a > 2 + 1e-9:
            continue
        event = local[(local[:, 0] >= a) & (local[:, 0] < b)]
        active = len(np.unique(event[:, 1]))
        if len(event) < 5 or active < max(3, int(np.ceil(.1*len(cell_ids)))):
            continue
        rows.append(dict(event_index=len(rows), start_s=a, end_s=b, peak_s=start+(peak+.5)*.001,
                         n_spikes=len(event), n_active_units=active, peak_mua_z=(density[peak]-mean)/sd))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS), summary


def extract_session(folder, output):
    folder = Path(folder)
    name = folder.name
    paths = {key: folder / filename for key, filename in dict(
        intervals="sessionIntervals.npy", spike_times="spike_times.npy", spike_clusters="spike_clusters.npy",
        info="cluster_info.tsv", groups="cluster_group.tsv", shanks=f"{name}.shank_neuron.csv",
        epochs=f"{name}.desen", regions=f"{name}.desel", sampling=f"{name}.sampling_rate_dat",
        position=f"{name}.pose.npy").items()}
    hashes = {key: sha256(path) for key, path in paths.items()}
    names = paths["epochs"].read_text().splitlines()
    intervals = np.load(paths["intervals"], allow_pickle=False)
    epoch, run, rest = foraging_rest_pair(names, intervals)
    units = select_ca1_units(pd.read_csv(paths["info"], sep="\t"), pd.read_csv(paths["groups"], sep="\t"),
                            pd.read_csv(paths["shanks"]), paths["regions"].read_text().splitlines(), name)
    ids = units.loc[units.included_ca1, "cluster_id"].to_numpy(int)
    if not len(ids):
        raise ValueError("no author-good CA1 units")
    fs = float(paths["sampling"].read_text().strip())
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("invalid native spike sample rate")
    samples = np.load(paths["spike_times"], mmap_mode="r", allow_pickle=False).reshape(-1)
    labels = np.load(paths["spike_clusters"], mmap_mode="r", allow_pickle=False).reshape(-1)
    if samples.shape != labels.shape or not np.issubdtype(samples.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("spike timestamp/label format mismatch")
    if (samples < 0).any() or (np.diff(samples.astype(np.int64)) < 0).any():
        raise ValueError("nonmonotone or negative native spike clock")
    times = samples.astype(float) / fs
    if len(times) and (times.min() < intervals[0, 0] - 1/fs or times.max() > intervals[-1, 1] + 1/fs):
        raise ValueError("native spike clock outside session intervals")
    keep = np.isin(labels, ids) & (((times >= run[0]) & (times < run[1])) | ((times >= rest[0]) & (times < rest[1])))
    spikes = np.column_stack([times[keep], labels[keep]])
    pose = np.load(paths["position"], allow_pickle=False)
    if pose.ndim != 2 or pose.shape[1] < 3 or not np.isfinite(pose[:, 0]).all() or (np.diff(pose[:, 0]) <= 0).any():
        raise ValueError("invalid native position clock")
    position = pose[(pose[:, 0] >= run[0]) & (pose[:, 0] < run[1]), :3]
    if len(position) < 2:
        raise ValueError("no position samples in first foraging epoch")
    events, detector = detect_rest_candidates(spikes, ids, rest)
    if any(sha256(paths[key]) != value for key, value in hashes.items()):
        raise RuntimeError("native data changed during extraction")
    target = Path(output) / folder.parent.name / name
    target.mkdir(parents=True, exist_ok=False)
    arrays = target / "native_inputs.npz"
    np.savez_compressed(arrays, position=position, spikes=spikes, cell_ids=ids,
                        run_interval=run, rest_interval=rest,
                        shank_ids=units.loc[units.included_ca1, "shank"].to_numpy(int))
    events.to_csv(target/"candidates.csv", index=False)
    units.to_csv(target/"native_units.csv", index=False)
    detail = dict(dataset="autopi_ca1", animal=folder.parent.name, session=name, epoch_index=int(epoch),
                  source_paths={key: str(path) for key, path in paths.items()}, source_sha256=hashes,
                  position_unit="cm_native_pose_no_rescaling", time_unit="native_seconds_no_shift",
                  sample_rate_hz=fs, detector=detector, n_author_good_ca1=len(ids),
                  event_definition="native_rest_high_mua_not_verified_sleep_or_ripple_or_immobility",
                  position_scope="first_circ80_only_not_task_or_rest", n_position_samples=len(position),
                  finite_position_fraction=float(np.isfinite(position[:, 1:3]).all(axis=1).mean()))
    (target/"native_manifest.json").write_text(json.dumps(detail, indent=2)+"\n")
    return dict(dataset="autopi_ca1", animal=folder.parent.name, session=name, status="extracted",
                candidates=len(events), n_author_good_ca1=len(ids), artifact_path=str(arrays),
                artifact_sha256=sha256(arrays), candidate_path=str(target/"candidates.csv"),
                candidate_sha256=sha256(target/"candidates.csv"), native_manifest_sha256=sha256(target/"native_manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exported-source-commit", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    folders = sorted(path for path in args.dataset_root.glob("*/*") if path.is_dir() and "-" in path.name)
    if not folders:
        raise ValueError("no native recording directories")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_at_utc=datetime.now(UTC).isoformat(), status="running",
                    exported_source_commit=args.exported_source_commit, extractor_sha256=sha256(__file__),
                    protocol_sha256=sha256(args.protocol), dataset_root=str(args.dataset_root.resolve()),
                    source_recordings=[str(path) for path in folders],
                    observation_model="none_native_extraction_only")
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    for folder in folders:
        try:
            result = extract_session(folder, args.output_dir)
        except (ValueError, OSError, KeyError) as exc:
            result = dict(dataset="autopi_ca1", animal=folder.parent.name, session=folder.name,
                          status="failed", candidates=np.nan, reason=str(exc))
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir/"native_sessions.csv", index=False)
        print(json.dumps(result), flush=True)
    unchanged = manifest["extractor_sha256"] == sha256(__file__) and manifest["protocol_sha256"] == sha256(args.protocol)
    manifest.update(status="complete" if unchanged else "failed", all_recordings_attempted=True,
                    failed_recordings=sum(row["status"] == "failed" for row in rows), inputs_unchanged=unchanged,
                    completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("extraction code/protocol changed")


if __name__ == "__main__":
    main()
