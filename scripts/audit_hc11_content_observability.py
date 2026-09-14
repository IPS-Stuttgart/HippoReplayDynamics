#!/usr/bin/env python3
"""Counts-only native hc-11 preflight for independent content validation."""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_population_content_stability import seed_for
from scripts.prepare_autopi_three_population_inputs import detect_rest_candidates

COHORT = (
    "Achilles_10252013", "Achilles_11012013", "Buddy_06272013",
    "Cicero_09012014", "Cicero_09102014", "Cicero_09172014",
    "Gatsby_08022013", "Gatsby_08282013",
)


def vector(value, name, n=None, integer=False):
    arr = np.atleast_1d(value).reshape(-1)
    if n is not None and len(arr) != n:
        raise ValueError(f"{name}: wrong length")
    if integer:
        if not np.isfinite(arr.astype(float)).all() or (arr.astype(float) != np.floor(arr.astype(float))).any():
            raise ValueError(f"{name}: not finite integer")
        arr = arr.astype(np.int64)
    return arr


def native_units(spikes, classes):
    ids = vector(spikes["UID"], "spike UID", integer=True)
    labels = vector(classes["UID"], "class UID", integer=True)
    if len(np.unique(ids)) != len(ids) or len(np.unique(labels)) != len(labels):
        raise ValueError("duplicate native UID")
    if (ids < 0).any() or not set(labels).issubset(ids):
        raise ValueError("invalid or extra native UID")
    pe, pi = (vector(classes[k], k, len(labels), True) for k in ("pE", "pI"))
    if not np.isin(pe, [0, 1]).all() or not np.isin(pi, [0, 1]).all() or ((pe+pi) > 1).any():
        raise ValueError("invalid/conflicting physiological labels")
    frame = pd.DataFrame({"unit_id": ids, "region": vector(spikes["region"], "region", len(ids)),
                          "shank_id": vector(spikes["shankID"], "shank", len(ids), True)})
    frame = frame.merge(pd.DataFrame({"unit_id": labels, "pE": pe, "pI": pi}),
                        on="unit_id", how="left", validate="one_to_one", sort=False)
    frame["native_included"] = (frame.region.isin(["CA1", "lCA1", "rCA1"])
                                 & frame.pE.eq(1) & frame.pI.eq(0))
    frame["exclusion_reason"] = np.where(frame.native_included, "",
        np.where(frame.pE.isna(), "missing_native_class", "not_explicit_excitatory_CA1"))
    return frame


def native_position(source):
    units = str(source["units"]).strip().lower()
    if units != "m":
        raise ValueError(f"unexpected native position units: {units}")
    t = vector(source["timestamps"], "position time").astype(float)
    x = vector(source["position"]["x"], "position x", len(t)).astype(float)
    y = vector(source["position"]["y"], "position y", len(t)).astype(float)
    if len(t) < 3 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("invalid/nonmonotone native position clock")
    epochs = np.array([source["Epochs"][k] for k in ("PREEpoch", "MazeEpoch", "POSTEpoch")], float)
    if (epochs.shape != (3, 2) or not np.isfinite(epochs).all()
            or (epochs[:, 1] <= epochs[:, 0]).any()
            or (epochs[1:, 0] < epochs[:-1, 1]-1e-8).any()):
        raise ValueError("invalid native epoch clock")
    if t[0] < epochs[1, 0]-1e-8 or t[-1] >= epochs[1, 1]+1e-8:
        raise ValueError("position clock outside native MazeEpoch")
    return np.column_stack([t, 100*x, 100*y]), epochs


def running_intervals(position, run):
    """Disjoint supported moving frame intervals, strictly before RUN midpoint."""
    position = np.asarray(position, float)
    t, xy = position[:, 0], position[:, 1:3]
    midpoint = float(np.mean(run))
    dt = np.diff(t)
    finite = np.isfinite(xy).all(axis=1) & (t >= run[0]) & (t < midpoint)
    edge = finite[:-1] & finite[1:] & (dt > 0) & (dt <= .1)
    speed = np.full(len(t), np.nan)
    good = edge[:-1] & edge[1:]
    indices = np.flatnonzero(good) + 1
    speed[indices] = np.linalg.norm(xy[indices+1]-xy[indices-1], axis=1)/(t[indices+1]-t[indices-1])
    keep = (edge & np.isfinite(speed[:-1]) & np.isfinite(speed[1:])
            & (speed[:-1] >= 10) & (speed[1:] >= 10)
            & (t[:-1] >= run[0]) & (t[1:] <= midpoint))
    return np.column_stack([t[:-1][keep], t[1:][keep]])


def count_windows(times, windows):
    times = np.asarray(times, float)
    windows = np.asarray(windows, float).reshape(-1, 2)
    if (not np.isfinite(times).all() or (np.diff(times) < 0).any()
            or not np.isfinite(windows).all() or (windows[:, 1] <= windows[:, 0]).any()):
        raise ValueError("invalid spike clock/window")
    return np.searchsorted(times, windows[:, 1], side="left") - np.searchsorted(times, windows[:, 0], side="left")


def acquisition_clock_limit(root, animal, session, expected_end, spike_end):
    """An analysis epoch need not span acquisition; verify any tail with EEG/XML."""
    if spike_end <= expected_end + 1e-8:
        return expected_end, dict(status="all_spikes_within_declared_epochs")
    folder = root.parent / "raw_eeg" / animal / session
    xml_path, eeg_path = folder / f"{session}.xml", folder / f"{session}.eeg"
    tree = ElementTree.parse(xml_path)
    n = int(tree.findtext(".//acquisitionSystem/nChannels"))
    bits = int(tree.findtext(".//acquisitionSystem/nBits"))
    fs = float(tree.findtext(".//fieldPotentials/lfpSamplingRate"))
    spike_fs = float(tree.findtext(".//acquisitionSystem/samplingRate"))
    stat = eeg_path.stat()
    if n <= 0 or bits <= 0 or bits % 8 or fs <= 0 or spike_fs <= 0 or stat.st_size % (n * bits // 8):
        raise ValueError("invalid acquisition XML/EEG length")
    end = stat.st_size / (n * bits // 8 * fs)
    if spike_end > end + 1/spike_fs or expected_end > end + 1/spike_fs:
        raise ValueError("native spikes/epoch outside independently verified acquisition")
    return end + 1/spike_fs, dict(status="tail_verified_against_native_EEG_XML_not_analyzed",
        analysis_post_end_s=float(expected_end), native_spike_end_s=float(spike_end), acquisition_duration_s=end,
        xml_path=str(xml_path), xml_sha256=file_sha256(xml_path), eeg_path=str(eeg_path),
        eeg_bytes=stat.st_size, eeg_mtime_ns=stat.st_mtime_ns,
        eeg_verification="length_and_XML_only_not_payload_checksum")


def partition(n, k, seed):
    if k not in (2, 3) or n // k < 5:
        raise ValueError("fewer than five activity-screened cells/group")
    p = np.random.default_rng(seed).permutation(n)
    size = n // k
    return [np.sort(p[i*size:(i+1)*size]) for i in range(k)], np.sort(p[k*size:])


def windows_for(events):
    if events.empty:
        return pd.DataFrame(columns=["event_index", "window", "start_s", "end_s", "available"])
    rows = []
    for e in events.itertuples(index=False):
        full_bins = int(np.floor((e.end_s-e.start_s)/.005+1e-8))
        end = min(e.start_s+full_bins*.005, e.end_s)
        rows.append(dict(event_index=int(e.event_index), window="endpoint", start_s=end-.02,
                         end_s=end, available=full_bins >= 4))
        rows.append(dict(event_index=int(e.event_index), window="peak_diagnostic", start_s=e.peak_s-.01,
                         end_s=e.peak_s+.01,
                         available=bool(e.peak_s-.01 >= e.start_s and e.peak_s+.01 <= e.end_s)))
    return pd.DataFrame(rows)


def group_readouts(counts, windows, cell_ids, animal, session, seed):
    frames, frozen, failures = [], [], []
    for k in (2, 3):
        for split in range(3):
            try:
                groups, dropped = partition(len(cell_ids), k, seed_for(seed, "hc11_preflight", session, k, split))
            except ValueError as exc:
                failures.append(dict(groups=k, split=split, reason=str(exc)))
                continue
            frozen.append(dict(groups=k, split=split, cell_ids=[cell_ids[g].tolist() for g in groups],
                               dropped_ids=cell_ids[dropped].tolist()))
            local = windows.copy()
            spike_counts = np.column_stack([counts[:, g].sum(axis=1) for g in groups])
            active = np.column_stack([(counts[:, g] > 0).sum(axis=1) for g in groups])
            local["groups"], local["split"], local["cells_per_group"] = k, split, len(groups[0])
            local["all_groups_supported"] = ((spike_counts >= 3) & (active >= 2)).all(axis=1) & local.available
            local["all_groups_silent"] = (spike_counts == 0).all(axis=1) & local.available
            local["any_group_silent"] = (spike_counts == 0).any(axis=1) & local.available
            local["minimum_group_spikes"] = spike_counts.min(axis=1)
            local["minimum_group_active_cells"] = active.min(axis=1)
            for i in range(k):
                local[f"group{i}_spikes"], local[f"group{i}_active_cells"] = spike_counts[:, i], active[:, i]
            frames.append(local)
    if not frames:
        return pd.DataFrame(), frozen, failures
    frame = pd.concat(frames, ignore_index=True)
    frame["animal"], frame["session"] = animal, session
    return frame, frozen, failures


def extract_one(root, session, output, seed):
    animal = session.split("_")[0]
    folder = root / animal / session
    paths = {k: folder / f"{session}.{s}.mat" for k, s in dict(
        spikes="spikes.cellinfo", classes="CellClass.cellinfo", position="position.behavior").items()}
    hashes = {k: file_sha256(p) for k, p in paths.items()}
    spikes = loadmat(paths["spikes"], simplify_cells=True)["spikes"]
    units = native_units(spikes, loadmat(paths["classes"], simplify_cells=True)["CellClass"])
    source = loadmat(paths["position"], simplify_cells=True)["position"]
    position, epochs = native_position(source)
    times = [spikes["times"]] if len(units) == 1 else np.atleast_1d(spikes["times"])
    if len(times) != len(units):
        raise ValueError("times/UID mismatch")
    by_id = {}
    for uid, raw in zip(vector(spikes["UID"], "UID", integer=True), times, strict=True):
        t = np.asarray(raw, float).reshape(-1)
        if (not np.isfinite(t).all() or (np.diff(t) < 0).any()
                or (len(t) and t[0] < epochs[0, 0]-1e-8)):
            raise ValueError(f"invalid native spike clock for UID {uid}")
        by_id[int(uid)] = t
    _, clock = acquisition_clock_limit(root, animal, session, epochs[-1, 1],
                                      max((t[-1] for t in by_id.values() if len(t)), default=0))
    ids = units.loc[units.native_included, "unit_id"].to_numpy(int)
    if not len(ids):
        raise ValueError("no explicit native CA1 excitatory units")
    rest = epochs[2]
    joined = np.concatenate([np.column_stack([by_id[i], np.full(len(by_id[i]), i)]) for i in ids])
    joined = joined[np.argsort(joined[:, 0], kind="stable")]
    events, detector = detect_rest_candidates(joined, ids, rest)
    intervals = running_intervals(position, epochs[1])
    run_duration = float(np.diff(intervals, axis=1).sum())
    if run_duration <= 0:
        raise ValueError("no first-half supported movement")
    units["first_half_run_spikes"] = [int(count_windows(by_id[i], intervals).sum()) for i in units.unit_id]
    units["first_half_run_rate_hz"] = units.first_half_run_spikes / run_duration
    units["activity_screen_pass"] = (units.native_included & units.first_half_run_spikes.ge(30)
                                      & units.first_half_run_rate_hz.le(4))
    eligible = units.loc[units.activity_screen_pass, "unit_id"].to_numpy(int)
    windows = windows_for(events)
    matrix = np.column_stack([count_windows(by_id[i], windows[["start_s", "end_s"]].to_numpy())
                              for i in eligible]) if len(eligible) else np.zeros((len(windows), 0), int)
    frames, frozen, failures = group_readouts(matrix, windows, eligible, animal, session, seed)
    # A separate scalar implementation checks every saved native count, including
    # exactly shared window boundaries, instead of reusing the vector counter.
    checked = 0
    for j, uid in enumerate(eligible):
        t = by_id[uid]
        for i, w in enumerate(windows.itertuples(index=False)):
            left = bisect_left(t, w.start_s)
            right = bisect_left(t, w.end_s)
            if len(t[left:right]) != matrix[i, j]:
                raise ValueError("native window recount mismatch")
            if right > left and not ((t[left:right] >= w.start_s) & (t[left:right] < w.end_s)).all():
                raise ValueError("half-open spike recount mismatch")
            checked += 1
    if any(file_sha256(paths[k]) != h for k, h in hashes.items()):
        raise ValueError("native input changed")
    if "xml_path" in clock:
        stat = Path(clock["eeg_path"]).stat()
        if (file_sha256(clock["xml_path"]) != clock["xml_sha256"]
                or stat.st_size != clock["eeg_bytes"] or stat.st_mtime_ns != clock["eeg_mtime_ns"]):
            raise ValueError("acquisition clock evidence changed")
    target = output / session
    target.mkdir(exist_ok=False)
    units.to_csv(target / "native_unit_audit.csv", index=False)
    events.to_csv(target / "candidate_events.csv", index=False)
    windows.to_csv(target / "windows.csv", index=False)
    frames.to_csv(target / "population_counts.csv.gz", index=False)
    np.savez_compressed(target / "count_arrays.npz", counts=matrix, cell_ids=eligible,
                        position_cm=position, epochs_s=epochs, moving_intervals_s=intervals,
                        spike_times_s=joined[:, 0], spike_unit_ids=joined[:, 1].astype(int))
    meta = dict(source_paths={k: str(p) for k, p in paths.items()}, source_sha256=hashes,
                population_assignments=frozen, population_failures=failures, detector=detector,
                position_conversion="native_m_times_100_to_cm", timestamps="native_seconds_no_shift",
                independently_recounted_cells_windows=checked, status="count_audited", acquisition_clock=clock,
                count_arrays_sha256=file_sha256(target / "count_arrays.npz"),
                output_sha256={p.name: file_sha256(p) for p in target.iterdir() if p.is_file()})
    (target / "manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    return dict(animal=animal, session=session, status="complete", candidates=len(events),
        native_units=len(units), native_ca1_excitatory_units=len(ids), activity_screen_units=len(eligible),
        first_half_supported_run_s=run_duration, maze_type=source.get("behaviorinfo", {}).get("MazeType", "unknown"),
        n_position_samples=len(position), finite_position_fraction=float(np.isfinite(position[:, 1:]).all(axis=1).mean()),
        native_nrem_ripple_table_present=(folder / f"{session}.ripplesNREM.event.mat").is_file(),
        artifact_dir=str(target), independent_recount_passed=True), frames


def summarize(sessions, counts):
    summaries = []
    for session in sessions.itertuples(index=False):
        for k in (2, 3):
            for window in ("endpoint", "peak_diagnostic"):
                for split in range(3):
                    if counts.empty:
                        local = pd.DataFrame()
                    else:
                        local = counts.loc[counts.session.eq(session.session) & counts.groups.eq(k)
                                           & counts.window.eq(window) & counts.split.eq(split)]
                    n = getattr(session, "candidates", np.nan)
                    row = dict(animal=session.animal, session=session.session, groups=k, window=window,
                               split=split, source_candidates=n, measured_candidates=len(local),
                               status="measured" if len(local) else "unavailable")
                    for name in ("all_groups_supported", "all_groups_silent", "any_group_silent", "available"):
                        row[f"{name}_fraction"] = (float(local[name].sum()/n) if n > 0 and len(local) else np.nan)
                    if n > 0 and local.empty:
                        row["all_groups_supported_fraction"] = 0.0
                        row["available_fraction"] = 0.0
                    row["minimum_group_spikes_mean"] = float(local.minimum_group_spikes.mean()) if len(local) else np.nan
                    row["minimum_group_active_cells_mean"] = float(local.minimum_group_active_cells.mean()) if len(local) else np.nan
                    row["cells_per_group"] = int(local.cells_per_group.iloc[0]) if len(local) else 0
                    summaries.append(row)
    by_session = pd.DataFrame(summaries)
    fields = [c for c in by_session if c.endswith("_fraction") or c.endswith("_mean")]
    by_animal = by_session.groupby(["animal", "groups", "window", "split"], as_index=False)[fields].mean()
    primary = by_session.loc[by_session.groups.eq(2) & by_session.window.eq("endpoint") & by_session.split.eq(0)]
    rats = by_animal.loc[by_animal.groups.eq(2) & by_animal.window.eq("endpoint") & by_animal.split.eq(0)]
    known = primary.source_candidates.notna().all()
    denominator = primary.source_candidates.sum()
    coverage = primary.measured_candidates.sum()/denominator if known and denominator > 0 else np.nan
    complete_animals = primary.loc[primary.measured_candidates.gt(0), "animal"].nunique()
    records = [
        ("all_eight_sources_complete", bool(len(sessions) == 8 and sessions.status.eq("complete").all()),
         int(sessions.status.eq("complete").sum()), "8"),
        ("all_four_animals_measurable", bool(complete_animals == 4), complete_animals, "4"),
        ("endpoint_measurability", bool(np.isfinite(coverage) and coverage >= .8), coverage, ">=0.80; unknown denominators fail"),
        ("both_halves_activity_support", bool(len(rats) == 4 and rats.all_groups_supported_fraction.ge(.2).sum() >= 3),
         int(rats.all_groups_supported_fraction.ge(.2).sum()), ">=0.20 activity-supported endpoints in >=3/4 rats"),
        ("independent_count_audit", bool(len(sessions) == 8 and sessions.independent_recount_passed.fillna(False).eq(True).all()),
         int(sessions.independent_recount_passed.fillna(False).eq(True).sum()), "all 8 native spike recounts"),
    ]
    gates = pd.DataFrame(records, columns=["gate", "passed", "observed", "required"])
    gates.loc[len(gates)] = ["ready_for_endpoint_encoding_validation", bool(gates.passed.all()),
                              "feasibility_only_not_content_validation", "all preflight gates"]
    return by_session, by_animal, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    inputs = dict(script=Path(__file__), detector=ROOT / "scripts/prepare_autopi_three_population_inputs.py",
                  seed_helper=ROOT / "scripts/measure_population_content_stability.py",
                  protocol=ROOT / "docs/hc11_content_observability_protocol.md")
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="running", created_at_utc=datetime.now(UTC).isoformat(), seed=args.seed,
                      cohort=list(COHORT), analysis="counts_only_no_content_or_model_fit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    rows, frames = [], []
    for session in COHORT:
        try:
            row, frame = extract_one(args.dataset_root, session, args.output_dir, args.seed)
            frames.append(frame)
        except (OSError, KeyError, ValueError, NotImplementedError) as exc:
            row = dict(animal=session.split("_")[0], session=session, status="failed", candidates=np.nan,
                       reason=f"{type(exc).__name__}: {exc}", independent_recount_passed=False)
        rows.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / "source_sessions.csv", index=False)
    counts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summaries, animals, gates = summarize(pd.DataFrame(rows), counts)
    summaries.to_csv(args.output_dir / "observability_by_session.csv", index=False)
    animals.to_csv(args.output_dir / "observability_by_animal.csv", index=False)
    gates.to_csv(args.output_dir / "observability_gate_summary.csv", index=False)
    lines = ["# hc-11 counts-only observability", "",
             "This is not a replay-content or remedy validation. Native POST population bursts;",
             "no ripple, immobility or replay label is implied. Linear/circular tracks, not open field.", "",
             "Only native CA1 excitatory labels and first-half RUN activity screened the populations.",
             "Place-field quality and spatial recovery have NOT been evaluated.", "", "## Primary split", "",
             "animal | groups | window | all groups supported (%) | minimum group spikes (mean)",
             "--- | ---: | --- | ---: | ---:"]
    for r in animals.loc[animals.split.eq(0)].itertuples(index=False):
        lines.append(f"{r.animal} | {r.groups} | {r.window} | {100*r.all_groups_supported_fraction:.2f} | {r.minimum_group_spikes_mean:.3f}")
    lines += ["", "## Feasibility gates", ""]
    for r in gates.itertuples(index=False):
        lines.append(f"- {r.gate}: {'pass' if r.passed else 'fail'}; observed={r.observed}; required={r.required}")
    lines += ["", "Peaks are diagnostic only and never replace candidate endpoints. A counts-only pass",
              "would permit encoding validation, not satisfy the active stability-remedy goal."]
    (args.output_dir / "observability_report.md").write_text("\n".join(lines)+"\n")
    unchanged = all(file_sha256(p) == provenance["input_file_sha256"][k] for k, p in inputs.items())
    provenance.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged,
                      completed_at_utc=datetime.now(UTC).isoformat(), source_results=rows,
                      output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir()
                                     if p.is_file() and p.name != "manifest.json"})
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("code/protocol changed")


if __name__ == "__main__":
    main()
