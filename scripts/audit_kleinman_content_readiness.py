#!/usr/bin/env python3
"""Inventory native Kleinman/Foster events and raw disjoint-population support."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts._provenance import build_script_provenance, file_sha256


def numeric_matrix(value, columns, name):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, columns), dtype=float)
    if array.ndim == 1 and array.size == columns:
        array = array.reshape(1, columns)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError("invalid " + name)
    return array


def snapshot_inputs(inputs):
    return {name: file_sha256(path) for name, path in inputs.items()}


def verify_inputs(inputs, expected):
    observed = snapshot_inputs(inputs)
    changed = sorted(name for name in expected if expected[name] != observed[name])
    if changed:
        raise ValueError("input changed during audit: " + ", ".join(changed))
    return observed


def parse_session(info):
    position = np.asarray(info["position"], dtype=float).reshape(-1)
    velocity = numeric_matrix(info["velocity"], 2, "velocity")
    if len(velocity) < 2 or not np.all(np.diff(velocity[:, 0]) > 0):
        raise ValueError("nonmonotonic or insufficient velocity timestamps")
    if not np.isfinite(position).all():
        raise ValueError("nonfinite position")
    if len(position) == len(velocity):
        alignment = "equal_length_not_independently_verified"
    elif len(position) == len(velocity) + 1:
        alignment = "one_extra_position_sample_unresolved"
    else:
        alignment = "incompatible_position_velocity_lengths"
    flags = {}
    for name in ("drug", "novel"):
        if name == "novel" and name not in info:
            flags[name] = None
            continue
        value = np.asarray(info[name]).reshape(-1)
        if len(value) != 1 or value[0] not in (0, 1):
            raise ValueError("invalid " + name)
        flags[name] = int(value[0])
    return position, velocity, alignment, flags


def unit_spike_trains(spikes):
    spikes = numeric_matrix(spikes, 3, "spikes")
    if len(spikes) == 0:
        return np.empty((0, 2), int), []
    if not np.equal(spikes[:, 1:], np.floor(spikes[:, 1:])).all():
        raise ValueError("noninteger unit identity")
    # Cluster IDs may be reused on another tetrode. Never merge those neurons.
    keys, inverse = np.unique(spikes[:, [2, 1]].astype(int), axis=0, return_inverse=True)
    trains = [np.sort(spikes[inverse == i, 0]) for i in range(len(keys))]
    return keys, trains


def counts_in_windows(trains, starts, ends):
    starts, ends = np.asarray(starts), np.asarray(ends)
    if starts.shape != ends.shape or np.any(ends <= starts):
        raise ValueError("invalid count windows")
    counts = np.empty((len(starts), len(trains)), dtype=np.int64)
    for i, times in enumerate(trains):
        counts[:, i] = np.searchsorted(times, ends, side="left") - np.searchsorted(times, starts, side="left")
    return counts


def split_units(n, identity, split, seed):
    value = f"{seed}|{identity}|{split}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(value).digest()[:8], "little"))
    order = rng.permutation(n)
    half = n // 2
    return np.sort(order[:half]), np.sort(order[half : 2 * half])


def event_windows(events, phase):
    events = numeric_matrix(events, 4, "events")
    valid = (events[:, 1] - events[:, 0] >= 0.02) & (events[:, 2] >= events[:, 0]) & (events[:, 2] <= events[:, 1])
    if phase == "native_endpoint":
        starts, ends = events[:, 1] - 0.02, events[:, 1]
    elif phase == "native_peak":
        starts, ends = events[:, 2] - 0.01, events[:, 2] + 0.01
        valid &= (starts >= events[:, 0]) & (ends <= events[:, 1])
    else:
        raise ValueError("unknown phase")
    return starts, ends, valid


def immobile_windows(velocity, starts, ends):
    t, speed = velocity[:, 0], np.abs(velocity[:, 1])
    result = []
    for start, end in zip(starts, ends, strict=True):
        left = max(0, np.searchsorted(t, start, side="right") - 1)
        right = min(len(t) - 1, np.searchsorted(t, end, side="left"))
        covered = t[left] <= start and t[right] >= end and right > left
        supported = covered and np.max(np.diff(t[left : right + 1])) <= 0.1
        result.append(bool(supported and np.max(speed[left : right + 1]) < 5))
    return np.array(result, dtype=bool)


def position_alignment_diagnostic(position, velocity, events):
    rows = []
    if not len(events):
        return rows
    mask = (events[:, 0] >= velocity[0, 0]) & (events[:, 0] <= velocity[-1, 0])
    candidates = (
        [("equal_length", position)]
        if len(position) == len(velocity)
        else ([("drop_last", position[:-1]), ("drop_first", position[1:])] if len(position) == len(velocity) + 1 else [])
    )
    for convention, xy in candidates:
        error = np.abs(events[mask, 3] - np.interp(events[mask, 0], velocity[:, 0], xy))
        rows.append(
            {
                "convention": convention,
                "events_with_clock_overlap": int(mask.sum()),
                "median_native_position_difference_cm": float(np.median(error)) if len(error) else np.nan,
                "p95_native_position_difference_cm": float(np.quantile(error, 0.95)) if len(error) else np.nan,
                "fraction_exact_to_1e_minus6": float(np.mean(error < 1e-6)) if len(error) else np.nan,
            }
        )
    return rows


def inspect_session(path, root, seed):
    relative = path.parent.relative_to(root)
    if len(relative.parts) != 3:
        raise ValueError("unexpected experiment/subject/session path")
    experiment, subject, session = relative.parts
    identity = {"experiment": experiment, "subject_label": subject, "recording_group": experiment + "/" + subject, "session": session}
    info = loadmat(path, simplify_cells=True)["session_info"]
    position, velocity, alignment, flags = parse_session(info)
    identity.update(flags)
    paths = [path]
    spike_file = path.parent / "spike_data.mat"
    spikes = loadmat(spike_file, simplify_cells=True)["spike_data"] if spike_file.is_file() else np.empty((0, 3))
    if spike_file.is_file():
        paths.append(spike_file)
    spikes = numeric_matrix(spikes, 3, "spikes")
    keys, trains = unit_spike_trains(spikes)
    span = velocity[-1, 0] - velocity[0, 0]
    row = dict(
        **identity,
        session_path=str(path.parent),
        status="readable",
        has_spikes=spike_file.is_file(),
        n_units_raw=len(keys),
        n_spikes_raw=len(spikes),
        n_nonpositive_identity_units=int(np.any(keys <= 0, axis=1).sum()),
        velocity_duration_s=float(span),
        median_position_dt_s=float(np.median(np.diff(velocity[:, 0]))),
        n_position_samples=len(position),
        n_velocity_samples=len(velocity),
        position_timing_status=alignment,
        position_span_cm=float(np.ptp(position)),
        decoder_ready=False,
        prospective_control_stratum=experiment == "Experiment_1" and subject.startswith("Con_") and flags["drug"] == 0,
        cell_type_and_place_field_qc_available=False,
    )
    events_out, alignment_rows, units_out = [], [], []
    for i, ((tetrode, cluster), train) in enumerate(zip(keys, trains, strict=True)):
        n_in_span = int(np.count_nonzero((train >= velocity[0, 0]) & (train < velocity[-1, 0])))
        units_out.append(
            dict(
                **identity,
                unit_index=i,
                tetrode=int(tetrode),
                cluster=int(cluster),
                n_spikes=len(train),
                first_spike_s=float(train[0]),
                last_spike_s=float(train[-1]),
                n_spikes_in_velocity_span=n_in_span,
                mean_rate_over_velocity_span_hz=n_in_span / span,
            )
        )
    for event_type in ("sdes", "ripple_events"):
        event_file = path.parent / (event_type + ".mat")
        if not event_file.is_file():
            row["n_" + event_type] = 0
            continue
        paths.append(event_file)
        events = numeric_matrix(loadmat(event_file, simplify_cells=True)[event_type], 4, event_type)
        row["n_" + event_type] = len(events)
        alignment_rows.extend(dict(**identity, event_type=event_type, **r) for r in position_alignment_diagnostic(position, velocity, events))
        for phase in ("native_endpoint", "native_peak"):
            starts, ends, valid = event_windows(events, phase)
            immobile = immobile_windows(velocity, starts, ends)
            counts = counts_in_windows(trains, starts, ends)
            for split in range(3):
                a, b = split_units(len(keys), str(relative), split, seed)
                ca, cb = counts[:, a], counts[:, b]
                sa, sb = ca.sum(axis=1), cb.sum(axis=1)
                aa, ab = np.count_nonzero(ca, axis=1), np.count_nonzero(cb, axis=1)
                for i in range(len(events)):
                    events_out.append(
                        dict(
                            **identity,
                            event_type=event_type,
                            event_index=i,
                            phase=phase,
                            split=split,
                            native_start_s=events[i, 0],
                            native_end_s=events[i, 1],
                            native_peak_s=events[i, 2],
                            start_s=starts[i],
                            end_s=ends[i],
                            valid_20ms_window=bool(valid[i]),
                            immobile=bool(immobile[i]),
                            n_units_per_half=len(a),
                            n_spikes_a=int(sa[i]),
                            n_spikes_b=int(sb[i]),
                            n_active_a=int(aa[i]),
                            n_active_b=int(ab[i]),
                            both_halves_have_3_spikes_2_units=bool(sa[i] >= 3 and sb[i] >= 3 and aa[i] >= 2 and ab[i] >= 2),
                        )
                    )
    return row, events_out, alignment_rows, units_out, paths


def summarize_support(events):
    columns = ["experiment", "subject_label", "recording_group", "session", "drug", "novel", "event_type", "phase", "split"]
    rows = []
    for values, local in events.groupby(columns, dropna=False):
        eligible = local.valid_20ms_window & local.immobile
        n = int(eligible.sum())
        supported = int((eligible & local.both_halves_have_3_spikes_2_units).sum())
        rows.append(
            dict(
                **dict(zip(columns, values, strict=True)),
                native_events=len(local),
                valid_immobile_windows=n,
                both_halves_supported=supported,
                support_fraction=supported / n if n else np.nan,
                units_per_half=int(local.n_units_per_half.iloc[0]),
                median_minimum_half_spikes=float(np.median(np.minimum(local.loc[eligible, "n_spikes_a"], local.loc[eligible, "n_spikes_b"]))) if n else np.nan,
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    paths = sorted(args.dataset_root.glob("Experiment_*/*/*/session_info.mat"))
    if not paths:
        raise ValueError("no native session catalog")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        "script": Path(__file__),
        "helper": ROOT / "scripts/_provenance.py",
        "readme": args.dataset_root / "README.md",
        "protocol": ROOT / "docs/kleinman_content_readiness_protocol.md",
    }
    for path in paths:
        for name in ("session_info.mat", "spike_data.mat", "sdes.mat", "ripple_events.mat"):
            source = path.parent / name
            if source.is_file():
                inputs[str(source.relative_to(args.dataset_root))] = source
    source_hashes = snapshot_inputs(inputs)
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="running", seed=args.seed, source_sha256_before=source_hashes)
    (args.output_dir / "frozen_inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    sessions, events, alignment, units = [], [], [], []
    for path in paths:
        try:
            row, ev, al, un, source_paths = inspect_session(path, args.dataset_root, args.seed)
            sessions.append(row)
            events.extend(ev)
            alignment.extend(al)
            units.extend(un)
            for source in source_paths:
                inputs[str(source.relative_to(args.dataset_root))] = source
        except (ValueError, KeyError, OSError) as exc:
            sessions.append({"session_path": str(path.parent), "status": "failed", "failure_reason": str(exc)})
        pd.DataFrame(sessions).to_csv(args.output_dir / "sessions.csv", index=False)
        print(json.dumps({"session": str(path.parent.relative_to(args.dataset_root)), "status": sessions[-1]["status"], "n_units": sessions[-1].get("n_units_raw")}), flush=True)
    ev = pd.DataFrame(events)
    if ev.empty:
        raise ValueError("no native events could be inspected")
    ev.to_csv(args.output_dir / "native_event_support.csv.gz", index=False)
    support = summarize_support(ev)
    support.to_csv(args.output_dir / "support_by_session.csv", index=False)
    pd.DataFrame(alignment).to_csv(args.output_dir / "position_alignment_unresolved.csv", index=False)
    pd.DataFrame(units).to_csv(args.output_dir / "raw_unit_inventory.csv", index=False)
    session_frame = pd.DataFrame(sessions)
    cohort = session_frame.loc[session_frame.status.eq("readable")]
    group = cohort.groupby(["experiment", "subject_label", "drug"], as_index=False).agg(
        sessions=("session", "size"),
        spiking_sessions=("has_spikes", "sum"),
        median_raw_units=("n_units_raw", "median"),
        max_raw_units=("n_units_raw", "max"),
        unresolved_position_sessions=("position_timing_status", lambda x: int(x.ne("independently_verified").sum())),
    )
    group.to_csv(args.output_dir / "subject_group_inventory.csv", index=False)
    control = cohort.loc[cohort.prospective_control_stratum & cohort.has_spikes]
    control_keys = set(zip(control.recording_group, control.session, strict=True))
    selected = support.loc[support.apply(lambda r: (r.recording_group, r.session) in control_keys, axis=1) & support.phase.eq("native_endpoint") & support.split.eq(0)]
    selected.to_csv(args.output_dir / "prospective_control_endpoint_support.csv", index=False)
    lines = [
        "# Kleinman/Foster native content-validation readiness",
        "",
        f"Readable sessions: {len(cohort)}/{len(paths)}. Sessions with spike files: {int(cohort.has_spikes.sum())}.",
        "",
        f"Prospective Experiment 1 control/saline stratum: {len(control)} spiking sessions, {control.subject_label.nunique()} within-experiment subject labels.",
        "",
        "No replay evidence, decoder output, model ranking, or biological classification was computed.",
        "",
        "Native candidate events are not validated replay labels. Support is measured in the last 20 ms at unchanged native endpoints; native peak-centered windows are a separate phase diagnostic, not substitute endpoints.",
        "",
        "Counts use every stored unit keyed by (tetrode, cluster), not place-field-qualified pyramidal cells. They are optimistic upper bounds on usable support after encoding QC. Three seeded disjoint equal halves are fixed from unit IDs; split0 is primary.",
        "",
        "Position timing remains unresolved when position has one extra sample relative to velocity timestamps. Both drop-first and drop-last conventions are compared descriptively with supplied event-onset positions; neither is silently selected. No session is declared decoder-ready.",
        "",
        "Subject labels are namespaced by experiment. Cross-experiment identity is not assumed independent or identical. Intervention conditions stay separate. Only Experiment 1 control animals with drug=0 define the prospective primary stratum, without selecting on replay outcomes.",
        "",
        "See support_by_session.csv for all cases; prospective_control_endpoint_support.csv preserves every control spiking session, including zero-support cases.",
        "",
    ]
    (args.output_dir / "readiness.md").write_text("\n".join(lines))
    verified_hashes = verify_inputs(inputs, source_hashes)
    provenance.update(
        status="complete" if len(cohort) == len(paths) else "incomplete",
        seed=args.seed,
        dataset="kleinman_foster_2025",
        decoder_ready=False,
        scientific_goal_achieved=False,
        native_session_files=len(paths),
        readable_sessions=len(cohort),
        spiking_sessions=int(cohort.has_spikes.sum()),
        native_spiking_sessions=sum((path.parent / "spike_data.mat").is_file() for path in paths),
        source_only_inventory=True,
        no_replay_outcome_selection=True,
        source_sha256_after=verified_hashes,
        inputs_unchanged=True,
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
