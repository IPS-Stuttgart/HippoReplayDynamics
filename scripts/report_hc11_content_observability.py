#!/usr/bin/env python3
"""Independently recount and summarize hc-11 activity support; never decode."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts._provenance import build_script_provenance, file_sha256


def histogram_recount(times, windows):
    """Independent half-open counting through a unique-boundary histogram/CDF."""
    windows = np.asarray(windows, float)
    if not len(windows):
        return np.empty(0, int)
    edges, index = np.unique(windows.ravel(), return_inverse=True)
    if (windows[:, 1] <= windows[:, 0]).any() or not np.isfinite(edges).all():
        raise ValueError("invalid audit windows")
    t = np.asarray(times, float)
    # numpy.histogram closes its last bin, whereas all source windows are half-open.
    hist = np.histogram(t[(t >= edges[0]) & (t < edges[-1])], bins=edges)[0]
    cumulative = np.r_[0, np.cumsum(hist)]
    bounds = index.reshape(-1, 2)
    return cumulative[bounds[:, 1]] - cumulative[bounds[:, 0]]


def audit_session(row, seed):
    folder = Path(row.artifact_dir)
    meta = json.loads((folder / "manifest.json").read_text())
    for key, digest in meta["source_sha256"].items():
        if file_sha256(meta["source_paths"][key]) != digest:
            raise ValueError("native source checksum changed")
    for name, digest in meta["output_sha256"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError("source output checksum changed")
    with np.load(folder / "count_arrays.npz", allow_pickle=False) as archive:
        ids, stored = archive["cell_ids"], archive["counts"]
    windows = pd.read_csv(folder / "windows.csv", float_precision="round_trip")
    candidates = pd.read_csv(folder / "candidate_events.csv", float_precision="round_trip")
    if candidates.event_index.duplicated().any() or len(candidates) != row.candidates:
        raise ValueError("candidate denominator mismatch")
    if windows.duplicated(["event_index", "window"]).any() or len(windows) != 2*len(candidates):
        raise ValueError("window completeness mismatch")
    times = windows[["start_s", "end_s"]].to_numpy()
    source = loadmat(meta["source_paths"]["spikes"], simplify_cells=True)["spikes"]
    uids = np.atleast_1d(source["UID"]).astype(int)
    by_id = dict(zip(uids, source["times"], strict=True))
    recounted = np.column_stack([histogram_recount(by_id[i], times) for i in ids])
    np.testing.assert_array_equal(recounted, stored, err_msg="independent native histogram count mismatch")
    native_classes = loadmat(meta["source_paths"]["classes"], simplify_cells=True)["CellClass"]
    physiology = {int(i): (int(e), int(h)) for i, e, h in zip(native_classes["UID"],
                      native_classes["pE"], native_classes["pI"], strict=True)}
    regions = dict(zip(uids, source["region"], strict=True))
    if not all(physiology.get(int(i)) == (1, 0) and regions[i] in {"CA1", "lCA1", "rCA1"} for i in ids):
        raise ValueError("included source cell is not native CA1 excitatory")
    for event in candidates.itertuples(index=False):
        local = windows.loc[windows.event_index.eq(event.event_index)].set_index("window")
        n = int(np.floor((event.end_s-event.start_s)/.005+1e-8))
        end = min(event.start_s+n*.005, event.end_s)
        np.testing.assert_allclose(local.loc["endpoint", ["start_s", "end_s"]].to_numpy(float),
                                   [end-.02, end], rtol=0, atol=1e-10)
        np.testing.assert_allclose(local.loc["peak_diagnostic", ["start_s", "end_s"]].to_numpy(float),
                                   [event.peak_s-.01, event.peak_s+.01], rtol=0, atol=1e-10)
        supported = event.peak_s-.01 >= event.start_s and event.peak_s+.01 <= event.end_s
        if bool(local.loc["peak_diagnostic", "available"]) != supported:
            raise ValueError("peak availability changed")
    frame = pd.read_csv(folder / "population_counts.csv.gz", float_precision="round_trip")
    if frame.duplicated(["event_index", "window", "groups", "split"]).any():
        raise ValueError("duplicate population readout")
    positions = {int(i): j for j, i in enumerate(ids)}
    for part in meta["population_assignments"]:
        k, split = part["groups"], part["split"]
        raw_seed = f"{seed}|hc11_preflight|{row.session}|{k}|{split}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(raw_seed).digest()[:8], "little"))
        permutation = rng.permutation(len(ids))
        size = len(ids)//k
        expected = [ids[np.sort(permutation[i*size:(i+1)*size])].tolist() for i in range(k)]
        if expected != part["cell_ids"] or ids[np.sort(permutation[k*size:])].tolist() != part["dropped_ids"]:
            raise ValueError("population selection differs from freeze")
        local = frame.loc[frame.groups.eq(k) & frame.split.eq(split)].reset_index(drop=True)
        pd.testing.assert_frame_equal(local[windows.columns], windows, check_dtype=False)
        totals, active = [], []
        for i, group in enumerate(part["cell_ids"]):
            sub = recounted[:, [positions[int(uid)] for uid in group]]
            total, cells = np.sum(sub, axis=1), np.sum(sub != 0, axis=1)
            np.testing.assert_array_equal(total, local[f"group{i}_spikes"])
            np.testing.assert_array_equal(cells, local[f"group{i}_active_cells"])
            totals.append(total)
            active.append(cells)
        totals, active = np.asarray(totals).T, np.asarray(active).T
        expected = dict(all_groups_supported=((totals >= 3) & (active >= 2)).all(axis=1) & windows.available,
                        all_groups_silent=(totals == 0).all(axis=1) & windows.available,
                        any_group_silent=(totals == 0).any(axis=1) & windows.available,
                        minimum_group_spikes=totals.min(axis=1), minimum_group_active_cells=active.min(axis=1))
        for key, value in expected.items():
            np.testing.assert_array_equal(value, local[key], err_msg=f"invalid derived {key}")
    if len(frame) != len(windows)*len(meta["population_assignments"]):
        raise ValueError("unexpected/missing population rows")
    record = dict(session=row.session, candidates=len(candidates), window_cell_counts=int(stored.size),
                  population_rows=len(frame), source_files_verified=len(meta["source_paths"]), status="passed")
    return record, frame


def audit_aggregates(sessions, counts, source):
    rows = []
    for (animal, session, groups, window, split), local in counts.groupby(
            ["animal", "session", "groups", "window", "split"], sort=True):
        n = int(sessions.set_index("session").loc[session, "candidates"])
        rows.append(dict(animal=animal, session=session, groups=groups, window=window, split=split,
            source_candidates=n, measured_candidates=len(local), status="measured",
            all_groups_supported_fraction=local.all_groups_supported.sum()/n,
            all_groups_silent_fraction=local.all_groups_silent.sum()/n,
            any_group_silent_fraction=local.any_group_silent.sum()/n, available_fraction=local.available.sum()/n,
            minimum_group_spikes_mean=local.minimum_group_spikes.mean(),
            minimum_group_active_cells_mean=local.minimum_group_active_cells.mean(),
            cells_per_group=int(local.cells_per_group.iloc[0])))
    expected = pd.DataFrame(rows)
    keys = ["animal", "session", "groups", "window", "split"]
    actual = pd.read_csv(source / "observability_by_session.csv")
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True)[expected.columns],
                                  expected.sort_values(keys).reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-12)
    fields = [c for c in expected if c.endswith("_fraction") or c.endswith("_mean")]
    animals = expected.groupby(["animal", "groups", "window", "split"], as_index=False)[fields].mean()
    pd.testing.assert_frame_equal(pd.read_csv(source / "observability_by_animal.csv"), animals,
                                  check_dtype=False, rtol=1e-10, atol=1e-12)
    primary = expected.loc[expected.groups.eq(2) & expected.window.eq("endpoint") & expected.split.eq(0)]
    rat_primary = animals.loc[animals.groups.eq(2) & animals.window.eq("endpoint") & animals.split.eq(0)]
    gates = pd.read_csv(source / "observability_gate_summary.csv").set_index("gate")
    expected_gates = dict(all_eight_sources_complete=len(sessions) == 8 and sessions.status.eq("complete").all(),
        all_four_animals_measurable=primary.animal.nunique() == 4,
        endpoint_measurability=primary.measured_candidates.sum()/primary.source_candidates.sum() >= .8,
        both_halves_activity_support=len(rat_primary) == 4 and rat_primary.all_groups_supported_fraction.ge(.2).sum() >= 3,
        independent_count_audit=len(sessions) == 8 and sessions.independent_recount_passed.eq(True).all())
    expected_gates["ready_for_endpoint_encoding_validation"] = all(expected_gates.values())
    expected_observed = dict(all_eight_sources_complete=int(sessions.status.eq("complete").sum()),
        all_four_animals_measurable=primary.animal.nunique(),
        endpoint_measurability=primary.measured_candidates.sum()/primary.source_candidates.sum(),
        both_halves_activity_support=int(rat_primary.all_groups_supported_fraction.ge(.2).sum()),
        independent_count_audit=int(sessions.independent_recount_passed.eq(True).sum()))
    if set(gates.index) != set(expected_gates):
        raise ValueError("gate set differs")
    for key, value in expected_gates.items():
        if bool(gates.loc[key, "passed"]) != bool(value):
            raise ValueError(f"gate does not match evidence: {key}")
        if key in expected_observed and not np.isclose(float(gates.loc[key, "observed"]), expected_observed[key]):
            raise ValueError(f"gate observation does not match evidence: {key}")
    return animals


def figure(animals, output):
    colors = {"endpoint": "#555960", "peak_diagnostic": "#007f82"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout="constrained")
    for ax, groups in zip(axes[0], (2, 3), strict=True):
        for window in ("endpoint", "peak_diagnostic"):
            local = animals.loc[animals.groups.eq(groups) & animals.split.eq(0) & animals.window.eq(window)]
            x = np.arange(len(local)) + (-.16 if window == "endpoint" else .16)
            ax.bar(x, 100*local.all_groups_supported_fraction, width=.3,
                   color=colors[window], label="Fixed endpoint" if window == "endpoint" else "Burst peak (diagnostic)")
        ax.set_xticks(np.arange(len(local)), local.animal)
        ax.set_ylim(0, 85)
        ax.set_ylabel("All populations activity-supported (%)")
        ax.set_title(f"{'A' if groups == 2 else 'B'}. {groups} disjoint populations")
        if groups == 2:
            ax.axhline(20, color="#a53563", ls="--", lw=1, label="Endpoint feasibility threshold")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax, metric, label, letter in (
        (axes[1, 0], "minimum_group_spikes_mean", "Spikes in lower-count half (mean)", "C"),
        (axes[1, 1], "any_group_silent_fraction", "At least one silent half (%)", "D"),
    ):
        for window in ("endpoint", "peak_diagnostic"):
            local = animals.loc[animals.groups.eq(2) & animals.split.eq(0) & animals.window.eq(window)]
            y = local[metric] * (100 if metric.endswith("fraction") else 1)
            ax.plot(local.animal, y, "o-", color=colors[window], lw=1.4)
        ax.set_ylabel(label)
        ax.set_title(f"{letter}. Two-population information budget")
        ax.set_ylim(bottom=0)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("hc-11: candidate endpoints have little activity after population splitting\n"
                 "First-half RUN activity screen; no decoded content or predictive-model fitting", fontsize=12)
    fig.savefig(output / "hc11_endpoint_observability.png", dpi=180)
    fig.savefig(output / "hc11_endpoint_observability.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir
    provenance = build_script_provenance(input_paths=dict(producer_manifest=source / "manifest.json",
                                                         reporter=Path(__file__)), cwd=ROOT)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["inputs_unchanged"]:
        raise ValueError("source run not complete/immutable")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("aggregate source checksum changed")
    sessions = pd.read_csv(source / "source_sessions.csv")
    if not sessions.status.eq("complete").all():
        raise ValueError("reporter requires complete source recount; retain failure artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audits, frames = [], []
    for row in sessions.itertuples(index=False):
        audit, frame = audit_session(row, int(manifest["seed"]))
        audits.append(audit)
        frames.append(frame)
        print(json.dumps(audit), flush=True)
    counts = pd.concat(frames, ignore_index=True)
    animals = audit_aggregates(sessions, counts, source)
    pd.DataFrame(audits).to_csv(args.output_dir / "independent_recount_by_session.csv", index=False)
    animals.to_csv(args.output_dir / "verified_observability_by_animal.csv", index=False)
    figure(animals, args.output_dir)
    provenance.update(status="passed", completed_at_utc=datetime.now(UTC).isoformat(),
                      candidates=int(sessions.candidates.sum()), animals=int(sessions.animal.nunique()),
                      window_cell_counts=sum(a["window_cell_counts"] for a in audits),
                      population_rows=sum(a["population_rows"] for a in audits),
                      validation_scope="native histogram recount; fixed partitions/windows; aggregates and all gates",
                      limits="no spatial encoding validation, no replay truth, no predictive model fitted",
                      outputs_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    if file_sha256(source / "manifest.json") != provenance["input_file_sha256"]["producer_manifest"]:
        raise ValueError("source manifest changed")
    (args.output_dir / "independent_audit.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    main()
