#!/usr/bin/env python3
"""Pre-decoding, rank-selected native-clock Tanni LFP/MUA examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hipporeplayimm.replay_coverage_data import array_sha256
from scipy.signal import butter, sosfiltfilt

from scripts._provenance import build_script_provenance, file_sha256
from scripts.prepare_replay_coverage_event_definitions import LFP_GROUP


def select_examples(windows):
    eligible = windows[windows.dataset.eq("tanni2022") & windows.eligible & windows.window_variant.eq("detected_core") & windows.ripple_status.eq("available")].copy()
    rows = []
    for animal in sorted(eligible.animal.unique()):
        for name in ["both_detectors", "mua_only", "ripple_only"]:
            frame = eligible[eligible.animal.eq(animal) & eligible.overlap_class.eq(name)]
            if name == "both_detectors":
                frame = frame[frame.detector.eq("source_high_mua")]
            frame = frame.sort_values(["n_spikes_qc_units", "session", "source_event_id"])
            if len(frame):
                row = frame.iloc[(len(frame) - 1) // 2].to_dict()
                rows.append({**row, "example_class": name, "class_candidates": len(frame), "rank": (len(frame) - 1) // 2 + 1,
                    "selection_rule": "median_QC_spike_count_rank_with_stable_session_source_ID_ties; no_decoding"})
    return pd.DataFrame(rows)


def run(root, out):
    manifest_path = root / "coverage_event_definition_manifest.json"
    meta = json.loads(manifest_path.read_text())
    audit = json.loads((root / "coverage_event_definition_reconstruction_audit.json").read_text())
    if meta["status"] not in {"complete", "complete_with_ripple_unavailable"} or audit["status"] != "pass" or audit["input_file_sha256"]["preparation_manifest"] != file_sha256(manifest_path):
        raise ValueError("linked terminal preparation and audit required")
    windows_path = root / "coverage_event_definition_windows.csv"
    if file_sha256(windows_path) != meta["output_sha256"][windows_path.name]:
        raise ValueError("window table changed")
    windows = pd.read_csv(windows_path)
    examples = select_examples(windows)
    if examples.empty:
        raise ValueError("no trace examples; refusing vacuous inspection")
    qc_path = root / "coverage_event_definition_channel_qc.csv"
    if file_sha256(qc_path) != meta["output_sha256"][qc_path.name]:
        raise ValueError("channel QC table changed")
    qc = pd.read_csv(qc_path)
    out.mkdir(parents=True, exist_ok=False)
    records = {(r["animal"], r["session"]): r for r in meta["results"]}
    panels = []
    for animal, group in examples.groupby("animal", sort=True):
        fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex="col", constrained_layout=True)
        for j, example in enumerate(group.itertuples(index=False)):
            record = records[(animal, example.session)]
            raw = record["raw_ripple_source"]
            if file_sha256(record["source_cache_path"]) != record["source_cache_sha256"] or file_sha256(raw["envelope_cache_path"]) != raw["envelope_cache_sha256"]:
                raise ValueError("trace source changed")
            with np.load(raw["envelope_cache_path"], allow_pickle=False) as f:
                times, z, fs = f["times_s"], f["z"], float(f["sampling_rate_hz"])
            lo, hi = example.peak_s - .5, example.peak_s + .5
            a, b = np.searchsorted(times, [lo, hi])
            with h5py.File(raw["path"], "r") as f:
                channels = np.asarray(f[f"{LFP_GROUP}/downsampling_info/downsampled_channels"])
                # Choose a fixed QC-passing channel ID, not the event maximum.
                available = qc[qc.animal.eq(animal) & qc.session.eq(example.session) & qc.selected]
                channel_id = int(available.native_channel_id.min())
                column = int(np.flatnonzero(channels == channel_id)[0])
                pa, pb = np.searchsorted(times, [lo - 1., hi + 1.])
                voltage = np.asarray(f[f"{LFP_GROUP}/downsampled_tetrode_data"][pa:pb, column], float)
            filtered = sosfiltfilt(butter(4, [150, 250], fs=fs, btype="bandpass", output="sos"), voltage)[a - pa:b - pa]
            raw_view = voltage[a - pa:b - pa]
            with np.load(record["source_cache_path"], allow_pickle=False) as f:
                spikes, ids = f["spikes"], f["cell_ids"][f["unit_qc_mask"]]
            st = spikes[(spikes[:, 0] >= lo) & (spikes[:, 0] < hi) & np.isin(spikes[:, 1], ids)]
            x = (times[a:b] - example.peak_s) * 1000
            axes[0, j].plot(x, raw_view, lw=.5, color="#555555")
            axes[1, j].plot(x, filtered, lw=.6, color="#176b8e")
            axes[2, j].plot(x, z[a:b], lw=1, color="#b94748")
            axes[2, j].axhline(0, color="black", lw=.5)
            axes[2, j].axhline(3, color="black", lw=.5, ls="--")
            axes[3, j].scatter((st[:, 0] - example.peak_s) * 1000, np.searchsorted(ids, st[:, 1]), s=3, color="black", marker="|")
            local = windows[windows.animal.eq(animal) & windows.session.eq(example.session) & windows.window_variant.eq("detected_core") & (windows.start_s < hi) & (windows.end_s > lo)]
            for w in local.itertuples(index=False):
                color = "#176b8e" if w.detector == "source_high_mua" else "#b94748"
                axes[2, j].axvspan((max(lo, w.start_s) - example.peak_s) * 1000, (min(hi, w.end_s) - example.peak_s) * 1000, color=color, alpha=.12 if w.eligible else .04)
            axes[0, j].set_title(f"{example.example_class}\n{example.session}\n{example.n_spikes_qc_units} QC spikes; channel {channel_id}", fontsize=10)
            axes[3, j].set_xlabel("Time from native event peak (ms)")
            panels.append({"animal": animal, "session": example.session, "window_uid": example.window_uid,
                "example_class": example.example_class, "channel_id": channel_id, "lfp_start_sample": a, "lfp_end_sample_exclusive": b,
                "time_shift_applied_s": 0, "display_raw_min": float(np.min(raw_view)), "display_raw_max": float(np.max(raw_view)),
                "raw_padded_slice_sha256": array_sha256(voltage), "raw_path": raw["path"],
                "raw_padded_start_sample": int(pa), "raw_padded_end_sample_exclusive": int(pb),
                "source_cache_sha256": record["source_cache_sha256"], "envelope_cache_sha256": raw["envelope_cache_sha256"],
                "display_aggregate_z_peak": float(np.nanmax(z[a:b])), "figure": f"traces_{animal}.png"})
        for i, label in enumerate(["LFP (native integer units)", "150-250 Hz (same units)", "Pooled envelope z", "RUN-QC cell index"]):
            axes[i, 0].set_ylabel(label)
        for ax in axes.flat:
            ax.set_xlim(-500, 500)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{animal}: pre-decoding detector inspection\nBlue shading = MUA; red = ripple-like; pale = excluded. One fixed channel, not a ripple-ground-truth label.", fontsize=12)
        fig.savefig(out / f"traces_{animal}.png", dpi=160)
        plt.close(fig)
    examples.to_csv(out / "coverage_lfp_mua_trace_selection.csv", index=False)
    pd.DataFrame(panels).to_csv(out / "coverage_lfp_mua_trace_manifest.csv", index=False)
    result = build_script_provenance(input_paths={"event_manifest": manifest_path, "windows": windows_path,
        "event_audit": root / "coverage_event_definition_reconstruction_audit.json", "channel_qc": qc_path,
        "script": Path(__file__), "protocol": ROOT / "docs/replay_coverage_detector_decoding_protocol.md"}, cwd=ROOT)
    result.update(status="complete", examples=len(panels), selection_uses_decoding=False,
        caveat="selected examples cannot establish detector specificity, no temporal offset, or artifact freedom; PF raw LFP unavailable",
        output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "coverage_lfp_mua_trace_provenance.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve())
