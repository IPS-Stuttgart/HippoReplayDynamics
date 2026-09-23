#!/usr/bin/env python3
"""Independent template likelihood/cross-fit check on the reused simulation bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from calibrate_kleinman_integrated_extent import condition_summary, screens


def reference_counts(model, row):
    n = round(row.duration_s * 1000)
    u = (np.arange(n) + 0.5) / n
    if row.profile == "cosine":
        u = 0.5 * (1 - np.cos(np.pi * u))
    elif row.profile == "pause_step":
        u = np.clip((u - 0.25) * 2, 0, 1)
    x = model["ends"][row.side] + (1 - 2 * row.side) * np.diff(model["ends"]).item() * row.extent * u
    b = np.digitize(x, model["edges"]) - 1 + row.side * (len(model["edges"]) - 1)
    intensity = model["rates"][b] * 0.001
    random = np.random.default_rng(np.random.SeedSequence([20260923, row.event_index]))
    fine = random.poisson(intensity * (row.expected_spikes / intensity.sum()))
    assert fine.sum() == row.n_spikes
    return np.array([fine[i : i + 10].sum(axis=0) for i in range(0, n, 10)])


def reference_library(model, duration):
    n = round(duration * 1000)
    u = (np.arange(n) + 0.5) / n
    metadata, logq = [], []
    for direction in [0, 1]:
        for si in range(9):
            for ei in range(9):
                for profile in ["static"] if si == ei else ["linear", "cosine"]:
                    progress = (1 - np.cos(np.pi * u)) / 2 if profile == "cosine" else u
                    x = model["ends"][0] + np.diff(model["ends"]).item() * (si + (ei - si) * progress) / 8
                    b = np.digitize(x, model["edges"]) - 1 + direction * (len(model["edges"]) - 1)
                    fraction = model["support"][b].mean()
                    if fraction < 0.95:
                        continue
                    probability = []
                    for start in range(0, n, 10):
                        rates = model["rates"][b[start : start + 10]].sum(axis=0)
                        probability.append(rates / rates.sum())
                    metadata.append({"start": si / 8, "end": ei / 8, "extent": abs(ei - si) / 8, "direction": direction, "profile": profile, "supported_fraction": fraction})
                    logq.append(np.log(probability))
    return np.array(logq), pd.DataFrame(metadata)


def reference_fit(counts, logq, templates):
    ll = np.array([(counts * q).sum() for q in logq])
    ties = ll >= ll.max() - 1e-10
    extent = templates.extent.to_numpy()
    scores = {
        "estimated_extent": extent[ties].mean(),
        "n_best_ties": int(ties.sum()),
        "estimated_start": templates.start.to_numpy()[ties].mean(),
        "estimated_end": templates.end.to_numpy()[ties].mean(),
        "likelihood_weighted_extent": np.exp(ll - logsumexp(ll)) @ extent,
        "best_conditional_log_likelihood": ll.max(),
    }
    delta = 0.0
    for parity in [0, 1]:
        train = np.arange(len(counts)) % 2 == parity
        train_ll = np.array([(counts[train] * q[train]).sum() for q in logq])
        test_ll = np.array([(counts[~train] * q[~train]).sum() for q in logq])
        f_scores = []
        for family in [extent == 0, extent > 0]:
            selected = family & (train_ll >= train_ll[family].max() - 1e-10)
            f_scores.append(test_ll[selected].mean())
        delta += f_scores[1] - f_scores[0]
    scores["crossfit_moving_minus_static"] = delta
    return scores


def verify(bank, output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert file_sha256(output / name) == digest, name
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(Path(path)) == manifest["input_file_sha256"][key], key
    frames, checked = [], 0
    for path in sorted(output.glob("*_estimates.csv")):
        f = pd.read_csv(path)
        assert len(f) == 2880 and not f.event_index.duplicated().any()
        frames.append(f)
        animal = f.animal.iloc[0]
        model = dict(np.load(bank / (animal + "_known_map.npz")))
        ids = np.random.default_rng(909).choice(f.event_index, 64, replace=False)
        for duration, sub in f.groupby("duration_s"):
            logq, meta = reference_library(model, duration)
            pd.testing.assert_frame_equal(meta, pd.read_csv(output / f"{animal}_{duration}_templates.csv"), check_dtype=False)
            for row in sub.loc[sub.event_index.isin(ids)].itertuples():
                ref = reference_fit(reference_counts(model, row), logq, meta)
                for name, value in ref.items():
                    np.testing.assert_allclose(getattr(row, name), value, rtol=1e-9, atol=1e-9, err_msg=f"{animal}/{row.event_index}/{name}")
                checked += 1
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 17280 and frame.event_index.nunique() == 17280 and frame.animal.nunique() == 6
    for name, data in [("kleinman_integrated_condition_summary.csv", condition_summary(frame)), ("kleinman_integrated_screens.csv", screens(frame))]:
        pd.testing.assert_frame_equal(pd.read_csv(output / name), data, check_dtype=False, rtol=1e-9, atol=1e-9)
    result = {
        "status": "passed",
        "producer_commit": manifest["code_commit"],
        "n_reused_events": len(frame),
        "independent_events_recomputed": checked,
        "independent_template_libraries": 18,
        "result_manifest_sha256": file_sha256(output / "manifest.json"),
        "verifier_sha256": file_sha256(Path(__file__)),
        "scope": "All source/output hashes and aggregate screens; independent scalar likelihood and cross-fitting on 64 events per animal. Not an independent re-fit of RUN maps or re-fit of every event.",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.bank_dir, args.output_dir)
