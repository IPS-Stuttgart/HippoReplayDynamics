#!/usr/bin/env python3
"""Recompute synthetic calibration from saved maps with a scalar reference decoder."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from calibrate_kleinman_replay_content import ARMS, readiness, summarize


def reference(row, model):
    count = round(row.duration_s * 1000)
    t = (np.arange(count) + 0.5) / count
    if row.profile == "cosine":
        p = 0.5 - 0.5 * np.cos(np.pi * t)
    elif row.profile == "pause_step":
        p = np.minimum(1, np.maximum(0, 2 * t - 0.5))
    else:
        p = t
    span = model["ends"][1] - model["ends"][0]
    truth = model["ends"][row.side] + (1 - row.side * 2) * row.extent * span * p
    n = len(model["edges"]) - 1
    b = np.digitize(truth, model["edges"]) - 1 + row.side * n
    intensity = model["rates"][b] / 1000
    gain = row.expected_spikes / intensity.sum()
    rng = np.random.default_rng(np.random.SeedSequence([20260923, row.event_index]))
    spikes = rng.poisson(gain * intensity)
    starts = list(range(0, count - 39, 5))
    centers = (np.array(starts) + 20) / 1000
    loc = (model["edges"][1:] + model["edges"][:-1]) / 2
    rates = model["rates"]
    probabilities = []
    true_means = []
    for start in starts:
        c = spikes[start : start + 40].sum(axis=0)
        if row.arm == "count_conditioned":
            ll = (c * np.log(rates / rates.sum(axis=1, keepdims=True))).sum(axis=1)
        else:
            use_gain = gain if row.arm == "matched_gain_poisson" else 1.0
            lam = 0.04 * use_gain * rates
            ll = (c * np.log(lam) - lam).sum(axis=1)
        ll[~model["support"]] = -np.inf
        probabilities.append(np.exp(ll - logsumexp(ll)))
        true_means.append(np.mean(truth[start : start + 40]))
    probabilities = np.array(probabilities)
    expected_position = (probabilities * np.tile(loc, 2)).sum(axis=1)
    first = centers <= centers[0] + 0.25 * (centers[-1] - centers[0])
    last = centers >= centers[0] + 0.75 * (centers[-1] - centers[0])
    delta = (expected_position[last].mean() - expected_position[first].mean()) * (1 - 2 * row.side) / span
    true_delta = (np.mean(np.array(true_means)[last]) - np.mean(np.array(true_means)[first])) * (1 - 2 * row.side) / span
    masses = probabilities.reshape(len(starts), 2, n).sum(axis=(0, 2)) / len(starts)
    dominant = int(masses.argmax())
    w = probabilities[:, dominant * n : (dominant + 1) * n]
    w = w / w.sum()
    mt = (w * centers[:, None]).sum()
    mx = (w * loc[None, :]).sum()
    cov = (w * (centers[:, None] - mt) * (loc[None, :] - mx)).sum()
    denom = np.sqrt((w * (centers[:, None] - mt) ** 2).sum() * (w * (loc[None, :] - mx) ** 2).sum())
    corr = cov / denom if denom > 1e-10 else 0.0
    return {
        "displacement_fraction": delta,
        "true_displacement_fraction": true_delta,
        "gain": gain,
        "n_spikes": int(spikes.sum()),
        "n_active_units": int((spikes.sum(axis=0) > 0).sum()),
        "true_support_fraction": float(model["support"][b].mean()),
        "incoming_direction_mass": masses[row.side],
        "weighted_correlation": corr,
        "reverse_content_call": bool(masses[dominant] >= 0.55 and abs(corr) >= 0.5 and corr * (2 * dominant - 1) < 0 and delta > 0.1),
    }


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert file_sha256(output / name) == digest, name
    files = sorted(output.glob("*_events.csv"))
    assert len(files) == 6
    frames, checked, errors = [], 0, []
    for path in files:
        f = pd.read_csv(path)
        frames.append(f)
        assert len(f) == 8640
        assert not f.duplicated(["event_index", "arm"]).any()
        assert set(f.arm) == set(ARMS)
        shapes = [(0.0, "linear")] + list(product([0.25, 0.5, 0.75], ["linear", "cosine", "pause_step"]))
        counts = f.groupby(["side", "extent", "profile", "duration_s", "expected_spikes", "arm"]).size()
        expected = {(s, e, p, d, c, a) for s, (e, p), d, c, a in product(range(2), shapes, [0.1, 0.2, 0.4], [24, 48, 96], ARMS)}
        assert set(counts.index) == expected and counts.eq(16).all()
        model = dict(np.load(output / (f.animal.iloc[0] + "_known_map.npz")))
        chosen = np.random.default_rng(771).choice(f.event_index.unique(), size=64, replace=False)
        for row in f.loc[f.event_index.isin(chosen)].itertuples():
            recomputed = reference(row, model)
            for name, value in recomputed.items():
                np.testing.assert_allclose(getattr(row, name), value, atol=1e-9, rtol=1e-9, err_msg=f"{path.name}/{row.event_index}/{row.arm}/{name}")
            errors.append(abs(row.displacement_fraction - recomputed["displacement_fraction"]))
            checked += 1
    frame = pd.concat(frames, ignore_index=True)
    assert frame.event_index.nunique() == 17280
    np.testing.assert_array_equal(np.sort(frame.event_index.unique()), np.arange(17280))
    recomputed_summary = summarize(frame)
    saved = pd.read_csv(output / "kleinman_content_condition_summary.csv")
    pd.testing.assert_frame_equal(saved, recomputed_summary, check_dtype=False, atol=1e-10, rtol=1e-10)
    gates, summary = readiness(frame)
    for name, table in [("stratum_gates", gates), ("readiness", summary)]:
        pd.testing.assert_frame_equal(pd.read_csv(output / ("kleinman_content_" + name + ".csv")), table, check_dtype=False)
    result = {
        "status": "passed",
        "producer_commit": manifest["code_commit"],
        "all_bank_combinations_verified": True,
        "synthetic_events": 17280,
        "independent_decoder_rows_recomputed": checked,
        "independent_events_recomputed": checked // 3,
        "maximum_displacement_fraction_error": max(errors),
        "verification_scope": "All output hashes, condition bank and summary/gate regeneration; independent direct-window/scalar-likelihood reference for 64 events per animal, all three arms. Not an independent map refit or all-event decoder reimplementation.",
        "verifier_sha256": file_sha256(Path(__file__)),
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    verify(parser.parse_args().output_dir)
