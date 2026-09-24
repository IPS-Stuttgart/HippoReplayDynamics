#!/usr/bin/env python3
"""Rebuild native candidate accounting and independently check sampled fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from audit_kleinman_native_adequacy import random_state, summarize
from validate_kleinman_run_decoder import matrix, split_units
from verify_kleinman_integrated_extent import reference_fit, reference_library


def read_csv(path):
    # Exact boundary timestamps must survive the CSV round trip.
    return pd.read_csv(path, float_precision="round_trip")


def independent_counts(trains, start, end, n_bins):
    edges = start + np.arange(n_bins + 1) * 0.01
    results = []
    for spikes in trains:
        selected = spikes[(spikes >= start) & (spikes < edges[-1])]
        results.append(np.histogram(selected, bins=edges)[0])
    return np.array(results).T


def saturation(counts):
    values = []
    for row in counts:
        positive = row[row > 0]
        values.append(sum(positive * np.log(positive / row.sum())) if len(positive) else 0.0)
    return sum(values)


def reference_diagnostic(counts, logq, best, rng):
    ll = np.array([(counts * q).sum() for q in logq])
    assert ll[best] >= ll.max() - 1e-8
    deviance = max(0, 2 * (saturation(counts) - ll.max()))
    draws = np.array([rng.multinomial(int(row.sum()), np.exp(logq[best, t]), size=99) for t, row in enumerate(counts)]).transpose(1, 0, 2)
    assert np.array_equal(draws.sum(axis=2), np.tile(counts.sum(axis=1), (99, 1)))
    replicas = []
    for draw in draws:
        best_ll = max(float((draw * q).sum()) for q in logq)
        replicas.append(max(0, 2 * (saturation(draw) - best_ll)))
    tail = (1 + sum(d >= deviance - 1e-10 for d in replicas)) / 100
    prediction = baseline = 0.0
    for parity in [0, 1]:
        train = np.arange(len(counts)) % 2 == parity
        scores = np.array([float((counts[train] * q[train]).sum()) for q in logq])
        chosen = np.flatnonzero(scores >= scores.max() - 1e-10)
        prediction += np.mean([float((counts[~train] * logq[i, ~train]).sum()) for i in chosen])
        probability = counts[train].sum(axis=0) + 0.5
        probability /= probability.sum()
        baseline += sum(float(row @ np.log(probability)) for row in counts[~train])
    return {
        "conditional_deviance": deviance,
        "replica_deviance_median": np.median(replicas),
        "replica_deviance_p95": np.quantile(replicas, 0.95),
        "deviance_tail_probability": tail,
        "crossfit_template_log_score": prediction,
        "crossfit_free_composition_log_score": baseline,
        "crossfit_template_minus_free": prediction - baseline,
        "crossfit_template_minus_free_per_spike": (prediction - baseline) / max(counts.sum(), 1),
    }


def compare_scores(row, counts, logq, templates, rng):
    ref = reference_fit(counts, logq, templates)
    ref.update(reference_diagnostic(counts, logq, int(row.best_template_index), rng))
    for key, value in ref.items():
        np.testing.assert_allclose(value, getattr(row, key), rtol=1e-8, atol=1e-8, err_msg=f"{row.animal}/{row.session}/{key}")
    assert bool(row.deviance_flag) == (ref["deviance_tail_probability"] <= 0.05)


def control_counts(model, row, logq, templates):
    i = row.control_id
    rng = random_state(row.animal, row.session, i, 1)
    static = i % 2 == 0
    choices = np.flatnonzero((templates.extent == 0).to_numpy() == static)
    index = int(rng.choice(choices))
    totals = rng.multinomial(48 if (i // 2) % 2 == 0 else 96, np.full(20, 0.05))
    q = np.exp(logq[index])
    biased = q.copy()
    cells = rng.choice(q.shape[1], max(1, q.shape[1] // 2), replace=False)
    biased[:, cells] *= 4
    biased /= biased.sum(axis=1, keepdims=True)
    namespace = 2 if row.condition == "matched" else 3
    truth = q if row.condition == "matched" else biased
    draw_rng = random_state(row.animal, row.session, i, namespace + 10)
    counts = np.array([draw_rng.multinomial(int(n), p, size=1)[0] for n, p in zip(totals, truth, strict=True)])
    assert counts.sum() == row.n_spikes
    assert templates.iloc[index].profile == row.generated_profile
    return counts, namespace


def run(args):
    out = args.output_dir
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert file_sha256(out / name) == digest, name
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(Path(path)) == manifest["input_file_sha256"][key], key
    inventory = read_csv(out / "kleinman_native_inventory.csv")
    fitted = read_csv(out / "kleinman_native_event_adequacy.csv")
    cal = read_csv(out / "kleinman_native_controls.csv")
    assert len(cal) == 1120 and cal.groupby(["animal", "session", "condition"]).size().eq(40).all()
    assert not fitted.duplicated(["animal", "session", "event_id"]).any()
    assert inventory.groupby(["animal", "session"]).ngroups == 127
    assert len(fitted) == len(inventory)
    pending = inventory.fit_status.eq("pending")
    assert fitted.loc[pending, "fit_status"].eq("scored").all()
    assert (fitted.loc[~pending, "fit_status"].to_numpy() == inventory.loc[~pending, "fit_status"].to_numpy()).all()
    checked, checked_cal, counted = 0, 0, 0
    for (animal, session), f in fitted.groupby(["animal", "session"]):
        folder = args.dataset_root / "Experiment_1" / animal / session
        native = matrix(loadmat(folder / "sdes.mat", simplify_cells=True)["sdes"], 4, "sdes")
        assert len(f) == len(native) and np.array_equal(f.event_id.to_numpy(), np.arange(len(native)))
        np.testing.assert_allclose(f[["start_s", "end_s", "peak_s", "animal_position_at_onset_cm"]], native, atol=1e-10)
        model = dict(np.load(args.cohort_dir / "sessions" / animal / session / "known_map.npz"))
        keys, all_trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
        lookup = {tuple(k): v for k, v in zip(keys, all_trains, strict=True)}
        trains = [lookup[tuple(k)] for k in model["units"]]
        for row in f.itertuples():
            counts = independent_counts(trains, row.start_s, row.end_s, int(row.n_fit_bins))
            assert counts.sum() == row.n_fit_spikes
            assert (counts.sum(axis=0) > 0).sum() == row.n_active_encoding_units
            assert counts[::2].sum() == row.even_bin_spikes and counts[1::2].sum() == row.odd_bin_spikes
            native_count = sum(np.count_nonzero((s >= row.start_s) & (s < row.end_s)) for s in trains)
            assert native_count == row.n_native_encoding_spikes
            assert native_count - counts.sum() == row.n_omitted_tail_spikes
            counted += 1
        eligible = f.loc[f.fit_status.eq("scored")]
        if eligible.empty:
            continue
        sample = eligible.iloc[np.unique(np.linspace(0, len(eligible) - 1, min(8, len(eligible)), dtype=int))]
        for row in sample.itertuples():
            logq, templates = reference_library(model, row.fit_duration_s)
            counts = independent_counts(trains, row.start_s, row.end_s, int(row.n_fit_bins))
            compare_scores(row, counts, logq, templates, random_state(animal, session, row.event_id, 0))
            checked += 1
        logq, templates = reference_library(model, 0.2)
        c = cal.loc[(cal.animal == animal) & (cal.session == session)]
        for row in c.loc[c.control_id.isin([0, 1, 38, 39])].itertuples():
            counts, namespace = control_counts(model, row, logq, templates)
            compare_scores(row, counts, logq, templates, random_state(animal, session, row.control_id, namespace))
            checked_cal += 1
        print(
            json.dumps({"animal": animal, "session": session, "counted_native_events": counted, "independent_native_fits": checked, "independent_control_fits": checked_cal}),
            flush=True,
        )
    summary = summarize(fitted)
    pd.testing.assert_frame_equal(summary, read_csv(out / "kleinman_native_session_summary.csv"), check_dtype=False, atol=1e-9, rtol=1e-9)
    grouped = (
        cal.groupby(["animal", "condition"])
        .agg(
            n_events=("control_id", "size"),
            flags=("deviance_flag", "sum"),
            alarm_fraction=("deviance_flag", "mean"),
            median_predictive_delta=("crossfit_template_minus_free_per_spike", "median"),
        )
        .reset_index()
    )
    pd.testing.assert_frame_equal(grouped, read_csv(out / "kleinman_native_control_summary.csv"), check_dtype=False, atol=1e-9, rtol=1e-9)
    assert read_csv(out / "kleinman_native_gates.csv").passed.all()
    result = {
        "status": "passed",
        "producer_commit": manifest["code_commit"],
        "manifest_sha256": file_sha256(out / "manifest.json"),
        "native_counts_rebuilt": counted,
        "independent_native_fits": checked,
        "independent_control_fits": checked_cal,
        "verifier_sha256": file_sha256(Path(__file__)),
        "scope": "All native event identities and encoding spike/bin counts reconstructed; all hashes and summaries; independent template and scalar likelihood/bootstrap checks on up to eight native events and eight controls per frozen session. RUN map estimation and source alignment reuse prior validated work; not every fit re-estimated.",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    run(p.parse_args())
