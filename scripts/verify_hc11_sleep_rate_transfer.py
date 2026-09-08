#!/usr/bin/env python3
"""Independent rate, score, fold and aggregate checks for sleep recalibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import verify_hc11_count_conditioned_prediction as independent
from _provenance import build_script_provenance, file_sha256


def assert_equal(actual, expected, name):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, atol=1e-9, rtol=1e-10, equal_nan=True):
        raise ValueError(f"{name} mismatch")


def reconstruct_gain(rates, calibration_counts, regime):
    reference = rates.sum(axis=1) / rates.sum()
    if regime == "run_original":
        return reference, np.ones(len(reference)), reference
    strength = {"sleep_alpha100": 100, "sleep_alpha1000": 1000}[regime]
    weights = calibration_counts.astype(float) + reference * strength
    probability = weights / weights.sum()
    return probability, probability / reference, reference


def global_score(counts, probabilities):
    p = probabilities / probabilities.sum()
    return float((counts @ np.log(p) + gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)).sum())


def verify_aggregates(scores, root):
    keys = ["session", "rat", "phase", "event_id", "split", "encoding_variant"]
    wide = scores.pivot(index=keys, columns=["regime", "map"], values=["score_first_order_imm", "score_iid_position", "score_static_location", "score_diffusion", "score_global"])
    counts = scores.groupby(keys).n_heldout_spikes.first()
    rows = []
    for regime in ("run_original", "sleep_alpha100", "sleep_alpha1000"):

        def value(model, map_name="real", r=regime):
            return wide[(f"score_{model}", r, map_name)]

        contrasts = {
            "imm_minus_iid": value("first_order_imm") - value("iid_position"),
            "imm_minus_static": value("first_order_imm") - value("static_location"),
            "diffusion_minus_iid": value("diffusion") - value("iid_position"),
            "imm_minus_global": value("first_order_imm") - value("global"),
            "iid_minus_global": value("iid_position") - value("global"),
            "real_minus_wrong_imm": value("first_order_imm") - value("first_order_imm", "population_code_permuted"),
        }
        if regime != "run_original":
            for model in ("first_order_imm", "global"):
                contrasts[f"adaptation_gain_{model}"] = value(model) - value(model, r="run_original")
            contrasts["adaptation_temporal_interaction"] = contrasts["imm_minus_iid"] - (value("first_order_imm", r="run_original") - value("iid_position", r="run_original"))
        for name, delta in contrasts.items():
            part = delta.rename("delta").to_frame()
            part["delta_per_heldout_spike"] = delta / counts.replace(0, np.nan)
            rows.append(part.reset_index().assign(regime=regime, contrast=name))
    split = pd.concat(rows, ignore_index=True)
    skeys = keys + ["regime", "contrast"]
    stored = pd.read_csv(root / "sleep_rate_transfer_split_contrasts.csv")
    joined = split.merge(stored, on=skeys, suffixes=("_check", ""), validate="one_to_one")
    if len(joined) != len(split) or len(joined) != len(stored):
        raise ValueError("missing split contrasts")
    assert_equal(joined.delta_check, joined.delta, "split contrasts")
    assert_equal(joined.delta_per_heldout_spike_check, joined.delta_per_heldout_spike, "normalized split contrasts")
    ekeys = [k for k in skeys if k != "split"]
    event = split.groupby(ekeys, as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    saved = pd.read_csv(root / "sleep_rate_transfer_event_contrasts.csv")
    ee = event.merge(saved, on=ekeys, suffixes=("_check", ""), validate="one_to_one")
    if len(ee) != len(event) or len(saved) != len(event):
        raise ValueError("missing event medians")
    assert_equal(ee.delta_check, ee.delta, "event medians")
    assert_equal(ee.delta_per_heldout_spike_check, ee.delta_per_heldout_spike, "normalized event medians")
    group_keys = ["phase", "regime", "encoding_variant", "contrast"]
    sessions = event.groupby(group_keys + ["rat", "session"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    animals = sessions.groupby(group_keys + ["rat"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    saved = pd.read_csv(root / "sleep_rate_transfer_by_animal.csv")
    aa = animals.merge(saved, on=group_keys + ["rat"], suffixes=("_check", ""), validate="one_to_one")
    if len(aa) != len(animals) or len(saved) != len(animals):
        raise ValueError("missing animal rows")
    assert_equal(aa.delta_check, aa.delta, "animal contrasts")
    summary = pd.read_csv(root / "sleep_rate_transfer_summary.csv").set_index(group_keys)
    errors = []
    for key, g in animals.groupby(group_keys):
        values = g.sort_values("rat").delta.to_numpy()
        if len(values) != 4:
            raise ValueError("missing animals")
        draws = np.random.default_rng(20260908).integers(0, 4, size=(5000, 4))
        ci = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
        expected = [values.mean(), *ci, (values > 0).sum(), 4, g.delta_per_heldout_spike.mean()]
        observed = summary.loc[key, ["mean", "ci_low", "ci_high", "positive_animals", "animals", "mean_per_heldout_spike"]].to_numpy(float)
        assert_equal(observed, expected, "aggregate / interval")
        errors.append(float(np.max(np.abs(observed - expected))))
    if len(errors) != len(summary):
        raise ValueError("extra summary rows")
    return {"split_contrasts": len(split), "event_contrasts": len(event), "aggregate_panels": len(errors), "max_aggregate_error": max(errors)}


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    manifest_path = root / "sleep_rate_transfer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("run incomplete")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed output: {name}")
    source = Path(manifest["source_dir"])
    parent_path = source / "hc11_conditional_manifest.json"
    if file_sha256(parent_path) != manifest["input_file_sha256"]["parent_manifest"]:
        raise ValueError("changed parent manifest")
    parent = json.loads(parent_path.read_text())
    selected = pd.read_csv(source / "frozen_selection.csv")
    if len(selected) != 320 or selected.session.nunique() != 8 or selected.animal.nunique() != 4:
        raise ValueError("incomplete frozen cohort")
    scored = []
    audit_rows = []
    global_errors = []
    raw_events = 0
    for item in parent["sessions"]:
        session = item["session"]
        cache_path = source / f"{session}_cache.npz"
        if file_sha256(cache_path) != parent["output_sha256"][cache_path.name]:
            raise ValueError("changed parent cache")
        with np.load(cache_path) as loaded:
            z = {k: loaded[k] for k in loaded.files}
        scores = pd.read_csv(root / f"{session}_scores.csv")
        scores["session"] = session
        scored.append(scores)
        all_cal = pd.read_csv(root / f"{session}_calibration.csv")
        metadata = pd.read_csv(root / f"{session}_folds.csv")
        spike_path = next(Path(p) for p in item["source_hashes"] if p.endswith("spikes.cellinfo.mat"))
        if file_sha256(spike_path) != item["source_hashes"][str(spike_path)]:
            raise ValueError("changed raw spikes")
        raw = loadmat(spike_path, squeeze_me=True, struct_as_record=False)["spikes"]
        by_unit = {int(i): np.asarray(t, float).ravel() for i, t in zip(np.asarray(raw.UID).ravel(), np.asarray(raw.times, object).ravel(), strict=True)}
        local = selected[selected.session.eq(session)]
        for e in local.itertuples(index=False):
            key = f"{e.phase}_{e.event_id}"
            edges = z[f"edges_{key}"]
            counts = np.column_stack([np.histogram(v[(v >= edges[0]) & (v < edges[-1])], bins=edges)[0] for v in (by_unit[int(u)] for u in z["unit_ids"])])
            assert_equal(counts, z[f"counts_{key}"], "raw counts")
            raw_events += 1
        for phase, events in local.groupby("phase"):
            events = events.sort_values(["start_time_s", "event_id"])
            for fold in (0, 1):
                test = events.iloc[10 * fold : 10 * (fold + 1)]
                candidates = events.iloc[10 * (1 - fold) : 10 * (2 - fold)]
                cal_ids, excluded = [], []
                for c in candidates.itertuples(index=False):
                    separated = all(c.end_time_s + 1 <= t.start_time_s or t.end_time_s + 1 <= c.start_time_s for t in test.itertuples(index=False))
                    (cal_ids if separated else excluded).append(c.event_id)
                if not cal_ids:
                    raise ValueError("empty calibration")
                meta = metadata[metadata.phase.eq(phase) & metadata.fold.eq(fold)]
                if len(meta) != 1:
                    raise ValueError("missing fold metadata")

                def ids(value):
                    return [] if pd.isna(value) else list(map(int, str(value).split(",")))

                if ids(meta.iloc[0].test_ids) != list(test.event_id) or ids(meta.iloc[0].calibration_ids) != cal_ids or ids(meta.iloc[0].excluded_ids) != excluded:
                    raise ValueError("calibration/test exclusion mismatch")
                totals = np.vstack([z[f"counts_{phase}_{i}"].sum(axis=0) for i in cal_ids]).sum(axis=0)
                for regime in ("run_original", "sleep_alpha100", "sleep_alpha1000"):
                    probability, gain, reference = reconstruct_gain(z["rates_pooled_0"], totals, regime)
                    saved_cal = all_cal[all_cal.phase.eq(phase) & all_cal.fold.eq(fold) & all_cal.regime.eq(regime)].set_index("unit_id").loc[z["unit_ids"]]
                    assert_equal(saved_cal.calibration_count, totals, "calibration counts")
                    assert_equal(saved_cal.reference_probability, reference, "reference probability")
                    assert_equal(saved_cal.calibrated_probability, probability, "recalibration probability")
                    assert_equal(saved_cal.gain, gain, "cell gains")
                    for e in test.itertuples(index=False):
                        key = f"{phase}_{e.event_id}"
                        counts, edges = z[f"counts_{key}"], z[f"edges_{key}"]
                        subset = scores[scores.phase.eq(phase) & scores.event_id.eq(e.event_id) & scores.regime.eq(regime)]
                        if len(subset) != 20:
                            raise ValueError("incomplete event factors")
                        for split in range(5):
                            held = z[f"held_{split}"]
                            expected = global_score(counts[:, held], probability[held])
                            observed = subset[subset.split.eq(split)].score_global.to_numpy()
                            assert_equal(observed, np.repeat(expected, len(observed)), "global baseline")
                            global_errors.extend(np.abs(observed - expected).tolist())
                        if e.event_id != test.iloc[0].event_id:
                            continue
                        kernels = independent.dense_kernels(z["centers"], edges, str(z["topology"]), float(z["track_length"]))
                        for row in subset[subset.split.isin([0, 4])].itertuples(index=False):
                            train, held = z[f"train_{row.split}"], z[f"held_{row.split}"]
                            if set(train) & set(held):
                                raise ValueError("overlapping neuron IDs")
                            order = z["permutation"] if row.map != "real" else np.arange(len(z["centers"]))
                            rates = [z[f"rates_{row.encoding_variant}_{d}"][:, order] * gain[:, None] for d in range(1 if row.encoding_variant == "pooled" else 2)]
                            tr = [independent.direct_parts(counts[:, train], r[train], np.diff(edges)) for r in rates]
                            he = [independent.direct_parts(counts[:, held], r[held], np.diff(edges)) for r in rates]
                            for model in ("iid_position", "static_location", "diffusion", "first_order_imm"):
                                actual = independent.direct_predict(tr, he, model, "count_conditioned", 1.0, kernels)["count_conditioned"]
                                stored = getattr(row, f"score_{model}")
                                assert_equal(actual, stored, "independent prediction")
                                audit_rows.append(
                                    {
                                        "session": session,
                                        "phase": phase,
                                        "fold": fold,
                                        "regime": regime,
                                        "event_id": e.event_id,
                                        "split": row.split,
                                        "map": row.map,
                                        "encoding_variant": row.encoding_variant,
                                        "model": model,
                                        "absolute_error": abs(actual - stored),
                                    }
                                )
        print(f"audited {session}", flush=True)
    scores = pd.concat(scored, ignore_index=True)
    expected_keys = ["session", "phase", "event_id", "regime", "encoding_variant", "map", "split"]
    if len(scores) != 19200 or scores.duplicated(expected_keys).any() or scores.heldout_used_for_inference.any() or not scores.posterior_unchanged.eq(True).all():
        raise ValueError("incomplete or contaminated prediction")
    original = pd.read_csv(source / "hc11_conditional_event_scores.csv")
    original = original[original.inference_temperature.eq(1) & original.observation.eq("count_conditioned")]
    old_keys = ["session", "phase", "event_index", "split", "encoding_variant", "map", "model"]
    current = scores[scores.regime.eq("run_original")].rename(columns={"event_id": "event_index"})
    melted = current.melt(
        id_vars=old_keys[:-1], value_vars=[f"score_{m}" for m in ("iid_position", "static_location", "diffusion", "first_order_imm")], var_name="model", value_name="current_score"
    )
    melted["model"] = melted.model.str.removeprefix("score_")
    joined = melted.merge(original, on=old_keys, validate="one_to_one")
    if len(joined) != 25600 or len(original) != len(joined):
        raise ValueError("missing original regression rows")
    assert_equal(joined.current_score, joined.conditional_heldout_log_score, "original regression")
    aggregated = verify_aggregates(scores, root)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite audit")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(out / "independent_predictive_audit.csv", index=False)
    provenance = build_script_provenance(input_paths={"run_manifest": manifest_path})
    provenance.update(
        status="passed",
        raw_events=raw_events,
        independent_scores=len(audit_rows),
        max_predictive_error=max(r["absolute_error"] for r in audit_rows),
        global_scores=len(global_errors),
        max_global_error=max(global_errors),
        original_score_regressions=len(joined),
        **aggregated,
        scope="All raw counts and recalibration/fold exclusions; all global predictions and original-model regression; 3072 separate dense neural predictions; all event contrasts and animal-only bootstrap intervals. Parent RUN maps are hash-pinned and previously audited, not newly fitted here.",
    )
    (out / "audit_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
