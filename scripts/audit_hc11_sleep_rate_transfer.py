#!/usr/bin/env python3
"""Frozen cross-event relative-rate recalibration of hc-11 sleep encoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import audit_hc11_count_conditioned_prediction as frozen
from _provenance import build_script_provenance, file_sha256

from hipporeplayimm.frozen_posterior_prediction import posterior_sha256

PARENT_DIGEST = "bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07"
REGIMES = {"run_original": None, "sleep_alpha100": 100.0, "sleep_alpha1000": 1000.0}
SEED = 20260908
GUARD = 1.0
CONTRASTS = {
    "imm_minus_iid": ("first_order_imm", "iid_position"),
    "imm_minus_static": ("first_order_imm", "static_location"),
    "diffusion_minus_iid": ("diffusion", "iid_position"),
    "imm_minus_global": ("first_order_imm", "global"),
    "iid_minus_global": ("iid_position", "global"),
}
KEYS = ["session", "rat", "phase", "event_id", "split", "encoding_variant"]


def folds(selected):
    selected = selected.sort_values(["start_time_s", "event_id"])
    if len(selected) != 20 or selected.event_id.duplicated().any():
        raise ValueError("each frozen session/phase must contain 20 unique events")
    intervals = selected[["start_time_s", "end_time_s"]].to_numpy(float)
    if not np.isfinite(intervals).all() or (intervals[:, 1] <= intervals[:, 0]).any():
        raise ValueError("invalid selected intervals")
    result = []
    for fold in (0, 1):
        test = selected.iloc[fold * 10 : (fold + 1) * 10]
        candidates = selected.iloc[(1 - fold) * 10 : (2 - fold) * 10]
        good = np.ones(len(candidates), dtype=bool)
        for row in test.itertuples(index=False):
            good &= (candidates.end_time_s.to_numpy() + GUARD <= row.start_time_s) | (candidates.start_time_s.to_numpy() >= row.end_time_s + GUARD)
        calibration = candidates[good]
        if calibration.empty:
            raise ValueError("empty guarded calibration fold")
        result.append((fold, test, calibration, candidates[~good]))
    return result


def calibrate(pooled_rates, calibration_counts, alpha):
    rates = np.asarray(pooled_rates, dtype=float)
    counts = np.asarray(calibration_counts)
    if rates.ndim != 2 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("positive finite rate matrix required")
    if counts.shape != (len(rates),) or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any() or counts.sum() == 0:
        raise ValueError("positive total integer calibration counts required")
    p0 = rates.mean(axis=1)
    p0 /= p0.sum()
    if alpha is None:
        return p0, np.ones(len(p0)), p0
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("strictly positive frozen pseudocount required")
    p = (counts + alpha * p0) / (counts.sum() + alpha)
    return p, p / p0, p0


def predict(counts, edges, rates, train, held, kernels, global_prob):
    if len(np.intersect1d(train, held)) or not len(train) or not len(held):
        raise ValueError("disjoint nonempty neural split required")
    durations = np.diff(edges)
    train_parts = [frozen.pf.likelihood_parts(counts[:, train], r[train], durations) for r in rates]
    posterior = {model: frozen.infer_training(train_parts, "count_conditioned", 1.0, model, kernels) for model in frozen.MODELS}
    before = {m: posterior_sha256(q) for m, q in posterior.items()}
    held_parts = [frozen.pf.likelihood_parts(counts[:, held], r[held], durations) for r in rates]
    scores = {f"score_{m}": frozen.heldout_score(q, held_parts, "count_conditioned") for m, q in posterior.items()}
    scores["score_global"] = float(frozen.pf.likelihood_parts(counts[:, held], global_prob[held, None], durations)["count_conditioned"].sum())
    if any(before[m] != posterior_sha256(posterior[m]) for m in posterior):
        raise ValueError("test spikes updated posterior")
    return scores, before


def score_session(task):
    session, selected, source, output = task
    source, output = Path(source), Path(output)
    rows, metadata, unit_rows = [], [], []
    with np.load(source / f"{session}_cache.npz") as z:
        cache = {k: z[k] for k in z.files}
    units = cache["unit_ids"]
    for phase, phase_events in selected.groupby("phase"):
        for fold, test, calibration, excluded in folds(phase_events):
            counts_cal = sum((cache[f"counts_{phase}_{int(i)}"].sum(axis=0) for i in calibration.event_id), start=np.zeros(len(units), dtype=int))
            metadata.append(
                {
                    "session": session,
                    "phase": phase,
                    "fold": fold,
                    "test_ids": ",".join(map(str, test.event_id)),
                    "calibration_ids": ",".join(map(str, calibration.event_id)),
                    "excluded_ids": ",".join(map(str, excluded.event_id)),
                    "n_calibration_spikes": int(counts_cal.sum()),
                    "test_start": test.start_time_s.min(),
                    "test_stop": test.end_time_s.max(),
                    "calibration_start": calibration.start_time_s.min(),
                    "calibration_stop": calibration.end_time_s.max(),
                }
            )
            for regime, alpha in REGIMES.items():
                p, gain, p0 = calibrate(cache["rates_pooled_0"], counts_cal, alpha)
                for index, uid in enumerate(units):
                    unit_rows.append(
                        {
                            "session": session,
                            "phase": phase,
                            "fold": fold,
                            "regime": regime,
                            "unit_id": uid,
                            "calibration_count": counts_cal[index],
                            "reference_probability": p0[index],
                            "calibrated_probability": p[index],
                            "gain": gain[index],
                        }
                    )
                for event in test.itertuples(index=False):
                    key = f"{phase}_{int(event.event_id)}"
                    counts, edges = cache[f"counts_{key}"], cache[f"edges_{key}"]
                    kernels = frozen.transitions(cache["centers"], edges, str(cache["topology"]), float(cache["track_length"]))
                    for variant in frozen.VARIANTS:
                        original = [cache[f"rates_{variant}_{d}"] for d in range(1 if variant == "pooled" else 2)]
                        for map_name in frozen.MAPS:
                            order = cache["permutation"] if map_name != "real" else np.arange(len(cache["centers"]))
                            rates = [r[:, order] * gain[:, None] for r in original]
                            for split in range(frozen.N_SPLITS):
                                train, held = cache[f"train_{split}"], cache[f"held_{split}"]
                                scores, hashes = predict(counts, edges, rates, train, held, kernels, p)
                                rows.append(
                                    {
                                        "session": session,
                                        "rat": event.animal,
                                        "phase": phase,
                                        "fold": fold,
                                        "event_id": event.event_id,
                                        "regime": regime,
                                        "encoding_variant": variant,
                                        "map": map_name,
                                        "split": split,
                                        **scores,
                                        "posterior_hashes": json.dumps(hashes, sort_keys=True),
                                        "train_cell_ids": ",".join(map(str, units[train])),
                                        "heldout_cell_ids": ",".join(map(str, units[held])),
                                        "n_spikes": counts.sum(),
                                        "n_train_spikes": counts[:, train].sum(),
                                        "n_heldout_spikes": counts[:, held].sum(),
                                        "heldout_used_for_inference": False,
                                        "posterior_unchanged": True,
                                        "status": "success",
                                    }
                                )
    pd.DataFrame(rows).to_csv(output / f"{session}_scores.csv", index=False)
    pd.DataFrame(unit_rows).to_csv(output / f"{session}_calibration.csv", index=False)
    pd.DataFrame(metadata).to_csv(output / f"{session}_folds.csv", index=False)
    return {"session": session, "rows": len(rows)}


def check_scores(scores, selection):
    if scores.empty or selection.empty:
        return False
    factors = {(r, v, m, s) for r in REGIMES for v in frozen.VARIANTS for m in frozen.MAPS for s in range(5)}
    keys = ["session", "phase", "event_id"]
    complete = all(set(g[["regime", "encoding_variant", "map", "split"]].itertuples(index=False, name=None)) == factors for _, g in scores.groupby(keys))
    values = scores[[f"score_{m}" for m in (*frozen.MODELS, "global")]].to_numpy(float)
    cells = scores[["train_cell_ids", "heldout_cell_ids"]].drop_duplicates()
    return bool(
        len(scores) == len(selection) * len(factors)
        and complete
        and not scores.duplicated(keys + ["regime", "encoding_variant", "map", "split"]).any()
        and set(scores[keys].itertuples(index=False, name=None)) == set(selection[keys].itertuples(index=False, name=None))
        and np.isfinite(values).all()
        and (values <= 1e-8).all()
        and scores.n_spikes.eq(scores.n_train_spikes + scores.n_heldout_spikes).all()
        and scores.status.eq("success").all()
        and scores.posterior_unchanged.eq(True).all()
        and scores.heldout_used_for_inference.eq(False).all()
        and all(not set(t.split(",")).intersection(h.split(",")) for t, h in cells.itertuples(index=False, name=None))
    )


def paired_contrasts(scores):
    real = scores[scores["map"].eq("real")].copy()
    records = []
    for name, (a, b) in CONTRASTS.items():
        records.append(real[KEYS + ["regime", "n_heldout_spikes"]].assign(contrast=name, delta=real[f"score_{a}"] - real[f"score_{b}"]))
    wrong = scores[scores["map"].eq("population_code_permuted")]
    both = real.merge(wrong, on=KEYS + ["regime"], suffixes=("_real", "_wrong"), validate="one_to_one")
    records.append(
        both[KEYS + ["regime"]].assign(
            n_heldout_spikes=both.n_heldout_spikes_real, contrast="real_minus_wrong_imm", delta=both.score_first_order_imm_real - both.score_first_order_imm_wrong
        )
    )
    base = real[real.regime.eq("run_original")]
    for regime in list(REGIMES)[1:]:
        matched = real[real.regime.eq(regime)].merge(base, on=KEYS, suffixes=("_new", "_old"), validate="one_to_one")
        for model in ("first_order_imm", "global"):
            records.append(
                matched[KEYS].assign(
                    regime=regime,
                    n_heldout_spikes=matched.n_heldout_spikes_new,
                    contrast=f"adaptation_gain_{model}",
                    delta=matched[f"score_{model}_new"] - matched[f"score_{model}_old"],
                )
            )
        interaction = (matched.score_first_order_imm_new - matched.score_iid_position_new) - (matched.score_first_order_imm_old - matched.score_iid_position_old)
        records.append(matched[KEYS].assign(regime=regime, n_heldout_spikes=matched.n_heldout_spikes_new, contrast="adaptation_temporal_interaction", delta=interaction))
    paired = pd.concat(records, ignore_index=True)
    paired["delta_per_heldout_spike"] = paired.delta / paired.n_heldout_spikes.replace(0, np.nan)
    keys = [k for k in KEYS if k != "split"] + ["regime", "contrast"]
    events = paired.groupby(keys, as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    return paired, events


def summarize(events):
    conditions = ["phase", "regime", "encoding_variant", "contrast"]
    session = events.groupby(conditions + ["rat", "session"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    animal = session.groupby(conditions + ["rat"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    rows = []
    for key, g in animal.groupby(conditions):
        values = g.sort_values("rat").delta.to_numpy()
        rng = np.random.default_rng(SEED)
        replicates = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        lo, hi = np.quantile(replicates, [0.025, 0.975])
        rows.append(
            dict(zip(conditions, key, strict=True))
            | {
                "mean": values.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "positive_animals": int((values > 0).sum()),
                "animals": len(values),
                "mean_per_heldout_spike": g.delta_per_heldout_spike.mean(),
            }
        )
    return pd.DataFrame(rows), animal, session


def run(args):
    source, out = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    parent_path = source / "hc11_conditional_manifest.json"
    if file_sha256(parent_path) != PARENT_DIGEST:
        raise ValueError("unexpected parent manifest")
    parent = json.loads(parent_path.read_text())
    selection = pd.read_csv(source / "frozen_selection.csv")
    frozen.validate_selection(selection, full_cohort=True)
    for name in ["frozen_selection.csv"] + [f"{s['session']}_cache.npz" for s in parent["sessions"]]:
        if file_sha256(source / name) != parent["output_sha256"][name]:
            raise ValueError("parent source changed")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite run")
    out.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / "docs/hc11_sleep_rate_transfer_protocol.md"
    manifest = build_script_provenance(input_paths={"parent_manifest": parent_path, "protocol": protocol})
    manifest.update(
        status="running",
        source_dir=str(source),
        regimes=REGIMES,
        guard_s=GUARD,
        seed=SEED,
        inference_temperature=1,
        heldout_temperature=1,
        claim_boundary="exploratory cross-event observation-transfer diagnostic; no mechanism or replication claim",
    )
    path = out / "sleep_rate_transfer_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        tasks = [(s, g, source, out) for s, g in selection.groupby("session")]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(score_session, t) for t in tasks]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        scores = pd.concat([pd.read_csv(out / f"{s}_scores.csv") for s in selection.session.unique()], ignore_index=True)
        passed = check_scores(scores, selection)
        pd.DataFrame([{"gate": "complete_proper_predictive_scores", "passed": passed}, {"gate": "overall", "passed": passed}]).to_csv(
            out / "sleep_rate_transfer_gates.csv", index=False
        )
        if not passed:
            raise ValueError("technical gates failed")
        paired, events = paired_contrasts(scores)
        summary, animal, session = summarize(events)
        for name, table in (("split_contrasts", paired), ("event_contrasts", events), ("summary", summary), ("by_animal", animal), ("by_session", session)):
            table.to_csv(out / f"sleep_rate_transfer_{name}.csv", index=False)
        primary = summary[summary.phase.eq("POST") & summary.regime.eq("sleep_alpha100") & summary.encoding_variant.eq("direction_mixture")].set_index("contrast")
        needed = ["imm_minus_iid", "imm_minus_static", "real_minus_wrong_imm"]
        promising = (primary.loc[needed, "ci_low"] > 0).all() and (primary.loc[needed, "positive_animals"] == 4).all()
        manifest.update(
            status="complete",
            elapsed_s=time.monotonic() - started,
            rows=len(scores),
            events=len(selection),
            decision="diagnostic_transfer_lead_requires_replication" if promising else "external_temporal_advantage_remains_unsupported",
        )
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != path}
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-20260908")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
