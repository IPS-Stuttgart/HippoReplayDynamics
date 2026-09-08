#!/usr/bin/env python3
"""Independent constraints, forecasts and paired-summary reconstruction."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.verify_2d_lagged_neural_prediction import ID, KEY, KINDS, Reference, assert_table, likelihood, spatial_kernels


def null_step(q, parameters):
    pi, mode, stay, u, v = [parameters[k] for k in ("pi", "mode", "stay", "u", "v")]
    m, n = pi.shape
    shaped = np.asarray(q).reshape(-1, m, n)
    weights = u.reshape(m, n, m)
    result = np.zeros_like(shaped)
    for source in range(m):
        for dest in range(m):
            flow = shaped[:, source] / pi[source] * weights[source, :, dest]
            result[:, dest] += shaped[:, source] * mode[source, dest] * stay[dest]
            result[:, dest] += (flow.sum(axis=1, keepdims=True) - flow) * v[dest]
    assert result.min() > -1e-12
    return np.maximum(result, 0).reshape(np.shape(q))


def verify_parameters(parameters, original):
    pi, mode, stay, u, v = [parameters[k] for k in ("pi", "mode", "stay", "u", "v")]
    m, n = pi.shape
    assert all(np.isfinite(a).all() and (a >= 0).all() for a in (pi, mode, stay, u, v))
    assert (pi > 0).all() and abs(pi.sum() - 1) < 1e-12
    np.testing.assert_allclose(original.step(pi.ravel()), pi.ravel(), atol=1e-12, rtol=0)
    if original.neural:
        np.testing.assert_allclose(stay[0], np.diag(original.a), atol=1e-12)
    else:
        np.testing.assert_allclose(mode, original.mode, atol=1e-12)
        for dest, k in enumerate(original.kernels):
            expected = np.full(n, 1 / n) if k is None else k.diagonal()
            np.testing.assert_allclose(stay[dest], expected, atol=1e-12)
    total_rows = np.zeros(m * n)
    for i in range(m * n):
        for dest in range(m):
            remaining = u[i, dest] / pi.ravel()[i] * (v[dest].sum() - v[dest, i % n])
            fixed = mode[i // n, dest] * stay[dest, i % n]
            assert abs(fixed + remaining - mode[i // n, dest]) < 1e-10
            total_rows[i] += fixed + remaining
    np.testing.assert_allclose(total_rows, 1, atol=1e-10, rtol=0)
    equilibrium = float(np.max(np.abs(null_step(pi.ravel(), parameters) - pi.ravel())))
    assert equilibrium < 1e-11
    return equilibrium


def null_forecast_scores(x, rates, train, held, original, parameters, h):
    ll = likelihood(x[:, train], rates[train])
    q = original.initial.copy()
    filtered = []
    for t, e in enumerate(ll):
        if t:
            q = original.step(q)
        q *= np.tile(np.exp(e - e.max()), original.modes)
        q /= q.sum()
        filtered.append(q.copy())
    forecast = np.array(filtered[:-h])
    for _ in range(h):
        forecast = null_step(forecast, parameters)
    p = np.array([original.marginal(q) for q in forecast])
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
    target = likelihood(x[h:, held], rates[held])
    with np.errstate(divide="ignore"):
        values = logsumexp(np.log(p) + target, axis=1)
    values[x[h:, held].sum(axis=1) == 0] = 0
    return float(values.sum())


def reconstruct(root, rows, old_summary):
    models = {name: rows[rows.model.eq(name)].set_index(KEY).sort_index() for name in KINDS}
    template = models[KINDS[0]][["n_target_bins", "n_heldout_target_spikes"]]
    contrasts = {}
    for name, g in models.items():
        assert g.index.equals(template.index)
        contrasts[name + "__dynamic_minus_matched"] = g.score_dynamic - g.score_occupancy_dwell
        contrasts[name + "__old_dwell_minus_matched"] = g.score_dwell_only - g.score_occupancy_dwell
    for kind in ("imm", "diffusion"):
        real, wrong = models[f"spatial_{kind}_real"], models[f"spatial_{kind}_permuted"]
        contrasts[f"spatial_{kind}__matched_map_interaction"] = real.score_dynamic - real.score_occupancy_dwell - wrong.score_dynamic + wrong.score_occupancy_dwell
    parts = []
    for name, delta in contrasts.items():
        p = template.copy()
        p["contrast"], p["delta"] = name, delta
        p["delta_per_spike"] = p.delta / p.n_heldout_target_spikes.where(p.n_heldout_target_spikes > 0)
        parts.append(p.reset_index())
    splits = pd.concat(parts, ignore_index=True)
    assert_table(splits, root / "occupancy_matched_splits.csv.gz", KEY + ["contrast"], ["delta", "delta_per_spike", "n_target_bins", "n_heldout_target_spikes"])
    ek = ID + ["event_id", "horizon", "contrast"]
    event = splits.groupby(ek)[["delta", "delta_per_spike"]].median().reset_index()
    event["valid_neural_splits"] = splits.groupby(ek).delta_per_spike.count().to_numpy()
    assert_table(event, root / "occupancy_matched_events.csv.gz", ek, ["delta", "delta_per_spike", "valid_neural_splits"])
    session = event.groupby(ID + ["horizon", "contrast"])[["delta", "delta_per_spike"]].mean().reset_index()
    assert_table(session, root / "occupancy_matched_sessions.csv.gz", ID + ["horizon", "contrast"], ["delta", "delta_per_spike"])
    ak = ["dataset", "animal", "horizon", "contrast"]
    animal = session.groupby(ak)[["delta", "delta_per_spike"]].mean().reset_index()
    assert_table(animal, root / "occupancy_matched_animals.csv.gz", ak, ["delta", "delta_per_spike"])
    summary = []
    for key, g in animal.groupby(["dataset", "horizon", "contrast"]):
        for metric in ("delta", "delta_per_spike"):
            v = g.sort_values("animal")[metric].to_numpy()
            assert np.isfinite(v).all()
            bootstrap = [np.mean(v[list(i)]) for i in itertools.product(range(len(v)), repeat=len(v))]
            ci = np.percentile(bootstrap, [2.5, 97.5])
            summary.append(
                {
                    "dataset": key[0],
                    "horizon": key[1],
                    "contrast": key[2],
                    "metric": metric,
                    "mean": v.mean(),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "positive_animals": int((v > 0).sum()),
                    "animals": len(v),
                }
            )
    summary = pd.DataFrame(summary)
    assert_table(summary, root / "occupancy_matched_summary.csv.gz", ["dataset", "horizon", "contrast", "metric"], ["mean", "ci_low", "ci_high", "positive_animals", "animals"])
    decisions = pd.read_csv(root / "occupancy_matched_decisions.csv").set_index("model")
    assert set(decisions.index) == {"spatial_imm_real", "spatial_diffusion_real", "learned_hmm"}
    main = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike")]
    old = old_summary[old_summary.horizon.eq(2) & old_summary.metric.eq("delta_per_spike")]
    for name, row in decisions.iterrows():
        p = main[main.contrast.eq(name + "__dynamic_minus_matched")]
        matched = len(p) == 2 and p.ci_low.gt(0).all() and p.positive_animals.eq(p.animals).all()
        p = old[old.contrast.isin([name + "__dynamic_minus_" + b for b in ("frozen", "no_history", "global")])]
        others = len(p) == 6 and p.ci_low.gt(0).all() and p.positive_animals.eq(p.animals).all()
        assert row.matched_destination_structure_lead == matched and row.other_original_controls_pass == others
        assert row.all_updated_controls_pass == (matched and others)
        assert not row.original_compound_verdict_changed and not row.independent_confirmation and not row.high_importance_discovery_established
    return len(splits), len(event)


def verify(root, output):
    manifest_path = root / "occupancy_matched_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete" and not manifest["git_dirty"]
    for name, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][name], name
    for name, digest in manifest["output_sha256"].items():
        assert file_sha256(root / name) == digest, name
    lagged = Path(manifest["input_file_paths"]["lagged_manifest"]).parent
    parent_manifest = json.loads((lagged / "lagged_prediction_manifest.json").read_text())
    parent, rate, learned = [Path(parent_manifest["input_file_paths"][k]).parent for k in ("parent", "rate", "learned")]
    frames = []
    checked = 0
    error = 0.0
    equilibrium = 0.0
    parameter_sets = 0
    for item in manifest["completed"]:
        tag = item["tag"]
        old = pd.read_csv(lagged / f"{tag}_forecast_scores.csv.gz")
        new = pd.read_csv(root / f"{tag}_matched_scores.csv.gz")
        keys = ["event_id", "split", "horizon", "model"]
        assert not new.duplicated(keys).any() and len(new) == len(old)
        a, b = new.set_index(keys).sort_index(), old.set_index(keys).sort_index()
        pd.testing.assert_frame_equal(a[b.columns], b, check_exact=False, rtol=1e-12, atol=1e-12)
        enough = new.status.eq("scored")
        assert new.loc[enough, "original_forecast_hash_matches"].all()
        assert new.loc[enough, "original_score_reconstruction_error"].le(1e-8).all()
        assert np.isfinite(new.loc[enough, "score_occupancy_dwell"]).all()
        assert new.loc[~enough, "score_occupancy_dwell"].isna().all()
        assert new.loc[enough, "score_occupancy_dwell"].le(1e-8).all()
        np.testing.assert_allclose(new.loc[enough & new.n_heldout_target_spikes.eq(0), "score_occupancy_dwell"], 0, atol=1e-12)
        cache = dict(np.load(parent / f"{tag}_cache.npz"))
        kernels = spatial_kernels(cache["centers"])
        spatial = {kind: Reference(kernels, imm=kind == "imm") for kind in ("imm", "diffusion")}
        params = {kind: dict(np.load(root / f"{tag}__{kind}_matched_null.npz")) for kind in spatial}
        for kind, op in spatial.items():
            equilibrium = max(equilibrium, verify_parameters(params[kind], op))
            parameter_sets += 1
        gains = pd.read_csv(rate / f"{tag}_gains.csv")
        folds = json.loads((lagged / f"{tag}_folds.json").read_text())
        lookup = new.set_index(keys).sort_index()
        for f in folds:
            fold = f["fold"]
            fit = dict(np.load(learned / f"{tag}__fold{fold}__k50_fit.npz"))
            neural = Reference(fit=fit)
            neural_params = dict(np.load(root / f"{tag}__fold{fold}_neural_matched_null.npz"))
            equilibrium = max(equilibrium, verify_parameters(neural_params, neural))
            parameter_sets += 1
            g = gains[gains.fold.eq(fold) & gains.alpha.eq(100)].set_index("unit_id")
            maps = cache["rates"] * g.loc[cache["unit_ids"], "gain"].to_numpy()[:, None]
            eid = int(f["test_ids"][0])
            x = cache[f"counts_{eid}"]
            if abs(np.diff(cache[f"edges_{eid}"])[-1] - 0.02) >= 1e-8:
                x = x[:-1]
            for split in range(5):
                train, held = cache[f"train_{split}"], cache[f"held_{split}"]
                for h in (1, 2, 4):
                    if h >= len(x):
                        continue
                    for model in KINDS:
                        if model == "learned_hmm":
                            op, p, rates = neural, neural_params, fit["probabilities"]
                        else:
                            kind = "imm" if model.startswith("spatial_imm") else "diffusion"
                            op, p = spatial[kind], params[kind]
                            rates = maps[:, cache["permutation"]] if model.endswith("permuted") else maps
                        score = null_forecast_scores(x, rates, train, held, op, p, h)
                        saved = lookup.loc[(eid, split, h, model), "score_occupancy_dwell"]
                        assert abs(score - saved) < 1e-8
                        error = max(error, abs(score - saved))
                        checked += 1
        frames.append(new)
        print(json.dumps({"tag": tag, "independent_scores": checked, "max_score_error": error, "max_equilibrium_error": equilibrium}), flush=True)
    rows = pd.concat(frames, ignore_index=True)
    assert len(rows) == 9225 * 75 and len(frames) == 33 and parameter_sets == 231
    split_count, event_count = reconstruct(root, rows, pd.read_csv(lagged / "lagged_prediction_summary.csv.gz"))
    provenance = build_script_provenance(
        input_paths={"run_manifest": manifest_path, "verifier": Path(__file__), "reference_filter": ROOT / "scripts/verify_2d_lagged_neural_prediction.py"}, cwd=ROOT
    )
    provenance.update(
        status="pass",
        independent_scores=checked,
        max_score_error=error,
        max_equilibrium_error=equilibrium,
        parameter_sets=parameter_sets,
        split_contrasts=split_count,
        event_contrasts=event_count,
        scope="All hashes, original rows, null constraints and aggregates checked. First event in each fold, all splits/horizons/models independently forecast. Native data not reopened; original model fits not repeated.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = verify(a.run_dir, a.output)
    print(json.dumps({k: v for k, v in r.items() if k not in ("input_file_paths", "input_file_sha256")}), flush=True)
