#!/usr/bin/env python3
"""Independent time-reversal forecasts, constraints and summary reconstruction."""

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


def reverse(q, original, pi):
    weighted = q / pi
    if original.neural:
        return (original.a @ weighted) * pi
    values = weighted.reshape(original.modes, original.n)
    output = np.zeros_like(values)
    for source in range(original.modes):
        for dest, kernel in enumerate(original.kernels):
            x = values[dest]
            back = np.full(original.n, x.sum() / original.n) if kernel is None else kernel.T @ x
            output[source] += original.mode[source, dest] * back
    return output.ravel() * pi


def constraints(original, pi):
    assert np.isfinite(pi).all() and (pi > 0).all() and abs(pi.sum() - 1) < 1e-12
    np.testing.assert_allclose(original.step(pi), pi, atol=1e-12, rtol=0)
    np.testing.assert_allclose(reverse(pi, original, pi), pi, atol=1e-12, rtol=0)
    if original.neural:
        a = original.a
        r = np.array([[pi[j] * a[j, i] / pi[i] for j in range(len(pi))] for i in range(len(pi))])
        b = (a + r) / 2
        np.testing.assert_allclose(r.sum(axis=1), 1, atol=1e-10, rtol=0)
        np.testing.assert_allclose(pi @ b, pi, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.diag(b), np.diag(a), atol=1e-12, rtol=0)
        f = pi[:, None] * a
        fb = pi[:, None] * b
        np.testing.assert_allclose(fb, fb.T, atol=1e-12, rtol=0)
        np.testing.assert_allclose(fb + fb.T, f + f.T, atol=1e-12, rtol=0)
        for i in range(len(pi)):
            np.testing.assert_allclose(reverse(np.eye(len(pi))[i], original, pi), r[i], atol=1e-12, rtol=0)
    else:
        # Check every position-pair flux in separate mode blocks, avoiding S*S storage.
        p = pi.reshape(original.modes, original.n)
        blocks = [np.full((original.n, original.n), 1 / original.n) if k is None else k.toarray().T for k in original.kernels]
        row_sums = np.zeros_like(p)
        for m in range(original.modes):
            for n in range(original.modes):
                f = p[m, :, None] * original.mode[m, n] * blocks[n]
                backward = (p[n, :, None] * original.mode[n, m] * blocks[m]).T
                r = backward / p[m, :, None]
                bflux = (f + backward) / 2
                counterpart = ((p[n, :, None] * original.mode[n, m] * blocks[m]) + f.T) / 2
                np.testing.assert_allclose(bflux, counterpart.T, atol=1e-12, rtol=0)
                row_sums[m] += r.sum(axis=1)
                if m == n:
                    np.testing.assert_allclose(np.diag(r), np.diag(original.mode[m, n] * blocks[n]), atol=1e-12)
                for x in np.unique(np.linspace(0, original.n - 1, min(16, original.n), dtype=int)):
                    basis = np.zeros(len(pi))
                    basis[m * original.n + x] = 1
                    ref = reverse(basis, original, pi).reshape(original.modes, original.n)[n]
                    np.testing.assert_allclose(ref, r[x], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row_sums, 1, atol=1e-10, rtol=0)
        if original.modes == 1:
            rng = np.random.default_rng(7)
            q = rng.dirichlet(np.ones(len(pi)))
            np.testing.assert_allclose(reverse(q, original, pi), original.step(q), atol=1e-11, rtol=0)


def scores(x, rates, train, held, op, pi, h):
    ll = likelihood(x[:, train], rates[train])
    q = op.initial.copy()
    filtered = []
    for t, emission in enumerate(ll):
        if t:
            q = op.step(q)
        q *= np.tile(np.exp(emission - emission.max()), op.modes)
        q /= q.sum()
        filtered.append(q.copy())
    target = likelihood(x[h:, held], rates[held])
    output = {}
    for condition in ("reverse", "reversible"):
        probabilities = []
        for origin in filtered[:-h]:
            q = origin.copy()
            for _ in range(h):
                r = reverse(q, op, pi)
                q = r if condition == "reverse" else (op.step(q) + r) / 2
            probabilities.append(op.marginal(q))
        p = np.array(probabilities)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
        with np.errstate(divide="ignore"):
            values = logsumexp(np.log(p) + target, axis=1)
        values[x[h:, held].sum(axis=1) == 0] = 0
        output["score_" + condition] = float(values.sum())
    return output


def reconstruct(root, rows):
    contrasts = {
        "dynamic_minus_reversible": ("score_dynamic", "score_reversible"),
        "dynamic_minus_reverse": ("score_dynamic", "score_reverse"),
        "reversible_minus_matched": ("score_reversible", "score_occupancy_dwell"),
    }
    parts = []
    for model in KINDS:
        frame = rows[rows.model.eq(model)].set_index(KEY).sort_index()
        for contrast, (a, b) in contrasts.items():
            p = frame[["n_heldout_target_spikes"]].copy()
            p["contrast"], p["delta"] = model + "__" + contrast, frame[a] - frame[b]
            p["delta_per_spike"] = p.delta / p.n_heldout_target_spikes.where(p.n_heldout_target_spikes > 0)
            parts.append(p.reset_index())
    splits = pd.concat(parts, ignore_index=True)
    assert_table(splits, root / "reversible_forecasts_splits.csv.gz", KEY + ["contrast"], ["delta", "delta_per_spike"])
    ek = ID + ["event_id", "horizon", "contrast"]
    events = splits.groupby(ek)[["delta", "delta_per_spike"]].median().reset_index()
    events["valid_neural_splits"] = splits.groupby(ek).delta_per_spike.count().to_numpy()
    assert_table(events, root / "reversible_forecasts_events.csv.gz", ek, ["delta", "delta_per_spike", "valid_neural_splits"])
    sessions = events.groupby(ID + ["horizon", "contrast"])[["delta", "delta_per_spike"]].mean().reset_index()
    assert_table(sessions, root / "reversible_forecasts_sessions.csv.gz", ID + ["horizon", "contrast"], ["delta", "delta_per_spike"])
    ak = ["dataset", "animal", "horizon", "contrast"]
    animals = sessions.groupby(ak)[["delta", "delta_per_spike"]].mean().reset_index()
    assert_table(animals, root / "reversible_forecasts_animals.csv.gz", ak, ["delta", "delta_per_spike"])
    summary = []
    for key, g in animals.groupby(["dataset", "horizon", "contrast"]):
        for metric in ("delta", "delta_per_spike"):
            v = g.sort_values("animal")[metric].to_numpy()
            draws = [np.mean(v[list(ix)]) for ix in itertools.product(range(len(v)), repeat=len(v))]
            ci = np.percentile(draws, [2.5, 97.5])
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
    assert_table(summary, root / "reversible_forecasts_summary.csv.gz", ["dataset", "horizon", "contrast", "metric"], ["mean", "ci_low", "ci_high", "positive_animals", "animals"])
    g = summary[
        summary.horizon.eq(2) & summary.metric.eq("delta_per_spike") & summary.contrast.isin(["learned_hmm__dynamic_minus_reverse", "learned_hmm__dynamic_minus_reversible"])
    ]
    assert len(g) == 4 and g.dataset.nunique() == 2
    d = pd.read_csv(root / "reversible_forecasts_decision.csv")
    assert len(d) == 1 and d.primary_model.iloc[0] == "learned_hmm" and d.horizon_ms.iloc[0] == 40
    assert d.directional_predictive_lead.iloc[0] == bool(g.ci_low.gt(0).all() and g.positive_animals.eq(g.animals).all())
    for column in ("original_compound_verdict_changed", "independent_confirmation", "high_importance_discovery_established"):
        assert not d[column].any()
    return len(splits), len(events)


def verify(root, output):
    mp = root / "reversible_forecasts_manifest.json"
    m = json.loads(mp.read_text())
    assert m["status"] == "complete" and not m["git_dirty"]
    for name, path in m["input_file_paths"].items():
        assert file_sha256(path) == m["input_file_sha256"][name], name
    for name, digest in m["output_sha256"].items():
        assert file_sha256(root / name) == digest, name
    source = Path(m["input_file_paths"]["source_manifest"]).parent
    sm = json.loads((source / "occupancy_matched_manifest.json").read_text())
    lagged = Path(sm["input_file_paths"]["lagged_manifest"]).parent
    lm = json.loads((lagged / "lagged_prediction_manifest.json").read_text())
    parent, rate, learned = [Path(lm["input_file_paths"][k]).parent for k in ("parent", "rate", "learned")]
    checked, nparams, error = 0, 0, 0.0
    frames = []
    for item in m["completed"]:
        tag = item["tag"]
        keys = ["event_id", "split", "horizon", "model"]
        old = pd.read_csv(source / f"{tag}_matched_scores.csv.gz").set_index(keys).sort_index()
        new = pd.read_csv(root / f"{tag}_reversible_scores.csv.gz").set_index(keys).sort_index()
        assert not new.index.duplicated().any() and len(new) == len(old)
        pd.testing.assert_frame_equal(new[old.columns], old, check_exact=False, atol=1e-12, rtol=1e-12)
        valid = new.status.eq("scored")
        assert new.loc[valid, "reconstructed_forecast_hash_matches"].all()
        assert np.isfinite(new.loc[valid, "reconstruction_error"]).all() and new.loc[valid, "reconstruction_error"].le(1e-8).all()
        for col in ("score_reverse", "score_reversible"):
            assert np.isfinite(new.loc[valid, col]).all() and new.loc[valid, col].le(1e-8).all()
            assert new.loc[~valid, col].isna().all()
            np.testing.assert_allclose(new.loc[valid & new.n_heldout_target_spikes.eq(0), col], 0, atol=1e-12)
        cache = dict(np.load(parent / f"{tag}_cache.npz"))
        kernels = spatial_kernels(cache["centers"])
        spatial = {kind: Reference(kernels, imm=kind == "imm") for kind in ("imm", "diffusion")}
        pis = {kind: np.load(source / f"{tag}__{kind}_matched_null.npz")["pi"].ravel() for kind in spatial}
        for kind, op in spatial.items():
            constraints(op, pis[kind])
            nparams += 1
        gains = pd.read_csv(rate / f"{tag}_gains.csv")
        folds = json.loads((lagged / f"{tag}_folds.json").read_text())
        for f in folds:
            fold = f["fold"]
            fit = dict(np.load(learned / f"{tag}__fold{fold}__k50_fit.npz"))
            neural = Reference(fit=fit)
            npi = np.load(source / f"{tag}__fold{fold}_neural_matched_null.npz")["pi"].ravel()
            constraints(neural, npi)
            nparams += 1
            g = gains[gains.fold.eq(fold) & gains.alpha.eq(100)].set_index("unit_id")
            maps = cache["rates"] * g.loc[cache["unit_ids"], "gain"].to_numpy()[:, None]
            eid = f["test_ids"][0]
            x = cache[f"counts_{eid}"]
            if abs(np.diff(cache[f"edges_{eid}"])[-1] - 0.02) >= 1e-8:
                x = x[:-1]
            for split in range(5):
                tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                for h in (1, 2, 4):
                    if h >= len(x):
                        continue
                    for model in KINDS:
                        if model == "learned_hmm":
                            op, pi, rates = neural, npi, fit["probabilities"]
                        else:
                            kind = "imm" if model.startswith("spatial_imm") else "diffusion"
                            op, pi = spatial[kind], pis[kind]
                            rates = maps[:, cache["permutation"]] if model.endswith("permuted") else maps
                        expected = scores(x, rates, tr, he, op, pi, h)
                        for col, value in expected.items():
                            discrepancy = abs(value - new.loc[(eid, split, h, model), col])
                            assert discrepancy < 1e-8
                            error = max(error, discrepancy)
                            checked += 1
        frames.append(new.reset_index())
        print(json.dumps({"tag": tag, "scores_checked": checked, "max_score_error": error}), flush=True)
    rows = pd.concat(frames, ignore_index=True)
    assert len(rows) == 9225 * 75 and len(frames) == 33 and nparams == 231
    split_n, event_n = reconstruct(root, rows)
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__), "reference": ROOT / "scripts/verify_2d_lagged_neural_prediction.py"}, cwd=ROOT)
    p.update(
        status="pass",
        independent_scores=checked,
        parameter_sets=nparams,
        max_score_error=error,
        split_contrasts=split_n,
        event_contrasts=event_n,
        scope="All hashes/original rows/aggregates checked; all neural edge constraints and spatial mode-block flux constraints checked, with 16 spatial adjoint probe positions per mode. First event per fold, all neural splits/horizons/models independently forecast. Native raw files and source fitting not repeated.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(p, indent=2) + "\n")
    return p


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = verify(a.run_dir, a.output)
    print(json.dumps({k: v for k, v in r.items() if k not in ("input_file_paths", "input_file_sha256")}), flush=True)
