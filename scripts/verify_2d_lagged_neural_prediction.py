#!/usr/bin/env python3
"""Independent forward recursion, predictive likelihood and table reconstruction."""

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
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import multinomial

from scripts._provenance import build_script_provenance, file_sha256

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split", "horizon"]
KINDS = ("spatial_imm_real", "spatial_imm_permuted", "spatial_diffusion_real", "spatial_diffusion_permuted", "learned_hmm")
BASELINES = ("dwell_only", "frozen", "no_history", "global", "same_time")


def likelihood(x, rates):
    p = (rates / rates.sum(axis=0)).T
    return np.stack([multinomial.logpmf(row, row.sum(), p) for row in x])


def spatial_kernels(centers):
    distances = cdist(centers, centers)
    result = []
    for sigma in (2.0, round(60 * np.sqrt(0.02), 10)):
        k = np.exp(-0.5 * (distances / sigma) ** 2)
        k[distances > 3 * sigma] = 0
        k /= k.sum(axis=0)
        result.append(csr_matrix(k))
    return result


class Reference:
    """Separate state recursion; no producer or production-filter imports."""

    def __init__(self, kernels=None, imm=True, fit=None):
        self.neural = fit is not None
        if self.neural:
            self.initial = fit["initial"].copy()
            self.a = fit["transition"].copy()
            w = fit["occupancy"]
            self.b = np.zeros_like(self.a)
            for i in range(len(w)):
                dest = np.arange(len(w)) != i
                self.b[i, i] = self.a[i, i]
                self.b[i, dest] = (1 - self.a[i, i]) * w[dest] / w[dest].sum()
            self.n = len(w)
            self.modes = 1
        else:
            self.n = kernels[0].shape[0]
            self.kernels = [*kernels, None] if imm else [kernels[1]]
            self.modes = len(self.kernels)
            self.initial = np.ones(self.n * self.modes) / (self.n * self.modes)
            self.mode = np.ones((1, 1))
            if imm:
                sticky = np.exp(-0.02 / 0.06)
                self.mode = np.full((3, 3), (1 - sticky) / 2)
                np.fill_diagonal(self.mode, sticky)

    def step(self, q, dwell=False):
        if self.neural:
            return (self.b if dwell else self.a).T @ q
        old = q.reshape(self.modes, self.n)
        new = np.zeros_like(old)
        for source in range(self.modes):
            for dest, kernel in enumerate(self.kernels):
                x = old[source]
                if kernel is None:
                    propagated = np.full(self.n, x.sum() / self.n)
                elif dwell and self.n > 1:
                    stay = kernel.diagonal()
                    leave = (1 - stay) * x / (self.n - 1)
                    propagated = stay * x + leave.sum() - leave
                else:
                    propagated = kernel @ x
                new[dest] += self.mode[source, dest] * propagated
        return new.ravel()

    def marginal(self, q):
        return q if self.neural else q.reshape(self.modes, self.n).sum(axis=0)

    def predict(self, ll, h):
        filtered, prior = [], []
        state = self.initial.copy()
        unconditioned = state.copy()
        for t, emission in enumerate(ll):
            if t:
                state = self.step(state)
                unconditioned = self.step(unconditioned)
            prior.append(unconditioned.copy())
            likelihood = np.tile(np.exp(emission - emission.max()), self.modes)
            state *= likelihood
            state /= state.sum()
            filtered.append(state.copy())
        values = {k: [] for k in ("dynamic", "dwell_only", "frozen", "no_history", "same_time")}
        for origin in range(len(ll) - h):
            dynamic, dwell = filtered[origin].copy(), filtered[origin].copy()
            for _ in range(h):
                dynamic, dwell = self.step(dynamic), self.step(dwell, True)
            for name, q in (("dynamic", dynamic), ("dwell_only", dwell), ("frozen", filtered[origin]), ("no_history", prior[origin + h]), ("same_time", filtered[origin + h])):
                values[name].append(self.marginal(q))
        return {k: np.array(v) for k, v in values.items()}


def reference_scores(x, rates, train, held, fit, kernels, permutation, horizon):
    result = []
    for model in KINDS:
        if model == "learned_hmm":
            op = Reference(fit=fit)
            maps = fit["probabilities"]
        else:
            op = Reference(kernels, imm=model.startswith("spatial_imm"))
            maps = rates[:, permutation] if model.endswith("permuted") else rates
        prediction = op.predict(likelihood(x[:, train], maps[train]), horizon)
        target = likelihood(x[horizon:, held], maps[held])
        scores = {}
        for name, q in prediction.items():
            with np.errstate(divide="ignore"):
                logq = np.log(q)
            scores["score_" + name] = float(logsumexp(logq + target, axis=1).sum())
        scores["score_global"] = float(likelihood(x[horizon:, held], fit["global_probability"][held, None]).sum())
        result.append({"model": model, **scores})
    return pd.DataFrame(result).set_index("model")


def assert_table(expected, path, keys, values):
    actual = pd.read_csv(path)
    expected = expected.copy()
    for column in ID:
        if column in keys:
            actual[column] = actual[column].astype(str)
            expected[column] = expected[column].astype(str)
    assert not actual.duplicated(keys).any(), path.name
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    assert a.index.equals(b.index), path.name
    np.testing.assert_allclose(a[values], b[values], atol=1e-8, rtol=1e-9, equal_nan=True)


def reconstruct_tables(root, rows):
    parts = []
    models = {m: rows[rows.model.eq(m)].set_index(KEY).sort_index() for m in KINDS}
    common = models[KINDS[0]][["n_target_bins", "n_heldout_target_spikes"]]
    for m in models.values():
        assert m.index.equals(common.index)
    contrasts = {}
    for name, m in models.items():
        for b in BASELINES:
            contrasts[name + "__dynamic_minus_" + b] = m.score_dynamic - m["score_" + b]
    for kind in ("imm", "diffusion"):
        r, w = models[f"spatial_{kind}_real"], models[f"spatial_{kind}_permuted"]
        contrasts[f"spatial_{kind}__real_minus_permuted"] = r.score_dynamic - w.score_dynamic
        contrasts[f"spatial_{kind}__map_route_interaction"] = r.score_dynamic - r.score_dwell_only - w.score_dynamic + w.score_dwell_only
    for other in ("learned_hmm", "spatial_diffusion_real"):
        contrasts["spatial_imm_minus_" + other] = models["spatial_imm_real"].score_dynamic - models[other].score_dynamic
    for name, delta in contrasts.items():
        p = common.copy()
        p["contrast"], p["delta"] = name, delta
        p["delta_per_spike"] = p.delta / p.n_heldout_target_spikes.where(p.n_heldout_target_spikes > 0)
        parts.append(p.reset_index())
    splits = pd.concat(parts, ignore_index=True)
    assert_table(splits, root / "lagged_prediction_splits.csv.gz", KEY + ["contrast"], ["delta", "delta_per_spike", "n_target_bins", "n_heldout_target_spikes"])
    ek = ID + ["event_id", "horizon", "contrast"]
    event = splits.groupby(ek)[["delta", "delta_per_spike"]].median().reset_index()
    event["valid_neural_splits"] = splits.groupby(ek).delta_per_spike.count().to_numpy()
    assert_table(event, root / "lagged_prediction_events.csv.gz", ek, ["delta", "delta_per_spike", "valid_neural_splits"])
    session = event.groupby(ID + ["horizon", "contrast"])[["delta", "delta_per_spike"]].mean().reset_index()
    assert_table(session, root / "lagged_prediction_sessions.csv.gz", ID + ["horizon", "contrast"], ["delta", "delta_per_spike"])
    ak = ["dataset", "animal", "horizon", "contrast"]
    animal = session.groupby(ak)[["delta", "delta_per_spike"]].mean().reset_index()
    assert_table(animal, root / "lagged_prediction_animals.csv.gz", ak, ["delta", "delta_per_spike"])
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
                    "positive_animals": (v > 0).sum(),
                    "animals": len(v),
                }
            )
    summary = pd.DataFrame(summary)
    assert_table(summary, root / "lagged_prediction_summary.csv.gz", ["dataset", "horizon", "contrast", "metric"], ["mean", "ci_low", "ci_high", "positive_animals", "animals"])
    decisions = pd.read_csv(root / "lagged_prediction_decisions.csv").set_index("model")
    assert set(decisions.index) == {"spatial_imm_real", "spatial_diffusion_real", "learned_hmm"}
    main = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike")]
    for model, row in decisions.iterrows():
        needed = {model + "__dynamic_minus_" + b for b in BASELINES if b != "same_time"}
        p = main[main.contrast.isin(needed)]
        passed = len(p) == 8 and p.ci_low.gt(0).all() and p.positive_animals.eq(p.animals).all()
        assert row.replicated_forecasting_lead == passed
        spatial = False
        if model.startswith("spatial"):
            stem = model[:-5]
            p = main[main.contrast.isin([stem + "__real_minus_permuted", stem + "__map_route_interaction"])]
            spatial = passed and len(p) == 4 and p.ci_low.gt(0).all() and p.positive_animals.eq(p.animals).all()
        assert row.spatial_route_specific_lead == spatial
        assert not row.independent_confirmation and not row.high_importance_discovery_established
    return len(splits), len(event)


def verify(root, output):
    mp = root / "lagged_prediction_manifest.json"
    manifest = json.loads(mp.read_text())
    assert manifest["status"] == "complete" and not manifest["git_dirty"]
    for name, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][name], name
    for name, digest in manifest["output_sha256"].items():
        assert file_sha256(root / name) == digest, name
    parent, rate, learned = (Path(manifest["input_file_paths"][k]).parent for k in ("parent", "rate", "learned"))
    frames, cover = [], []
    maximum, checked, sampled_events = 0.0, 0, 0
    for item in manifest["completed"]:
        tag = item["tag"]
        cache = dict(np.load(parent / f"{tag}_cache.npz"))
        selection = pd.read_csv(parent / f"{tag}_selection.csv").sort_values(["start_s", "event_id"])
        published = json.loads((root / f"{tag}_folds.json").read_text())
        actual = pd.read_csv(root / f"{tag}_forecast_scores.csv.gz")
        assert len(actual) == len(selection) * 75
        assert set(actual.event_id) == set(selection.event_id)
        assert not actual.duplicated(["event_id", "split", "model", "horizon"]).any()
        assert not actual.forecast_uses_heldout.any() and not actual.forecast_uses_future_training.any()
        for col in ID:
            assert actual[col].eq(item[col]).all()
        gains = pd.read_csv(rate / f"{tag}_gains.csv")
        kernels = spatial_kernels(cache["centers"])
        lookup = actual.set_index(["event_id", "split", "horizon", "model"]).sort_index()
        factors = set(itertools.product(range(5), (1, 2, 4), KINDS))
        for e in selection.itertuples():
            counts, edges = cache[f"counts_{e.event_id}"], cache[f"edges_{e.event_id}"]
            widths = np.diff(edges)
            assert len(edges) == len(counts) + 1 and (widths > 0).all()
            np.testing.assert_allclose(widths[:-1], 0.02, rtol=0, atol=1e-8)
            assert widths[-1] <= 0.02000001
            nf = len(counts) if abs(widths[-1] - 0.02) < 1e-8 else len(counts) - 1
            assert counts.sum() == e.n_spikes_qc_units
            sub = lookup.loc[e.event_id]
            assert set(sub.index) == factors
            for split in range(5):
                train, held = cache[f"train_{split}"], cache[f"held_{split}"]
                assert len(train) and len(held) and sorted([*train, *held]) == list(range(counts.shape[1]))
                for h in (1, 2, 4):
                    g = sub.loc[(split, h)]
                    nt = max(nf - h, 0)
                    assert g.n_full_bins.eq(nf).all() and g.n_target_bins.eq(nt).all()
                    assert g.n_heldout_target_spikes.eq(counts[h:nf, held].sum()).all()
                    assert g.discarded_partial_bin_spikes.eq(counts[nf:].sum()).all()
                    assert g.horizon_ms.eq(h * 20).all() and g.unobserved_gap_ms.eq((h - 1) * 20).all()
                    columns = ["score_dynamic", *["score_" + b for b in BASELINES]]
                    assert g.status.eq("scored" if nt else "insufficient_full_bins").all()
                    if nt:
                        assert np.isfinite(g[columns]).all().all() and g[columns].le(1e-8).all().all()
                        if not counts[h:nf, held].sum():
                            np.testing.assert_allclose(g[columns], 0, atol=1e-10)
                    else:
                        assert g[columns].isna().all().all()
                    if split == 0:
                        cover.append(
                            {k: item[k] for k in ID}
                            | {"horizon": h, "selected_events": 1, "eligible_events": int(nt > 0), "target_bins": nt, "discarded_partial_spikes": counts[nf:].sum()}
                        )
        for fold, index in enumerate(np.array_split(np.arange(len(selection)), 5)):
            test = selection.iloc[index]
            possible = selection[~selection.event_id.isin(test.event_id)]
            keep = np.array([all(c.end_s + 1 <= t.start_s or c.start_s >= t.end_s + 1 for t in test.itertuples()) for c in possible.itertuples()])
            cal, excluded = possible[keep], possible[~keep]
            stem = f"{tag}__fold{fold}__k50"
            source = json.loads((learned / f"{stem}_manifest.json").read_text())
            detail = next(d for d in published if d["fold"] == fold)
            for name, frame in (("test_ids", test), ("calibration_ids", cal), ("excluded_ids", excluded)):
                assert source[name] == detail[name] == frame.event_id.tolist()
            assert source["fit_converged"]
            assert actual[actual.event_id.isin(test.event_id)].fold.eq(fold).all()
            fit = dict(np.load(learned / f"{stem}_fit.npz"))
            g = gains[gains.fold.eq(fold) & gains.alpha.eq(100)].set_index("unit_id")
            assert set(g.index) == set(cache["unit_ids"]) and not g.index.duplicated().any()
            rates = cache["rates"] * g.loc[cache["unit_ids"], "gain"].to_numpy()[:, None]
            eid = int(test.iloc[0].event_id)
            x = cache[f"counts_{eid}"]
            if abs(np.diff(cache[f"edges_{eid}"])[-1] - 0.02) >= 1e-8:
                x = x[:-1]
            sampled_events += 1
            for split in range(5):
                for h in (1, 2, 4):
                    if len(x) <= h:
                        continue
                    expected = reference_scores(x, rates, cache[f"train_{split}"], cache[f"held_{split}"], fit, kernels, cache["permutation"], h)
                    saved = lookup.loc[(eid, split, h)].reindex(expected.index)
                    np.testing.assert_allclose(saved[expected.columns], expected, rtol=1e-9, atol=1e-8)
                    maximum = max(maximum, float(np.max(np.abs(saved[expected.columns].to_numpy() - expected.to_numpy()))))
                    checked += expected.size
        frames.append(actual)
        print(json.dumps({"tag": tag, "checked_scores": checked, "maximum_error": maximum}), flush=True)
    coverage = pd.DataFrame(cover).groupby(ID + ["horizon"], as_index=False).sum()
    assert_table(coverage, root / "lagged_prediction_coverage.csv", ID + ["horizon"], ["selected_events", "eligible_events", "target_bins", "discarded_partial_spikes"])
    rows = pd.concat(frames, ignore_index=True)
    assert len(rows) == 9225 * 75 and len(frames) == 33
    split_rows, event_rows = reconstruct_tables(root, rows)
    report = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    report.update(
        status="pass",
        independent_score_reconstructions=checked,
        sampled_events=sampled_events,
        maximum_absolute_score_error=maximum,
        reconstructed_split_contrasts=split_rows,
        reconstructed_event_contrasts=event_rows,
        events=9225,
        sessions=33,
        scope="All input/output hashes, folds, temporal/cell support, contrasts and summaries. First chronological event per fold, all splits/horizons/models independently rescored. Native files not reopened; parameter fitting not repeated.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.run_dir, a.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_file_paths", "input_file_sha256")}), flush=True)
