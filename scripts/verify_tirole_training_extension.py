"""Independent audit of training-only simulations, unchanged tests and mixture recovery."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts.verify_tirole_content_bank import direct_posterior
from scripts.verify_tirole_crossfit_mixture import close, digest, reference_fit, reference_templates, reference_weights, seed
from scripts.verify_tirole_measurement_calibration import draw, path


def reference_lookup(q, components):
    result = np.full((*q.shape, 2), np.nan)
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for a in range(len(q)):
        for s in range(q.shape[1]):
            if np.isfinite(q[a, s]).all():
                for t in range(2):
                    j = min(np.searchsorted(edges, q[a, s, t], side="right") - 1, 4)
                    result[a, s, t] = components[s, :, j]
    return result


def audit(folder, output):
    if output.exists():
        raise ValueError("new independent audit output required")
    m = json.loads((folder / "manifest.json").read_text())
    bank = Path(m["bank_dir"])
    source = Path(m["source_manifest"]).parent
    if m["status"] != "complete" or m["git_dirty"] or m["real_data_corrected"] or m["test_spikes_regenerated"] or m["test_trajectories_regenerated"]:
        raise ValueError("invalid frozen calibration metadata")
    for p, h in [(bank / "manifest.json", m["bank_manifest_sha256"]), (source / "manifest.json", m["source_manifest_sha256"])]:
        if digest(p) != h:
            raise ValueError("changed source manifest")
    for base, key in [(bank, "outputs_sha256"), (source, "output_sha256"), (folder, "output_sha256")]:
        for name, h in json.loads((base / "manifest.json").read_text())[key].items():
            if digest(base / name) != h:
                raise ValueError("source/output hash mismatch")
    test = pd.read_csv(source / "count_anchors.csv").sort_values("event_id")
    saved_test = pd.read_csv(folder / "frozen_test_anchors.csv")
    pd.testing.assert_frame_equal(test.reset_index(drop=True), saved_test, check_exact=False, atol=1e-9)
    training = pd.read_csv(folder / "training_anchors.csv")
    events = pd.read_csv(bank / "candidate_events.csv")
    eligible = []
    for e in events.itertuples():
        if not e.ripple_supported or e.event_id in set(test.event_id):
            continue
        if any(e.start_s < t.end_s + 1.0 and e.end_s > t.start_s - 1.0 for t in test.itertuples()):
            continue
        eligible.append(e.event_id)
    selected = []
    for _, g in events[events.event_id.isin(eligible)].groupby("epoch"):
        selected.extend(sorted(g.event_id, key=lambda e: seed(20260918, m["session"], e, "training-only-composition-anchor"))[:80])
    if sorted(selected) != training.event_id.tolist() or set(selected) & set(test.event_id):
        raise ValueError("training/test overlap or score-dependent anchor selection")
    maps = np.load(bank / "RUN_maps.npz")
    c = np.load(bank / "event_counts.npz")
    parts = json.loads((bank / "partitions.json").read_text())["splits"]
    gen = pd.read_csv(folder / "training_readouts.csv")
    keys = ["anchor_event", "split", "truth_track"]
    ix = pd.MultiIndex.from_product([training.event_id, range(5), [1, 2]], names=keys)
    if len(gen) != len(ix) or gen.duplicated(keys).any():
        raise ValueError("missing/duplicate training simulations")
    indexed = gen.set_index(keys).reindex(ix)
    checks = {"regenerated_training_rows": 0, "point_fits": 0, "selection_fits": 0, "bootstrap_fits": 0, "primary_intervals": 0}
    maxerr = 0.0
    for eid in training.event_id:
        original = c["counts"][c["offsets"][eid] : c["offsets"][eid + 1]]
        for s, part in enumerate(parts):
            a, b = np.array(part["inference"]), np.array(part["evaluation"])
            if set(a) & set(b):
                raise ValueError("cell-role leakage")
            positions = path(maps["valid_bins"], len(original), np.random.default_rng(seed(20260918, m["session"], eid, s, "known-path")))
            for t in range(2):
                rng = np.random.default_rng(seed(20260918, m["session"], eid, s, t, "known-track-spikes"))
                generated = np.zeros_like(original)
                for ids in (a, b):
                    rates = np.maximum(maps["rates"][t][:, positions].T[:, ids], 1e-4)
                    generated[:, ids] = draw(original[:, ids].sum(axis=1), rates, rng)
                r = indexed.loc[(eid, s, t + 1)]
                if r.n_evaluation_spikes != generated[:, b].sum():
                    raise ValueError("count-preservation failure")
                for name, conditional in [("poisson", False), ("conditional_count", True)]:
                    use = generated[:, b]
                    use = use[use.sum(axis=1) > 0]
                    if len(use):
                        posterior = direct_posterior(use, maps["rates"][:, b], maps["valid_bins"], conditional)
                        mass = posterior.sum(axis=(0, 2))
                        value = mass[1] / mass.sum()
                    else:
                        value = np.nan
                    maxerr = max(maxerr, close(value, r[name]))
                span = abs(maps["bin_centers_cm"][positions[-1]] - maps["bin_centers_cm"][positions[0]])
                maxerr = max(maxerr, close(span, r.true_path_span_cm))
                checks["regenerated_training_rows"] += 1
    raw = pd.read_csv(source / "known_track_calibration.csv")
    raw["poisson"] = raw.evaluation_track2_probability
    raw["conditional_count"] = expit(-np.where(raw.truth_track == 1, 1, -1) * raw.conditional_true_signed_log_odds)
    tx = pd.MultiIndex.from_product([test.event_id, range(5), [1, 2]], names=keys)
    fixed = raw[(raw.generator == "ordered") & raw.fraction.eq(1)].set_index(keys).reindex(tx)
    reduced = raw[(raw.generator == "ordered") & raw.fraction.eq(0.5)].set_index(keys).reindex(tx)
    shape = (len(test), 5, 2)
    counts = fixed.n_evaluation_spikes.to_numpy().reshape(shape)[:, :, 0]
    spans = fixed.true_path_span_cm.to_numpy().reshape(shape)[:, :, 0]
    full, half = fixed.sequence_accepted.to_numpy().reshape(shape), reduced.sequence_accepted.to_numpy().reshape(shape)
    recovery = pd.read_csv(folder / "recovery.csv")
    selection = pd.read_csv(folder / "selection_stress.csv")
    boot = pd.read_csv(folder / "bootstrap.csv")
    component_rows = pd.read_csv(folder / "components.csv")
    llrows = pd.read_csv(folder / "test_likelihoods.csv")
    for name in ["poisson", "conditional_count"]:
        train_q = indexed[name].to_numpy().reshape(len(training), 5, 2)
        q = fixed[name].to_numpy().reshape(shape)
        component = reference_templates(train_q, np.zeros(len(training), int), np.ones(len(training)))[1]
        ll = reference_lookup(q, component)
        observed = llrows[llrows.readout == name].set_index(keys).reindex(tx)
        maxerr = max(maxerr, close(q.reshape(-1), observed.posterior_mass.to_numpy()), close(ll.reshape(-1, 2), observed[["likelihood_track1", "likelihood_track2"]].to_numpy()))
        for r in component_rows[component_rows.readout == name].itertuples():
            maxerr = max(maxerr, close(component[r.split, r.truth_track - 1], [getattr(r, f"bin{i}") for i in range(5)]))
            if r.n_unique_training_anchors != np.isfinite(train_q[:, r.split]).all(axis=1).sum():
                raise ValueError("training support mismatch")
        conditions = {
            "all": np.ones(counts.shape, bool),
            "count_1_to_4": (counts >= 1) & (counts <= 4),
            "count_at_least_5": counts >= 5,
            "path_under_100cm": spans < 100,
            "path_at_least_100cm": spans >= 100,
        }
        for r in recovery[recovery.readout == name].itertuples():
            mask = ((test.epoch == r.epoch) & (test.primary_ripple_candidate == r.ripple_positive)).to_numpy()[:, None] & conditions[r.condition]
            weights = reference_weights(q, mask, r.true_pi)
            estimate, complete = reference_fit(ll, weights)
            maxerr = max(maxerr, close(estimate, r.estimated_pi))
            if complete != r.complete_point_likelihood:
                raise ValueError("point completeness mismatch")
            checks["point_fits"] += 1
            if r.recovery_gate != "not_primary_target":
                draws = boot[(boot.readout == name) & boot.true_pi.eq(r.true_pi)]
                if len(draws) != 2000 or set(draws.draw) != set(range(2000)):
                    raise ValueError("missing/duplicate bootstrap fits")
                good = draws.estimated_pi.dropna().to_numpy()
                maxerr = max(maxerr, close(len(good) / 2000, r.finite_bootstrap_fraction))
                if len(good) >= 1900:
                    lo, hi = np.quantile(good, [0.025, 0.975])
                    maxerr = max(maxerr, close([lo, hi], [r.ci025, r.ci975]))
                    direction = hi < 0.5 if r.true_pi < 0.5 else lo > 0.5 if r.true_pi > 0.5 else True
                    gate = "pass" if complete and np.isfinite(estimate) and lo <= r.true_pi <= hi and direction else "fail"
                else:
                    gate = "fail_insufficient_bootstrap"
                if gate != r.recovery_gate:
                    raise ValueError("gate reconstruction mismatch")
                checks["primary_intervals"] += 1
        for r in selection[selection.readout == name].itertuples():
            mask = ((test.epoch == r.epoch) & (test.primary_ripple_candidate == r.ripple_positive)).to_numpy()[:, None] & np.ones((1, 5), bool)
            weight = reference_weights(q, mask, accepted={"full": full, "retained": full & half, "half": half}[r.group])
            estimate, complete = reference_fit(ll, weight)
            truth = weight[:, :, 1].sum() / weight.sum() if weight.sum() else np.nan
            maxerr = max(maxerr, close([estimate, truth], [r.estimated_pi, r.true_weighted_pi]))
            if complete != r.complete_likelihood:
                raise ValueError("selection completeness mismatch")
            checks["selection_fits"] += 1
        trng = np.random.default_rng(seed(20260918, m["session"], name, "independent-training-bootstrap"))
        qrng = np.random.default_rng(seed(20260918, m["session"], name, "frozen-test-bootstrap"))
        tr = trng.multinomial(len(train_q), np.full(len(train_q), 1 / len(train_q)), size=2000)
        te = qrng.multinomial(len(q), np.full(len(q), 1 / len(q)), size=2000)
        mask = (test.epoch.eq("POST") & test.primary_ripple_candidate).to_numpy()[:, None] & np.ones((1, 5), bool)
        lookup = boot[boot.readout == name].set_index(["true_pi", "draw"])
        for j in range(2000):
            params = reference_templates(train_q, np.zeros(len(train_q), int), tr[j])[1]
            ref = reference_lookup(q, params)
            for pi in [0.25, 0.5, 0.75]:
                weight = reference_weights(q, mask, pi) * te[j, :, None, None]
                estimate, _ = reference_fit(ref, weight)
                maxerr = max(maxerr, close(estimate, lookup.loc[(pi, j), "estimated_pi"]))
                checks["bootstrap_fits"] += 1
    if checks != {"regenerated_training_rows": len(training) * 10, "point_fits": 200, "selection_fits": 24, "bootstrap_fits": 12000, "primary_intervals": 6}:
        raise ValueError("incomplete verification")
    result = {
        "status": "pass",
        "session": m["session"],
        "manifest_sha256": digest(folder / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "max_absolute_error": maxerr,
        "scope": "source/output hashes; all training anchor membership, synthetic counts/content, unchanged test scores; independent histogram/root solve for all point fits and all refitted bootstrap draws; not biological transport",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.experiment_dir, a.output)
