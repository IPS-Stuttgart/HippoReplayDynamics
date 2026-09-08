#!/usr/bin/env python3
"""Independent reconstruction of the frozen Tanni cross-context diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

KEYS = ["animal", "session", "context", "phase", "event_id"]
CONTRASTS = {
    "mix_iid_minus_current_iid": ("mix_iid", "current_iid"),
    "mix_iid_minus_mix_global": ("mix_iid", "mix_global"),
    "mix_iid_minus_event_global": ("mix_iid", "event_global"),
    "mix_iid_minus_blind_iid": ("mix_iid", "blind_iid"),
    "current_iid_minus_current_global": ("current_iid", "current_global"),
    "mix_global_minus_current_global": ("mix_global", "current_global"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def close(left, right, tolerance=1e-8):
    np.testing.assert_allclose(left, right, atol=tolerance, rtol=1e-10, equal_nan=True)


def reference_prediction(counts, maps, compositions, train, held, labels, current, baseline):
    """Direct normalized multinomial equations; no production inference import."""
    labels = np.asarray(labels)
    categories = np.unique(labels)
    prior = np.array([1 / len(categories) / np.sum(labels == c) for c in labels])

    def emission(neurons, rates):
        observations = counts[:, neurons]
        selected = rates[neurons]
        p = selected / selected.sum(axis=0)
        coefficient = gammaln(observations.sum(axis=1) + 1) - gammaln(observations + 1).sum(axis=1)
        return observations @ np.log(p) + coefficient[:, None]

    train_spatial = [emission(train, m) for m in maps]
    train_global = [emission(train, c[:, None])[:, 0] for c in compositions]
    norms = [logsumexp(x, axis=1) for x in train_spatial]
    q = [x - z[:, None] for x, z in zip(train_spatial, norms, strict=True)]
    weight_ll = {
        "iid": np.array([(z - np.log(m.shape[1])).sum() for z, m in zip(norms, maps, strict=True)]),
        "global": np.array([x.sum() for x in train_global]),
    }
    result = {}
    for model in ("iid", "global"):
        logw = np.log(prior) + weight_ll[model]
        logw -= logsumexp(logw)
        if model == "iid":
            components = np.array([logsumexp(post + emission(held, m), axis=1) for post, m in zip(q, maps, strict=True)])
        else:
            components = np.array([emission(held, c[:, None])[:, 0] for c in compositions])
        result[f"mix_{model}"] = logsumexp(components + logw[:, None], axis=0).sum()
        keep = labels == current
        restricted = logw[keep] - logsumexp(logw[keep])
        result[f"current_{model}"] = logsumexp(components[keep] + restricted[:, None], axis=0).sum()
        result[f"blind_{model}"] = logsumexp(components + np.log(prior)[:, None], axis=0).sum()
        for c in categories:
            result[f"{model}_p_{c}"] = np.exp(logw[labels == c]).sum()
    result["event_global"] = emission(held, baseline[:, None]).sum()
    return result


def recount(times_by_unit, edges):
    result = np.empty((len(edges) - 1, len(times_by_unit)), dtype=np.int64)
    for i, times in enumerate(times_by_unit):
        lo, hi = np.searchsorted(times, edges[[0, -1]], side="left")
        result[:, i] = np.histogram(times[lo:hi], edges)[0]
    return result


def guarded_ids(events, fold):
    ordered = events.sort_values(["start_s", "event_id"])
    test = ordered.iloc[np.array_split(np.arange(len(ordered)), 5)[fold]]
    cal, excluded = [], []
    for row in ordered.itertuples(index=False):
        if row.event_id in set(test.event_id):
            continue
        separated = all(row.end_s + 1 <= t.start_s or row.start_s >= t.end_s + 1 for t in test.itertuples(index=False))
        (cal if separated else excluded).append(row.event_id)
    return set(test.event_id), set(cal), set(excluded)


def csv_ids(value):
    if pd.isna(value) or value == "":
        return set()
    return {int(float(x)) for x in str(value).split(",")}


def primary_pass(summary):
    primary = summary.loc[list(CONTRASTS)[:3]]
    return bool((primary["mean"] > 0).all() and (primary.ci_low > 0).all() and primary.positive_animals.eq(5).all())


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "tanni_multicontext_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["output_sha256"].items():
        assert digest(root / name) == expected, name
    scores = pd.read_csv(root / "tanni_multicontext_split_contrasts.csv")
    assert not scores.duplicated(KEYS + ["split"]).any()
    assert scores.groupby(KEYS).split.apply(lambda x: set(x) == set(range(5))).all()
    assert not scores.heldout_used_for_inference.any()
    sampled_scores = raw_bins = raw_windows = 0
    maximum_error = 0.0
    for animal in sorted(scores.animal.unique()):
        bank = load_npz(root / f"{animal}_bank.npz")
        units = bank["unit_ids"]
        maps = [bank[f"rates_{i}"] for i in range(5)]
        compositions = [bank[f"composition_{i}"] for i in range(5)]
        labels = bank["context_ids"]
        templates = pd.read_csv(root / f"{animal}_templates.csv").sort_values("template_index")
        selection = pd.read_csv(root / f"{animal}_selection.csv")
        folds = pd.read_csv(root / f"{animal}_calibration_folds.csv")
        sources = []
        for template in templates.itertuples(index=False):
            assert digest(template.source_path) == template.source_sha256
            assert digest(template.source_metadata_path) == template.source_metadata_sha256
            assert digest(template.selection_path) == template.selection_sha256
            sources.append(load_npz(template.source_path))
        shared = sorted(set.intersection(*(set(x["cell_ids"]) for x in sources)))
        first_maps, occupied = [], []
        for a in sources:
            ind = np.array([list(a["cell_ids"]).index(u) for u in shared])
            support = a["occupancy_first_half_s"] >= 0.05
            first_maps.append(a["rates_first_half_hz"][ind][:, support])
            occupied.append(a["occupancy_first_half_s"][support])
        expected = np.stack([(m * occ).sum(axis=1) for m, occ in zip(first_maps, occupied, strict=True)]).sum(axis=0)
        mean = expected / np.concatenate(occupied).sum()
        peak = np.concatenate(first_maps, axis=1).max(axis=1)
        keep = (expected >= 30) & (mean <= 4) & (peak >= 2)
        np.testing.assert_array_equal(np.asarray(shared)[keep], units)
        for i, (m, occ) in enumerate(zip(first_maps, occupied, strict=True)):
            close(m[keep], maps[i], 0)
            close((m[keep] * occ).sum(axis=1) / occ.sum(), compositions[i])
        for split in range(5):
            raw_seed = int.from_bytes(hashlib.sha256(f"20260908|{animal}|common-context-cells|{split}".encode()).digest()[:8], "little")
            permutation = np.random.default_rng(raw_seed).permutation(len(units))
            nheld = round(0.3 * len(units))
            np.testing.assert_array_equal(bank[f"held_{split}"], np.sort(permutation[:nheld]))
            np.testing.assert_array_equal(bank[f"train_{split}"], np.sort(permutation[nheld:]))
        for i, (template, a) in enumerate(zip(templates.itertuples(index=False), sources, strict=True)):
            observations = load_npz(root / f"{animal}__{template.session}_observations.npz")
            meta = json.loads(Path(template.source_metadata_path).read_text())
            close(meta["rate_map_half_split_time_s"], template.rate_map_cutoff_s)
            area = np.prod(a["arena_bounds_cm"][1] - a["arena_bounds_cm"][0]) / 10000
            assert abs(np.log2(area / 1.09375) - "ABCD".index(template.context)) < 0.001
            assert labels[i] == template.context
            current_selection = selection[selection.session.eq(template.session)]
            mua = current_selection[current_selection.phase.eq("MUA")]
            source_selection = pd.read_csv(template.selection_path)
            assert set(mua.event_id) == set(source_selection.event_id)
            times = [np.sort(a["spikes"][a["spikes"][:, 1] == u, 0]) for u in units]
            for event in mua.itertuples(index=False):
                counts, edges = observations[f"counts_{event.event_id}"], observations[f"edges_{event.event_id}"]
                close(edges[[0, -1]], [event.start_s, event.end_s])
                np.testing.assert_array_equal(counts, recount(times, edges))
                raw_bins += len(counts)
                raw_windows += 1
            windows = observations["run_windows"]
            assert len(windows) == template.run_windows >= 50
            # CSV parsing can round the cutoff upward; selection used this exact JSON value.
            assert (windows[:, 0] >= meta["rate_map_half_split_time_s"] + 1).all()
            assert (windows[1:, 0] >= windows[:-1, 1] - 1e-9).all()
            close(windows[:, 1] - windows[:, 0], 0.2)
            for j, (start, end) in enumerate(windows):
                edges = start + np.arange(11) * 0.02
                edges[-1] = end
                np.testing.assert_array_equal(observations["run_counts"][j], recount(times, edges))
                raw_bins += 10
                raw_windows += 1
                p = a["position"]
                speed = np.hypot(np.gradient(p[:, 1], p[:, 0]), np.gradient(p[:, 2], p[:, 0]))
                support = (p[:, 0] >= start - 1e-9) & (p[:, 0] <= end + 1e-9)
                assert support.sum() >= 2 and (speed[support] >= 10).all()
                assert (np.diff(p[support, 0]) <= 0.1).all()
            session_rows = scores[scores.animal.eq(animal) & scores.session.eq(template.session)]
            for row in session_rows.itertuples(index=False):
                c = observations["run_counts"][row.event_id] if row.phase == "RUN" else observations[f"counts_{row.event_id}"]
                assert row.n_train_cells == len(bank[f"train_{row.split}"])
                assert row.n_heldout_cells == len(bank[f"held_{row.split}"])
                assert row.n_train_spikes == c[:, bank[f"train_{row.split}"]].sum()
                assert row.n_heldout_spikes == c[:, bank[f"held_{row.split}"]].sum()
            reference = compositions[i] / compositions[i].sum()
            for row in folds[folds.session.eq(template.session)].itertuples(index=False):
                test_ids, calibration, excluded = guarded_ids(mua, row.fold)
                assert test_ids == csv_ids(row.test_ids)
                assert calibration == csv_ids(row.calibration_ids) and calibration
                assert excluded == csv_ids(row.guard_excluded_ids)
                pooled = np.stack([observations[f"counts_{eid}"].sum(axis=0) for eid in sorted(calibration)]).sum(axis=0)
                close(observations[f"event_global_{row.fold}"], (pooled + 100 * reference) / (pooled.sum() + 100))
            for phase, selected_ids in (("RUN", [0, len(windows) - 1]), ("MUA", mua.event_id.to_numpy()[np.unique(np.linspace(0, len(mua) - 1, min(3, len(mua))).astype(int))])):
                for eid in selected_ids:
                    event = current_selection[current_selection.phase.eq(phase) & current_selection.event_id.eq(eid)].iloc[0]
                    counts = observations["run_counts"][eid] if phase == "RUN" else observations[f"counts_{eid}"]
                    baseline = reference if phase == "RUN" else observations[f"event_global_{event.fold}"]
                    chosen = scores[scores.animal.eq(animal) & scores.session.eq(template.session) & scores.phase.eq(phase) & scores.event_id.eq(eid)]
                    for row in chosen.itertuples(index=False):
                        direct = reference_prediction(counts, maps, compositions, bank[f"train_{row.split}"], bank[f"held_{row.split}"], labels, template.context, baseline)
                        error = max(abs(float(getattr(row, k)) - v) for k, v in direct.items())
                        maximum_error = max(maximum_error, error)
                        assert error < 1e-8
                        sampled_scores += 1
            print(f"verified {animal} {template.session}", flush=True)
    for metric, (left, right) in CONTRASTS.items():
        close(scores[metric], scores[left] - scores[right])
        close(scores[metric + "_per_spike"], (scores[left] - scores[right]) / scores.n_heldout_spikes.replace(0, np.nan))
    metrics = [v for k in CONTRASTS for v in (k, k + "_per_spike")]
    events = scores.groupby(KEYS, as_index=False)[metrics + ["n_heldout_spikes", "n_train_spikes", "duration_s"]].median()
    saved_events = pd.read_csv(root / "tanni_multicontext_event_contrasts.csv")
    close(events.sort_values(KEYS)[metrics], saved_events.sort_values(KEYS)[metrics])
    sessions = events.groupby(["animal", "session", "context", "phase"])[metrics].mean()
    contexts = sessions.groupby(["animal", "context", "phase"])[metrics].mean()
    animals = contexts.groupby(["animal", "phase"])[metrics].mean()
    for name, data in (("sessions", sessions), ("contexts", contexts), ("animals", animals)):
        stored = pd.read_csv(root / f"tanni_multicontext_{name}.csv").set_index(data.index.names)
        close(data.sort_index(), stored.sort_index()[metrics])
    summary = pd.read_csv(root / "tanni_multicontext_summary.csv").set_index("metric")
    run = scores[scores.phase.eq("RUN")].copy()
    for model in ("iid", "global"):
        context_index = run[[f"{model}_p_{c}" for c in "ABCD"]].to_numpy().argmax(axis=1)
        actual = np.array(list("ABCD"))[context_index] == run.context.to_numpy()
        np.testing.assert_array_equal(actual, run[f"{model}_context_correct"])
    accuracies = run.groupby(["animal", "session", "context"])[["iid_context_correct", "global_context_correct", "iid_correct_context_probability"]].mean()
    accuracies = accuracies.groupby(["animal", "context"]).mean().groupby("animal").mean()
    saved_run = pd.read_csv(root / "tanni_multicontext_RUN_validation.csv").set_index("animal")
    close(accuracies, saved_run[accuracies.columns])
    run_pass = bool(accuracies.iid_context_correct.gt(0.5).all() and animals.xs("RUN", level="phase").current_iid_minus_current_global.gt(0).all())
    decisions = pd.read_csv(root / "tanni_multicontext_decisions.csv").iloc[0]
    assert decisions.RUN_context_validation_passed == run_pass
    assert decisions.all_candidate_predictive_contrasts_passed == primary_pass(summary)
    assert decisions.cross_context_spatial_reactivation_lead == (run_pass and primary_pass(summary))
    mua_animals = animals.xs("MUA", level="phase")
    exact_draws = np.array(list(itertools.product(range(5), repeat=5)))
    rows = []
    for metric in metrics:
        values = mua_animals[metric].to_numpy()
        close(summary.loc[metric, "mean"], values.mean())
        assert summary.loc[metric, "positive_animals"] == (values > 0).sum()
        lo, hi = np.quantile(values[exact_draws].mean(axis=1), [0.025, 0.975])
        rows.append({"metric": metric, "mean": values.mean(), "animal_only_ci_low": lo, "animal_only_ci_high": hi})
    pd.DataFrame(rows).to_csv(out / "independent_animal_only_bootstrap.csv", index=False)
    report = {
        "status": "passed",
        "source_manifest_sha256": digest(manifest_path),
        "raw_cached_timestamp_windows_recounted": raw_windows,
        "raw_cached_timestamp_bins_recounted": raw_bins,
        "independent_prediction_rows": sampled_scores,
        "maximum_prediction_error": maximum_error,
        "event_split_rows": len(scores),
        "MUA_events": int(events.phase.eq("MUA").sum()),
        "RUN_windows": int(events.phase.eq("RUN").sum()),
        "source_rate_maps_refitted": False,
        "original_NWB_reopened": False,
        "hierarchical_CI_exactly_reconstructed": False,
        "independent_animal_only_CI_sensitivity": True,
        "scope": "All source/output hashes, training-only map-bank construction, unit alignment, raw cached timestamp counts, calibration exclusion, fixed cell splits, sampled independent predictive equations, all event/split/group means. Parent native/map audits reused. Animal-only bootstrap is separate sensitivity, not the hierarchical CI.",
    }
    (out / "tanni_multicontext_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    verify(args.input_dir, args.output_dir)
