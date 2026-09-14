"""Recompute the frozen error-trained context report without importing its code."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True)


def check(report):
    report = Path(report)
    manifest = json.loads((report / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["primary_method"] == "error_trained_context"
    for key, path in manifest["input_file_paths"].items():
        assert digest(path) == manifest["input_file_sha256"][key], key
    for name, value in manifest["output_sha256"].items():
        assert digest(report / name) == value, name
    producer = json.loads(Path(manifest["input_file_paths"]["producer"]).read_text())
    audit = json.loads(Path(manifest["input_file_paths"]["audit"]).read_text())
    assert producer["status"] == "complete" and audit["status"] == "passed"
    assert audit["input_file_sha256"]["producer"] == digest(manifest["input_file_paths"]["producer"])
    frames = [pd.read_csv(Path(x["artifact_dir"]) / "event_readouts.csv.gz", float_precision="round_trip") for x in producer["results"]]
    raw = pd.concat(frames, ignore_index=True)
    assert raw.dataset.nunique() == 1 and raw.session.nunique() == 8 and raw.animal.nunique() == 4
    assert set(raw.method) == {"independent", "unconditional_context", "error_trained_context", "entropy_matched"}
    ids = ["dataset", "animal", "session", "split", "source", "method"]
    assert not raw.duplicated(ids + ["event_index"]).any()
    core = ["separation_cm", "regional_tv", "a_entropy", "b_entropy"]
    truth = ["a_error", "b_error", "a_brier", "b_brier", "a_nll", "b_nll"]
    rows = []
    for key, group in raw.groupby(ids, sort=True):
        metrics = core if key[4] == "real" else core + truth
        for name in metrics:
            values = group[name].to_numpy()
            assert np.isfinite(values).all()
            rows.append((*key, name, np.sum(values) / len(values), len(values)))
        if key[4] != "real":
            for side in ("a", "b"):
                rows.append((*key, side + "_error_p90", np.percentile(group[side + "_error"], 90), len(group)))
    sessions = pd.DataFrame(rows, columns=ids + ["metric", "value", "n_events"])
    saved = pd.read_csv(report / "by_session.csv").set_index(ids + ["metric"]).sort_index()
    expected = sessions.set_index(ids + ["metric"]).sort_index()
    assert expected.index.equals(saved.index)
    close(saved[["value", "n_events"]], expected[["value", "n_events"]])
    animal_keys = ["dataset", "animal", "split", "source", "method", "metric"]
    animals = sessions.groupby(animal_keys).value.mean()
    saved_animals = pd.read_csv(report / "by_animal.csv").set_index(animal_keys).value.sort_index()
    assert animals.index.equals(saved_animals.index)
    close(animals, saved_animals)
    summary = pd.read_csv(report / "summary.csv")
    lookup = animals.to_dict()
    reconstructed = {}
    for row in summary.itertuples(index=False):
        base = np.array([lookup[(row.dataset, rat, row.split, row.source, "independent", row.metric)] for rat in sorted(raw.animal.unique())])
        values = np.array([lookup[(row.dataset, rat, row.split, row.source, row.method, row.metric)] for rat in sorted(raw.animal.unique())])
        delta = base - values
        draws = np.random.default_rng(20260914).integers(0, 4, size=(5000, 4))
        lo, hi = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        expected = [base.mean(), values.mean(), delta.mean(), delta.mean() / base.mean() if base.mean() > 0 else np.nan, lo, hi, 4, np.sum(delta > 0)]
        close([row.baseline, row.value, row.reduction, row.relative_reduction, row.reduction_ci_low, row.reduction_ci_high, row.animals, row.animals_improved], expected)
        reconstructed[(row.split, row.source, row.method, row.metric)] = expected
    assert len(reconstructed) == len(summary) == 768
    diag_keys = ["dataset", "animal", "session", "split", "source", "side"]
    diagnostics = pd.read_csv(report / "diagnostic_by_session.csv").set_index(diag_keys)
    expected_diag = {(*key, side) for key in raw.groupby(ids[:-1]).groups for side in ("a", "b")}
    assert diagnostics.index.is_unique and set(diagnostics.index) == expected_diag
    auc_rows = []
    grouped = {k: g.set_index("event_index").sort_index() for k, g in raw.groupby(ids)}
    for key, row in diagnostics.iterrows():
        dataset, rat, session, split, source, side = key
        baseline = grouped[(dataset, rat, session, split, source, "independent")]
        context = grouped[(dataset, rat, session, split, source, "unconditional_context")]
        assert baseline.index.equals(context.index)
        eligible = (baseline[side + "_spikes"].to_numpy() >= 3) & (baseline[side + "_active"].to_numpy() >= 2)
        gain = (baseline[side + "_error"] - context[side + "_error"]).to_numpy()
        score = baseline[side + "_predicted_physical_gain"].to_numpy()[eligible]
        value = gain[eligible]
        labels = value > 1e-8
        n1, n0 = int(labels.sum()), int((~labels).sum())
        available = len(value) > 0 and np.isfinite(value).all() and n1 > 0 and n0 > 0
        auc = (rankdata(score)[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if available else np.nan
        rho = (
            np.corrcoef(rankdata(score), rankdata(value))[0, 1]
            if len(value) > 2 and np.isfinite(value).all() and len(np.unique(value)) > 1 and len(np.unique(score)) > 1
            else np.nan
        )
        use = baseline[side + "_use_context"].to_numpy(dtype=bool)
        selected = gain[use]
        close(
            [row.events, row.eligible_events, row.context_events, row.eligible_fraction, row.context_fraction, row.gain_auroc, row.gain_spearman],
            [len(baseline), eligible.sum(), use.sum(), eligible.mean(), use.mean(), auc, rho],
        )
        assert bool(row.auroc_available) == bool(available)
        close(
            [row.selected_mean_truth_gain, row.selected_harm_fraction],
            [selected.mean() if len(selected) else np.nan, np.mean(selected < -1e-8) if len(selected) and np.isfinite(selected).all() else np.nan],
        )
        auc_rows.append((dataset, rat, split, source, side, auc))
    auc_table = pd.DataFrame(auc_rows, columns=["dataset", "animal", "split", "source", "side", "auc"])
    animal_diag_keys = ["dataset", "animal", "split", "source", "side"]
    saved_diag_animals = pd.read_csv(report / "diagnostic_by_animal.csv").set_index(animal_diag_keys)
    groups = diagnostics.reset_index().groupby(animal_diag_keys)
    assert saved_diag_animals.index.is_unique and set(saved_diag_animals.index) == set(groups.groups)
    for key, values in groups:
        actual = saved_diag_animals.loc[key]
        close(
            [
                actual.sessions,
                actual.events,
                actual.eligible_events,
                actual.context_events,
                actual.eligible_fraction,
                actual.context_fraction,
                actual.gain_auroc,
                actual.gain_spearman,
            ],
            [
                len(values),
                values.events.sum(),
                values.eligible_events.sum(),
                values.context_events.sum(),
                values.eligible_fraction.mean(),
                values.context_fraction.mean(),
                values.gain_auroc.mean() if values.auroc_available.all() else np.nan,
                values.gain_spearman.mean() if values.gain_spearman.notna().all() else np.nan,
            ],
        )
    checks = {"independently_verified": True}
    for metric in ("separation_cm", "regional_tv"):
        value = reconstructed[(0, "real", "error_trained_context", metric)]
        control = reconstructed[(0, "real", "entropy_matched", metric)][1]
        checks["real_reduction_" + metric] = value[3] >= 0.1 and value[7] >= 3 and value[4] > 0
        checks["beats_entropy_control_" + metric] = control > 0 and (control - value[1]) / control >= 0.05
    for side in ("a", "b"):
        checks["no_increased_" + side + "_entropy"] = reconstructed[(0, "real", "error_trained_context", side + "_entropy")][2] >= -1e-8
        for source in sorted(set(raw.source) - {"real"}):
            for metric in (side + "_error", side + "_error_p90", side + "_brier"):
                checks[source + "_no_worse_" + metric] = reconstructed[(0, source, "error_trained_context", metric)][2] >= -1e-8
    controls = raw.loc[(raw.split == 0) & (raw.method == "error_trained_context"), ["a_entropy_control_available", "b_entropy_control_available"]]
    checks["all_primary_entropy_controls_available"] = not controls.empty and controls.notna().all().all() and controls.eq(True).all().all()
    for source in ("run_q4", "sim_late_jump"):
        for side in ("a", "b"):
            per_rat = []
            for rat in sorted(raw.animal.unique()):
                values = auc_table.loc[(auc_table.split == 0) & (auc_table.source == source) & (auc_table.side == side) & (auc_table.animal == rat), "auc"].to_numpy()
                nsessions = raw.loc[raw.animal == rat, "session"].nunique()
                per_rat.append(values.mean() if len(values) == nsessions and np.isfinite(values).all() else np.nan)
            per_rat = np.array(per_rat)
            checks[f"{source}_{side}_predicts_known_gain"] = np.isfinite(per_rat).all() and per_rat.mean() >= 0.6 and np.sum(per_rat > 0.5) >= 3
    checks["advance_external_validation"] = all(checks.values())
    saved_gates = pd.read_csv(report / "gate_summary.csv")
    assert not saved_gates.gate.duplicated().any()
    assert {k: bool(v) for k, v in checks.items()} == dict(zip(saved_gates.gate, saved_gates.passed, strict=True))
    assert manifest["primary_advanced"] == bool(checks["advance_external_validation"])
    result = dict(
        status="passed",
        raw_rows=len(raw),
        session_metrics=len(sessions),
        animal_metrics=len(animals),
        summary_rows=len(summary),
        diagnostic_rows=len(diagnostics),
        gates=len(checks),
        report_sha256=digest(report / "manifest.json"),
        verifier_sha256=digest(__file__),
        imports_producer_or_reporter=False,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verified = check(sys.argv[1])
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(json.dumps(verified, indent=2) + "\n")
