import json
from itertools import product

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.metric_finite_recovery import score_batch
from scripts._provenance import file_sha256
from scripts.report_2d_metric_finite_recovery import report
from scripts.run_2d_metric_finite_recovery import aggregate, reduce_trials
from scripts.verify_2d_metric_finite_recovery import KEYS, MODELS, assert_table, reference_reductions, reference_scores


def test_independent_score_reconstruction_and_corruption():
    rng = np.random.default_rng(717)
    rates = rng.gamma(2, 3, (8, 6)) + 0.01
    x, y = rng.poisson(1, (9, 4)), rng.poisson(2, (9, 4))
    states, offsets = rng.integers(6, size=9), np.array([0, 4, 9])
    a = rng.dirichlet(np.ones(6), 6)
    b = rng.dirichlet(np.ones(6), 6)
    kernels = {"physical": a @ a, "neural": b @ b}
    expected = reference_scores(x, y, rates[:4], rates[4:], states, offsets, kernels)
    actual = pd.DataFrame(score_batch(x, y, rates[:4], rates[4:], states, offsets, kernels))
    assert_table(actual, expected, ["simulation_index", "origin"])
    actual.loc[0, "score_neural"] += 0.01
    with pytest.raises(AssertionError):
        assert_table(actual, expected, ["simulation_index", "origin"])


def example_scores(ids):
    rows = list(product(MODELS, range(64), ("matched", "gain_drift", "map_error"), (1, 4), ("decoded", "known"), range(5)))
    base = pd.DataFrame(rows, columns=["generator", "trial", "condition", "support", "origin", "split"])
    base["phase"] = np.where(base.trial < 32, "calibration", "evaluation")
    base["n_held_target_spikes"], base["n_train_origin_spikes"] = 7, 18
    for model in MODELS:
        base["score_" + model] = -10.0
    for generator in MODELS[:2]:
        mask = base.generator.eq(generator)
        base.loc[mask, ["score_physical", "score_neural"]] = -3.0
        base.loc[mask, "score_" + generator] = -2.0
    return pd.concat([base.assign(dataset=d, animal=a, session=s) for d, a, s in ids], ignore_index=True)


def test_all_recording_reductions_and_missing_rows():
    ids = [("pfeiffer_foster", f"Rat{a}", f"{a}day{s}") for a in range(4) for s in range(2)]
    ids += [("tanni2022", f"R{a}", f"{a}day{s}") for a in range(5) for s in range(5)]
    scores = example_scores(ids)
    expected, thresholds = reference_reductions(scores)
    actual = reduce_trials(scores)
    assert_table(actual, expected[actual.columns], KEYS)
    assert thresholds.finite_calibration.all()
    with pytest.raises(AssertionError):
        reference_reductions(scores.iloc[:-1])


def test_report_smoke_and_changed_input_rejected(tmp_path):
    scores = example_scores([("pfeiffer_foster", "Rat1", "day1"), ("tanni2022", "R2470", "day1")])
    data = aggregate(scores)
    run = tmp_path / "run"
    run.mkdir()
    names = ("trials", "thresholds", "sessions", "animals", "summary", "checks", "decision", "winners")
    for name, frame in zip(names, data, strict=True):
        frame.to_csv(run / f"metric_finite_recovery_{name}.csv.gz", index=False)
    manifest = {
        "status": "complete",
        "code_commit": "synthetic-fixture",
        "completed": [{}, {}],
        "rows": len(scores),
        "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()},
    }
    mp = run / "metric_finite_recovery_manifest.json"
    mp.write_text(json.dumps(manifest))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"status": "pass", "input_file_sha256": {"run_manifest": file_sha256(mp)}, "predictive_scores_checked": len(scores) * 4, "max_absolute_score_error": 1e-12})
    )
    out = tmp_path / "report"
    report(run, audit, out)
    text = (out / "metric_finite_recovery_report.md").read_text()
    assert "no real replay event was rescored" in text
    assert "not biological population intervals" in text
    assert (out / "metric_finite_recovery_primary.png").stat().st_size > 10000
    (run / "metric_finite_recovery_checks.csv.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed audited artifact"):
        report(run, audit, tmp_path / "bad")
