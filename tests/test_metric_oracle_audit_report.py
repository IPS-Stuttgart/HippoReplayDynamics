import json
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.metric_oracle_recovery import whole_evidence
from scripts._provenance import file_sha256
from scripts.report_2d_metric_oracle_recovery import report
from scripts.run_2d_metric_oracle_recovery import MODELS, OBSERVED, aggregate, classify
from scripts.verify_2d_metric_finite_recovery import assert_table
from scripts.verify_2d_metric_oracle_recovery import KEYS, backward_evidence, reference_classification


def test_independent_backward_and_log_library():
    from hmmlearn import _hmmc

    rng = np.random.default_rng(975)
    ll = np.log(rng.uniform(0.001, 0.9, (9, 4))) - 1000
    a, b = [rng.dirichlet(np.ones(4), 4) for _ in range(2)]
    offsets = np.array([0, 3, 9])
    got = whole_evidence(ll, offsets, {"physical": a, "neural": b})
    for model, kernel in (("physical", a), ("neural", b)):
        independent = backward_evidence(ll, offsets, kernel)
        np.testing.assert_allclose(got[model], independent, atol=1e-10, rtol=0)
        z, _ = _hmmc.forward_log(np.ones(4) / 4, kernel, ll[3:])
        assert z == pytest.approx(got[model][1], abs=1e-10)
    assert got["iid"][0] == pytest.approx((logsumexp(ll[:3], axis=1) - np.log(4)).sum())


def example(ids):
    factors = [("latent_path", 0, -1)] + list(product(OBSERVED, (1, 4), range(5)))
    base = pd.DataFrame([(g, t, *f) for g in MODELS for t in range(64) for f in factors], columns=["generator", "trial", "method", "support", "split"])
    base["phase"] = np.where(base.trial < 32, "calibration", "evaluation")
    base["informative"] = True
    base["score_kind"] = np.where(
        base.method == "latent_path", "latent_path_log_probability", np.where(base.method.str.startswith("whole_"), "joint_event_log_probability", "sum_40ms_predictive_log_scores")
    )
    for m in MODELS:
        base["score_" + m] = -10.0
    for g in MODELS[:2]:
        mask = base.generator.eq(g)
        base.loc[mask, ["score_physical", "score_neural"]] = -3.0
        base.loc[mask, "score_" + g] = -2.0
    return pd.concat([base.assign(dataset=d, animal=a, session=s) for d, a, s in ids], ignore_index=True)


def test_realization_reductions_and_corruption():
    ids = [("pfeiffer_foster", f"Rat{a}", f"{a}d{s}") for a in range(4) for s in range(2)]
    ids += [("tanni2022", f"R{a}", f"{a}d{s}") for a in range(5) for s in range(5)]
    scores = example(ids)
    actual = classify(scores)
    independent = reference_classification(scores)
    assert_table(actual[0], independent[0], KEYS + ["split"])
    assert_table(actual[1], independent[1], ["dataset", "animal", "session", "method", "support", "split"])
    assert_table(actual[2], independent[2], KEYS)
    with pytest.raises(AssertionError):
        reference_classification(scores.iloc[:-1])


def test_report_smoke_and_unverified_run_rejected(tmp_path):
    scores = example([("pfeiffer_foster", "Rat1", "day1"), ("tanni2022", "R2470", "day1")])
    outputs = aggregate(scores)
    names = ("decisions", "thresholds", "trials", "sessions", "animals", "summary", "paired", "checks", "decision")
    run = tmp_path / "run"
    run.mkdir()
    for name, frame in zip(names, outputs, strict=True):
        frame.to_csv(run / f"metric_oracle_{name}.csv.gz", index=False)
    mp = run / "metric_oracle_manifest.json"
    mp.write_text(
        json.dumps({"status": "complete", "code_commit": "synthetic", "completed": [{}, {}], "rows": len(scores), "output_sha256": {f.name: file_sha256(f) for f in run.iterdir()}})
    )
    audit = tmp_path / "audit.json"
    a = {
        "status": "pass",
        "input_file_sha256": {"run_manifest": file_sha256(mp)},
        "independently_reconstructed_scores": len(scores),
        "verified_parent_copies": 100,
        "hmmlearn_log_crosschecks": 12,
        "max_absolute_score_error": 1e-12,
    }
    audit.write_text(json.dumps(a))
    output = tmp_path / "report"
    report(run, audit, output)
    text = (output / "metric_oracle_report.md").read_text()
    assert "No real replay event was rescored" in text
    assert "not held-out predictive validation" in text
    assert (output / "metric_oracle_recovery.png").stat().st_size > 10000
    a["status"] = "fail"
    audit.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="verified completed"):
        report(run, audit, tmp_path / "bad")
