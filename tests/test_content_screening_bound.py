from argparse import Namespace
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import bound_content_screening as m
from scripts import audit_content_screening_bound as a


def example(known=True):
    # Identical observations can have different true localization errors.
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1], float)
    high = np.r_[np.full(4, 0.2), np.full(4, 0.8)]
    low = np.full(8, 0.1)
    frame = pd.DataFrame(
        dict(
            high_home=high,
            low_home=low,
            separation=np.r_[np.full(4, 0.1), np.ones(4)],
            regional_tv=high - low,
            high_entropy=np.r_[np.full(4, 0.2), np.full(4, 0.8)],
            low_entropy=np.r_[np.full(4, 0.2), np.full(4, 0.8)],
            high_error=[5, 25, 5, 25, 5, 5, 5, 5],
            low_error=[5, 25, 5, 25, 5, 5, 5, 5],
            high_brier=(high - labels) ** 2,
            low_brier=(low - labels) ** 2,
            true_home=labels,
        )
    )
    if not known:
        frame["true_home"] = np.nan
        frame[list(m.LOSSES)] = np.nan
    counts = np.array([[1, 0]] * 4 + [[0, 1]] * 4)
    return frame, counts


def test_identical_codes_share_probability_but_free_oracle_does_not(tmp_path):
    frame, counts = example()
    tied = m.solve_case(frame, counts, 0.5, "count_pattern", "truth_guarded", tmp_path / "tied.npz")
    free = m.solve_case(frame, counts, 0.5, "free_event", "truth_guarded", tmp_path / "free.npz")
    assert tied["max_progress"] < 1e-6
    assert not tied["targets_attainable"]
    assert free["max_progress"] == pytest.approx(1.875)
    assert free["targets_attainable"]
    for oracle, path in (("count_pattern", "tied"), ("free_event", "free")):
        problem = a.rebuild(frame, counts, 0.5, oracle, "truth_guarded")
        assert a.verify_certificate(problem, a.load(tmp_path / f"{path}.npz"))["duality_gap"] < 1e-7


def test_truth_guards_prevent_agreement_by_harming_accuracy(tmp_path):
    frame, counts = example()
    only = m.solve_case(frame, counts, 0.5, "count_pattern", "agreement_only", tmp_path / "only.npz")
    assert only["targets_attainable"]
    assert only["retained_balanced_high_error"] > only["baseline_balanced_high_error"]


def test_equal_true_class_retention_and_constant_feasibility():
    frame, counts = example()
    for q in m.COVERAGES:
        p, _, _ = m.build_problem(frame, counts, q, "count_pattern", "truth_guarded")
        x = np.r_[np.full(len(p["c"]) - 1, q), 0]
        np.testing.assert_allclose(p["E"] @ x, p["f"])
        assert np.max(p["A"] @ x - p["b"]) < 1e-10


def test_zero_disagreement_is_not_a_nonvacuous_remedy(tmp_path):
    frame, counts = example(False)
    frame["low_home"] = frame.high_home
    frame[["regional_tv", "separation"]] = 0
    row = m.solve_case(frame, counts, 0.5, "count_pattern", "agreement_only", tmp_path / "zero.npz")
    assert not row["baseline_target_nonzero"] and not row["targets_attainable"]


@pytest.mark.parametrize("change", ["empty", "fractional", "negative", "nan"])
def test_invalid_counts_rejected(change):
    counts = example()[1].astype(float)
    if change == "empty":
        counts = counts[:0]
    else:
        counts[0, 0] = {"fractional": 0.5, "negative": -1, "nan": np.nan}[change]
    with pytest.raises(ValueError):
        m.groups(counts, "count_pattern")


def test_unknown_replay_truth_cannot_be_used_as_guard():
    frame, counts = example(False)
    with pytest.raises(ValueError, match="unknown"):
        m.build_problem(frame, counts, 0.5, "count_pattern", "truth_guarded")


def test_primal_and_dual_tampering_rejected(tmp_path):
    frame, counts = example()
    m.solve_case(frame, counts, 0.5, "free_event", "truth_guarded", tmp_path / "proof.npz")
    p = a.rebuild(frame, counts, 0.5, "free_event", "truth_guarded")
    cert = a.load(tmp_path / "proof.npz")
    bad = deepcopy(cert)
    bad["x"][-1] += 0.2
    with pytest.raises(AssertionError):
        a.verify_certificate(p, bad)
    bad = deepcopy(cert)
    bad["upper_dual"][:] = 1
    with pytest.raises(AssertionError):
        a.verify_certificate(p, bad)


def test_nonvacuous_weighted_metrics():
    with pytest.raises(ValueError):
        m.metrics(example()[0], np.zeros(8))


def test_collectively_material_tiny_coefficients_are_not_dropped():
    n = 2000
    problem = dict(
        c=np.r_[np.zeros(n), -1.0],
        A=np.array([np.r_[np.full(n, 1e-10), 1.0]]),
        b=np.array([1.0]),
        E=np.array([np.r_[np.ones(n), 0.0]]),
        f=np.array([float(n)]),
        lo=np.zeros(n + 1),
        hi=np.r_[np.ones(n), 5.0],
        group=np.arange(n),
    )
    solved = m.solve_program(problem)
    assert solved.status == 0
    assert solved.x[-1] == pytest.approx(1 - n * 1e-10, abs=1e-10)
    certificate = dict(
        x=solved.x,
        inequality_dual=solved.ineqlin.marginals,
        equality_dual=solved.eqlin.marginals,
        lower_dual=solved.lower.marginals,
        upper_dual=solved.upper.marginals,
        group=problem["group"],
    )
    assert a.verify_certificate(problem, certificate)["primal_residual"] < 1e-10


def test_full_artifact_certificates_and_rehashed_summary_tamper(tmp_path):
    source, result = tmp_path / "source", tmp_path / "existing_result"
    source.mkdir()
    result.mkdir()
    input_hashes, output_hashes = {}, {}
    for session in m.SESSIONS:
        slug = session.replace("/", "_")
        folder = source / slug
        folder.mkdir()
        np.savez(folder / "encoding.npz", high_indices=np.array([0]), low_indices=np.array([1]))
        events = []
        for src in m.REAL + m.TRUTH:
            frame, counts = example(src in m.TRUTH)
            event_ids = np.array([f"e{i:03}" for i in range(8)])
            np.savez(folder / f"{src}.npz", counts=counts, event_ids=event_ids)
            for enc in ("early_run", "full_run") if src in m.REAL else ("early_run",):
                events.append(frame.assign(event_id=event_ids, observation_index=np.arange(8), session=session, source=src, encoding=enc, method="baseline"))
        path = result / f"{slug}_events.csv.gz"
        pd.concat(events, ignore_index=True).to_csv(path, index=False)
        output_hashes[path.name] = a.sha(path)
        for path in folder.iterdir():
            input_hashes[str(path)] = a.sha(path)
    source_manifest = result / "manifest.json"
    source_manifest.write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=input_hashes, output_sha256=output_hashes, synthetic_fixture=True)))
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text(json.dumps(dict(status="pass", manifest_sha256=a.sha(source_manifest), synthetic_fixture=True)))
    dest = tmp_path / "measurement"
    m.measure(Namespace(result_dir=result, audit=source_audit, output_dir=dest))
    verified = a.audit(dest, tmp_path / "audit.json")
    assert verified["status"] == "pass" and verified["cases"] == 384
    table = pd.read_csv(dest / "bounds.csv")
    table.loc[0, "max_progress"] += 0.05
    table.to_csv(dest / "bounds.csv", index=False)
    manifest = json.loads((dest / "manifest.json").read_text())
    manifest["output_sha256"]["bounds.csv"] = a.sha(dest / "bounds.csv")
    (dest / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        a.audit(dest, tmp_path / "bad_audit.json")
