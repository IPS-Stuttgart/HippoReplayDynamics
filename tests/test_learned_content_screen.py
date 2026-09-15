import json
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import learned_content_screen as m
from scripts.audit_content_screening_bound import verify_certificate
from scripts import audit_learned_content_screen as a
from scripts import report_learned_content_screen as r


def calibration():
    frames = []
    for rat in m.RATS:
        for source in m.CAL:
            f = np.repeat(np.linspace(0, 1, 256), 2)
            label = np.tile([0, 1], 256)
            high, low = 0.6 + 0.3 * f, 0.6 + 0.1 * f
            raw = pd.DataFrame(
                dict(
                    high_home=high,
                    low_home=low,
                    high_entropy=0.3 + 0.5 * f,
                    low_entropy=0.4 + 0.5 * f,
                    high_spikes=1 + 20 * (1 - f),
                    low_spikes=2 + 15 * (1 - f),
                    high_active=1 + 5 * (1 - f),
                    low_active=1 + 4 * (1 - f),
                    separation=10 + 60 * f,
                    regional_tv=0.1 + 0.5 * f,
                    high_error=20 + 40 * f,
                    low_error=25 + 45 * f,
                    high_brier=(high - label) ** 2,
                    low_brier=(low - label) ** 2,
                    true_home=label,
                )
            )
            frames.append(m.enrich(raw, 200).assign(session=rat + "/Open1", animal=rat, source=source, observation_index=np.arange(len(f))))
    return pd.concat(frames, ignore_index=True)


def test_tree_policy_is_frozen_and_held_rat_cannot_affect_it(tmp_path):
    frame = calibration()
    model = m.fit_model(frame, "Rat1", tmp_path)
    assert model["maximum_training_progress"] > 0.9
    assert model["training_sessions"] == ["Rat2/Open1", "Rat4/Open1"]
    changed = frame.copy()
    changed.loc[changed.animal.eq("Rat1"), list(m.FEATURES) + list(set(m.COSTS) - set(m.FEATURES))] = 1000
    second = m.fit_model(changed, "Rat1", tmp_path)
    assert second == model
    scored = m.predict(json.loads(json.dumps(model)), frame)
    no_truth = frame.drop(columns=["true_home", "high_error", "low_error", "high_brier", "low_brier", "normalized_high_error", "normalized_low_error"])
    np.testing.assert_array_equal(scored, m.predict(model, no_truth))
    assert np.isfinite(scored).all() and (scored >= -1e-8).all() and (scored <= 1 + 1e-8).all()
    chosen = m.selection_frame(frame[frame.source.eq(m.CAL[0]) & frame.animal.eq("Rat1")], model)
    assert chosen.predictive_half.sum() == 256


def test_calibration_weights_and_certificates(tmp_path):
    frame = calibration()
    model = m.fit_model(frame, "Rat1", tmp_path)
    frame = frame[frame.animal.ne("Rat1")].reset_index(drop=True)
    weights = m.training_weights(frame)
    check = frame.assign(weight=weights).groupby(["source", "animal", "true_home"]).weight.sum()
    np.testing.assert_allclose(check, 1 / 12)
    _, ids = np.unique(m.tree_leaves(model["tree"], frame[list(m.FEATURES)]), return_inverse=True)
    first = m.build_problem(frame, ids, weights)
    with np.load(tmp_path / "Rat1_maximal.npz") as z:
        cert = dict(z)
    assert verify_certificate(first, cert)["primal_residual"] < 1e-7
    second = m.conservative_problem(first, cert["x"][-1], np.bincount(ids, weights=weights))
    with np.load(tmp_path / "Rat1_conservative.npz") as z:
        cert2 = dict(z)
    assert verify_certificate(second, cert2)["duality_gap"] < 1e-6


def test_missing_calibration_class_and_unknown_rat_rejected(tmp_path):
    frame = calibration()
    with pytest.raises(ValueError, match="class"):
        m.training_weights(frame[~(frame.animal.eq("Rat4") & frame.source.eq("run_q3") & frame.true_home.eq(1))])
    with pytest.raises(ValueError, match="held rat"):
        m.fit_model(frame, "Rat9", tmp_path)


def test_invalid_features_and_corrupt_tree_rejected():
    frame = calibration()
    with pytest.raises(ValueError):
        m.enrich(frame, 0)
    tree = dict(left=[0], right=[0], feature=[0], threshold=[0.5])
    with pytest.raises(ValueError, match="cyclic"):
        m.tree_leaves(tree, frame[list(m.FEATURES)])


def test_exact_half_ties_do_not_depend_on_truth(tmp_path):
    frame = calibration()
    model = m.fit_model(frame, "Rat1", tmp_path)
    frame = frame[frame.source.eq("run_q3") & frame.animal.eq("Rat1")].iloc[:101].copy()
    model["probabilities"] = [0.5] * len(model["leaf_nodes"])
    first = m.selection_frame(frame, model)
    frame["true_home"] = np.nan
    second = m.selection_frame(frame, model)
    np.testing.assert_array_equal(first.predictive_half, second.predictive_half)
    assert second.predictive_half.sum() == 51


def test_independent_refit_and_model_tamper(tmp_path):
    cal = calibration()
    model = m.fit_model(cal, "Rat1", tmp_path)
    residuals = a.refit_and_verify(cal, "Rat1", model, tmp_path)
    assert max(v["duality_gap"] for v in residuals) < 1e-6
    bad = deepcopy(model)
    bad["probabilities"][0] += 0.1
    with pytest.raises(AssertionError):
        a.refit_and_verify(cal, "Rat1", bad, tmp_path)


def test_full_artifact_reconstruction_and_rehashed_selection_tamper(tmp_path):
    source, previous = tmp_path / "source", tmp_path / "previous"
    source.mkdir()
    previous.mkdir()
    rng = np.random.default_rng(346)
    inputs, outputs = {}, {}
    for session in m.base.SESSIONS:
        slug = session.replace("/", "_")
        folder = source / slug
        folder.mkdir()
        grid = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
        rates = rng.uniform(1, 30, (12, 4))
        enc = dict(early_run=rates, full_run=rates * 1.05, grid_cm=grid, near=np.array([True, True, False, False]), high_indices=np.arange(6), low_indices=np.arange(6, 12))
        np.savez(folder / "encoding.npz", **enc)
        events = []
        for src in m.CAL + m.base.REAL + m.base.TRUTH:
            n = 512 if src in m.CAL else 40
            positions = rng.integers(0, 4, n)
            counts = rng.poisson(0.02 * rates[:, positions].T)
            truth = grid[positions] if src not in m.base.REAL else np.full((n, 2), np.nan)
            bank = dict(counts=counts, truth_cm=truth, event_ids=np.array([f"e{j:04}" for j in range(n)]))
            if src not in m.base.REAL:
                bank["labels"] = enc["near"][positions]
            if src == "run_q3":
                bank["parent_ids"] = np.repeat(np.arange(n // 2), 2)
            np.savez(folder / f"{src}.npz", **bank)
            if src not in m.CAL:
                for encoding in ("early_run", "full_run") if src in m.base.REAL else ("early_run",):
                    values = m.score_pair(bank, enc, encoding, dict(high=[], low=[]))
                    events.append(
                        pd.DataFrame(values).assign(
                            session=session,
                            animal=session.split("/")[0],
                            source=src,
                            encoding=encoding,
                            observation_index=np.arange(n),
                            event_id=bank["event_ids"],
                            method="baseline",
                        )
                    )
        path = previous / f"{slug}_events.csv.gz"
        pd.concat(events, ignore_index=True).drop(columns="animal").to_csv(path, index=False)
        outputs[path.name] = a.sha(path)
        inputs.update({str(p): a.sha(p) for p in folder.iterdir()})
    (previous / "manifest.json").write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=inputs, output_sha256=outputs, synthetic_fixture=True)))
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text(json.dumps(dict(status="pass", manifest_sha256=a.sha(previous / "manifest.json"), synthetic_fixture=True)))
    root = tmp_path / "measurement"
    m.measure(Namespace(result_dir=previous, audit=source_audit, output_dir=root))
    verified = a.audit(root, tmp_path / "audit.json")
    assert verified["models_refitted"] == 3 and verified["status"] == "pass"
    assert not verified["validated_remedy"] and not verified["external_validation"]
    report_dir = tmp_path / "report"
    r.report(root, tmp_path / "audit.json", report_dir)
    assert (report_dir / "learned_content_screen.png").stat().st_size > 1000
    report_manifest = json.loads((report_dir / "report_manifest.json").read_text())
    assert not report_manifest["validated_remedy"] and not report_manifest["external_validation"]
    assert "NOT ESTABLISHED" in (report_dir / "report.md").read_text()
    frame = pd.read_csv(root / "event_selection.csv.gz")
    frame.loc[0, "predictive_half"] = not frame.loc[0, "predictive_half"]
    frame.to_csv(root / "event_selection.csv.gz", index=False)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["output_sha256"]["event_selection.csv.gz"] = a.sha(root / "event_selection.csv.gz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        a.audit(root, tmp_path / "bad_audit.json")
