import json
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import local_content_screen as m
from scripts import audit_local_content_screen as a
from scripts import report_local_content_screen as r


def calibration():
    frames = []
    for session in m.old.base.SESSIONS:
        for source in m.old.CAL:
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
            frames.append(m.old.enrich(raw, 200).assign(session=session, animal=session.split("/")[0], source=source, observation_index=np.arange(len(f))))
    return pd.concat(frames, ignore_index=True)


def test_local_training_excludes_other_sessions_and_truth_not_a_feature(tmp_path):
    cal = calibration()
    session = m.old.base.SESSIONS[0]
    model = m.fit_model(cal, session, tmp_path)
    assert model["training_sessions"] == [session]
    assert "true_home" not in model["features"] and "true_home" in model["target_names"]
    changed = cal.copy()
    changed.loc[changed.session.ne(session), "high_home"] = -1000
    assert m.fit_model(changed, session, tmp_path) == model
    x = cal.loc[cal.session.eq(session), list(m.old.FEATURES)]
    np.testing.assert_array_equal(m.old.predict(model, x), m.old.predict(model, x.assign(true_home=-1000)))
    assert max(r["duality_gap"] for r in a.refit(cal, session, model, tmp_path)) < 1e-6
    bad = deepcopy(model)
    bad["probabilities"][0] += 0.1
    with pytest.raises(AssertionError):
        a.refit(cal, session, bad, tmp_path)


def test_class_guard_rejects_loss_hidden_by_balanced_mean():
    source = calibration().iloc[:4].copy()
    for col in list(m.old.COSTS):
        source[col] = 0.25
    source["true_home"] = [0, 0, 1, 1]
    source["normalized_high_error"] = [0.01, 0.09, 0.01, 0.09]
    source["normalized_low_error"] = source.normalized_high_error
    frame = pd.concat([source.assign(source=c) for c in m.old.CAL], ignore_index=True)
    w = m.old.training_weights(frame)
    groups = np.arange(len(frame))
    policy = np.r_[np.tile([1, 0, 0, 1], 3), 0]
    first = m.old.build_problem(frame, groups, w)
    assert (first["A"] @ policy <= first["b"] + 1e-10).all()
    np.testing.assert_allclose(first["E"] @ policy, first["f"])
    protected = m.build_problem(frame, groups, w)
    assert (protected["A"] @ policy - protected["b"]).max() > 0.1
    const = np.r_[np.full(len(frame), 0.5), 0]
    assert (protected["A"] @ const <= protected["b"] + 1e-10).all()


def test_missing_local_class_and_unknown_session_rejected(tmp_path):
    frame = calibration()
    session = m.old.base.SESSIONS[0]
    with pytest.raises(ValueError, match="class"):
        m.fit_model(frame[~(frame.session.eq(session) & frame.source.eq("run_q3") & frame.true_home.eq(1))], session, tmp_path)
    with pytest.raises(ValueError, match="session"):
        m.fit_model(frame, "Rat9/Open1", tmp_path)


def test_hard_half_and_ties_do_not_use_truth(tmp_path):
    cal = calibration()
    session = m.old.base.SESSIONS[0]
    model = m.fit_model(cal, session, tmp_path)
    model["probabilities"] = [0.5] * len(model["probabilities"])
    frame = cal[cal.session.eq(session) & cal.source.eq("run_q3")].iloc[:101].assign(encoding="early_run")
    first = m.select(frame, model)
    second = m.select(frame.assign(true_home=np.nan), model)
    assert first.predictive_half.sum() == 51
    np.testing.assert_array_equal(first.predictive_half, second.predictive_half)
    with pytest.raises(ValueError, match="matching session"):
        m.select(frame.assign(session="Rat9/Open1"), model)


def test_region_gate_is_nonvacuous_and_detects_one_local_failure():
    rows = []
    for session in m.old.base.SESSIONS:
        for source in m.old.base.TRUTH:
            for label in (0, 1):
                for method in ("all", "local_half"):
                    rows.append(dict(session=session, source=source, true_home=label, method=method, **{key: 1.0 for key in m.CLASS_METRICS}))
    table = pd.DataFrame(rows)
    assert all(m.regional_flags(table).values())
    assert m.regional_flags(table) == a.class_flags(table)
    table.loc[1, "high_brier"] = 2
    assert not m.regional_flags(table)["regional_run_q4_high_brier"]
    assert m.regional_flags(table) == a.class_flags(table)
    assert not any(m.regional_flags(table.iloc[:0]).values())


def test_complete_pipeline_and_rehashed_selection_tamper(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    cal = calibration()
    cal.to_csv(previous / "calibration.csv.gz", index=False)
    frames = []
    for session in m.old.base.SESSIONS:
        template = cal[cal.session.eq(session) & cal.source.eq("cal_conditional")].iloc[:80].copy()
        for source in m.old.base.REAL + m.old.base.TRUTH:
            for encoding in ("early_run", "full_run") if source in m.old.base.REAL else ("early_run",):
                frame = template.assign(source=source, encoding=encoding, event_id=[f"event{j}" for j in range(len(template))])
                frame["all"] = True
                frame["predictive_half"] = False
                frame["spike_half"] = False
                frame["entropy_half"] = False
                frame["pair_score"] = 0.5
                if source in m.old.base.REAL:
                    frame["true_home"] = np.nan
                frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(previous / "event_selection.csv.gz", index=False)
    manifest = dict(input_file_sha256={}, output_sha256={p.name: a.sha(p) for p in previous.iterdir()}, synthetic_fixture=True)
    (previous / "manifest.json").write_text(json.dumps(manifest))
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=a.sha(previous / "manifest.json"), synthetic_fixture=True)))
    root = tmp_path / "measurement"
    m.measure(Namespace(result_dir=previous, audit=audit, output_dir=root))
    result = a.audit(root, tmp_path / "audit.json")
    assert result["status"] == "pass" and result["models_refitted"] == 4
    assert not result["external_validation"] and not result["validated_remedy"]
    r.report(root, tmp_path / "audit.json", tmp_path / "report")
    assert (tmp_path / "report/local_content_screen.png").stat().st_size > 1000
    assert "NOT ESTABLISHED" in (tmp_path / "report/report.md").read_text()
    events = pd.read_csv(root / "event_selection.csv.gz")
    events.loc[0, "predictive_half"] = not events.loc[0, "predictive_half"]
    events.to_csv(root / "event_selection.csv.gz", index=False)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["output_sha256"]["event_selection.csv.gz"] = a.sha(root / "event_selection.csv.gz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        a.audit(root, tmp_path / "bad_audit.json")
