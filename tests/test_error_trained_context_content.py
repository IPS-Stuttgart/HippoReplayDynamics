import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts._provenance import file_sha256
from scripts.audit_error_trained_context_content import forest_predict, verify_session, verify_training
from scripts.audit_predictive_context_content import verify_one
from scripts.error_trained_context_content import (
    FEATURES,
    METHODS,
    PRIMARY,
    SOURCES,
    choose,
    evaluate_session,
    features,
    fit_model,
    fit_states,
    load_models,
    predict_tree_model,
    training_rows,
    training_weights,
    validate_development,
)
from scripts.measure_edge_support_content import read_event
from scripts.measure_predictive_context_content import measure_session
from scripts.report_predictive_context_content import combined_gates, diagnostic_summary
from scripts.report_temporal_endpoint_content import TRUTH, summaries


def training_fixture():
    rng = np.random.default_rng(25)
    rows = []
    for rat in range(4):
        for session in range(2):
            for source in sorted(set(SOURCES) - {"real"}):
                for side in ("a", "b"):
                    for event in range(20):
                        x = rng.normal(size=len(FEATURES))
                        rows.append(
                            dict(zip(FEATURES, x, strict=True))
                            | dict(
                                animal=f"Rat{rat}",
                                session=f"Rat{rat}/Open{session}",
                                source=source,
                                side=side,
                                split=0,
                                event_index=event,
                                physical_gain=x[0] - x[1],
                                brier_gain=x[2] / 10,
                            )
                        )
    return pd.DataFrame(rows)


def test_serialized_forest_independent_walk_and_training_weights():
    frame = training_fixture()
    frame = frame.loc[~((frame.animal == "Rat0") & (frame.event_index < 9))].reset_index(drop=True)
    weights = training_weights(frame)
    weighted = frame.assign(weight=weights)
    np.testing.assert_allclose(weighted.groupby("animal").weight.sum(), 0.25)
    np.testing.assert_allclose(weighted.groupby(["animal", "session", "source", "side"]).weight.sum(), 1 / 80)
    state = fit_model(frame[list(FEATURES)], frame.physical_gain, weights)
    np.testing.assert_allclose(predict_tree_model(frame[list(FEATURES)], state), forest_predict(frame[list(FEATURES)], state), atol=1e-12)
    with pytest.raises(ValueError):
        fit_model(frame[list(FEATURES)], frame.physical_gain, np.full(len(frame), np.nan))


@pytest.mark.parametrize("change", ["replay", "split", "duplicate", "source"])
def test_training_rejects_leakage_and_missing_sources(change):
    frame = training_fixture()
    if change == "replay":
        frame.loc[0, "source"] = "real"
    elif change == "split":
        frame.loc[0, "split"] = 1
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame = frame.loc[~((frame.animal == "Rat0") & (frame.source == "sim_late_jump"))]
    with pytest.raises(ValueError):
        training_weights(frame)


def test_all_sessions_of_heldout_rat_excluded_from_fitting():
    training = training_fixture()
    before = fit_states(training)
    training.loc[training.animal == "Rat0", ["physical_gain", "brier_gain"]] = 1000
    after = fit_states(training)
    assert before["Rat0"] == after["Rat0"]
    assert before["external"] != after["external"]
    for rat in ("Rat0", "Rat1", "Rat2", "Rat3"):
        assert rat not in before[rat]["training_animals"]
        assert all(row["animal"] != rat for row in before[rat]["training_ids"])


def make_base(tmp_path, n_events=2):
    source, output = tmp_path / "edge", tmp_path / "base"
    source.mkdir()
    output.mkdir()
    grid = np.array([[4.001, 36.001], [4.001, 116.001], [12.001, 36.001]])
    rates = np.random.default_rng(87).uniform(0.1, 25, (12, 3))
    ids = np.arange(10, 130, 10)
    counts = np.random.default_rng(42).integers(0, 3, (n_events * 8, 12))
    offsets, starts, event_ids = np.arange(n_events + 1) * 8, np.arange(n_events, dtype=float), np.arange(n_events) + 100
    spikes = []
    for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        for k in range(lo, hi):
            for u, cell in enumerate(ids):
                spikes.extend([[starts[j] + (k - lo + 0.5) * 0.005, cell]] * int(counts[k, u]))
    native = tmp_path / "encoding.npz"
    np.savez_compressed(
        native,
        spikes=np.asarray(spikes),
        candidate_event_indices=event_ids,
        candidate_offsets=offsets,
        candidate_base_starts_s=np.concatenate([j + np.arange(8) * 0.005 for j in starts]),
        candidate_base_durations_s=np.full(len(counts), 0.005),
    )
    (tmp_path / "encoding_manifest.json").write_text(json.dumps(dict(training_only=True, holdout_spikes_used_for_rate_or_unit_selection=False)))
    groups = [dict(split=j, a_ids=ids[:6].tolist(), b_ids=ids[6:].tolist()) for j in range(3)]
    freeze = dict(seed=20260914, groups=groups, encoding_path=str(native), encoding_sha256=file_sha256(native))
    (source / "frozen_measurement.json").write_text(json.dumps(freeze))
    previous = []
    for name in SOURCES:
        truth = np.full((len(counts), 2), np.nan) if name == "real" else np.tile(grid[0], (len(counts), 1))
        np.savez_compressed(
            source / f"{name}_audit.npz", counts=counts, truth_base_cm=truth, offsets=offsets, starts_s=starts, event_ids=event_ids, rates_hz=rates, grid_cm=grid, cell_ids=ids
        )
        for part in groups:
            for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
                values = read_event(counts[lo:hi], rates, grid, (np.arange(6), np.arange(6, 12)), starts[j], event_ids[j], None if name == "real" else truth[lo:hi])
                previous.extend(dict(v, source=name, split=part["split"]) for v in values)
    pd.DataFrame(previous).to_csv(source / "edge_readouts.csv.gz", index=False)
    (source / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in source.iterdir()}))
    row = SimpleNamespace(artifact_dir=str(source), dataset="pfeiffer_foster", animal="Rat0", session="Rat0/Open0", selected_candidates=n_events)
    measured = measure_session(row, output)
    verify_one(SimpleNamespace(**measured))
    return measured


def constant_state(value):
    return dict(initial=value, learning_rate=0.05, trees=[])


def test_choices_own_features_dense_audit_and_resealed_tamper(tmp_path):
    row = make_base(tmp_path)
    path = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
    frame = pd.read_csv(path)
    local = frame.loc[(frame.source == "real") & (frame.split == 0)]
    baseline = local.loc[local.method == "independent"].set_index("event_index")
    context = local.loc[local.method == "unconditional_context"].set_index("event_index")
    x = features(baseline, context, "a", 100)
    changed = baseline.copy()
    for name in changed:
        if name.startswith("b_") or name in ("a_error", "a_brier", "animal", "session"):
            changed[name] = 987
    np.testing.assert_array_equal(features(changed, context, "a", 100), x)
    np.testing.assert_array_equal(choose(baseline, [1, 1], [1, -1], "a"), [True, False])
    changed = baseline.copy()
    changed.loc[:, "a_active"] = 1
    assert not choose(changed, [1, 1], [1, 1], "a").any()
    with pytest.raises(ValueError):
        choose(baseline, [np.nan, 1], [1, 1], "a")
    with pytest.raises(ValueError):
        features(baseline, context.iloc[::-1], "a", 100)
    states = {"Rat0": dict(excluded_animal="Rat0", training_animals=["Rat1", "Rat2", "Rat3"], physical=constant_state(1), brier=constant_state(1))}
    output = tmp_path / "new"
    output.mkdir()
    measured = evaluate_session(row, states, output)
    audited = verify_session(measured, states)
    assert audited["rows"] == 144 and audited["posterior_rows"] == 288
    folder = Path(measured["artifact_dir"])
    path = folder / "event_readouts.csv.gz"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "a_predicted_physical_gain"] += 0.1
    frame.to_csv(path, index=False)
    hashes = json.loads((folder / "outputs.json").read_text())
    hashes[path.name] = file_sha256(path)
    (folder / "outputs.json").write_text(json.dumps(hashes))
    with pytest.raises(AssertionError):
        verify_session(measured, states)


def test_full_training_reconstruction_and_refit(tmp_path):
    original = make_base(tmp_path, n_events=20)
    folder = Path(original["artifact_dir"])
    raw = pd.read_csv(folder / "event_readouts.csv.gz", float_precision="round_trip")
    results = []
    for rat in range(4):
        for session in range(2):
            local = tmp_path / f"alias{rat}{session}"
            local.mkdir()
            (local / "frozen_input.json").write_text((folder / "frozen_input.json").read_text())
            frame = raw.assign(animal=f"Rat{rat}", session=f"Rat{rat}/Open{session}")
            frame.to_csv(local / "event_readouts.csv.gz", index=False)
            results.append(dict(original, animal=f"Rat{rat}", session=f"Rat{rat}/Open{session}", artifact_dir=str(local)))
    base = dict(status="complete", results=results)
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    training = training_rows(base)
    states = fit_states(training)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    training.to_csv(frozen / "training_rows.csv.gz", index=False)
    (frozen / "models.json").write_text(json.dumps(states))
    manifest = dict(
        status="frozen",
        primary_method=PRIMARY,
        features=list(FEATURES),
        no_replay_labels=True,
        input_file_paths=dict(base=str(base_path)),
        input_file_sha256=dict(base=file_sha256(base_path)),
        outputs={p.name: file_sha256(p) for p in frozen.iterdir()},
    )
    (frozen / "manifest.json").write_text(json.dumps(manifest))
    checked, n = verify_training(frozen)
    assert n == 1600 and checked == load_models(frozen)
    states["Rat0"]["training_animals"].append("Rat0")
    (frozen / "models.json").write_text(json.dumps(states))
    manifest["outputs"]["models.json"] = file_sha256(frozen / "models.json")
    (frozen / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        verify_training(frozen)


def report_fixture():
    rows = []
    for animal in range(4):
        for session in range(2):
            for split in range(3):
                for source in SOURCES:
                    for method in METHODS:
                        for event in range(8):
                            scale = 0.7 if method == PRIMARY else 1
                            row = dict(
                                dataset="pfeiffer_foster",
                                animal=str(animal),
                                session=f"{animal}/{session}",
                                split=split,
                                source=source,
                                method=method,
                                event_index=event,
                                original_start_s=float(event),
                                original_end_s=event + 0.02,
                                context_ms=200,
                                separation_cm=30 * scale,
                                regional_tv=0.5 * scale,
                                a_entropy=0.8 * scale,
                                b_entropy=0.8 * scale,
                            )
                            for side in ("a", "b"):
                                row.update(
                                    {
                                        side + "_spikes": 6,
                                        side + "_active": 4,
                                        side + "_predicted_physical_gain": event - 2,
                                        side + "_use_context": event > 2,
                                        side + "_entropy_control_available": True,
                                    }
                                )
                            for metric in TRUTH:
                                value = 16 if method == PRIMARY and event > 2 else 20
                                if method == "unconditional_context":
                                    value = 24 - 2 * event
                                row[metric] = np.nan if source == "real" else value
                            rows.append(row)
    return pd.DataFrame(rows)


def test_new_diagnostic_gates_and_jump_falsification():
    frame = report_fixture()
    _, _, summary = summaries(frame, methods=METHODS)
    _, diagnostic = diagnostic_summary(frame, prediction_suffix="_predicted_physical_gain")
    assert combined_gates(summary, frame, diagnostic, True, primary=PRIMARY).passed.all()
    frame.loc[(frame.source == "sim_late_jump") & (frame.method == PRIMARY), "a_error"] = 40
    _, _, summary = summaries(frame, methods=METHODS)
    gates = combined_gates(summary, frame, diagnostic, True, primary=PRIMARY).set_index("gate")
    assert not gates.loc["sim_late_jump_no_worse_a_error", "passed"]
    assert not gates.loc["advance_external_validation", "passed"]


def test_external_cannot_use_wrong_model_or_truncated_passing_gates(tmp_path):
    frame = report_fixture()
    _, _, summary = summaries(frame, methods=METHODS)
    _, diagnostic = diagnostic_summary(frame, prediction_suffix="_predicted_physical_gain")
    gates = combined_gates(summary, frame, diagnostic, True, primary=PRIMARY)
    gate_path = tmp_path / "gate_summary.csv"
    gates.to_csv(gate_path, index=False)
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    producer = tmp_path / "producer.json"
    producer.write_text(
        json.dumps(
            dict(
                status="complete",
                primary_method=PRIMARY,
                input_file_sha256=dict(frozen=file_sha256(models / "manifest.json")),
                results=frame[["dataset", "animal", "session"]].drop_duplicates().to_dict("records"),
            )
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(status="passed", input_file_sha256=dict(producer=file_sha256(producer)))))
    manifest = dict(
        status="complete",
        primary_method=PRIMARY,
        primary_advanced=True,
        input_file_paths=dict(producer=str(producer), audit=str(audit)),
        input_file_sha256=dict(producer=file_sha256(producer), audit=file_sha256(audit)),
        output_sha256={gate_path.name: file_sha256(gate_path)},
    )
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    validate_development(tmp_path, models)
    manifest["primary_method"] = "predictive_context"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wrong or failed"):
        validate_development(tmp_path, models)
    manifest["primary_method"] = PRIMARY
    gates.loc[gates.gate == "advance_external_validation"].to_csv(gate_path, index=False)
    manifest["output_sha256"][gate_path.name] = file_sha256(gate_path)
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete development"):
        validate_development(tmp_path, models)
