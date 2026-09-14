import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from hipporeplayimm.state_space_utils import _gaussian_transition_matrix
from scripts._provenance import file_sha256
from scripts.audit_predictive_context_content import dense_log_posteriors, rebuild_population, verify_one
from scripts.audit_temporal_endpoint_content import dense_gaussian
from scripts.measure_edge_support_content import read_event
from scripts.measure_predictive_context_content import (
    METHODS,
    SOURCES,
    cell_folds,
    choose_context,
    infer_two,
    integrated_score,
    measure_session,
    own_population,
    source_readouts,
)
from scripts.measure_predictive_context_content import validate_development
from scripts.measure_temporal_endpoint_content import reset_transition
from scripts.report_predictive_context_content import combined_gates, diagnostic_summary
from scripts.report_temporal_endpoint_content import TRUTH, summaries


def tiny():
    grid = np.array([[4.001, 36.001], [4.001, 116.001], [12.001, 36.001]])
    rates = np.random.default_rng(87).uniform(0.1, 25, (12, 3))
    transition = reset_transition(_gaussian_transition_matrix(grid, 20, 4), 0.25)
    return grid, rates, transition


def test_joint_predictive_integral_is_not_product_of_cell_marginals():
    p = np.array([0.4, 0.6])
    mean = np.array([[0.05, 2.0], [2.0, 0.05]])
    counts = np.array([2, 1])
    per_cell = poisson.pmf(counts[:, None], mean)
    expected = sum(p[x] * per_cell[:, x].prod() for x in range(2))
    actual = integrated_score(np.log(p), np.log(per_cell).sum(axis=0))
    np.testing.assert_allclose(np.exp(actual), expected)
    wrong = np.prod(per_cell @ p)
    assert not np.isclose(wrong, expected)
    with pytest.raises(ValueError):
        integrated_score(np.log(p) + 1, np.log(per_cell).sum(axis=0))


def test_frozen_folds_and_strict_supported_positive_choice():
    f = cell_folds(np.arange(11), "test", 0, "a")
    np.testing.assert_array_equal(f, cell_folds(np.arange(11), "test", 0, "a"))
    assert sorted(np.bincount(f)) == [3, 4, 4]
    assert choose_context([1, -1, 2], [2, 1])
    assert not choose_context([1, -1, -2], [2, 1])
    assert not choose_context([1, 2, 3], [3, 0])
    assert not choose_context([1, 2, 3], [1, 1])
    assert not choose_context([1e-12] * 3, [2, 1])
    with pytest.raises(ValueError):
        choose_context([1, np.nan, 2], [2, 1])
    with pytest.raises(ValueError):
        cell_folds(np.arange(5), "test", 0, "a")


def test_inner_posterior_never_receives_validation_spikes(monkeypatch):
    from scripts import measure_predictive_context_content as module

    _, rates, transition = tiny()
    rates = rates[:6]
    blocks = np.arange(18).reshape(3, 6)
    fold = np.array([0, 0, 1, 1, 2, 2])
    calls = []
    original = module.infer_two

    def spy(n, r, t):
        calls.append((n.copy(), r.copy()))
        return original(n, r, t)

    monkeypatch.setattr(module, "infer_two", spy)
    own_population(blocks, rates, transition, fold)
    assert len(calls) == 4
    for k in range(3):
        np.testing.assert_array_equal(calls[k][0], blocks[:, fold != k])
        np.testing.assert_array_equal(calls[k][1], rates[fold != k])
    np.testing.assert_array_equal(calls[-1][0], blocks)
    before, _ = original(blocks[:, fold != 0], rates[fold != 0], transition)
    blocks[:, fold == 0] += 100
    after, _ = original(blocks[:, fold != 0], rates[fold != 0], transition)
    np.testing.assert_array_equal(before, after)


def test_dense_reference_matches_each_fold_and_one_bin_fallback():
    grid, rates, transition = tiny()
    rates = rates[:6]
    fold = np.array([0, 1, 2, 0, 1, 2])
    rng = np.random.default_rng(66)
    events = [rng.integers(0, 3, (n, 6)) for n in (1, 3, 5)]
    full, scores, differences, choices = rebuild_population(events, rates, dense_gaussian(grid), fold)
    log_full = dense_log_posteriors(events, rates, dense_gaussian(grid))
    for j, event in enumerate(events):
        bank, diagnostic, _ = own_population(event, rates, transition, fold)
        lp, _ = infer_two(event, rates, transition)
        np.testing.assert_allclose(lp, log_full[j], atol=1e-11)
        np.testing.assert_allclose(diagnostic["fold_scores"], scores[j], atol=1e-11)
        np.testing.assert_allclose(diagnostic["fold_delta"], differences[j], atol=1e-11)
        assert diagnostic["use_context"] == choices[j]
        np.testing.assert_allclose(bank["predictive_context"], full[j, int(choices[j])], atol=1e-12)
    assert not choices[0]


def test_other_population_and_truth_do_not_select_or_infer_a():
    grid, rates, transition = tiny()
    rng = np.random.default_rng(8)
    arrays = dict(
        counts=rng.integers(0, 3, (12, 12)),
        offsets=np.array([0, 12]),
        truth_base_cm=np.full((12, 2), np.nan),
        starts_s=np.array([0.0]),
        event_ids=np.array([40]),
        rates_hz=rates,
        grid_cm=grid,
        cell_ids=np.arange(12),
    )
    groups = [np.arange(6), np.arange(6, 12)]
    folds = [cell_folds(g, "x", 0, s) for s, g in zip(("a", "b"), groups)]
    _, before = source_readouts(arrays, groups, folds, transition)
    arrays["counts"][:, 6:] *= 20
    arrays["truth_base_cm"] = np.tile(grid[2], (12, 1))
    _, after = source_readouts(arrays, groups, folds, transition)
    for key in before:
        if key.startswith("a_"):
            np.testing.assert_array_equal(before[key], after[key])


def test_full_producer_independent_audit_roundtrip_and_score_tamper(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    grid, rates, _ = tiny()
    ids = np.arange(10, 130, 10)
    base = np.random.default_rng(42).integers(0, 3, (13, 12))
    offsets, starts, event_ids = np.array([0, 5, 13]), np.array([0.0, 1.0]), np.array([100, 101])
    spikes = []
    for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        for k in range(lo, hi):
            for u, cell in enumerate(ids):
                spikes.extend([[starts[j] + (k - lo + 0.5) * 0.005, cell]] * int(base[k, u]))
    native = tmp_path / "encoding.npz"
    np.savez_compressed(
        native,
        spikes=np.asarray(spikes),
        candidate_event_indices=event_ids,
        candidate_offsets=offsets,
        candidate_base_starts_s=np.r_[np.arange(5) * 0.005, 1 + np.arange(8) * 0.005],
        candidate_base_durations_s=np.full(13, 0.005),
    )
    (tmp_path / "encoding_manifest.json").write_text(json.dumps(dict(training_only=True, holdout_spikes_used_for_rate_or_unit_selection=False)))
    groups = [dict(split=j, a_ids=ids[:6].tolist(), b_ids=ids[6:].tolist()) for j in range(3)]
    freeze = dict(seed=20260914, groups=groups, encoding_path=str(native), encoding_sha256=file_sha256(native))
    (source / "frozen_measurement.json").write_text(json.dumps(freeze))
    previous = []
    for name in SOURCES:
        truth = np.full((13, 2), np.nan) if name == "real" else np.tile(grid[0], (13, 1))
        np.savez_compressed(
            source / f"{name}_audit.npz", counts=base, truth_base_cm=truth, offsets=offsets, starts_s=starts, event_ids=event_ids, rates_hz=rates, grid_cm=grid, cell_ids=ids
        )
        for part in groups:
            for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
                values = read_event(base[lo:hi], rates, grid, (np.arange(6), np.arange(6, 12)), starts[j], event_ids[j], None if name == "real" else truth[lo:hi])
                previous.extend(dict(v, source=name, split=part["split"]) for v in values)
    pd.DataFrame(previous).to_csv(source / "edge_readouts.csv.gz", index=False)
    (source / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in source.iterdir()}))
    row = SimpleNamespace(artifact_dir=str(source), dataset="pfeiffer_foster", animal="Rat1", session="Rat1/Open1", selected_candidates=2)
    measured = measure_session(row, output)
    audited = verify_one(SimpleNamespace(**measured))
    assert audited["rows"] == 144 and audited["posterior_rows"] == 288 and audited["predictive_scores"] == 432
    json.dumps(audited)
    folder = Path(measured["artifact_dir"])
    path = folder / "real_split0_audit.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["a_fold_scores"][0, 0, 0] += 1
    np.savez_compressed(path, **arrays)
    checks = json.loads((folder / "outputs.json").read_text())
    checks[path.name] = file_sha256(path)
    (folder / "outputs.json").write_text(json.dumps(checks))
    with pytest.raises(AssertionError):
        verify_one(SimpleNamespace(**measured))


def report_fixture():
    rows = []
    for animal in range(4):
        for session in range(2):
            for split in range(3):
                for source in SOURCES:
                    for method in METHODS:
                        for event in range(8):
                            context = event > 2
                            scale = 0.7 if method == "predictive_context" else 1.0
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
                                        side + "_median_predictive_delta": event - 2,
                                        side + "_use_context": context,
                                        side + "_entropy_control_available": True,
                                    }
                                )
                            for metric in TRUTH:
                                value = 20 * (0.8 if method == "predictive_context" and context else 1)
                                if method == "unconditional_context":
                                    value = 24 - 2 * event
                                row[metric] = np.nan if source == "real" else value
                            rows.append(row)
    return pd.DataFrame(rows)


def test_diagnostic_and_remedy_gates_both_required():
    data = report_fixture()
    _, _, summary = summaries(data, methods=METHODS)
    _, diag = diagnostic_summary(data)
    assert combined_gates(summary, data, diag, True).passed.all()
    diag.loc[(diag.source == "sim_late_jump") & (diag.side == "a"), "gain_auroc"] = np.nan
    result = combined_gates(summary, data, diag, True).set_index("gate")
    assert not result.loc["sim_late_jump_a_predicts_known_gain", "passed"]
    assert not result.loc["advance_external_validation", "passed"]
    data.loc[(data.source == "sim_late_jump") & (data.method == "predictive_context"), "a_error"] = 30
    _, _, summary = summaries(data, methods=METHODS)
    _, diag = diagnostic_summary(data)
    assert not combined_gates(summary, data, diag, True).set_index("gate").loc["advance_external_validation", "passed"]


def test_zero_eligibility_and_missing_models_do_not_pass():
    data = report_fixture()
    data["a_spikes"] = 0
    data["a_active"] = 0
    data["a_use_context"] = False
    _, _, summary = summaries(data, methods=METHODS)
    sessions, diag = diagnostic_summary(data)
    assert sessions.loc[sessions.side == "a", "gain_auroc"].isna().all()
    assert not combined_gates(summary, data, diag, True).set_index("gate").loc["advance_external_validation", "passed"]
    with pytest.raises(ValueError):
        summaries(data.loc[data.method != "predictive_context"], methods=METHODS)


def test_external_requires_correct_audited_development_method(tmp_path):
    data = report_fixture()
    _, _, summary = summaries(data, methods=METHODS)
    _, diag = diagnostic_summary(data)
    path = tmp_path / "gate_summary.csv"
    combined_gates(summary, data, diag, True).to_csv(path, index=False)
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps(dict(status="complete", primary_method="predictive_context", results=[dict(dataset="pfeiffer_foster")])))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(status="passed", input_file_sha256=dict(producer=file_sha256(producer)))))
    report = dict(
        status="complete",
        primary_method="predictive_context",
        primary_advanced=True,
        input_file_paths=dict(producer=str(producer), audit=str(audit)),
        input_file_sha256=dict(producer=file_sha256(producer), audit=file_sha256(audit)),
        output_sha256={path.name: file_sha256(path)},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(report))
    validate_development(path)
    report["primary_method"] = "diffusion_reset"
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="wrong or failed"):
        validate_development(path)
    report["primary_method"] = "predictive_context"
    manifest.write_text(json.dumps(report))
    path.write_text("gate,passed\nadvance_external_validation,True\n")
    with pytest.raises(ValueError, match="changed development gates"):
        validate_development(path)
