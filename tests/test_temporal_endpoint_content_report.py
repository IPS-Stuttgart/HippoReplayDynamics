import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scripts._provenance import file_sha256
from scripts.audit_temporal_endpoint_content import dense_bank, verify_one
from scripts.measure_edge_support_content import read_event
from scripts.measure_temporal_endpoint_content import METHODS, endpoint_bank, measure_session
from scripts.report_temporal_endpoint_content import SOURCES, TRUTH, gates, summaries


def fixture():
    records = []
    for animal in range(4):
        for session in range(2):
            for split in range(3):
                for source in SOURCES:
                    for method in METHODS:
                        for event in range(2):
                            scale = 0.7 if method == "diffusion_reset" else 1.0
                            value = dict(
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
                                a_entropy_control_available=True,
                                b_entropy_control_available=True,
                            )
                            value.update({key: 20 * scale if source != "real" else np.nan for key in TRUTH})
                            records.append(value)
    return pd.DataFrame(records)


def test_success_requires_truth_and_independent_audit():
    data = fixture()
    _, _, table = summaries(data)
    assert gates(table, data, audited=True).passed.all()
    assert not gates(table, data, audited=False).set_index("gate").loc["advance_external_validation", "passed"]
    selected = data.source.eq("sim_late_jump") & data.method.eq("diffusion_reset")
    data.loc[selected, "a_error"] = 25
    _, _, table = summaries(data)
    result = gates(table, data, audited=True).set_index("gate")
    assert result.loc["real_reduction_separation_cm", "passed"]
    assert not result.loc["sim_late_jump_no_worse_a_error", "passed"]
    assert not result.loc["advance_external_validation", "passed"]


@pytest.mark.parametrize("kind", ["empty", "missing", "time", "nan", "duplicate"])
def test_incomplete_or_shifted_inputs_fail(kind):
    data = fixture()
    if kind == "empty":
        data = data.iloc[:0]
    elif kind == "missing":
        data = data.loc[data.method.ne("diffusion_reset")]
    elif kind == "time":
        data.loc[data.method.eq("diffusion_reset"), "original_end_s"] += 0.005
    elif kind == "nan":
        data.loc[data.method.eq("diffusion_reset"), "separation_cm"] = np.nan
    else:
        data = pd.concat([data, data.iloc[:1]])
    with pytest.raises((ValueError, AssertionError)):
        summaries(data)


def test_equal_animal_weight_and_no_vacuous_entropy_match():
    data = fixture()
    data.loc[data.animal.eq("3"), "separation_cm"] *= 2
    _, _, first = summaries(data)
    extra = data.loc[data.animal.eq("3")].copy()
    extra.event_index += 20
    extra.original_start_s += 20
    extra.original_end_s += 20
    _, _, second = summaries(pd.concat([data, extra]))
    np.testing.assert_allclose(first.value, second.value)
    data.loc[data.animal.eq("3"), "a_entropy_control_available"] = False
    assert not gates(first, data, audited=True).set_index("gate").loc["advance_external_validation", "passed"]


def test_independent_dense_auditor_matches_sparse_producer():
    grid = np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 8.0]])
    rates = np.array([[15.0, 0.2, 0.4], [0.2, 0.6, 20.0]])
    distance2 = ((grid[:, None] - grid[None]) ** 2).sum(axis=2)
    g = np.exp(-distance2 / 800) * (distance2 <= 6400)
    g /= g.sum(axis=0)
    events = [np.array([[0, 1]]), np.array([[3, 0], [0, 3]]), np.array([[1, 0], [0, 0], [2, 1]])]
    bank, _ = dense_bank(events, rates, grid)
    for i, event in enumerate(events):
        expected, _ = endpoint_bank(event, rates, csr_matrix(g))
        for method in expected:
            np.testing.assert_allclose(bank[method][i], expected[method], atol=1e-12)


def test_full_session_roundtrip_and_json_serialization(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    ids = np.array([10, 20, 30, 40])
    grid = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    rates = np.array([[20.0, 1.0, 2.0], [2.0, 20.0, 1.0], [10.0, 1.0, 15.0], [1.0, 10.0, 20.0]])
    base = np.random.default_rng(42).integers(0, 3, size=(13, 4))
    offsets, starts, event_ids = np.array([0, 5, 13]), np.array([0.0, 1.0]), np.array([100, 101])
    spikes = []
    for e, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        for b in range(lo, hi):
            for u, cell in enumerate(ids):
                spikes.extend([[starts[e] + (b - lo + 0.5) * 0.005, cell]] * int(base[b, u]))
    native = tmp_path / "encoding.npz"
    np.savez_compressed(native, spikes=np.array(spikes))
    (tmp_path / "encoding_manifest.json").write_text(json.dumps(dict(training_only=True, holdout_spikes_used_for_rate_or_unit_selection=False)))
    groups = [dict(split=i, a_ids=[10, 20], b_ids=[30, 40]) for i in range(3)]
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
                values = read_event(base[lo:hi], rates, grid, ([0, 1], [2, 3]), starts[j], event_ids[j], None if name == "real" else truth[lo:hi])
                previous.extend(dict(value, source=name, split=part["split"]) for value in values)
    pd.DataFrame(previous).to_csv(source / "edge_readouts.csv.gz", index=False)
    (source / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in source.iterdir()}))
    row = SimpleNamespace(artifact_dir=str(source), dataset="pfeiffer_foster", animal="Rat1", session="Rat1/Open1", selected_candidates=2)
    measured = measure_session(row, output)
    checked = verify_one(SimpleNamespace(**measured))
    assert checked["rows"] == 6 * 3 * 5 * 2
    assert checked["native_context_blocks"] == 6
    json.dumps(checked)
