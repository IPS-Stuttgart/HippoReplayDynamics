from argparse import Namespace
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import reserve_cell_content as m
from scripts import audit_reserve_cell_content as a


def fixture():
    rng = np.random.default_rng(918)
    grid = np.column_stack([np.arange(6) * 8.0, np.zeros(6)])
    rates = rng.uniform(1, 40, (6, 6))
    enc = dict(
        early_run=rates, full_run=rates * 1.1, cell_ids=np.arange(100, 106), high_indices=np.array([0, 1]), low_indices=np.array([1, 2]), grid_cm=grid, near=np.arange(6) < 3
    )
    position = np.tile(np.array([0, 3, 1, 4, 2, 5]), 10)
    bank = dict(
        counts=rng.poisson(0.02 * rates[:, position].T),
        truth_cm=grid[position],
        labels=position < 3,
        parent_ids=np.arange(len(position)),
        event_ids=np.array([f"e{x:03}" for x in range(len(position))]),
    )
    return enc, bank


def test_equal_budget_and_no_overlap_added():
    enc, q3 = fixture()
    result = m.select_cells(enc, q3, "Rat1/Open1")
    assert result["quota"] == 1
    m.validate_assignments(enc, result)
    assert set(result["methods"]) == set(m.METHODS)
    a.check_assignment(enc, q3, result)


def test_selection_ignores_later_maps_and_no_target_input():
    enc, q3 = fixture()
    first = m.select_cells(enc, q3, "Rat1/Open1")
    changed = deepcopy(enc)
    changed["full_run"][:] = np.nan
    changed["late_rates"] = np.full_like(enc["early_run"], 1e10)
    assert first == m.select_cells(changed, q3, "Rat1/Open1")


def test_first_bin_per_parent_and_independent_parent_gate():
    _, bank = fixture()
    bank["parent_ids"] = np.repeat(np.arange(30), 2)
    bank["labels"] = np.repeat(np.arange(30) % 2, 2)
    rows = m.calibration_rows(bank)
    np.testing.assert_array_equal(rows, np.arange(0, 60, 2))
    bank["parent_ids"][:] = 1
    with pytest.raises(ValueError, match="parents"):
        m.calibration_rows(bank)


def test_numerical_tie_breaking_is_fixed():
    enc, q3 = fixture()
    enc["early_run"][:] = 5
    result = m.select_cells(enc, q3, "Rat1/Open1")
    assert result["methods"]["targeted"] == dict(high=[3], low=[4])
    a.check_assignment(enc, q3, result)


@pytest.mark.parametrize("change", ["repeated", "existing", "both"])
def test_invalid_added_assignment_rejected(change):
    enc, q3 = fixture()
    result = m.select_cells(enc, q3, "Rat1/Open1")
    pair = result["methods"]["targeted"]
    if change == "repeated":
        pair["high"] *= 2
    elif change == "existing":
        pair["high"] = [0]
    else:
        pair["low"] = pair["high"].copy()
    with pytest.raises(ValueError):
        m.validate_assignments(enc, result)


def test_source_count_immutability_and_independent_likelihood():
    enc, bank = fixture()
    saved = bank["counts"].copy()
    pair = dict(high=[3], low=[4])
    expected = a.values_for(bank, enc, "early_run", pair)
    measured = m.score_pair(bank, enc, "early_run", pair)
    for key in expected:
        np.testing.assert_allclose(measured[key], expected[key], atol=1e-12)
    np.testing.assert_array_equal(bank["counts"], saved)


def passing_rows():
    rows = []
    candidate = dict(zip(m.SESSIONS, (361, 613, 328, 534), strict=True))
    accepted = dict(zip(m.SESSIONS, (100, 100, 100, 213), strict=True))
    for s in m.SESSIONS:
        for src in m.REAL + m.TRUTH:
            for enc in ("early_run", "full_run") if src in m.REAL else ("early_run",):
                for method in m.METHODS:
                    value = 1 if method == "baseline" else 0.7 if method == "targeted" else 0.9
                    r = dict(
                        session=s,
                        animal=s.split("/")[0],
                        source=src,
                        encoding=enc,
                        method=method,
                        events=candidate[s] if src == m.REAL[0] else accepted[s] if src == m.REAL[1] else 60,
                    )
                    r.update({k: value for k in m.METRICS})
                    r.update({k: 3.0 for k in ("high_spikes", "low_spikes", "high_active", "low_active")})
                    rows.append(r)
    return pd.DataFrame(rows)


def test_all_gates_positive_and_independently_reconstructed():
    rows = passing_rows()
    combined, animal, summary = m.aggregate(rows)
    gates = m.gates(rows, animal, summary)
    assert gates.passed.all()
    tables, expected = a.rebuild_tables(rows.to_dict("records"))
    assert dict(zip(gates.gate, gates.passed, strict=True)) == expected
    a.compare_frame(combined, tables["session_summary"], ["session", "source", "encoding", "method"])


def test_one_rat_harm_prevents_promotion():
    rows = passing_rows()
    mask = rows.animal.eq("Rat4") & rows.source.eq("run_q4") & rows.method.eq("targeted")
    rows.loc[mask, "balanced_high_error"] = 1.01
    _, animal, summary = m.aggregate(rows)
    gates = m.gates(rows, animal, summary).set_index("gate").passed
    assert not gates["truth_run_q4_balanced_high_error"]
    assert not gates["development_numerical_screen"]


def test_missing_random_draw_fails_nonvacuously():
    rows = passing_rows().iloc[1:]
    _, animal, summary = m.aggregate(rows)
    assert not m.gates(rows, animal, summary).set_index("gate").passed["source_and_random_coverage"]


def test_class_balancing_does_not_hide_rare_region_error():
    enc, bank = fixture()
    values = m.score_pair(bank, enc, "early_run", dict(high=[], low=[]))
    values["true_home"][:] = 0
    values["true_home"][0] = 1
    values["high_error"][:] = 0
    values["high_error"][0] = 100
    row = m.summarize_pair(values, "Rat1/Open1", "run_q4", "early_run", "baseline")
    assert row["balanced_high_error"] == 50


def test_error_decomposition_retains_shared_bias_and_known_geometry():
    from scripts.report_reserve_cell_content import error_decomposition

    frame = pd.DataFrame(
        dict(
            session=["Rat1/Open1"] * 2,
            source=["run_q4"] * 2,
            encoding=["early_run"] * 2,
            method=["baseline"] * 2,
            true_home=[0, 1],
            high_error=[3.0, 3.0],
            low_error=[1.0, 1.0],
            separation=[2.0, 2.0],
        )
    )
    _, summary = error_decomposition(frame)
    assert summary.sum_position_mse_cm2.item() == 10
    assert summary.pair_separation_mse_cm2.item() == 4
    assert summary.twice_error_dot_cm2.item() == 6
    frame["separation"] = 9
    with pytest.raises(ValueError, match="geometric"):
        error_decomposition(frame)


def test_artifact_roundtrip_and_rehashed_tamper_rejected(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "manifest.json").write_text("{}\n")
    for session in m.SESSIONS:
        folder = src / session.replace("/", "_")
        folder.mkdir()
        enc, q3 = fixture()
        np.savez(folder / "encoding.npz", **enc)
        np.savez(folder / "run_q3.npz", **q3)
        for source in m.REAL + m.TRUTH:
            bank = deepcopy(q3)
            if source in m.REAL:
                bank.pop("labels")
                bank["truth_cm"][:] = np.nan
            for encoding in ("early_run", "full_run"):
                values = a.values_for(bank, enc, encoding, dict(high=[], low=[]))
                bank[f"{encoding}_scores"] = np.column_stack([values["high_home"], values["low_home"]])
            np.savez(folder / f"{source}.npz", **bank)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({str(p): a.sha(p) for p in src.rglob("*") if p.is_file()}))
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text(json.dumps(dict(status="pass", input_file_sha256=dict(source=a.sha(src / "manifest.json")))))
    output = tmp_path / "measurement"
    m.measure(Namespace(source_dir=src, source_snapshot=snapshot, source_audit=source_audit, output_dir=output))
    result = a.audit(output, tmp_path / "audit.json")
    assert result["status"] == "pass"
    assert not result["gates"]["development_numerical_screen"]  # Miniature event denominators.
    from scripts.report_reserve_cell_content import report

    tables = report(output, tmp_path / "audit.json", tmp_path / "report")
    assert len(tables["acquisition_summary"]) == 4
    assert (tmp_path / "report/reserve_cell_content.png").stat().st_size > 1000
    report_text = (tmp_path / "report/report.md").read_text()
    assert "Development screen fails" in report_text
    assert "240 candidate endpoints" in report_text
    summary = pd.read_csv(output / "summary.csv")
    summary.loc[0, "home_gap"] += 0.01
    summary.to_csv(output / "summary.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["output_sha256"]["summary.csv"] = hashlib.sha256((output / "summary.csv").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        report(output, tmp_path / "audit.json", tmp_path / "tampered_report")
    with pytest.raises(AssertionError):
        a.audit(output, tmp_path / "tampered_audit.json")
