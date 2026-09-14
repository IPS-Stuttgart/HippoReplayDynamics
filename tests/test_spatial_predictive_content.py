import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from scripts.spatial_predictive_content import (
    REAL,
    TRUTH,
    SESSIONS,
    consistency,
    decode,
    event_table,
    make_folds,
    measure,
    select_half,
    summarize,
)
from scripts._provenance import file_sha256


def test_shared_location_wins_and_opposed_locations_lose():
    rates = np.array([[20, 1], [1, 20]] * 3, float)
    folds = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
    same = consistency(np.array([[4, 0] * 3]), rates, folds)
    opposing = consistency(np.array([[4, 0, 4, 0, 0, 4]]), rates, folds)
    assert np.min(same) > 0.6
    assert opposing[0, 2] < -5


def test_zero_spikes_do_not_create_shared_silence_evidence():
    rates = np.array([[30, 1], [2, 15], [40, 1]], float)
    n = np.array([[0, 0, 0], [3, 0, 0]])
    np.testing.assert_array_equal(consistency(n, rates, make_folds(3, "S", "high")), 0)


def test_gain_invariance_and_neutral_spatially_flat_maps():
    rng = np.random.default_rng(3)
    n, r = rng.poisson(2, (10, 12)), rng.uniform(0.1, 100, (12, 20))
    folds = make_folds(12, "S", "low")
    np.testing.assert_allclose(consistency(n, r, folds), consistency(n, 30 * r, folds), atol=1e-10)
    np.testing.assert_allclose(consistency(n, np.ones_like(r), folds), 0, atol=1e-10)


def test_folds_are_frozen_and_cover_cells():
    a = make_folds(17, "Rat1/Open1", "high")
    b = make_folds(17, "Rat1/Open1", "high")
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(17))
    with pytest.raises(ValueError):
        make_folds(2, "S", "high")


def test_invalid_folds_fail():
    with pytest.raises(ValueError):
        consistency(np.ones((3, 6)), np.ones((6, 4)), [np.array([0, 1])] * 3)


def test_fixed_half_ties_and_odd_denominator():
    np.testing.assert_array_equal(select_half([0, 1, 1, 0, 1], [5, 9, 2, 3, 4]), [False, True, True, False, True])
    np.testing.assert_array_equal(select_half([0, 0, 0, 0], [4, 3, 2, 1]), [False, False, True, True])
    with pytest.raises(ValueError):
        select_half([], [])


def test_decoder_unchanged_poisson():
    r = np.array([[1.0, 50, 3], [30.0, 2.0, 1.0]])
    n, grid = np.array([[1, 2], [0, 0]]), np.array([[0, 0], [10, 10], [20, 0]])
    near = np.array([False, True, False])
    out = decode(n, r, grid, near, grid[:2])
    p = softmax(n @ np.log(r) - 0.02 * r.sum(axis=0), axis=1)
    np.testing.assert_allclose(out["home"], p[:, 1])
    np.testing.assert_allclose(out["mean"], p @ grid)


def test_truth_cannot_affect_scores_or_selection():
    rng = np.random.default_rng(99)
    r = rng.uniform(0.1, 30, (12, 9))
    n, grid = rng.poisson(1, (20, 12)), rng.uniform(0, 100, (9, 2))
    groups, near = [np.arange(6), np.arange(6, 12)], np.arange(9) < 3
    enc = dict(early_run=r, grid_cm=grid, near=near, high_indices=groups[0], low_indices=groups[1])
    s = np.column_stack([softmax(n[:, g] @ np.log(r[g]) - 0.02 * r[g].sum(axis=0), axis=1)[:, near].sum(axis=1) for g in groups])
    bank = dict(counts=n, truth_cm=rng.uniform(0, 100, (20, 2)), labels=np.arange(20) % 2 == 0, early_run_scores=s)
    a = event_table("Rat1/Open1", "run_q4", "early_run", bank, enc)
    bank["truth_cm"] *= -100
    bank["labels"] = ~bank["labels"]
    b = event_table("Rat1/Open1", "run_q4", "early_run", bank, enc)
    cols = [c for c in a.columns if "score" in c or "gain" in c or "half" in c]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def fixture_rows():
    records = []
    for session in SESSIONS:
        for source in REAL + TRUTH:
            for encoding in ("early_run", "full_run") if source in REAL else ("early_run",):
                for j in range(40):
                    chosen = j < 20
                    label = j % 2 if source in TRUTH else np.nan
                    rows = dict(
                        animal=session.split("/")[0],
                        session=session,
                        source=source,
                        encoding=encoding,
                        observation_index=j,
                        pair_score=40 - j,
                        true_home=label,
                        all=True,
                        predictive_half=chosen,
                        spike_half=chosen,
                        entropy_half=chosen,
                        high_home=0.2 if chosen else 0.5,
                        low_home=0.15 if chosen else 0.1,
                        separation=10 + j,
                        regional_tv=0.1 if chosen else 0.6,
                        high_entropy=0.2 if chosen else 0.8,
                        low_entropy=0.2 if chosen else 0.8,
                    )
                    for name in ("high_error", "low_error", "high_brier", "low_brier"):
                        rows[name] = (0.05 if chosen else 0.5) if source in TRUTH else np.nan
                    records.append(rows)
    return pd.DataFrame(records)


def test_complete_fixture_has_nonvacuous_success():
    g = summarize(fixture_rows())["gates"]
    assert g.passed.all()


def test_discarding_true_home_cannot_manufacture_a_pass():
    d = fixture_rows()
    m = d.source.isin(TRUTH)
    d.loc[m, "predictive_half"] = d.loc[m, "true_home"].eq(0)
    g = summarize(d)["gates"].set_index("gate")
    assert not g.loc["both_truth_classes_retained", "passed"]
    assert not g.loc["development_numerical_screen", "passed"]


def test_single_rat_harm_and_duplicates_fail():
    d = fixture_rows()
    mask = d.animal.eq("Rat4") & d.source.eq("test_poisson_gain4") & d.predictive_half
    d.loc[mask, "high_error"] = 20
    g = summarize(d)["gates"].set_index("gate")
    assert not g.loc["truth_test_poisson_gain4_balanced_high_error", "passed"]
    d = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    assert not summarize(d)["gates"].set_index("gate").loc["complete_source_coverage", "passed"]


def test_full_measurement_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(771)
    for session in SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir()
        rates, grid = rng.uniform(0.1, 20, (12, 9)), np.array([[x * 8, y * 8] for x in range(3) for y in range(3)], float)
        groups, near = [np.arange(6), np.arange(6, 12)], np.arange(9) < 3
        np.savez_compressed(
            folder / "encoding.npz", early_run=rates, full_run=1.1 * rates, grid_cm=grid, near=near, high_indices=groups[0], low_indices=groups[1], cell_ids=np.arange(12)
        )
        for name in REAL + TRUTH:
            states = np.tile(np.arange(9), 10)
            n = rng.poisson(0.02 * rates[:, states].T)
            scores = {}
            for enc, r in (("early_run", rates), ("full_run", 1.1 * rates)):
                scores[f"{enc}_scores"] = np.column_stack([softmax(n[:, g] @ np.log(r[g]) - 0.02 * r[g].sum(axis=0), axis=1)[:, near].sum(axis=1) for g in groups])
            data = dict(counts=n, truth_cm=grid[states], **scores)
            if name in TRUTH:
                data["labels"] = near[states]
            else:
                data["truth_cm"][:] = np.nan
            np.savez_compressed(folder / f"{name}.npz", **data)
        (folder / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in folder.glob("*.npz")}))
    (source / "manifest.json").write_text('{"synthetic_fixture": true}')
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps(dict(status="pass", input_file_sha256={"source": file_sha256(source / "manifest.json")})))
    output = tmp_path / "results"
    measure(SimpleNamespace(source_dir=source, source_audit=audit, output_dir=output))
    d = pd.read_csv(output / "events.csv.gz")
    assert len(d) == 3600
    assert (d.groupby(["session", "source", "encoding"]).predictive_half.sum() == 45).all()
    assert json.loads((output / "manifest.json").read_text())["external_validation"] is False
    from scripts.audit_spatial_predictive_content import audit

    checked = audit(output)
    assert checked["status"] == "pass"
    table = pd.read_csv(output / "summary.csv")
    table.loc[0, "home_gap"] += 0.1
    table.to_csv(output / "summary.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["output_sha256"]["summary.csv"] = file_sha256(output / "summary.csv")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit(output)
