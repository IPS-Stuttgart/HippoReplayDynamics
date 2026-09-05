#!/usr/bin/env python3
"""Independently reconstruct excluded-animal fits and all predictive intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from hipporeplayimm.replay_speed_population_transfer import (
    GROUP,
    IDENTITY,
    METRICS,
    PANEL_KEY,
    READOUT,
    paired_comparison,
    panel_digest,
    summarize_animals,
)
from scipy.stats import t

from scripts._provenance import build_script_provenance, file_sha256
from scripts.validate_replay_speed_population_transfer import load_source


def independent_fit(training, calibration):
    finite = training[np.isfinite(training.statistic)]
    x, y = finite.statistic.to_numpy(), finite.gradient.to_numpy()
    sxx = np.square(x - x.mean()).sum() if len(x) else np.nan
    status = "insufficient_fit" if len(x) < 20 else "uninformative_fit" if sxx < 1e-12 else "fitted"
    model = {"status": status, "n_fit": len(x), "intercept": np.nan, "slope": np.nan,
        "x_mean": np.nan, "sxx": np.nan, "residual_sd": np.nan, "calibration_radius": np.inf}
    if status != "fitted":
        return model
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    residual_sd = np.linalg.norm(y - intercept - slope * x) / np.sqrt(len(x) - 2)
    rank = int(np.ceil(.95 * (len(calibration) + 1)))
    errors = np.full(len(calibration), np.inf)
    valid = np.isfinite(calibration.statistic.to_numpy())
    errors[valid] = np.abs(calibration.gradient.to_numpy()[valid] - intercept - slope * calibration.statistic.to_numpy()[valid])
    radius = np.partition(errors, rank - 1)[rank - 1] if rank <= len(errors) else np.inf
    return {"status": status, "n_fit": len(x), "intercept": intercept, "slope": slope,
        "x_mean": x.mean(), "sxx": sxx, "residual_sd": residual_sd, "calibration_radius": radius}


def verify_intervals(rows, model):
    x, truth = rows.statistic.to_numpy(), rows.gradient.to_numpy()
    valid = np.isfinite(x) & (model["status"] == "fitted")
    point, lo, hi = np.full(len(x), np.nan), np.full(len(x), -np.inf), np.full(len(x), np.inf)
    if valid.any():
        point[valid] = model["intercept"] + model["slope"] * x[valid]
        width = np.full(len(x), model["calibration_radius"])
        gaussian = valid & rows.method.eq("inverse_gaussian").to_numpy()
        width[gaussian] = t.isf(.025, model["n_fit"] - 2) * np.sqrt(model["residual_sd"] ** 2 * (
            1 + 1 / model["n_fit"] + np.square(x[gaussian] - model["x_mean"]) / model["sxx"]))
        lo[valid], hi[valid] = point[valid] - width[valid], point[valid] + width[valid]
    np.testing.assert_allclose(rows[["estimate", "lower", "upper"]].to_numpy(),
        np.column_stack([point, lo, hi]), rtol=1e-9, atol=1e-9, equal_nan=True)
    finite = np.isfinite(lo) & np.isfinite(hi)
    np.testing.assert_array_equal(rows.finite_interval, finite)
    np.testing.assert_array_equal(rows.covered, (lo <= truth) & (truth <= hi))
    np.testing.assert_allclose(rows.interval_width, hi - lo, rtol=1e-9, atol=1e-9)
    nonzero = finite & ((lo > 0) | (hi < 0))
    np.testing.assert_array_equal(rows.nonzero_claim, nonzero)
    for bound in [.10, .25, .50]:
        inside = np.abs(truth) < bound
        claim = finite & (lo > -bound) & (hi < bound)
        np.testing.assert_array_equal(rows[f"inside_{bound:.2f}"], inside)
        np.testing.assert_array_equal(rows[f"equivalence_{bound:.2f}"], claim)
        if bound == .25:
            np.testing.assert_array_equal(rows.truth_inside_equivalence, inside)
            np.testing.assert_array_equal(rows.equivalence_claim, claim)
            np.testing.assert_array_equal(rows.false_equivalence, claim & ~inside)
            expected = np.select([claim, nonzero, finite], ["equivalent_in_surrogate", "nonzero_in_surrogate", "inconclusive"], default="abstain")
            np.testing.assert_array_equal(rows.decision, expected)


def audit(input_dir):
    manifest = input_dir / "speed_population_transfer_manifest.json"
    meta = json.loads(manifest.read_text())
    assert meta["status"] == "complete"
    for key, path in meta["input_file_paths"].items():
        assert file_sha256(path) == meta["input_file_sha256"][key], key
    for name, digest in {**meta["output_sha256"], **meta["snapshot_sha256"]}.items():
        assert file_sha256(input_dir / name) == digest, name
    panels, reference, _, _ = load_source(Path(meta["input_file_paths"]["source_manifest"]).parent)
    def read(name, compressed=False):
        suffix = ".csv.gz" if compressed else ".csv"
        return pd.read_csv(input_dir / f"speed_population_transfer_{name}{suffix}", float_precision="round_trip")
    fits, decisions = read("fits"), read("decisions", True)
    assert len(fits) == meta["fits"] and len(decisions) == meta["decisions"]
    assert not fits.duplicated(["dataset", "heldout_animal"] + READOUT).any()
    assert not decisions.duplicated(PANEL_KEY + ["method"]).any()
    assert decisions.method.isin(["inverse_gaussian", "inverse_conformal"]).all()
    assert decisions.calibration_scope.eq("leave_one_animal_out").all()
    test = panels[panels.phase.eq("test")].sort_values(PANEL_KEY).reset_index(drop=True)
    for method in ["inverse_gaussian", "inverse_conformal"]:
        actual = decisions[decisions.method.eq(method)].sort_values(PANEL_KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[test.columns], test, check_dtype=False, rtol=1e-12, atol=1e-12)
    for record in fits.to_dict("records"):
        choose = panels.dataset.eq(record["dataset"])
        for name in READOUT:
            choose &= panels[name].eq(record[name])
        others = panels[choose & ~panels.animal.eq(record["heldout_animal"])].sort_values(PANEL_KEY)
        train, calibration = [others[others.phase.eq(phase)] for phase in ["fit", "calibration"]]
        assert set(train.animal) == set(json.loads(record["source_animals"]))
        assert set(map(tuple, json.loads(record["source_sessions"]))) == set(zip(train.animal, train.session, strict=True))
        assert record["heldout_animal"] not in set(train.animal) | set(calibration.animal)
        assert len(train) == record["n_fit_scheduled"] and len(calibration) == record["n_calibration"]
        assert np.isfinite(calibration.statistic).sum() == record["finite_calibration"]
        assert panel_digest(train) == record["fit_input_sha256"] and panel_digest(calibration) == record["calibration_input_sha256"]
        expected = independent_fit(train, calibration)
        assert expected["status"] == record["status"] and expected["n_fit"] == record["n_fit"]
        names = [k for k in expected if k not in ["status", "n_fit"]]
        np.testing.assert_allclose([record[k] for k in names], [expected[k] for k in names], rtol=1e-9, atol=1e-9, equal_nan=True)
        selected = decisions.dataset.eq(record["dataset"]) & decisions.animal.eq(record["heldout_animal"])
        for name in READOUT:
            selected &= decisions[name].eq(record[name])
        verify_intervals(decisions[selected], expected)
    sessions = read("session_summary")
    keys = IDENTITY + GROUP
    assert not sessions.duplicated(keys).any()
    new = sessions[sessions.calibration_scope.eq("leave_one_animal_out")].set_index(keys).sort_index()
    groups = decisions.groupby(keys, sort=True)
    counts = groups.agg(panels=("covered", "size"), finite_panels=("finite_interval", "sum"),
        coverage=("covered", "mean"), finite_fraction=("finite_interval", "mean"),
        nonzero_fraction=("nonzero_claim", "mean"), equivalence_fraction=("equivalence_claim", "mean"))
    finite_stats = decisions[decisions.finite_interval].groupby(keys).agg(
        finite_coverage=("covered", "mean"), median_finite_width=("interval_width", "median"))
    counts = counts.join(finite_stats)
    for bound in [.10, .25, .50]:
        for inside, label in [(True, "true"), (False, "false")]:
            subset = decisions[decisions[f"inside_{bound:.2f}"].eq(inside)].groupby(keys)
            number = subset.size().reindex(counts.index, fill_value=0)
            counts[f"{'inside' if inside else 'outside'}_panels_{bound:.2f}"] = number
            counts[f"{label}_equivalence_fraction_{bound:.2f}"] = subset[f"equivalence_{bound:.2f}"].mean().reindex(counts.index)
    pd.testing.assert_frame_equal(new[counts.columns], counts, check_dtype=False, rtol=1e-12, atol=1e-12)
    reference["calibration_scope"] = "within_session"
    local = sessions[sessions.calibration_scope.eq("within_session")].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(local[reference.columns], reference.sort_values(keys).reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
    animal, summary = summarize_animals(sessions, meta["seed"], meta["bootstraps"])
    direct = sessions.groupby(["dataset", "animal"] + GROUP, sort=True)[METRICS].mean()
    pd.testing.assert_frame_equal(animal.set_index(["dataset", "animal"] + GROUP)[METRICS], direct, check_dtype=False)
    for name, expected in [("animal_summary", animal), ("summary", summary),
        ("paired_comparison", paired_comparison(sessions, meta["seed"], meta["bootstraps"]))]:
        pd.testing.assert_frame_equal(read(name), expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    provenance = build_script_provenance(input_paths={"manifest": manifest, "auditor": Path(__file__)}, cwd=ROOT)
    provenance.update(status="pass", fits_verified=len(fits), intervals_verified=len(decisions),
        scope="source hashes/linkage; independent least-squares and residual ranks; all intervals/decisions and panel values; direct grouped counts/means; shared seeded-bootstrap reconstruction; no new raw-spike audit")
    (input_dir / "speed_population_transfer_reconstruction_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ["status", "fits_verified", "intervals_verified"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    audit(parser.parse_args().input_dir)
