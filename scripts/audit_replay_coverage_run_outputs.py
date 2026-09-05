#!/usr/bin/env python3
"""Independently recount held-out RUN observations without decoding again."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts._provenance import file_sha256


def direct_window_support(spikes, cell_ids, centers, width):
    """Assign individual spikes to nonoverlapping windows, then count units."""
    centers = np.asarray(centers, dtype=float)
    cells = np.asarray(cell_ids, dtype=int)
    if len(centers) == 0 or not np.isfinite(centers).all() or width <= 0 or not np.isfinite(width):
        raise ValueError("invalid test windows")
    if np.any(np.diff(centers) < width - 1e-9) or not np.all(np.diff(cells) > 0):
        raise ValueError("windows overlap or unit IDs are not unique and sorted")
    raw = spikes[np.isin(spikes[:, 1], cells)]
    starts, stops = centers - width / 2, centers + width / 2
    event = np.searchsorted(starts, raw[:, 0], side="right") - 1
    keep = (event >= 0) & (raw[:, 0] < stops[np.clip(event, 0, len(stops) - 1)])
    event = event[keep]
    cell = np.searchsorted(cells, raw[keep, 1])
    n_spikes = np.bincount(event, minlength=len(centers))
    unique_pairs = np.unique(event * len(cells) + cell)
    n_units = np.bincount(unique_pairs // len(cells), minlength=len(centers))
    return n_spikes, n_units


def audit(root):
    manifest = json.loads((root / "coverage_RUN_validation_manifest.json").read_text())
    folds = pd.read_csv(root / "coverage_RUN_validation_folds.csv")
    predictions = pd.read_csv(root / "coverage_RUN_validation_predictions.csv")
    records = pd.read_csv(manifest["input_file_paths"]["cache_sessions"])
    keys = ["dataset", "animal", "session"]
    expected = manifest["sessions"] * manifest["folds"]
    if not len(folds) == expected or not folds.status.eq("scored").all():
        raise ValueError("incomplete validation folds")
    if folds.duplicated([*keys, "fold"]).any() or predictions.duplicated([*keys, "fold", "window_s", "likelihood", "center_s"]).any():
        raise ValueError("duplicate validation identities")
    identities = folds[keys].drop_duplicates()
    sources = identities.merge(records, on=keys, how="left", validate="one_to_one")
    rows = []
    for source in sources.itertuples(index=False):
        if file_sha256(source.artifact_path) != source.artifact_sha256:
            raise ValueError("source cache hash mismatch")
        with np.load(source.artifact_path, allow_pickle=False) as cache:
            spikes, position = cache["spikes"], cache["position"]
        session_folds = folds[folds.dataset.eq(source.dataset) & folds.animal.eq(source.animal) & folds.session.eq(source.session)]
        for fold in session_folds.itertuples(index=False):
            model_path = Path(fold.model_path)
            meta = json.loads(model_path.with_suffix(".json").read_text())
            if file_sha256(model_path) != meta["model_sha256"]:
                raise ValueError("training fold model hash mismatch")
            with np.load(model_path, allow_pickle=False) as model:
                centers, cells = model["test_centers_s"], model["cell_ids"]
                intervals = model["training_intervals"]
                if not np.all((intervals[:, 1] < meta["test_start_s"] - meta["guard_s"]) | (intervals[:, 0] > meta["test_end_s"] + meta["guard_s"])):
                    raise ValueError("training interval touches excluded block")
            local = predictions[predictions.dataset.eq(source.dataset) & predictions.animal.eq(source.animal) & predictions.session.eq(source.session) & predictions.fold.eq(fold.fold)]
            if len(local) != len(centers) * 4 or len(centers) != fold.n_test_windows:
                raise ValueError("test-window denominator mismatch")
            for width in [.02, .25]:
                count, active = direct_window_support(spikes, cells, centers, width)
                for likelihood in ["poisson", "conditional_multinomial"]:
                    observed = local[local.window_s.eq(width) & local.likelihood.eq(likelihood)].sort_values("center_s")
                    np.testing.assert_allclose(observed.center_s, centers, rtol=0, atol=1e-9)
                    np.testing.assert_array_equal(observed.n_spikes, count)
                    np.testing.assert_array_equal(observed.n_active_units, active)
                    for axis, column in [(1, "truth_x_cm"), (2, "truth_y_cm")]:
                        np.testing.assert_allclose(observed[column], np.interp(centers, position[:, 0], position[:, axis]), rtol=0, atol=1e-8)
            rows.append({"dataset": source.dataset, "animal": source.animal, "session": source.session,
                         "fold": fold.fold, "unique_centers": len(centers), "prediction_rows_verified": len(local)})
    if sum(row["prediction_rows_verified"] for row in rows) != len(predictions):
        raise ValueError("extra predictions not belonging to a verified fold")
    checked = pd.DataFrame(rows)
    checked.to_csv(root / "coverage_RUN_validation_independent_audit.csv", index=False)
    result = {"passed": True, "folds": len(rows), "unique_test_centers": int(checked.unique_centers.sum()),
              "verified_prediction_rows": len(predictions), "audit_script_sha256": file_sha256(__file__),
              "validation_manifest_sha256": file_sha256(root / "coverage_RUN_validation_manifest.json"),
              "python_version": sys.version,
              "library_versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "matplotlib"]},
              "scope": "raw-spike support, center tracking truth, fold identity, guard intervals and model/cache hashes; no rescoring"}
    (root / "coverage_RUN_validation_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    audit(parser.parse_args().validation_dir.resolve())
