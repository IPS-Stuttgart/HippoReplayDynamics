import json

import numpy as np
import pandas as pd
import pytest

from scripts import audit_pf_count_conditioned_prediction as scorer
from scripts import verify_pf_count_conditioned_prediction as verifier


def fixture_output(root):
    rows = [
        {
            "session": f"{rat}/s",
            "rat": rat,
            "event_index": e,
            "split": split,
            "inference_temperature": 1.0,
            "heldout_temperature": 1.0,
            "map": map_name,
            "observation": obs,
            "model": model,
            "conditional_heldout_log_score": -10.0 if model == "first_order_imm" else -12.0,
            "poisson_heldout_log_score": -15.0,
            "status": "success",
            "cells_disjoint": True,
            "heldout_used_for_inference": False,
            "posterior_unchanged": True,
            "train_cell_ids": "1,2,3",
            "heldout_cell_ids": "4,5",
            "n_train_cells": 3,
            "n_heldout_cells": 2,
            "training_counts_sha256": "same",
            "n_train_spikes": 20,
            "n_heldout_spikes": 10,
            "n_time_bins": 12,
            "n_heldout_nonzero_bins": 8,
        }
        for rat in ["A", "B", "C", "D"]
        for e in [1, 2]
        for split in range(2)
        for map_name in scorer.MAPS
        for obs in scorer.OBSERVATIONS
        for model in scorer.MODELS
    ]
    scores = pd.DataFrame(rows)
    split, events = scorer.contrasts(scores)
    summary, _ = scorer.summarize(events, replicates=10)
    frames = {
        "split_scores.csv": scores,
        "frozen_event_set.csv": scores[["session", "event_index"]].drop_duplicates(),
        "split_contrasts.csv": split,
        "event_medians.csv": events,
        "contrast_summary.csv": summary,
    }
    for name, frame in frames.items():
        frame.to_csv(root / name, index=False)
    source = root / "source_fixture.mat"
    source.write_bytes(b"synthetic provenance fixture, not an Axona/MAT recording")
    manifest = {
        "status": "completed",
        "arguments": {"n_splits": 2, "temperatures": [1.0]},
        "source_mat_sha256": {str(source): verifier.digest(source)},
        "outputs_sha256": {name: verifier.digest(root / name) for name in frames},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_independent_reconstruction(tmp_path):
    fixture_output(tmp_path)
    report, normalized = verifier.verify(tmp_path)
    assert report["status"] == "pass"
    assert report["score_rows"] == 480
    values = normalized[normalized.contrast == "count_conditioned:imm_minus_diffusion"]
    np.testing.assert_allclose(values.delta_per_heldout_spike, 0.2)
    assert not report["raw_posteriors_independently_recomputed"]


@pytest.mark.parametrize(
    "fault", ["powered", "positive_log_probability", "overlapping_cells", "incorrect_event_median", "missing_row", "unknown_split", "unknown_temperature", "mutated_posterior"]
)
def test_semantic_corruption_not_hidden_by_updated_file_hash(tmp_path, fault):
    manifest = fixture_output(tmp_path)
    filename = "event_medians.csv" if fault == "incorrect_event_median" else "split_scores.csv"
    frame = pd.read_csv(tmp_path / filename)
    if fault == "powered":
        frame.loc[0, "heldout_temperature"] = 0.3
    elif fault == "positive_log_probability":
        frame.loc[0, "conditional_heldout_log_score"] = 1
    elif fault == "overlapping_cells":
        frame.loc[0, "heldout_cell_ids"] = "1,5"
    elif fault == "incorrect_event_median":
        frame.loc[0, "delta"] += 1
    elif fault == "unknown_split":
        frame.loc[frame.split.eq(1), "split"] = 9
    elif fault == "unknown_temperature":
        frame["inference_temperature"] = 2
    elif fault == "mutated_posterior":
        frame.loc[0, "posterior_unchanged"] = False
    else:
        frame = frame.iloc[1:]
    frame.to_csv(tmp_path / filename, index=False)
    manifest["outputs_sha256"][filename] = verifier.digest(tmp_path / filename)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        verifier.verify(tmp_path)
