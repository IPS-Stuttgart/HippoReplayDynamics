import json

import numpy as np
import pandas as pd
import pytest
from test_two_track_content import maps_and_counts

from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import split_populations
from scripts.score_tirole_content_coverage import chosen_events, run, validate_bank


def write_bank(root, boost_evaluation=False, overlap=False):
    root.mkdir()
    rates, counts, valid, _, _ = maps_and_counts()
    detector, splits = split_populations(np.arange(30), "synthetic")
    counts = counts.copy()
    if boost_evaluation:
        counts[:, splits[0][1]] *= 3
    np.savez_compressed(root / "event_counts.npz", counts=np.concatenate([counts, counts[::-1]]), offsets=[0, 20, 40], unit_ids=np.arange(30))
    np.savez_compressed(root / "RUN_maps.npz", rates=rates, valid_bins=valid, bin_centers_cm=np.arange(20), unit_ids=np.arange(30), common_units=np.arange(30))
    parts = {"detector": detector.tolist(), "splits": [{"inference": a.tolist(), "evaluation": b.tolist()} for a, b in splits], "seed": 20260918}
    if overlap:
        parts["splits"][0]["inference"].append(parts["splits"][0]["evaluation"][0])
    (root / "partitions.json").write_text(json.dumps(parts))
    pd.DataFrame([{"event_id": i, "epoch": "POST" if i else "PRE", "primary_ripple_candidate": True, "ripple_supported": True, "ripple_peak_z": 4} for i in range(2)]).to_csv(
        root / "candidate_events.csv", index=False
    )
    manifest = {"session": "synthetic", "animal": "synthetic", "cohort_stratum": "synthetic_test", "outputs_sha256": {p.name: file_sha256(p) for p in root.iterdir()}}
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_full_pipeline_does_not_use_evaluation_spikes_for_sequence(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_bank(a)
    write_bank(b, boost_evaluation=True)
    run(a, tmp_path / "out_a", 0, 0, limit=1)
    run(b, tmp_path / "out_b", 0, 0, limit=1)
    x = pd.read_csv(tmp_path / "out_a/event_content_coverage_scores.csv")
    y = pd.read_csv(tmp_path / "out_b/event_content_coverage_scores.csv")
    cols = [c for c in x if c.startswith(("track", "sequence_")) or c in {"best_weighted_correlation", "inferred_track"}]
    pd.testing.assert_frame_equal(x[cols], y[cols])
    np.testing.assert_equal(y.evaluation_n_eval_spikes.to_numpy(), 3 * x.evaluation_n_eval_spikes.to_numpy())
    assert len(x) == 12
    m = json.loads((tmp_path / "out_a/manifest.json").read_text())
    assert m["technical_pilot_only"] and m["expected_score_rows"] == 12
    assert not m["biological_claim_gate"]


def test_cell_role_overlap_stops_scoring(tmp_path):
    write_bank(tmp_path / "bank", overlap=True)
    with pytest.raises(ValueError, match="cell-role"):
        validate_bank(tmp_path / "bank")


def test_modified_bank_stops_scoring(tmp_path):
    root = tmp_path / "bank"
    write_bank(root)
    (root / "partitions.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        validate_bank(root)


def test_zero_selection_not_a_pass(tmp_path):
    root = tmp_path / "bank"
    write_bank(root)
    with pytest.raises(ValueError, match="zero selected"):
        run(root, tmp_path / "empty", 0, 0, "mua_only")


def test_cap_is_balanced_and_independent_of_scores():
    rows = pd.DataFrame({"event_id": np.arange(20), "epoch": ["PRE"] * 10 + ["POST"] * 10, "primary_ripple_candidate": True, "ripple_supported": True})
    a = chosen_events(rows, "s", "ripple", 3)
    assert a.groupby("epoch").size().to_dict() == {"POST": 3, "PRE": 3}
    rows["future_outcome"] = np.arange(20)
    b = chosen_events(rows, "s", "ripple", 3)
    assert a.event_id.tolist() == b.event_id.tolist()
