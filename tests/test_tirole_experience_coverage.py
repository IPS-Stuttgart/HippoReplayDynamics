import json

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.tirole_two_track import file_sha256
from scripts.audit_tirole_experience_coverage import coverage_subsets, run_information
from scripts.report_tirole_experience_coverage import ARMS, bootstrap_weights, event_contributions, run, statistics


def test_information_is_rate_scale_invariant_and_constant_fields_have_zero_information():
    rates = np.random.default_rng(4).uniform(0.1, 5, (2, 9, 20))
    occupancy = np.ones((2, 20))
    valid = occupancy.astype(bool)
    info, mean = run_information(rates, occupancy, valid)
    scaled, mean_scaled = run_information(3 * rates, occupancy, valid)
    np.testing.assert_allclose(info, scaled)
    np.testing.assert_allclose(mean_scaled, 3 * mean)
    flat, _ = run_information(np.ones_like(rates), occupancy, valid)
    np.testing.assert_allclose(flat, 0.0)


def test_targeted_subsets_use_one_cell_per_rate_pair_and_flip_with_track_labels():
    information = np.array([np.arange(9), np.arange(9)[::-1]], float)
    rate = np.arange(1.0, 10.0)
    arms, pairs, singleton = coverage_subsets(np.arange(9), information, rate, "fixture", 0)
    flipped, _, _ = coverage_subsets(np.arange(9), information[::-1], rate, "fixture", 0)
    assert len(arms) == 13 and singleton.tolist() == [8]
    np.testing.assert_array_equal(arms["track1_information"], flipped["track2_information"])
    for name, units in arms.items():
        if name == "full":
            continue
        assert len(units) == 5 and 8 in units
        assert all(len(set(pair) & set(units)) == 1 for pair in pairs)
    for r in range(5):
        assert set(arms[f"random_{r}_a"]) & set(arms[f"random_{r}_b"]) == {8}


def scores():
    rows = []
    for event in range(6):
        for split in range(5):
            for arm in ARMS:
                selected = event < 3 if arm == "track1_information" else event >= 3
                rows.append(
                    {
                        "session": "fixture",
                        "animal": "RAT",
                        "event_id": event,
                        "split": split,
                        "arm": arm,
                        "epoch": "POST",
                        "primary_ripple_candidate": True,
                        "start_s": event * 61.0,
                        "n_inference_units": 10 if arm == "full" else 5,
                        "sequence_accepted": selected,
                        "sequence_eligible": True,
                        "inferred_track": 1 if event < 3 else 2,
                        "evaluation_conditional_track2_probability": 0.2 if event < 3 else 0.8,
                        "evaluation_conditional_track2_identity_z": -1.0 if event < 3 else 1.0,
                        "duration_s": 0.2,
                        "n_evaluation_spikes": 10,
                        "n_inference_spikes": 20,
                    }
                )
    return pd.DataFrame(rows)


def test_original_event_weight_and_independent_readout_contrast():
    data = event_contributions(scores())
    result = statistics(data, np.ones(6))
    i, j = ARMS.index("track1_information"), ARMS.index("track2_information")
    assert result["conditional_track2_score_mean"][0, j] - result["conditional_track2_score_mean"][0, i] == pytest.approx(0.6)
    assert result["selected_event_equivalents"][0, i] == 3
    assert len(data) == 6 * len(ARMS)


def test_readout_changes_with_inference_arm_fail():
    frame = scores()
    frame.loc[0, "evaluation_conditional_track2_probability"] = 0.99
    with pytest.raises(ValueError, match="readout changes"):
        event_contributions(frame)


def test_zero_selected_events_remain_undefined():
    frame = scores()
    frame["sequence_accepted"] = False
    result = statistics(event_contributions(frame), np.ones(6))
    assert np.isnan(result["conditional_track2_score_mean"]).all()
    assert (result["selected_event_equivalents"] == 0).all()


def test_no_candidates_and_missing_arm_scores_fail():
    with pytest.raises(ValueError, match="zero selected"):
        event_contributions(scores().iloc[:0])
    with pytest.raises(ValueError, match="missing/duplicate"):
        event_contributions(scores().iloc[1:])


def test_block_resampling_keeps_shared_block_events_together():
    frame = scores()
    frame.loc[frame.event_id.isin([0, 1]), "start_s"] = 1.0
    contributions = event_contributions(frame)
    weights, n_blocks = bootstrap_weights(contributions, "fixture", "POST", True, n_boot=30)
    assert n_blocks == 5
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])


@pytest.mark.parametrize("coherent", [True, False])
def test_complete_report_requires_independent_direction_not_just_classifier_labels(tmp_path, coherent):
    source = tmp_path / "source"
    source.mkdir()
    frames = []
    for copy in range(4):
        frame = scores()
        frame["event_id"] += 6 * copy
        frame["start_s"] += 400 * copy
        if not coherent:
            frame["evaluation_conditional_track2_probability"] = 1 - frame.evaluation_conditional_track2_probability
            frame["evaluation_conditional_track2_identity_z"] *= -1
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(source / "event_coverage_scores.csv", index=False)
    pd.DataFrame([{"split": split, "arm": arm, "mean_track1_minus_track2_information": 1 if arm == "track1_information" else -1} for split in range(5) for arm in ARMS]).to_csv(
        source / "RUN_only_subsets.csv", index=False
    )
    manifest = {
        "status": "complete",
        "git_dirty": False,
        "subset_selection_RUN_only": True,
        "evaluation_readout_reused_unchanged": True,
        "session": "fixture",
        "selected_event_ids": sorted(frame.event_id.unique().tolist()),
        "n_score_rows": len(frame),
        "output_sha256": {p.name: file_sha256(p) for p in source.iterdir()},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "report"
    run(source, out)
    gates = pd.read_csv(out / "gate_summary.csv").set_index("gate").passed
    assert bool(gates.session_mechanistic_readout) == coherent
    assert gates.five_supported_events_each_targeted_arm
    assert not json.loads((out / "manifest.json").read_text())["real_experience_fractions_corrected"]
    manifest["selected_event_ids"].append(999)
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen candidate manifest"):
        run(source, tmp_path / "must_not_exist")
