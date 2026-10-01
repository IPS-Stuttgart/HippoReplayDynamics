import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from scripts.odor_place_feasibility_core import clean
from scripts.odor_place_original_source_audit import (
    PREFIX, associate_nwb, coverage_passed, dispatch, epochs, geometry,
    infer_cue, intervals, inventory, pause_window, reconstruct_epoch,
    report, transitions, verify,
)
from scripts.odor_place_source_io import atomic_json, digest

PROTOCOL = json.loads((Path(__file__).parents[1] / "docs/odor_place_source_v3_protocol.json").read_text())


def fixture(abort=False):
    nose = [(1., 1.6), (5., 5.6)]
    if abort:
        nose.insert(1, (4.5, 4.6))
    channels = np.empty((1, 23), dtype=object)
    for i in range(23):
        channels[0, i] = SimpleNamespace(_fieldnames=["time", "state"], time=np.array([]), state=np.array([]))
    for number, windows in {5: nose, 22: [(1.001, 1.601)], 23: [(5.001, 5.601)],
                            1: [(6.5, 7.5)], 2: [(2.5, 3.5)]}.items():
        channels[0, number - 1].time = np.array([v for pair in windows for v in pair])
        channels[0, number - 1].state = np.tile([1., 0.], len(windows))
    original = {"allTriggers": [1., 5.], "leftTriggers": [1.], "rightTriggers": [5.],
                "correctTriggers": [], "incorrectTriggers": [1., 5.]}
    times = np.arange(0., 10., .05)
    xy = np.tile([0., 20.], (len(times), 1))
    xy[(times >= 2.) & (times < 4.)] = [10., 0.]
    xy[(times >= 6.) & (times < 8.)] = [-10., 0.]
    data = np.column_stack([times, xy, np.zeros((len(times), 2))])
    centers = {"left": np.array([-10., 0.]), "right": np.array([10., 0.])}
    rows = reconstruct_epoch(channels, original, np.array([[1., 1.6], [5., 5.6]]), data, centers, PROTOCOL)
    for row in rows:
        row.update(animal="CS31", source_day=1, source_epoch=2, primary_condition=True)
    headers = {"asset": {"subject": "Symanski-CS31", "trial_arrays": {"start_time": [1., 5.], "id": [1, 2], "rewarded": [False, False]}}}
    associate_nwb(rows, headers, .005)
    return channels, original, data, centers, rows, headers


def test_changed_cue_correction_is_not_next_trial_task_correctness():
    *_, rows, _ = fixture()
    result = transitions(rows, PROTOCOL)
    assert result[0]["eligible_source_transition"]
    assert result[0]["cue_condition"] == "changed"
    assert result[0]["outcome_relative_to_prior_error_cue"] == "correction_to_prior_cue_arm"
    assert result[0]["next_task_correct"] is False
    assert rows[0]["pump_onset_s"] is None
    assert rows[0]["task_correct"] is False


def test_aborted_sample_is_not_skipped_to_the_next_completed_trial():
    *_, rows, _ = fixture(abort=True)
    result = transitions(rows, PROTOCOL)
    assert not result[0]["eligible_source_transition"]
    assert result[0]["exclusion_reason"] == "intervening_unresolved_or_premature_sample"
    assert result[0]["next_sample_index"] == 1


def test_cue_cannot_be_reconstructed_from_choice_and_correctness():
    channels, original, data, centers, _, _ = fixture()
    channels[0, 21].time = np.array([])
    channels[0, 21].state = np.array([])
    rows = reconstruct_epoch(channels, original, np.array([[1., 1.6], [5., 5.6]]), data, centers, PROTOCOL)
    assert not rows[0]["cue_verified"] and rows[0]["cue"] is None
    assert rows[0]["source_error_annotation"] is True
    assert rows[0]["exclusion_reason"] == "missing_or_multiple_independent_odor_onsets"


def test_missing_raw_sample_keeps_original_error_visible():
    channels, original, data, centers, _, _ = fixture()
    channels[0, 4].time = np.array([5., 5.6])
    channels[0, 4].state = np.array([1., 0.])
    rows = reconstruct_epoch(channels, original, np.array([[1., 1.6], [5., 5.6]]), data, centers, PROTOCOL)
    orphan = next(r for r in rows if r["source_sample_index"] < 0)
    assert orphan["source_error_annotation"] is True
    assert orphan["exclusion_reason"] == "original_trigger_without_unique_raw_nosepoke"


def test_conflicting_original_cue_fails_instead_of_using_outcome():
    channels, original, data, centers, _, _ = fixture()
    original["leftTriggers"] = []
    original["rightTriggers"] = [1., 5.]
    rows = reconstruct_epoch(channels, original, np.array([[1., 1.6], [5., 5.6]]), data, centers, PROTOCOL)
    assert rows[0]["exclusion_reason"] == "conflicting_independent_cue_or_sample_annotations"
    assert not rows[0]["trial_verified"]


@pytest.mark.parametrize("times,states", [([1., 1.], [1, 0]), ([2., 1.], [1, 0]), ([1., 2.], [1, 1]), ([1., np.nan], [1, 0]), ([1.], [2])])
def test_invalid_digital_chronology_is_explicit(times, states):
    with pytest.raises(ValueError):
        intervals(times, states)


def test_missing_offset_and_multiple_odors_are_not_guessed():
    assert intervals([1.], [1]) == [(1., None)]
    assert infer_cue(1., 1.6, {"left": [(1., 1.6)], "right": [(1.01, 1.6)]}, .005)[0] is None


def test_clock_drift_or_duplicate_assets_never_receive_fitted_offsets():
    *_, rows, headers = fixture()
    headers["asset"]["trial_arrays"]["start_time"] = [1.02, 5.02]
    associate_nwb(rows, headers, .005)
    assert all(r["nwb_clock_status"] == "unmatched" for r in rows)
    headers["asset"]["trial_arrays"]["start_time"] = [1., 5.]
    headers["duplicate"] = headers["asset"]
    associate_nwb(rows, headers, .005)
    assert all(r["nwb_clock_status"] == "ambiguous_across_assets" for r in rows)


def test_tracking_gap_or_earlier_visit_is_not_bridged():
    _, _, data, centers, _, _ = fixture()
    altered = data[(data[:, 0] < 2.45) | (data[:, 0] > 2.6)]
    assert pause_window(altered, centers["right"], 1.6, 2.5, 5., PROTOCOL)["pause_exclusion_reason"] == "well_sensor_tracking_gap"
    earlier = data.copy()
    earlier[(earlier[:, 0] >= 1.7) & (earlier[:, 0] < 1.8), 1:3] = centers["right"]
    assert pause_window(earlier, centers["right"], 1.6, 2.5, 5., PROTOCOL)["pause_exclusion_reason"] == "earlier_wrong_well_visit_before_sensor_choice"


def test_short_or_fast_pause_cannot_count_as_eligible_exposure():
    _, _, data, centers, _, _ = fixture()
    data[:, 4] = 5.
    result = pause_window(data, centers["right"], 1.6, 2.5, 5., PROTOCOL)
    assert result["usable_exposure_s"] == 0.
    assert result["pause_exclusion_reason"] == "insufficient_immobile_wrong_well_exposure"


def test_day_and_epoch_reset_cannot_create_a_transition():
    *_, rows, _ = fixture()
    rows[1]["source_epoch"] = 4
    result = transitions(rows, PROTOCOL)
    assert not any(r["eligible_source_transition"] for r in result)
    assert all(r["exclusion_reason"] == "no_next_trial_before_epoch_end" for r in result)


def test_matlab_empty_days_keep_original_indices():
    outer = np.empty((1, 3), dtype=object)
    outer[0, 0] = outer[0, 1] = np.empty((0, 0))
    inner = np.empty((1, 4), dtype=object)
    for i in range(4):
        inner[0, i] = np.array([[2.]]) if i == 3 else np.empty((0, 0))
    outer[0, 2] = inner
    assert list(epochs(outer, 3)) == [4]
    assert epochs(outer, 2) == {}


def test_geometry_requires_explicit_shared_graph():
    left = np.array([[0., 20.], [0., 0.], [-20., 0.]])[:, :, None]
    right = left.copy()
    right[-1, 0] = 20.
    routes = np.empty((1, 2), dtype=object)
    routes[0] = [left, right]
    assert geometry({"linearcoord": routes})["left"].tolist() == [-20., 0.]
    right[0, 0] = 99.
    with pytest.raises(ValueError, match="share"):
        geometry({"linearcoord": routes})


def test_zero_transitions_and_missing_condition_cannot_pass_screen():
    screen = PROTOCOL["screening"]
    assert coverage_passed([], [], [], screen) == (False, [])
    animals = [{"animal": a, "source_full_maze": True, "both_conditions": True} for a in PROTOCOL["full_maze_animals_source_code"]]
    cells = [{"outcome_relative_to_prior_error_cue": o, "n_source_transitions": 25} for _ in range(2)
             for o in ["correction_to_prior_cue_arm", "repeated_mistaken_arm"]]
    assert coverage_passed(animals, list(range(100)), cells, screen)[0]
    cells[-1]["n_source_transitions"] = 19
    assert not coverage_passed(animals, list(range(100)), cells, screen)[0]


def test_source_stage_rejects_neural_processing():
    with pytest.raises(ValueError, match="source-only"):
        dispatch(SimpleNamespace(stage="run-qc"), PROTOCOL)


def test_real_mat_zip_inventory_and_saved_accounting(tmp_path):
    channels, original, data, centers, _, headers = fixture()
    root, out, ref = [tmp_path / p for p in ["dataset", "output", "reference"]]
    (root / "metadata").mkdir(parents=True)
    out.mkdir()
    (ref / "checkpoints").mkdir(parents=True)
    identity = ref / (PREFIX + "inventory_identity.json")
    identity.write_text("{}")
    protocol = {**PROTOCOL, "v1_reference_inventory_sha256": digest(identity)}
    for i in range(38):
        observed = headers["asset"] if i == 0 else {"subject": "Symanski-CS33", "trial_arrays": {"start_time": [], "id": [], "rewarded": []}}
        packet = {"status": "readable", "observations": observed}
        import hashlib
        atomic_json(ref / "checkpoints" / f"asset{i}.json", {"packet": packet, "packet_sha256": hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest()})
    routes = np.empty((1, 2), dtype=object)
    for i, side in enumerate(["left", "right"]):
        routes[0, i] = np.array([[0., 20.], [0., 0.], centers[side]])[:, :, None]
    contents = {"task": {"type": "run", "environment": "odorplace", "linearcoord": routes},
                "DIO": channels, "odorTriggers": original, "nosepokeWindow": np.array([[1., 1.6], [5., 5.6]]),
                "pos": {"fields": "time x y dir vel", "data": data}}
    archive = root / "Figure1-6.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for role, value in contents.items():
            if role == "DIO":
                serialized = np.empty_like(channels)
                for i in range(channels.size):
                    serialized.ravel()[i] = {"time": channels.ravel()[i].time, "state": channels.ravel()[i].state}
                value = serialized
            inner = np.empty((1, 2), dtype=object)
            inner[0, 0], inner[0, 1] = np.empty((0, 0)), value
            outer = np.empty((1, 1), dtype=object)
            outer[0, 0] = inner
            memory = io.BytesIO()
            savemat(memory, {"dio" if role == "DIO" else role: outer})
            handle.writestr(f"Figure1-6/CS31task/CS31{role}01.mat", memory.getvalue())
    atomic_json(root / "metadata/source_verified.json", {"sha256": digest(archive), "published_digest": protocol["source_published_md5"]})
    path = tmp_path / "protocol.json"
    atomic_json(path, protocol)
    args = SimpleNamespace(output_dir=out, dataset_root=root, reference_inventory=ref, source_archive=archive, protocol=path)
    result = inventory(args, protocol)
    assert result["n_source_errors"] == 2
    assert result["n_eligible_source_transitions"] == 1
    assert not result["ready_for_calibration"]
    assert verify(args, protocol)["n_eligible_source_transitions"] == 1
    assert report(args, protocol)["status"] == "inconclusive_source_feasibility"
    assert clean(json.loads((out / (PREFIX + "source_sensor_checkpoints.json")).read_text()))[0]["source_sensor_trials"][0]["nwb_asset_id"] == "asset0"
    (out / (PREFIX + "decision.json")).write_text("{}")
    with pytest.raises(ValueError, match="decision differs"):
        verify(args, protocol)
