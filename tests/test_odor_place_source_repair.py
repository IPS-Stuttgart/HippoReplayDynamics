import json
from pathlib import Path

import numpy as np
import pytest

from scripts.odor_place_original_source_audit import (
    assign_source_identity, channel, intervals, reconstruct_epoch,
    transitions, unresolved_source_trials,
)
from tests.test_odor_place_original_source_audit import fixture, PROTOCOL

AMENDED = json.loads((Path(__file__).parents[1] / "docs/odor_place_source_v3_neural_feasibility_protocol.json").read_text())


def test_observed_three_zero_duration_pulses_do_not_invalidate_later_samples():
    times = [1545.06, 1545.729, 1557.684, 1558.321, 1618.414, 1618.414,
             1618.525, 1618.526, 1618.528, 1618.528, 1618.542, 1618.542, 1652.174, 1652.773]
    states = np.tile([1, 0], 7)
    with pytest.raises(ValueError):
        intervals(times, states)
    result = intervals(times, states, allow_zero_duration_on_off=True)
    assert len(result) == 7
    assert sum(a == b for a, b in result) == 3
    assert result[-1] == (1652.174, 1652.773)


@pytest.mark.parametrize("times,states", [
    ([2, 1], [1, 0]), ([1, 1], [0, 1]), ([1, 1, 1], [1, 0, 1]),
    ([1, 2], [1, 1]), ([1, np.nan], [1, 0]), ([1, 2], [1, 3]),
])
def test_repair_does_not_admit_other_sensor_errors(times, states):
    with pytest.raises(ValueError):
        intervals(times, states, allow_zero_duration_on_off=True)


def test_exception_cannot_be_applied_to_other_digital_channels():
    channels, *_ = fixture()
    with pytest.raises(ValueError, match="only.*channel 5"):
        channel(channels, 22, allow_zero_duration_on_off=True)


def test_zero_duration_sample_interrupts_error_to_next_trial_transition():
    channels, original, data, centers, _, headers = fixture()
    channels[0, 4].time = np.array([1., 1.6, 4.5, 4.5, 5., 5.6])
    channels[0, 4].state = np.tile([1., 0.], 3)
    rows = reconstruct_epoch(channels, original, np.array([[1., 1.6], [5., 5.6]]), data, centers, AMENDED)
    from scripts.odor_place_original_source_audit import associate_nwb
    for row in rows:
        row.update(animal="CS31", source_day=1, source_epoch=2, primary_condition=True)
    associate_nwb(rows, headers, .005)
    assert rows[1]["exclusion_reason"] == "zero_duration_nosepoke_pulse"
    assert not rows[1]["trial_verified"]
    result = transitions(rows, AMENDED)[0]
    assert not result["eligible_source_transition"]
    assert result["next_sample_index"] == 1
    assert result["exclusion_reason"] == "intervening_unresolved_or_premature_sample"


def test_unaffected_source_rows_are_identical_with_new_policy():
    channels, original, data, centers, _, _ = fixture()
    windows = np.array([[1., 1.6], [5., 5.6]])
    assert reconstruct_epoch(channels, original, windows, data, centers, PROTOCOL) == reconstruct_epoch(channels, original, windows, data, centers, AMENDED)


def test_trigger_identity_survives_epoch_reconstruction_and_row_changes():
    _, original, _, _, reconstructed, _ = fixture()
    unresolved = unresolved_source_trials(original, "old_epoch_failure")
    for rows in [reconstructed, unresolved]:
        assign_source_identity(rows, original, "CS31", 1, 2, .005)
    assert [r["source_trial_key"] for r in reconstructed] == [r["source_trial_key"] for r in unresolved]
    assert reconstructed[0]["source_sample_index"] != unresolved[0]["source_sample_index"]
    assert reconstructed[0]["source_trigger_index"] == 0


def test_frozen_floors_and_source_pins_are_unchanged():
    for field in ["screening", "minimum_odor_sample_s", "maximum_tracking_gap_s", "well_radius_cm",
                  "immobility_max_speed_cm_s", "minimum_pause_exposure_s", "maximum_pause_s",
                  "dandi_version", "figshare_version", "source_published_md5", "animals"]:
        assert AMENDED[field] == PROTOCOL[field]
    assert AMENDED["seed"] == 20261001
    assert AMENDED["no_association_stage"]
    assert AMENDED["allow_zero_duration_nosepoke_on_off"]


def test_amended_inventory_is_independently_verified_against_raw_mat_edges(tmp_path, monkeypatch):
    from tests import test_odor_place_original_source_audit as original
    protocol = dict(AMENDED)
    protocol.pop("previous_audit_identity_sha256")
    monkeypatch.setattr(original, "PROTOCOL", protocol)
    original.test_real_mat_zip_inventory_and_saved_accounting(tmp_path)
    saved = json.loads((tmp_path / "output/odor_place_post_error_verification.json").read_text())
    assert saved["raw_source_reconciliation"]["status"] == "verified_against_raw_digital_edges"
    assert saved["raw_source_reconciliation"]["verified_trials"] == 2
