import hashlib

import numpy as np
import pytest
from scipy.io import savemat

from scripts.acquire_roscow2025 import destination, verify, wanted
from scripts.preflight_roscow2025 import audit_session, trial_table, unit_table


def behavior():
    values = np.empty(5, dtype=object)
    for i in range(5):
        values[i] = np.array([["Optimal", "Legitimate", "Illegitimate"],
                              ["Illegitimate", "Optimal", "Legitimate"]], dtype=object)
    return {
        "n_trials": np.full(5, 2),
        "actions": [np.array([1, 1])] * 5,
        "rewarded": [np.array([0, 0])] * 4 + [np.array([1, 0])],
        "reward_probs": [np.array(["high", "medium", "low"])] * 4 + [np.array(["low", "medium", "high"])],
        "arm_values": values,
    }


def test_sparse_session_number_and_illegitimate_outcome():
    df = trial_table(behavior(), 5, [10, 20], [5, 25])
    assert df.rewarded.tolist() == [1, 0]
    assert df.chosen_probability_label.tolist() == ["low", "low"]
    assert df.eligible_probabilistic_outcome.tolist() == [True, False]
    assert df.probability_assignment_changed_from_session1.all()


@pytest.mark.parametrize("change", ["missing_reward", "arrival_count", "outside_task", "illegitimate_reward"])
def test_trial_mismatch_is_not_silently_repaired(change):
    data = behavior()
    times = [10, 20]
    if change == "missing_reward":
        data["rewarded"][4] = np.array([])
    elif change == "arrival_count":
        times = [10]
    elif change == "outside_task":
        times = [10, 40]
    else:
        data["rewarded"][4] = [1, 1]
    with pytest.raises(ValueError):
        trial_table(data, 5, times, [5, 25])


def test_squeezed_single_trial():
    b = {"n_trials": 1, "actions": 1, "rewarded": 0,
         "reward_probs": ["high", "medium", "low"],
         "arm_values": ["Optimal", "Legitimate", "Illegitimate"]}
    assert len(trial_table(b, 1, 12.0, [5, 25])) == 1


def test_regions_and_phase_spikes():
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0] = np.array([1, 11, 22])
    cells[0, 1] = np.array([2, 12, 23, 24])
    df = unit_table(cells, 1, np.array([[0, 5], [10, 15], [20, 30]]))
    assert df.region.tolist() == ["vStr", "CA1"]
    assert df.post_spikes.tolist() == [1, 2]
    cells[0, 1] = np.array([3, 1])
    with pytest.raises(ValueError, match="unsorted"):
        unit_table(cells, 1, np.array([[0, 5], [10, 15], [20, 30]]))


@pytest.mark.parametrize("folder_name", ["Session5", "session5"])
def test_whole_mat_session_audit(tmp_path, folder_name):
    folder = tmp_path / "Rat_S" / folder_name
    folder.mkdir(parents=True)
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0], cells[0, 1] = np.array([1, 11, 32]), np.array([2, 12, 33])
    for file, data in {
        "spiketimes": {"spiketimes": cells}, "nNAcUnits": {"nNAcUnits": 1},
        "startEndTimes": {"startEndTimes": [[0, 5], [6, 25], [30, 40]]},
        "RarrivalTimes": {"Rarrival": [10, 20]},
        "CPentryTimes": {"central_platform_entry": [8, 18]},
        "CPexitTimes": {"CPexit": [9, 19]},
        "ripples3std": {"rippleStart": [1, 31], "ripplePeak": [1.05, 31.05], "rippleStop": [1.1, 31.1]},
    }.items():
        savemat(folder / (file + ".mat"), data)
    row, trials, units = audit_session(folder, behavior())
    assert row["status"] == "metadata_pass"
    assert row["post_native_ripples"] == 1
    assert row["n_illegitimate_trials"] == 1
    assert not row["sequence_content_validated"]
    assert len(trials) == len(units) == 2


def test_download_hash_and_path(tmp_path):
    p = tmp_path / "payload"
    p.write_bytes(b"abc")
    item = {"path": "data/a.mat", "size": 3, "sha": hashlib.sha1(b"blob 3\0abc").hexdigest()}
    assert verify(p, item) == hashlib.sha256(b"abc").hexdigest()
    p.write_bytes(b"abd")
    with pytest.raises(ValueError, match="blob mismatch"):
        verify(p, item)
    with pytest.raises(ValueError, match="Unsafe"):
        destination(tmp_path, {**item, "path": "../escape"})


def test_explicit_download_scope():
    assert wanted("data/ephys_data/Rat_Q/Session1/spiketimes.mat")
    assert wanted("data/behavioural_data/behaviour_quirinius.mat")
    assert not wanted("data/ephys_data/Rat_Q/Session1/binnedfr.mat")
    assert not wanted("data/explainedVarianceReactivationAnalysis.mat")


@pytest.mark.parametrize("probabilities", [["high", "medium"], ["high", "high", "low"]])
def test_invalid_probability_labels(probabilities):
    b = behavior()
    b["reward_probs"][4] = probabilities
    with pytest.raises(ValueError, match="probability"):
        trial_table(b, 5, [10, 20], [5, 25])


def test_missing_arrival_timestamp_is_not_imputed():
    with pytest.raises(ValueError, match="nonfinite"):
        trial_table(behavior(), 5, [10, np.nan], [5, 25])
