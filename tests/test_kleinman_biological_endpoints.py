import json

import numpy as np
import pytest
from scipy.io import savemat

from scripts.preflight_kleinman_biological_endpoints import (
    inspect_session,
    main,
    summarize_inventory,
)


def fixture(tmp_path, experiment="Experiment_1", **info):
    folder = tmp_path / experiment / "Con_3" / "20220617_run2"
    folder.mkdir(parents=True)
    path = folder / "session_info.mat"
    savemat(path, {"session_info": {"drug": 1, **info}})
    return path


def test_optional_novelty_and_planned_not_delivered(tmp_path):
    path = fixture(tmp_path, "Experiment_2", reward_schedule=np.zeros((100, 2)),
                   left_visit=np.ones((28, 2)), right_visit=np.ones((28, 2)))
    row = inspect_session(path)
    assert row["status"] == "metadata_readable"
    assert not row["novel_label_present"]
    assert row["novel"] == ""
    assert row["reward_schedule_rows"] == 100
    assert row["left_visit_rows"] == 28
    assert row["zero_reward_schedule_entries"] == 200
    assert not row["schedule_alignment_validated"]
    assert summarize_inventory([row])[0]["sessions_with_omission_schedule_and_spikes"] == 0


def test_spike_inventory_is_not_unit_identity_validation(tmp_path):
    path = fixture(tmp_path, novel=0, epoch_change=np.ones((2, 2)), incr_end=1)
    savemat(path.parent / "spike_data.mat", {"spike_data": np.ones((5, 3))})
    row = inspect_session(path)
    assert row["status"] == "metadata_readable"
    assert row["spike_data_present"]
    assert json.loads(row["spike_data_shape"]) == [5, 3]
    assert not row["unit_identity_validated"]
    assert row["novel"] == 0
    assert row["epoch_change_present"]


@pytest.mark.parametrize("info", [{"novel": 3}, {"drug": np.nan},
                                  {"reward_schedule": np.ones((3, 3))}])
def test_bad_metadata_reported_not_silently_defaulted(tmp_path, info):
    assert inspect_session(fixture(tmp_path, **info))["status"] == "failed"


def test_cli_provenance_and_no_overwrite(tmp_path):
    fixture(tmp_path / "raw", novel=1)
    args = ["--dataset-root", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "out")]
    main(args)
    manifest = json.loads((tmp_path / "out" / "inventory_manifest.json").read_text())
    assert manifest["summary"][0]["sessions"] == 1
    assert manifest["real_events_scored"] is False
    assert manifest["output_sha256"]
    with pytest.raises(FileExistsError):
        main(args)
