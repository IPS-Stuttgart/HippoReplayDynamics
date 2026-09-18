import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from test_score_tirole_content_coverage import write_bank

from hipporeplayimm.tirole_two_track import file_sha256
from scripts.score_tirole_content_coverage import run as score
from scripts.verify_tirole_content_bank import run as verify


def raw_fixture(root, bank):
    root.mkdir()
    write_bank(bank)
    d = np.load(bank / "event_counts.npz")
    st = []
    ids = []
    events = pd.read_csv(bank / "candidate_events.csv")
    for i in range(2):
        c = d["counts"][i * 20 : (i + 1) * 20]
        events.loc[i, "start_s"] = 10.0 + i
        events.loc[i, "n_time_bins"] = 20
        for t, u in zip(*np.nonzero(c), strict=True):
            st.extend([10.0 + i + (t + 0.5) * 0.02] * int(c[t, u]))
            ids.extend([u] * int(c[t, u]))
    savemat(root / "synthetic_extracted_clusters.mat", {"clusters": {"spike_times": np.array(st), "spike_id": np.array(ids)}})
    events.to_csv(bank / "candidate_events.csv", index=False)
    m = json.loads((bank / "manifest.json").read_text())
    m["outputs_sha256"]["candidate_events.csv"] = file_sha256(bank / "candidate_events.csv")
    (bank / "manifest.json").write_text(json.dumps(m))


def test_independent_verifier_reconstructs_raw_and_nulls(tmp_path):
    raw, bank = tmp_path / "raw", tmp_path / "bank"
    raw_fixture(raw, bank)
    score(bank, tmp_path / "scores", 0, 0, limit=1)
    verify(raw, bank, tmp_path / "audit.json", tmp_path / "scores")
    result = json.loads((tmp_path / "audit.json").read_text())
    assert result["status"] == "pass" and result["count_mismatch_entries"] == 0
    assert result["independent_sequence_null_checks"]


def test_raw_spike_mismatch_fails(tmp_path):
    raw, bank = tmp_path / "raw", tmp_path / "bank"
    raw_fixture(raw, bank)
    d = np.load(bank / "event_counts.npz")
    c = d["counts"].copy()
    c[0, 0] += 1
    np.savez_compressed(bank / "event_counts.npz", counts=c, offsets=d["offsets"], unit_ids=d["unit_ids"])
    with pytest.raises(SystemExit):
        verify(raw, bank, tmp_path / "audit.json")
    assert json.loads((tmp_path / "audit.json").read_text())["count_mismatch_entries"] == 1
