import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("joint_readout", Path(__file__).parents[1] / "scripts/calibrate_dandi000978_joint_readout.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture(autouse=True)
def smaller(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "virtual_studies", 300)


def test_score_matrix_preserves_reference_and_donor_axes():
    v = np.arange(32).reshape(2, 4, 4)
    references = np.array([[3, 2, 1, 0], [0, 0, 1, 1]])
    result = m.score_matrix(v, references)
    for b in range(2):
        for i in range(4):
            for j in range(4):
                assert result[b, i, j] == v[b, j, references[b, i]]


def test_oracle_does_not_assume_decoded_reference_is_truth():
    v = np.eye(4)[None]
    wrong = np.array([[1, 2, 3, 0]])
    result = m.score_matrix(v, wrong)
    assert np.diagonal(result, axis1=1, axis2=2).sum() == 0
    oracle = m.score_matrix(v, np.arange(4)[None])
    assert np.diagonal(oracle, axis1=1, axis2=2).sum() == 4


def test_strong_joint_information_detected_and_null_calibrated():
    matrix = np.broadcast_to(np.eye(4), (12, 4, 4))
    targets = np.repeat(["a", "b", "c", "d"], 3)
    result, audit = m.group_simulation(matrix, targets, ["a", "b", "c", "d"], ("strong",))
    assert result["positive_control_rejection"] > 0.95
    assert result["null_false_positive_rate"] <= 0.08
    assert np.asarray(audit["rows"]).shape == (300, 4)


def test_one_event_cannot_supply_strong_route_pairing_evidence():
    matrix = np.eye(4)[None]
    result, _ = m.group_simulation(matrix, np.array(["a"]), ["a"], ("single",))
    assert result["positive_control_rejection"] == 0


def test_uninformative_scores_never_pass():
    matrix = np.zeros((12, 4, 4))
    result, _ = m.group_simulation(matrix, np.repeat(["a", "b", "c", "d"], 3), 4, ("null",))
    assert result["positive_control_rejection"] == 0
    assert result["null_false_positive_rate"] == 0


def test_missing_primary_target_not_silently_substituted():
    result, audit = m.group_simulation(np.eye(4)[None], np.array(["a"]), ["a", "missing"], ("missing",))
    assert result["status"] == "unsupported" and result["n_requested"] == 2
    assert result["missing_target_count"] == 1 and audit is None


def test_sensitivity_targets_unique_and_seed_deterministic():
    targets = np.repeat(["a", "b", "c", "d"], 3)
    matrix = np.broadcast_to(np.eye(4), (12, 4, 4))
    first, a = m.group_simulation(matrix, targets, 3, ("deterministic",))
    second, b = m.group_simulation(matrix, targets, 3, ("deterministic",))
    assert first == second
    assert np.array_equal(a["rows"], b["rows"])
    for draw in a["rows"]:
        assert len(set(targets[draw])) == 3


def test_insufficient_unique_targets_explicit():
    result, _ = m.group_simulation(np.eye(4)[None], np.array(["a"]), 3, ("few",))
    assert result["status"] == "unsupported"


def test_invalid_matrix_shape_rejected():
    with pytest.raises(ValueError, match="Four-route"):
        m.score_matrix(np.zeros((2, 3, 4)), np.zeros((2, 3), dtype=int))


def paired_bank():
    frame = pd.DataFrame(
        {"target_index": [0] * 4, "repeat": [0] * 4, "scope": ["window_250ms"] * 4, "route": range(4), "trial_id": range(4), "matched": [True] * 4, "target_count": [5] * 4}
    )
    base = {
        "target_event_ids": np.array(["event"]),
        "target_counts": np.array([5]),
        "training_trial_ids": np.array([99]),
        "test_trial_ids": np.arange(4),
        "native": np.full((4, 4), 3),
        "sparse": np.ones((4, 4), dtype=int) + np.eye(4, dtype=int),
        "rates": np.ones((4, 4)) + 4 * np.eye(4),
    }
    ca = {**base, "source_unit_ids": np.arange(4)}
    pf = {**base, "source_unit_ids": np.arange(4) + 10}
    return frame, frame.copy(), ca, pf


def test_joint_bank_references_and_same_eligible_controls():
    ca, pf, cz, pz = paired_bank()
    blocks, supported, bank = m.build_bank(ca, pf, cz, pz)
    assert blocks.matched_block.all() and len(supported) == 1
    assert bank["sleep_matched_ca1_accuracy"][0] == 1
    assert np.array_equal(bank["sleep_matched_decoded"], bank["sleep_matched_oracle"])
    assert np.diagonal(bank["sleep_matched_decoded"][0]).min() > 0


def test_any_unmatched_route_excludes_whole_block_not_only_bad_route():
    ca, pf, cz, pz = paired_bank()
    pz["native"] = pz["native"].copy()
    pz["sparse"] = pz["sparse"].copy()
    pz["native"][0] = 1
    pz["sparse"][0] = 0
    pf.loc[0, "matched"] = False
    blocks, supported, bank = m.build_bank(ca, pf, cz, pz)
    assert not blocks.matched_block.any()
    assert supported.empty and bank["sleep_matched_decoded"].shape == (0, 4, 4)


@pytest.mark.parametrize("corruption", ["overlap", "test_leakage", "wrong_order", "extra_spikes"])
def test_corrupt_sources_fail(corruption):
    ca, pf, cz, pz = paired_bank()
    if corruption == "overlap":
        pz["source_unit_ids"] = cz["source_unit_ids"]
    elif corruption == "test_leakage":
        cz["training_trial_ids"] = np.array([0, 99])
    elif corruption == "wrong_order":
        pf = pf.iloc[::-1].reset_index(drop=True)
    else:
        pz["sparse"] = pz["sparse"].copy()
        pz["sparse"][0, 0] = 5
    with pytest.raises(ValueError):
        m.validate_pair(ca, pf, cz, pz)
