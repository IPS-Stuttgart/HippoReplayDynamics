from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm.ground_truth as gt_module
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.ground_truth_candidate_support import _score_joint_for_ground_truth
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def _emissions(log_likelihood: np.ndarray) -> LogEmissionTensor:
    log_likelihood = np.asarray(log_likelihood, dtype=float)
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((log_likelihood.shape[0], 1), dtype=int),
        times=np.arange(log_likelihood.shape[0], dtype=float) * 0.01,
        dt=0.01,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_ground_truth_candidate_support_patch_preserves_occupancy_for_state_space(
    monkeypatch,
) -> None:
    train_emissions = _emissions(
        np.log(np.array([[0.70, 0.20, 0.10], [0.20, 0.60, 0.20]], dtype=float))
    )
    joint_emissions = _emissions(
        np.log(np.array([[0.40, 0.40, 0.20], [0.10, 0.20, 0.70]], dtype=float))
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    occupancy_s = np.array([1.0, 0.0, 1.0], dtype=float)
    captured: dict[str, object] = {}

    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(mode="momentum", momentum_candidate_top_k=2),
    )

    def fake_score(
        emissions,
        centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory=True,
    ):
        captured["emissions"] = emissions
        captured["centers"] = centers
        captured["candidate_indices"] = candidate_indices
        captured["occupancy_s"] = occupancy_s
        captured["return_trajectory"] = return_trajectory
        return "score"

    monkeypatch.setattr(model, "score", fake_score)

    assert (
        _score_joint_for_ground_truth(
            model,
            train_emissions,
            joint_emissions,
            bin_centers,
            occupancy_s=occupancy_s,
        )
        == "score"
    )
    assert captured["emissions"] is joint_emissions
    assert captured["centers"] is bin_centers
    assert captured["occupancy_s"] is occupancy_s
    assert len(captured["candidate_indices"]) == train_emissions.n_time


def test_ground_truth_sensitivity_forwards_clusterless_mark_group_by(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_compare_scores_to_ground_truth(root, scores, **kwargs):
        captured["clusterless_mark_group_by"] = kwargs.get("clusterless_mark_group_by")
        return pd.DataFrame(
            {"session": ["Rat1/Open1"], "event_index": [0], "model": ["clusterless"]}
        )

    def fake_generate_behavioral_ground_truth(root, config):
        return pd.DataFrame(
            {"session": ["Rat1/Open1"], "event_index": [0], "valid_label": [True]}
        )

    def fake_add_ground_truth_metrics(comparison, decoded, gt_frame):
        out = comparison.copy()
        out["valid_label"] = True
        out["goal_correct"] = True
        out["active_goal_correct"] = True
        out["endpoint_error_cm"] = 0.0
        out["true_well_posterior"] = 1.0
        return out

    monkeypatch.setattr(
        gt_module, "compare_scores_to_ground_truth", fake_compare_scores_to_ground_truth
    )
    monkeypatch.setattr(
        gt_module, "generate_behavioral_ground_truth", fake_generate_behavioral_ground_truth
    )
    monkeypatch.setattr(gt_module, "_add_ground_truth_metrics", fake_add_ground_truth_metrics)

    result = gt_module.compare_scores_to_ground_truth_sensitivity(
        "root",
        pd.DataFrame(
            {"session": ["Rat1/Open1"], "event_index": [0], "model": ["clusterless"]}
        ),
        sensitivity_config=gt_module.GroundTruthSensitivityConfig(
            visit_radii_cm=(10.0,), min_dwells_s=(0.2,), future_horizons_s=(30.0,)
        ),
        clusterless_mark_group_by="tetrode",
    )

    assert captured["clusterless_mark_group_by"] == "tetrode"
    assert not result.rows.empty
