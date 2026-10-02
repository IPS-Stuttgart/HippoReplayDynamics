"""Independent numerical score checks; not endpoint or biological calibration."""
import json
from pathlib import Path

import numpy as np

from scripts import _run_pair_rate_glm as glm
from scripts.verify_run_pair_rate_prediction_audit import independent_score


def test_independent_scores_match_full_and_additive_models_at_both_penalties():
    root = Path(__file__).parents[1] / "docs"
    base = json.loads((root / "run_pair_coordination_endpoint_v3_protocol.json").read_text())
    t = np.arange(.0005, 20, .001)
    cov = {"position": np.column_stack((5 + np.sin(t), 5 + np.cos(t))),
        "direction": np.mod(t, 2 * np.pi), "speed": np.full(len(t), 15.),
        "theta": np.column_stack((np.mod(t * 16 * np.pi, 2 * np.pi) - np.pi,
                                  np.mod(t * 16 * np.pi + .4, 2 * np.pi) - np.pi))}
    y = np.random.default_rng(17).poisson(.02 * np.exp(np.cos(cov["theta"][:, :1])), size=(len(t), 2))
    for penalty in (1., 100.):
        for interaction in (False, True):
            p = {**base, "glm_workers": 1, "glm_l2_penalty": penalty,
                "glm_spatial_direction_interaction": interaction, "glm_spatial_theta_interaction": interaction}
            _, qc, prediction, _ = glm.crossfit(y, t, np.zeros(len(t), int), .001, p, cov, return_predictions=True)
            scores, independent = independent_score(y, t, cov, p)
            np.testing.assert_allclose(scores.sum(), qc["heldout_poisson_improvement_over_global"], rtol=1e-3, atol=2e-5)
            np.testing.assert_allclose(prediction.mean(axis=0), independent.mean(axis=0), rtol=1e-3, atol=1e-8)


def test_strong_penalty_protocol_changes_only_penalty_and_documentation():
    root = Path(__file__).parents[1] / "docs"
    original = json.loads((root / "run_pair_coordination_endpoint_v3_protocol.json").read_text())
    stronger = json.loads((root / "run_pair_coordination_endpoint_v3_strong_penalty_development_protocol.json").read_text())
    changed = {"protocol_id", "frozen_before", "scope", "source_and_support", "claim_boundary", "glm_l2_penalty"}
    for key in set(original) & set(stronger) - changed:
        assert original[key] == stronger[key]
    assert stronger["glm_l2_penalty"] == 100
