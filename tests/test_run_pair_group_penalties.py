"""Group-penalty algebra and independent fitting, not biological calibration."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import _run_pair_rate_glm as glm
from scripts.verify_run_pair_rate_glm_stress import features, independent_penalties, independent_residuals


def inputs():
    p = json.loads((Path(__file__).parents[1] / "docs/run_pair_coordination_endpoint_v4_group_penalty_protocol.json").read_text())
    p["glm_workers"] = 1
    times = np.arange(.0005, 20, .001)
    c = {"position": np.column_stack((5 + np.sin(times), 5 + np.cos(times))),
         "direction": np.mod(times, 2 * np.pi), "speed": np.full(len(times), 15.),
         "theta": np.column_stack((np.mod(times * 16 * np.pi, 2 * np.pi) - np.pi,
                                   np.mod(times * 16 * np.pi + .4, 2 * np.pi) - np.pi))}
    return times, c, p


def test_scaled_design_equals_unscaled_explicit_penalty_objective():
    t, c, p = inputs()
    train, target = [{k: v[mask] for k, v in c.items()} for mask in (t < 10, t >= 10)]
    x, z, _ = glm.design(train, target, p)
    a, b = features(train, target, p)
    penalties = independent_penalties(train, p, a.shape[1])
    scale = np.sqrt(p["glm_main_effect_l2_penalty"] / penalties)
    np.testing.assert_allclose(x.toarray(), a.toarray() * scale, rtol=0, atol=1e-14)
    np.testing.assert_allclose(z.toarray(), b.toarray() * scale, rtol=0, atol=1e-14)
    beta = np.random.default_rng(9).normal(size=a.shape[1])
    transformed = beta / scale
    np.testing.assert_allclose(x @ transformed, a @ beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p["glm_main_effect_l2_penalty"] * np.sum(transformed ** 2), penalties @ (beta ** 2))


def test_equal_group_penalties_preserve_default_design_and_predictions():
    t, c, p = inputs()
    original = {k: v for k, v in p.items() if k != "glm_main_effect_l2_penalty"}
    equal = {**original, "glm_main_effect_l2_penalty": original["glm_l2_penalty"]}
    y = np.random.default_rng(2).poisson(.03, size=(len(t), 1))
    first = glm.crossfit(y, t, np.zeros(len(t), int), .001, original, c, return_predictions=True)
    second = glm.crossfit(y, t, np.zeros(len(t), int), .001, equal, c, return_predictions=True)
    np.testing.assert_array_equal(first[2], second[2])
    assert first[1] == second[1]


def test_group_fit_matches_independent_unscaled_penalty_vector():
    t, c, p = inputs()
    y = np.random.default_rng(13).poisson(.03 * np.exp(np.cos(c["theta"][:, :1])), size=(len(t), 2))
    residual, qc, prediction, _ = glm.crossfit(y, t, np.zeros(len(t), int), .001, p, c, return_predictions=True)
    other, mean = independent_residuals(y, t, c, p, return_prediction=True)
    np.testing.assert_allclose(prediction, mean, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(residual, other, rtol=1e-3, atol=1e-5)
    assert qc["predicted_bins"] == qc["total_bins"] == len(t)


@pytest.mark.parametrize("penalty", [0., -1., np.nan, np.inf, 101.])
def test_invalid_main_penalty_is_rejected_before_fitting(penalty):
    t, c, p = inputs()
    with pytest.raises(ValueError, match="penalty"):
        glm.prepare(t, c, {**p, "glm_main_effect_l2_penalty": penalty})


def test_group_protocol_changes_no_clock_support_or_cohort_settings():
    _, _, p = inputs()
    p["glm_workers"] = 4
    old = json.loads((Path(__file__).parents[1] / "docs/run_pair_coordination_endpoint_v3_protocol.json").read_text())
    changed = {"protocol_id", "frozen_before", "scope", "rate_model", "source_and_support", "claim_boundary", "glm_l2_penalty"}
    assert set(p) - set(old) == {"glm_main_effect_l2_penalty"} and not set(old) - set(p)
    assert all(p[k] == old[k] for k in set(old) - changed)
