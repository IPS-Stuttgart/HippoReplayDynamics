"""Independent basis/likelihood arithmetic, not scientific null calibration."""
import json
from pathlib import Path

import numpy as np

from scripts import _run_pair_rate_glm as producer
from scripts import verify_run_pair_rate_glm_stress as verifier


def protocol():
    p = json.loads((Path(__file__).parents[1] / "docs/run_pair_coordination_endpoint_v2_protocol.json").read_text())
    return {**p, "glm_workers": 1}


def covariates(times):
    return {"position": np.column_stack((np.sin(times) + 5, np.cos(times) + 5)),
            "direction": np.mod(times, 2 * np.pi), "speed": np.full(len(times), 15.),
            "theta": np.column_stack((np.mod(2 * np.pi * 8 * times, 2 * np.pi) - np.pi,
                                      np.mod(2 * np.pi * 8 * times + .4, 2 * np.pi) - np.pi))}


def test_independent_scipy_basis_matches_position_phase_direction_and_speed_features():
    times = np.arange(.0005, 20, .001)
    train, target = covariates(times[:10000]), covariates(times[10000:])
    target["position"][::100] += 100
    train["speed"] += np.sin(times[:10000])
    target["speed"] += np.sin(times[10000:])
    a, b, _ = producer.design(train, target, protocol())
    x, z = verifier.features(train, target, protocol())
    np.testing.assert_allclose(x.toarray(), a.toarray(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z.toarray(), b.toarray(), rtol=1e-12, atol=1e-12)


def test_independent_poisson_objective_matches_crossfit_without_producer_rate_code():
    times = np.arange(.0005, 20, .001)
    c, p = covariates(times), protocol()
    counts = np.random.default_rng(19).poisson(.03 * np.exp(np.cos(c["theta"][:, :1])))
    original, _ = producer.crossfit(counts, times, np.zeros(len(times), int), .001, p, c)
    independent = verifier.independent_residuals(counts, times, c, p)
    np.testing.assert_allclose(independent, original, rtol=2e-3, atol=2e-4)
