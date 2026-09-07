import numpy as np
import pytest
from numpy.testing import assert_allclose

from scripts import audit_hc11_count_conditioned_prediction as scorer
from scripts import verify_hc11_count_conditioned_prediction as verifier


@pytest.mark.parametrize("topology", ["linear", "circular"])
@pytest.mark.parametrize("model", scorer.MODELS)
def test_independent_prediction_matches(topology, model):
    rng = np.random.default_rng(872)
    edges = np.array([0.0, 0.02, 0.04, 0.06, 0.079])
    centers = np.arange(5) * 4.0 + 2.0
    kernels = scorer.transitions(centers, edges, topology, 20.0)
    dense = verifier.dense_kernels(centers, edges, topology, 20.0)
    tr, he = [], []
    for _ in range(2):
        rates = rng.uniform(0.05, 20, (4, 5))
        train = rng.poisson(2, (4, 4))
        held = rng.poisson(3, (4, 4))
        t = verifier.direct_parts(train, rates, np.diff(edges))
        h = verifier.direct_parts(held, rates, np.diff(edges))
        for obs, expected in scorer.pf.likelihood_parts(train, rates, np.diff(edges)).items():
            assert_allclose(t[obs], expected, atol=1e-12)
        tr.append(t)
        he.append(h)
    for temp in scorer.TEMPERATURES:
        for obs in scorer.OBSERVATIONS:
            q = scorer.infer_training(tr, obs, temp, model, kernels)
            independent = verifier.direct_predict(tr, he, model, obs, temp, dense)
            for target in ("count_conditioned", "full_poisson"):
                assert_allclose(independent[target], scorer.heldout_score(q, he, target), atol=1e-10, rtol=0)


def test_unknown_model_fails():
    parts = [verifier.direct_parts(np.ones((2, 2)), np.ones((2, 3)), np.ones(2))]
    with pytest.raises(ValueError):
        verifier.direct_predict(parts, parts, "missing", "count_conditioned", 1.0, {})
