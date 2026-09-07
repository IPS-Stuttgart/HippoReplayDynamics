from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, logsumexp

from hipporeplayimm.encoding import EmissionConfig, LogEmissionTensor
from hipporeplayimm.frozen_posterior_prediction import frozen_smoothed_marginal_log_score
from hipporeplayimm.state_space import StateSpaceDecoderConfig
from scripts import audit_pf_count_conditioned_prediction as audit


def test_exact_poisson_factorization_unequal_durations():
    k = np.array([[2, 0, 1], [0, 0, 0], [1, 4, 0]])
    r = np.array([[1.0, 8.0], [5.0, 1.0], [0.2, 6.0]])
    dt = np.array([0.02, 0.02, 0.003])
    parts = audit.likelihood_parts(k, r, dt, 2)
    mu = 2 * dt[:, None, None] * r[None, :, :]
    direct = (k[:, :, None] * np.log(mu) - mu - gammaln(k[:, :, None] + 1)).sum(axis=1)
    np.testing.assert_allclose(parts["full_poisson"], direct, atol=1e-12)
    np.testing.assert_allclose(parts["count_conditioned"][1], 0, atol=1e-12)


def test_multinomial_is_normalized_not_powered():
    k = np.array([v for v in product(range(5), repeat=3) if sum(v) == 4])
    rates = np.array([[1.0, 2.0], [4.0, 5.0], [3.0, 0.4]])
    ll = audit.likelihood_parts(k, rates, np.full(len(k), 0.02))["count_conditioned"]
    np.testing.assert_allclose(logsumexp(ll, axis=0), 0, atol=1e-12)
    assert np.any(abs(logsumexp(ll / 0.3, axis=0)) > 0.1)


def test_position_dependent_population_gain_cancels():
    rng = np.random.default_rng(4)
    k = rng.poisson(2, (10, 5))
    r = rng.uniform(0.1, 10, (5, 7))
    dt = np.full(10, 0.004)
    original = audit.likelihood_parts(k, r, dt)
    scaled = audit.likelihood_parts(k, r * np.exp(rng.normal(size=7)), dt, 4.0)
    np.testing.assert_allclose(original["count_conditioned"], scaled["count_conditioned"], atol=1e-12)
    assert not np.allclose(original["total_rate_only"], scaled["total_rate_only"])


@pytest.mark.parametrize("model", audit.MODELS[:2])
def test_analytic_models_are_map_permutation_invariant(model):
    rng = np.random.default_rng(18)
    train, held = rng.normal(size=(2, 10, 20))
    permutation = rng.permutation(20)
    posterior = audit.analytic_posterior(train, model)
    wrong = audit.analytic_posterior(train[:, permutation], model)
    a = frozen_smoothed_marginal_log_score(posterior, held).total_log_score
    b = frozen_smoothed_marginal_log_score(wrong, held[:, permutation]).total_log_score
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize(
    "rates,counts,durations",
    [
        ([[0.0, 1.0]], [[1]], [0.02]),
        ([[1.0, 1.0]], [[-1]], [0.02]),
        ([[1.0, 1.0]], [[0.2]], [0.02]),
        ([[1.0, 1.0]], [[1]], [0.0]),
        ([[1.0, 1.0]], [[1]], [np.nan]),
        ([[1.0, np.inf]], [[1]], [0.02]),
    ],
)
def test_invalid_likelihood_inputs_fail(rates, counts, durations):
    with pytest.raises(ValueError):
        audit.likelihood_parts(counts, rates, durations)


def test_no_vacuous_pass():
    gate = audit.gates(pd.DataFrame(), pd.DataFrame(columns=["session", "event_index"]), 5, [1.0])
    assert not gate.passed.any()


def test_inference_never_sees_heldout_counts(monkeypatch):
    """Perturb held-out activity: every inferred posterior remains identical."""

    class FakeEncoding:
        cell_ids = np.array([1, 2, 3, 4])
        rates_hz = np.array([[2.0, 7.0], [5.0, 1.0], [3.0, 8.0], [1.0, 9.0]])
        bin_centers = np.array([[0.0, 0.0], [8.0, 0.0]])
        n_bins = 2

        def select_cells(self, ids):
            return SimpleNamespace(rates_hz=self.rates_hz[np.asarray(ids) - 1], cell_ids=np.asarray(ids))

    phase = {"held_built": False, "perturb": False}

    def build(session, encoding, event, config):
        is_held = 4 in encoding.cell_ids
        if is_held:
            phase["held_built"] = True
        else:
            phase["held_built"] = False
        counts = np.ones((3, len(encoding.cell_ids)), dtype=int)
        if phase["perturb"] and is_held:
            counts[:, 0] = [9, 2, 7]
        return LogEmissionTensor(np.zeros((3, 2)), counts, np.array([0.002, 0.006, 0.010]), 0.004, encoding.cell_ids, int(counts.sum()))

    class FakeModel:
        def __init__(self, **kwargs):
            self.mode = kwargs["mode"]

        def score(self, emissions, centers):
            assert not phase["held_built"]
            assert 4 not in emissions.cell_ids
            model = "static_location" if self.mode == "stationary" else "iid_position"
            return SimpleNamespace(trajectory_log_posterior=audit.analytic_posterior(emissions.log_likelihood, model))

    monkeypatch.setattr(audit, "load_replay_session", lambda _: SimpleNamespace(rat="Rat1"))
    monkeypatch.setattr(audit, "fit_place_field_encoding", lambda *_: FakeEncoding())
    monkeypatch.setattr(audit, "population_code_permuted_encoding", lambda e, **_: (e, "hash"))
    monkeypatch.setattr(audit, "_split_cells", lambda *_: (np.array([1, 2]), np.array([3, 4])))
    monkeypatch.setattr(audit, "build_emissions", build)
    monkeypatch.setattr(audit, "SortedSpikeStateSpaceReplayModel", FakeModel)
    task = {
        "configs": (None, EmissionConfig(), StateSpaceDecoderConfig()),
        "dataset_root": "/unused",
        "session": "Rat1/session",
        "events": [4],
        "n_splits": 1,
        "temperatures": [1.0, 0.3],
    }
    first = pd.DataFrame(audit._task(task))
    phase["perturb"] = True
    second = pd.DataFrame(audit._task(task))
    assert first.posterior_sha256.tolist() == second.posterior_sha256.tolist()
    assert not np.allclose(first.conditional_heldout_log_score, second.conditional_heldout_log_score)
    gate = audit.gates(first, pd.DataFrame({"session": ["Rat1/session"], "event_index": [4]}), 1, [1.0, 0.3])
    assert gate.passed.all()
    corrupted = first.copy()
    corrupted.loc[0, "model"] = "unknown"
    assert not audit.gates(corrupted, pd.DataFrame({"session": ["Rat1/session"], "event_index": [4]}), 1, [1.0, 0.3]).passed.all()


def test_event_medians_do_not_weight_more_split_rows():
    rows = [
        {
            "session": f"{rat}/s",
            "rat": rat,
            "event_index": e,
            "split": split,
            "inference_temperature": 1.0,
            "map": map_name,
            "observation": obs,
            "model": model,
            "conditional_heldout_log_score": (-10 if model == "first_order_imm" else -12),
        }
        for rat in ["A", "B", "C", "D"]
        for e in [1, 2]
        for split in range(3)
        for map_name in audit.MAPS
        for obs in audit.OBSERVATIONS
        for model in audit.MODELS
    ]
    _, event = audit.contrasts(pd.DataFrame(rows))
    summary, _ = audit.summarize(event, replicates=50)
    row = summary[summary.contrast == "count_conditioned:imm_minus_diffusion"].iloc[0]
    assert row.events == 8
    assert row.equal_animal_mean_event_median_delta == 2
    assert row.rat_signflip_one_sided_p == 0.0625
