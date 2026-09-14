import numpy as np
from scipy.stats import multinomial, poisson
from scipy.special import softmax

from hipporeplayimm.replay_coverage import decode_independent
from scripts.measure_count_conditioned_content import poisson_sources, readout
from scripts.measure_encoding_uncertainty_content import entropy
from scripts.audit_count_conditioned_content import conditional_posterior, raw_endpoints


def fixture():
    rng = np.random.default_rng(330)
    n = rng.poisson(.4, (15, 10))
    n[0] = 0
    rates = rng.uniform(.001, 30, (10, 25))
    grid = np.array([[x, y] for x in np.arange(5)*8. for y in np.arange(5)*8.])
    return n, rates, grid


def test_conditional_decoder_matches_independent_scipy_multinomial():
    n, rates, grid = fixture()
    p = rates/rates.sum(axis=0)
    logits = np.array([[multinomial.logpmf(counts, counts.sum(), p[:, i]) for i in range(len(grid))] for counts in n])
    actual = decode_independent(n, rates, grid, .02, likelihood='conditional_multinomial')['posterior']
    np.testing.assert_allclose(actual, softmax(logits, axis=1), atol=1e-11)
    np.testing.assert_allclose(conditional_posterior(n, rates), actual, atol=1e-11)


def test_poisson_factorizes_into_conditional_identities_and_total_count():
    n, rates, grid = fixture()
    p = rates/rates.sum(axis=0)
    conditional = np.array([[multinomial.logpmf(counts, counts.sum(), p[:, i]) for i in range(len(grid))] for counts in n])
    full = poisson.logpmf(n[:, :, None], .02*rates[None]).sum(axis=1)
    total = poisson.logpmf(n.sum(axis=1)[:, None], .02*rates.sum(axis=0)[None])
    np.testing.assert_allclose(full, conditional+total, atol=1e-10)


def test_conditioning_removes_common_not_cell_specific_gain():
    n, rates, grid = fixture()
    def decode(r):
        return decode_independent(n, r, grid, .02, likelihood='conditional_multinomial')['posterior']
    baseline = decode(rates)
    np.testing.assert_allclose(decode(rates*20), baseline, atol=1e-12)
    np.testing.assert_allclose(decode(rates*np.linspace(.3, 5, 25)[None]), baseline, atol=1e-12)
    assert not np.allclose(decode(rates*np.arange(1, 11)[:, None]), baseline)


def test_zero_counts_remain_uniform_not_localized_and_entropy_is_controlled():
    n, rates, grid = fixture()
    rows, bank = readout(n, np.full((len(n), 2), np.nan), rates, grid, [np.arange(5), np.arange(5, 10)])
    for p in bank:
        np.testing.assert_allclose(p['conditional_multinomial'][0], 1/25)
        np.testing.assert_allclose(entropy(p['conditional_multinomial']), entropy(p['poisson_entropy_matched']), atol=1e-7)
    assert all(r.a_entropy_control_available.all() and r.b_entropy_control_available.all() for r in rows)


def test_b_spikes_cannot_inform_a_posterior_or_total_count_diagnostic():
    n, rates, grid = fixture()
    truth = np.full((len(n), 2), np.nan)
    groups = [np.arange(5), np.arange(5, 10)]
    a, pa = readout(n, truth, rates, grid, groups)
    n[:, 5:] += 8
    b, pb = readout(n, truth, rates, grid, groups)
    for k in pa[0]: np.testing.assert_array_equal(pa[0][k], pb[0][k])
    for x, y in zip(a, b, strict=True):
        for key in ('a_total_count_reliance_tv', 'a_matched_log_temperature', 'a_entropy'):
            np.testing.assert_array_equal(x[key], y[key])


def test_unconditional_generators_preserve_truth_not_imposed_real_counts():
    _, rates, grid = fixture()
    source = dict(counts=np.zeros((12, 10), int), offsets=np.array([0, 4, 8, 12]),
        truth_base_cm=np.repeat(grid[[0, 12, 24]], 4, axis=0), starts_s=np.arange(3),
        event_ids=np.array([3, 40, 2]), grid_cm=grid, rates_hz=rates)
    first = poisson_sources(source, 'test', 's1')
    second = poisson_sources(source, 'test', 's1')
    base, truth, start = raw_endpoints(source)
    np.testing.assert_array_equal(base, 0)
    np.testing.assert_array_equal(truth, grid[[0, 12, 24]])
    np.testing.assert_array_equal(start, source['starts_s'])
    for k, a in first.items():
        for field in a: np.testing.assert_array_equal(a[field], second[k][field])
        np.testing.assert_allclose(a['expected_counts'], .02*float(a['gain'])*rates[:, [0, 12, 24]].T)
        np.testing.assert_array_equal(a['truth_cm'], grid[[0, 12, 24]])
        assert a['counts'].sum() > 0
    assert first['sim_poisson_stationary_gain20']['counts'].sum() > first['sim_poisson_stationary']['counts'].sum()
