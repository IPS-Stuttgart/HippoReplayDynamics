"""Known-path tests for a fixed-endpoint context, not biological recovery tests."""

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.special import softmax

from hipporeplayimm.state_space_first_order import _forward_backward_first_order
from hipporeplayimm.state_space_utils import _gaussian_transition_matrix
from scripts.measure_temporal_endpoint_content import context_blocks, endpoint_bank, reset_transition


def test_nonoverlap_right_alignment_and_context_cap():
    base = np.arange(47 * 3).reshape(47, 3)
    blocks = context_blocks(base)
    assert blocks.shape == (10, 3)
    np.testing.assert_array_equal(blocks.sum(axis=0), base[-40:].sum(axis=0))
    np.testing.assert_array_equal(blocks[-1], base[-4:].sum(axis=0))
    np.testing.assert_array_equal(context_blocks(base[-7:]), base[-4:].sum(axis=0)[None])


@pytest.mark.parametrize("base", [np.zeros((3, 2)), [[1, -1]] * 4, [[1, 0.5]] * 4, [[1, np.nan]] * 4])
def test_bad_counts_fail(base):
    with pytest.raises(ValueError):
        context_blocks(base)


@pytest.mark.parametrize("reset", [0.0, 0.25, 1.0])
def test_exact_recursion_matches_all_latent_paths(reset):
    grid = np.array([[0.0], [8.0], [16.0]])
    gaussian = _gaussian_transition_matrix(grid, 10, 4)
    operator = reset_transition(gaussian, reset)
    matrix = (1 - reset) * gaussian.toarray() + reset / len(grid)
    np.testing.assert_allclose(matrix.sum(axis=0), 1)
    probe = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(operator @ probe, matrix @ probe)
    np.testing.assert_allclose(operator.T @ probe, matrix.T @ probe)
    ll = np.log([[0.9, 0.3, 0.2], [0.4, 0.8, 0.3], [0.2, 0.1, 0.9]])
    total = np.zeros(3)
    for path in itertools.product(range(3), repeat=3):
        probability = np.exp(sum(ll[t, x] for t, x in enumerate(path))) / 3
        for t in range(1, 3):
            probability *= matrix[path[t], path[t - 1]]
        total[path[-1]] += probability
    evidence, posterior = _forward_backward_first_order(ll, operator)
    np.testing.assert_allclose(np.exp(evidence), total.sum())
    np.testing.assert_allclose(np.exp(posterior[-1]), total / total.sum())
    if reset == 1:
        np.testing.assert_allclose(np.exp(posterior[-1]), softmax(ll[-1]))


def test_last_bin_is_not_shifted_and_reset_allows_late_jump():
    rates = np.array([[100.0, 0.001], [0.001, 100.0]])
    gaussian = csr_matrix(np.eye(2))
    blocks = np.array([[8, 0]] * 8 + [[0, 8]])
    bank, ll = endpoint_bank(blocks, rates, gaussian)
    assert bank["diffusion_reset"][1] > 0.99
    assert bank["pooled_static"][0] > 0.99
    assert bank["diffusion_no_reset"][0] > 0.99
    np.testing.assert_allclose(bank["independent"], softmax(ll))


def test_one_bin_methods_equal_and_other_population_not_used():
    rates = np.array([[10.0, 0.1], [0.1, 10.0]])
    gaussian = csr_matrix([[0.9, 0.1], [0.1, 0.9]])
    bank, _ = endpoint_bank(np.array([[1, 0]]), rates, gaussian)
    for value in bank.values():
        np.testing.assert_allclose(value, bank["independent"])
    whole = np.array([[2, 0, 20], [0, 2, 1]])
    first, _ = endpoint_bank(whole[:, :2], rates, gaussian)
    whole[:, 2] = 1000
    second, _ = endpoint_bank(whole[:, :2], rates, gaussian)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_reset_validation():
    with pytest.raises(ValueError):
        reset_transition(csr_matrix(np.eye(2)), np.nan)
