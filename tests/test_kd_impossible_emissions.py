from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm import kd_reference
from hipporeplayimm.kd_reference import (
    kd_diffusion_log_evidence_from_transition,
    kd_momentum_log_evidence_from_transitions,
)


def test_scaled_emission_all_negative_infinity_row_becomes_zero_mass():
    apply_runtime_patches()

    scaled, offset = kd_reference._scaled_emission(np.array([[-np.inf, -np.inf]]), 0)

    assert np.isneginf(offset)
    assert np.array_equal(scaled, np.zeros(2))


def test_scaled_emission_rejects_nan_or_positive_infinity_rows():
    apply_runtime_patches()

    invalid_log_emissions = (
        np.array([[0.0, np.nan]], dtype=float),
        np.array([[0.0, np.inf]], dtype=float),
    )

    for log_emissions in invalid_log_emissions:
        with pytest.raises(ValueError, match=r"NaN or \+inf"):
            kd_reference._scaled_emission(log_emissions, 0)


def test_stationary_gaussian_kd_preserves_disjoint_latent_support():
    apply_runtime_patches()

    transition = np.eye(2)
    log_emissions = np.array(
        [
            [0.0, -np.inf, -np.inf, -np.inf],
            [-np.inf, 0.0, -np.inf, -np.inf],
        ],
        dtype=float,
    )

    score = kd_reference.kd_stationary_gaussian_log_evidence_from_transitions(
        log_emissions,
        2,
        2,
        transition,
    )

    assert np.isneginf(score)
    assert not np.isnan(score)


def test_stationary_gaussian_kd_keeps_feasible_exact_support_score():
    apply_runtime_patches()

    transition = np.eye(2)
    log_emissions = np.array(
        [
            [0.0, -np.inf, -np.inf, -np.inf],
            [0.0, -np.inf, -np.inf, -np.inf],
        ],
        dtype=float,
    )

    score = kd_reference.kd_stationary_gaussian_log_evidence_from_transitions(
        log_emissions,
        2,
        2,
        transition,
    )

    assert score == pytest.approx(-np.log(4.0))


def test_first_order_kd_returns_negative_infinity_for_impossible_emission_rows():
    transition = np.eye(2)
    impossible = [-np.inf, -np.inf, -np.inf, -np.inf]
    feasible = [0.0, -np.inf, -np.inf, -np.inf]

    for log_emissions in (
        np.array([impossible, feasible], dtype=float),
        np.array([feasible, impossible], dtype=float),
    ):
        score = kd_diffusion_log_evidence_from_transition(log_emissions, 2, 2, transition)

        assert np.isneginf(score)
        assert not np.isnan(score)


def test_second_order_kd_scores_single_bin_emissions():
    apply_runtime_patches()

    initial = np.eye(2)
    transition = np.zeros((2, 2, 2), dtype=float)
    log_emissions = np.array([[0.0, -np.inf, -np.inf, -np.inf]], dtype=float)

    score = kd_momentum_log_evidence_from_transitions(log_emissions, 2, initial, transition)

    assert score == pytest.approx(-np.log(4.0))


def test_second_order_kd_returns_negative_infinity_for_impossible_emission_rows():
    initial = np.eye(2)
    transition = np.zeros((2, 2, 2), dtype=float)
    for prev_prev in range(2):
        for prev in range(2):
            transition[prev, prev, prev_prev] = 1.0

    impossible = [-np.inf, -np.inf, -np.inf, -np.inf]
    feasible = [0.0, -np.inf, -np.inf, -np.inf]

    for log_emissions in (
        np.array([impossible, feasible], dtype=float),
        np.array([feasible, impossible], dtype=float),
    ):
        score = kd_momentum_log_evidence_from_transitions(log_emissions, 2, initial, transition)

        assert np.isneginf(score)
        assert not np.isnan(score)
