from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
import hipporeplayimm.kd_reference as kd
from hipporeplayimm.kd_impossible_emission_patch import _scaled_emission as patched_scaled_emission
from hipporeplayimm.kd_single_bin_momentum_patch2 import apply_kd_single_bin_momentum_patch2


def test_kd_runtime_patches_restore_replaced_aliases():
    def stale_scaled_emission(log_emissions, time_index):
        raise RuntimeError("stale scaled-emission alias")

    def stale_second_order(log_emissions, n_bins, initial, transition):
        raise RuntimeError("stale second-order alias")

    kd._scaled_emission = stale_scaled_emission
    kd._second_order_separable_log_evidence = stale_second_order

    hipporeplayimm.apply_runtime_patches()

    assert kd._scaled_emission is patched_scaled_emission
    assert getattr(kd._second_order_separable_log_evidence, "_kd_single_bin_momentum_wrapper", False)

    log_emissions = np.array([[0.0, -1.0, -2.0, -3.0]], dtype=float)
    initial = np.eye(2, dtype=float)
    transition = np.ones((2, 2, 2), dtype=float) / 2.0

    actual = kd._second_order_separable_log_evidence(log_emissions, 2, initial, transition)

    assert actual == pytest.approx(kd.kd_random_log_evidence(log_emissions))


def test_single_bin_patch_refreshes_dependency_when_wrapper_already_installed():
    apply_kd_single_bin_momentum_patch2()
    wrapped_second_order = kd._second_order_separable_log_evidence

    def stale_scaled_emission(log_emissions, time_index):
        raise RuntimeError("stale scaled-emission alias")

    kd._scaled_emission = stale_scaled_emission

    apply_kd_single_bin_momentum_patch2()

    assert kd._scaled_emission is patched_scaled_emission
    assert kd._second_order_separable_log_evidence is wrapped_second_order

    log_emissions = np.array([[0.0, -np.inf, -np.inf, -np.inf]], dtype=float)

    actual = kd._second_order_separable_log_evidence(
        log_emissions,
        2,
        np.eye(2, dtype=float),
        np.ones((2, 2, 2), dtype=float) / 2.0,
    )

    assert actual == pytest.approx(-np.log(4.0))
