import sys
import types

import hipporeplayimm
from hipporeplayimm import encoding, kd_reference, state_space_utils
from hipporeplayimm.sparse_momentum_bin_center_validation import (
    _synchronize_imported_aliases as _synchronize_sparse_center_aliases,
)
from hipporeplayimm.sparse_momentum_valid_bin_mask_validation import (
    _synchronize_imported_aliases as _synchronize_valid_bin_mask_aliases,
)
from hipporeplayimm.state_space_sigma_validation import (
    _patch_state_space_utils_mode_transition,
    _patch_state_space_utils_sigma,
)


def test_duration_builder_sync_respects_package_namespace(monkeypatch):
    external = types.ModuleType("hipporeplayimm_extension")
    external_build_emissions = object()
    external_build_kd_emissions = object()
    external.build_emissions = external_build_emissions
    external.build_kd_emissions = external_build_kd_emissions
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = types.ModuleType("hipporeplayimm._runtime_patch_test")
    internal.build_emissions = object()
    internal.build_kd_emissions = object()
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    hipporeplayimm._synchronize_duration_patched_emission_builders()

    assert external.build_emissions is external_build_emissions
    assert external.build_kd_emissions is external_build_kd_emissions
    assert internal.build_emissions is encoding.build_emissions
    assert internal.build_kd_emissions is kd_reference.build_kd_emissions


def test_sparse_alias_sync_respects_package_namespace(monkeypatch):
    external = types.ModuleType("hipporeplayimm_extension")
    internal = types.ModuleType("hipporeplayimm._runtime_patch_test")
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    stale_mask = object()
    active_mask = state_space_utils._coerce_valid_bin_mask
    external._coerce_valid_bin_mask = stale_mask
    internal._coerce_valid_bin_mask = stale_mask

    stale_centers = object()
    active_centers = object()
    external._as_2d_centers = stale_centers
    internal._as_2d_centers = stale_centers

    _synchronize_valid_bin_mask_aliases(active_mask)
    _synchronize_sparse_center_aliases(stale_centers, active_centers)

    assert external._coerce_valid_bin_mask is stale_mask
    assert external._as_2d_centers is stale_centers
    assert internal._coerce_valid_bin_mask is active_mask
    assert internal._as_2d_centers is active_centers


def test_state_space_scalar_alias_sync_respects_package_namespace(monkeypatch):
    def stale_sigma(sigma_cm_sqrt_s, dt_s):
        return float(sigma_cm_sqrt_s) * float(dt_s)

    def stale_mode_transition(n_modes, stickiness):
        return n_modes, stickiness

    external = types.ModuleType("hipporeplayimm_extension")
    external._per_bin_sigma = stale_sigma
    external._mode_transition_matrix = stale_mode_transition
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = types.ModuleType("hipporeplayimm._runtime_patch_test")
    internal._per_bin_sigma = stale_sigma
    internal._mode_transition_matrix = stale_mode_transition
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    monkeypatch.setattr(state_space_utils, "_per_bin_sigma", stale_sigma)
    _patch_state_space_utils_sigma()
    active_sigma = state_space_utils._per_bin_sigma

    assert active_sigma is not stale_sigma
    assert external._per_bin_sigma is stale_sigma
    assert internal._per_bin_sigma is active_sigma

    monkeypatch.setattr(
        state_space_utils,
        "_mode_transition_matrix",
        stale_mode_transition,
    )
    _patch_state_space_utils_mode_transition()
    active_mode_transition = state_space_utils._mode_transition_matrix

    assert active_mode_transition is not stale_mode_transition
    assert external._mode_transition_matrix is stale_mode_transition
    assert internal._mode_transition_matrix is active_mode_transition
