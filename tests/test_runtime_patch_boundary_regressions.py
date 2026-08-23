import sys
import types

import numpy as np
import pytest

from hipporeplayimm import state_space_utils
from hipporeplayimm.ground_truth_float_metadata import (
    _patch_direct_ground_truth_numeric_helpers,
    _synchronize_coerce_bool_series_aliases,
)
from hipporeplayimm.model_parameter_validation import (
    _apply_state_space_mode_transition_validation_patch,
)


def _package_alias_snapshot(attribute):
    snapshot = []
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name != "hipporeplayimm" and not module_name.startswith("hipporeplayimm."):
            continue
        if hasattr(module, attribute):
            snapshot.append((module, getattr(module, attribute)))
    return snapshot


def _restore_package_aliases(attribute, snapshot):
    for module, value in snapshot:
        setattr(module, attribute, value)


def test_ground_truth_bool_alias_sync_respects_package_namespace(monkeypatch):
    stale = object()
    active = object()

    external = types.ModuleType("hipporeplayimm_extension")
    external._coerce_bool_series = stale
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = types.ModuleType("hipporeplayimm._runtime_patch_test_ground_truth")
    internal._coerce_bool_series = stale
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    snapshot = _package_alias_snapshot("_coerce_bool_series")
    try:
        _synchronize_coerce_bool_series_aliases(active)

        assert external._coerce_bool_series is stale
        assert internal._coerce_bool_series is active
    finally:
        _restore_package_aliases("_coerce_bool_series", snapshot)


def test_model_parameter_alias_sync_respects_package_namespace(monkeypatch):
    def stale_mode_transition(n_modes, stickiness):
        return n_modes, stickiness

    external = types.ModuleType("hipporeplayimm_extension")
    external._mode_transition_matrix = stale_mode_transition
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = types.ModuleType("hipporeplayimm._runtime_patch_test_model_parameter")
    internal._mode_transition_matrix = stale_mode_transition
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    snapshot = _package_alias_snapshot("_mode_transition_matrix")
    try:
        monkeypatch.setattr(
            state_space_utils,
            "_mode_transition_matrix",
            stale_mode_transition,
        )

        _apply_state_space_mode_transition_validation_patch()
        active_mode_transition = state_space_utils._mode_transition_matrix

        assert active_mode_transition is not stale_mode_transition
        assert external._mode_transition_matrix is stale_mode_transition
        assert internal._mode_transition_matrix is active_mode_transition
    finally:
        _restore_package_aliases("_mode_transition_matrix", snapshot)


def test_active_goal_direct_helper_validates_well_ids():
    def active_goal_at_time(session, time_s):
        return 1

    def first_post_ripple_well_visit(
        position,
        wells,
        ripple_peak,
        *,
        visit_radius_cm,
        min_dwell_s,
        future_horizon_s,
    ):
        return None

    def infer_well_locations_from_arrays(
        position,
        well_sequence,
        well_arrival_window_s=1.0,
    ):
        return None

    ground_truth = types.SimpleNamespace(
        active_goal_at_time=active_goal_at_time,
        first_post_ripple_well_visit=first_post_ripple_well_visit,
        infer_well_locations_from_arrays=infer_well_locations_from_arrays,
    )
    _patch_direct_ground_truth_numeric_helpers(ground_truth)

    session = types.SimpleNamespace(
        well_sequence=np.array([[0.0, 1.5]], dtype=float),
    )

    with pytest.raises(ValueError, match="well IDs"):
        ground_truth.active_goal_at_time(session, 1.0)
