import sys
from types import ModuleType

import numpy as np
import pytest

import hipporeplayimm.candidate_active_support_validation as candidate_validation
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_candidate_source_rejects_rows_without_active_support() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
                [0.0, -np.inf],
            ],
            dtype=float,
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 0.02, 0.04], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    model = StateSpaceReplayModel(
        mode="momentum",
        config=StateSpaceDecoderConfig(
            mode="momentum",
            momentum_candidate_source="emission",
        ),
    )

    with pytest.raises(ValueError, match="active support"):
        model.candidate_indices(
            emissions,
            centers,
            valid_bin_mask=np.array([False, True], dtype=bool),
        )


def test_transition_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    def original(*args, **kwargs):
        return None

    def replacement(*args, **kwargs):
        return None

    package_module = ModuleType("hipporeplayimm._candidate_active_support_alias_test")
    package_module._state_transition_matrix = original
    sibling_module = ModuleType("hipporeplayimm_extension")
    sibling_module._state_transition_matrix = original

    monkeypatch.setitem(sys.modules, package_module.__name__, package_module)
    monkeypatch.setitem(sys.modules, sibling_module.__name__, sibling_module)

    candidate_validation._synchronize_transition_aliases(
        "_state_transition_matrix",
        original,
        replacement,
    )

    assert package_module._state_transition_matrix is replacement
    assert sibling_module._state_transition_matrix is original
