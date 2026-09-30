import numpy as np
import pytest

from scripts.audit_regional_blind_bank import check_templates, independent_saved_path_dwell


def test_independent_moving_dwell():
    v, tol = independent_saved_path_dwell(np.array([0., .1]), np.array([0, 1]),
        np.array([[-40., 0.], [40., 0.]]), "moving", np.array([0., 0.]), .1)
    assert abs(v-.05) <= tol


def test_independent_jump_dwell():
    v, tol = independent_saved_path_dwell(np.array([0., .015, .08, .1]), np.array([0, 1, 0, 0]),
        np.array([[0., 0.], [100., 0.]]), "jumping", np.array([0., 0.]), .1)
    assert abs(v-.035) <= tol


@pytest.mark.parametrize("corrupt", [False, True])
def test_original_timestamp_templates_are_enforced(tmp_path, corrupt):
    np.savez(tmp_path/"source_inputs.npz", template_offsets=[0, 2], template_times=[.03, .06],
             candidate_start=[0.], windows=[[.08, .1]])
    np.savez(tmp_path/"templates.npz", source_indices=[0], offsets=[0, 2],
             times=[.03, .06+(.001 if corrupt else 0.)], starts=[0.], endpoints=[.1])
    if corrupt:
        with pytest.raises(AssertionError):
            check_templates(tmp_path)
    else:
        check_templates(tmp_path)
