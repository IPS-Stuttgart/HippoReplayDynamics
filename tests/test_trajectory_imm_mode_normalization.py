from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.trajectory_imm_single_bin_diagnostics import _normalized_mode_row


def _trajectory_imm_stub():
    return SimpleNamespace(
        _TRAJECTORY_IMM_MODES=("stationary", "diffusion", "fragmented", "momentum"),
    )


def test_normalized_mode_row_scales_before_summing_extreme_masses() -> None:
    maximum = np.finfo(float).max
    row = np.array([maximum, maximum, maximum / 2.0, 0.0], dtype=float)

    normalized = _normalized_mode_row(_trajectory_imm_stub(), row)

    np.testing.assert_allclose(normalized, [0.4, 0.4, 0.2, 0.0])
    assert normalized.sum() == pytest.approx(1.0)


def test_normalized_mode_row_still_rejects_zero_mass() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        _normalized_mode_row(_trajectory_imm_stub(), np.zeros(4, dtype=float))
