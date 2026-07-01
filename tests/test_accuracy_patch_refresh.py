from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.accuracy_replay_gain_gamma_patch import (
    _GAMMA_WRAPPER_FLAG,
    _PATCHED_FLAG,
    apply_accuracy_replay_gain_gamma_patch,
)


def _unvalidated(*args, **kwargs):
    del args, kwargs
    return "unvalidated"


def test_accuracy_patch_refreshes_stale_gamma_wrapper(monkeypatch):
    import hipporeplayimm.accuracy_upgrades as accuracy

    monkeypatch.setattr(accuracy, _PATCHED_FLAG, True, raising=False)
    monkeypatch.setattr(accuracy, "gamma_poisson_predictive_log_emissions", _unvalidated)

    apply_accuracy_replay_gain_gamma_patch()

    assert getattr(accuracy.gamma_poisson_predictive_log_emissions, _GAMMA_WRAPPER_FLAG, False)
    with pytest.raises(ValueError, match="dt"):
        accuracy.gamma_poisson_predictive_log_emissions(
            np.zeros((1, 1), dtype=int),
            np.ones((1, 1), dtype=float),
            np.ones((1, 1), dtype=float),
            float("nan"),
        )
