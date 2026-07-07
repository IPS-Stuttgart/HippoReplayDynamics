from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import candidate_support_normalization_validation as patch
from hipporeplayimm import models, state_space, state_space_model, state_space_utils

_SCORE_PATCHED_FLAG = "_candidate_support_score_validation_patch_applied"


def _legacy_top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    del log_emission, top_k
    return np.array([0], dtype=int)


def _legacy_mass_retaining_candidate_indices(
    log_emission: np.ndarray,
    mass_threshold: float | None = None,
    *,
    top_k: int | None = None,
    min_k: int = 1,
    max_k: int = 0,
) -> np.ndarray:
    del log_emission, mass_threshold, top_k, min_k, max_k
    return np.array([0], dtype=int)


def test_candidate_support_patch_refreshes_stale_state_space_score_helpers(monkeypatch) -> None:
    for module in (state_space_utils, state_space, state_space_model):
        monkeypatch.setattr(module, "_top_candidate_indices", _legacy_top_candidate_indices)
        monkeypatch.setattr(module, "_mass_retaining_candidate_indices", _legacy_mass_retaining_candidate_indices)
    setattr(state_space_utils, _SCORE_PATCHED_FLAG, True)

    patch.apply_candidate_support_normalization_validation_patch()

    for module in (state_space_utils, state_space, state_space_model):
        with pytest.raises(ValueError, match="at least one finite"):
            module._top_candidate_indices(np.array([np.nan, np.nan]), 1)
        with pytest.raises(TypeError, match="mass_threshold"):
            module._mass_retaining_candidate_indices(np.array([0.0, -1.0]), True)


def test_candidate_support_patch_refreshes_stale_model_score_helper(monkeypatch) -> None:
    monkeypatch.setattr(models, "_top_candidate_indices", _legacy_top_candidate_indices)
    setattr(models, _SCORE_PATCHED_FLAG, True)

    patch.apply_candidate_support_normalization_validation_patch()

    with pytest.raises(ValueError, match="at least one finite"):
        models._top_candidate_indices(np.array([np.nan, np.nan]), 1)
