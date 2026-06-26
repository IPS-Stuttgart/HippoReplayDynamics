from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _stale_candidate_log_masses(*args, **kwargs) -> list[float]:
    del args, kwargs
    return [0.0]


def test_candidate_log_mass_patch_reinstalls_after_alias_overwrite(monkeypatch) -> None:
    import hipporeplayimm.models as models
    import hipporeplayimm.state_space as state_space
    import hipporeplayimm.state_space_candidates as state_space_candidates
    import hipporeplayimm.state_space_candidates_momentum as state_space_candidates_momentum
    import hipporeplayimm.state_space_utils as state_space_utils
    from hipporeplayimm.candidate_log_mass_validation import (
        apply_candidate_log_mass_validation_patch,
    )

    monkeypatch.setattr(
        state_space,
        "_candidate_log_mass_validation_patch_applied",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        state_space_utils,
        "_candidate_log_masses",
        _stale_candidate_log_masses,
    )
    monkeypatch.setattr(
        state_space,
        "_candidate_log_masses",
        _stale_candidate_log_masses,
        raising=False,
    )
    monkeypatch.setattr(
        state_space_candidates,
        "_candidate_log_masses",
        _stale_candidate_log_masses,
    )
    monkeypatch.setattr(
        state_space_candidates_momentum,
        "_candidate_log_masses",
        _stale_candidate_log_masses,
    )
    monkeypatch.setattr(
        models,
        "_candidate_log_masses",
        _stale_candidate_log_masses,
        raising=False,
    )

    apply_candidate_log_mass_validation_patch()

    log_likelihood = np.asarray([[0.0, -np.inf]], dtype=float)
    candidates = [np.asarray([1], dtype=int)]

    with pytest.raises(ValueError, match="select no finite"):
        state_space_utils._candidate_log_masses(log_likelihood, candidates)
    with pytest.raises(ValueError, match="select no finite"):
        models._candidate_log_masses(
            SimpleNamespace(log_likelihood=log_likelihood),
            candidates,
        )

    assert state_space._candidate_log_masses is state_space_utils._candidate_log_masses
    assert (
        state_space_candidates._candidate_log_masses
        is state_space_utils._candidate_log_masses
    )
    assert (
        state_space_candidates_momentum._candidate_log_masses
        is state_space_utils._candidate_log_masses
    )
