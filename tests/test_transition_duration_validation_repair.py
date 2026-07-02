import numpy as np
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches


def test_transition_duration_validation_rechecks_stale_module_flag(monkeypatch) -> None:
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum
    from hipporeplayimm.duration_occupancy_metadata_guard import (
        apply_duration_occupancy_metadata_guard_patch,
        _coerce_transition_durations,
    )

    def stale_coerce_transition_durations(*args, **kwargs):
        return np.array([], dtype=float)

    def stale_duration_adjusted_decays(config, durations, reference_dt):
        return np.asarray(durations, dtype=float)

    monkeypatch.setattr(sparse_momentum, "_coerce_transition_durations", stale_coerce_transition_durations, raising=False)
    monkeypatch.setattr(sparse_momentum, "_duration_adjusted_decays", stale_duration_adjusted_decays, raising=False)
    monkeypatch.setattr(sparse_momentum, "_transition_duration_validation_patch_applied", True, raising=False)

    apply_duration_occupancy_metadata_guard_patch()

    assert sparse_momentum._coerce_transition_durations is _coerce_transition_durations
    assert getattr(sparse_momentum._duration_adjusted_decays, "_transition_duration_validation_wrapped", False)
    with pytest.raises(ValueError, match="not boolean"):
        sparse_momentum._duration_adjusted_decays(object(), np.array([True], dtype=object), 0.02)
