import pytest

import hipporeplayimm
from hipporeplayimm.models import CandidateKinematicModel


def test_candidate_kinematic_velocity_decay_guard_survives_restored_shared_validator(monkeypatch):
    """Reject amplifying momentum even if shared validator aliases are restored."""

    hipporeplayimm.apply_runtime_patches()
    import hipporeplayimm.models as models

    original_nonnegative = getattr(
        models._validate_nonnegative_parameter,
        "__hipporeplayimm_original__",
        models._validate_nonnegative_parameter,
    )
    monkeypatch.setattr(models, "_validate_nonnegative_parameter", original_nonnegative)

    with pytest.raises(ValueError, match="velocity_decay"):
        CandidateKinematicModel(mode="momentum", velocity_decay=1.1)


def test_candidate_kinematic_velocity_decay_accepts_unit_interval_boundaries():
    hipporeplayimm.apply_runtime_patches()

    assert CandidateKinematicModel(mode="momentum", velocity_decay=0.0).velocity_decay == 0.0
    assert CandidateKinematicModel(mode="momentum", velocity_decay=1.0).velocity_decay == 1.0
