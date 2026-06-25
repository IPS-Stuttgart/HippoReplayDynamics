import numpy as np
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm.duration_occupancy import (
    _candidate_selection_emissions,
    _uniform_probabilities,
)
from hipporeplayimm.duration_occupancy_metadata_guard import _coerce_transition_durations
from hipporeplayimm.encoding import LogEmissionTensor


def _emissions_with_metadata() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.1], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        metadata={"source": "original"},
    )


def test_candidate_selection_emissions_copies_metadata_for_masked_support():
    emissions = _emissions_with_metadata()

    restricted = _candidate_selection_emissions(
        emissions,
        np.array([True, False, True], dtype=bool),
    )

    assert restricted is not emissions
    assert restricted.metadata == {"source": "original"}
    assert restricted.metadata is not emissions.metadata
    restricted.metadata["derived"] = True
    assert emissions.metadata == {"source": "original"}
    np.testing.assert_allclose(
        restricted.log_likelihood[:, [0, 2]],
        emissions.log_likelihood[:, [0, 2]],
    )
    assert np.all(np.isneginf(restricted.log_likelihood[:, 1]))


def test_candidate_selection_emissions_reuses_input_when_no_masking_needed():
    emissions = _emissions_with_metadata()

    assert _candidate_selection_emissions(emissions, None) is emissions
    assert _candidate_selection_emissions(
        emissions,
        np.array([True, True, True], dtype=bool),
    ) is emissions


@pytest.mark.parametrize(
    "bad_n_bins",
    [0, -1, True, np.bool_(True), 1.5, "2.5", np.array([2])],
)
def test_uniform_probabilities_rejects_invalid_bin_count(bad_n_bins):
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _uniform_probabilities(bad_n_bins)


def test_uniform_probabilities_are_bootstrapped_into_duration_occupancy_module():
    import hipporeplayimm.duration_occupancy as duration_occupancy

    assert callable(duration_occupancy._uniform_probabilities)
    np.testing.assert_allclose(
        duration_occupancy._uniform_probabilities(2),
        np.array([0.5, 0.5], dtype=float),
    )


def test_transition_duration_validation_patch_recovers_from_partial_module_flags(monkeypatch) -> None:
    import hipporeplayimm.state_space_displacement_imm as displacement_imm
    import hipporeplayimm.state_space_displacement_momentum as displacement_momentum
    import hipporeplayimm.state_space_sparse_momentum as sparse_momentum
    import hipporeplayimm.state_space_trajectory_imm as trajectory_imm
    from hipporeplayimm.duration_occupancy_metadata_guard import (
        apply_duration_occupancy_metadata_guard_patch,
        _coerce_transition_durations,
    )

    monkeypatch.setattr(sparse_momentum, "_transition_duration_validation_patch_applied", True, raising=False)
    for module in (trajectory_imm, displacement_momentum, displacement_imm):
        monkeypatch.setattr(module, "_transition_duration_validation_patch_applied", False, raising=False)

    apply_duration_occupancy_metadata_guard_patch()

    for module in (trajectory_imm, displacement_momentum, displacement_imm):
        assert getattr(module, "_transition_duration_validation_patch_applied", False)
        assert module._coerce_transition_durations is _coerce_transition_durations
        assert getattr(module._duration_adjusted_decays, "_transition_duration_validation_wrapped", False)


def test_transition_duration_guard_fills_only_missing_metadata() -> None:
    durations = _coerce_transition_durations([], n_time=3, fallback_dt=0.02)

    np.testing.assert_allclose(durations, np.array([0.02, 0.02], dtype=float))


def test_transition_duration_guard_rejects_malformed_nonempty_metadata() -> None:
    with pytest.raises(ValueError, match="one finite positive value per transition"):
        _coerce_transition_durations([0.02], n_time=3, fallback_dt=0.02)

    with pytest.raises(ValueError, match="one-dimensional"):
        _coerce_transition_durations(np.ones((2, 1)), n_time=3, fallback_dt=0.02)

    with pytest.raises(ValueError, match="finite and positive"):
        _coerce_transition_durations([0.02, float("nan")], n_time=3, fallback_dt=0.02)
