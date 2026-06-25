import numpy as np
import pytest

from hipporeplayimm.state_space import _mass_retaining_candidate_indices


_PATCH_ATTR = "__hipporeplayimm_bin_count_validation_patch__"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def test_mass_retaining_candidate_support_rejects_noninteger_count_bounds():
    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.05], dtype=float))

    invalid_bounds = (
        {"top_k": 1.5},
        {"top_k": True},
        {"min_k": 2.5},
        {"min_k": np.bool_(True)},
        {"max_k": 2.5},
        {"max_k": np.array([2])},
    )
    for kwargs in invalid_bounds:
        with pytest.raises(TypeError, match="must be an integer"):
            _mass_retaining_candidate_indices(log_emission, 0.95, **kwargs)


def test_mass_retaining_candidate_support_rejects_negative_top_k():
    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.05], dtype=float))

    for mass_threshold in (None, 0.0, 0.95):
        with pytest.raises(ValueError, match="top_k"):
            _mass_retaining_candidate_indices(
                log_emission,
                mass_threshold,
                top_k=-1,
            )


def test_mass_retaining_candidate_support_accepts_integer_valued_count_bounds():
    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.05], dtype=float))

    selected = _mass_retaining_candidate_indices(
        log_emission,
        0.95,
        top_k=1.0,
        min_k=2.0,
        max_k=3.0,
    )

    assert list(selected) == [0, 1, 2]


def test_bin_count_patch_recovers_candidate_count_validator_after_partial_patch(monkeypatch):
    import hipporeplayimm.state_space as state_space
    import hipporeplayimm.state_space_utils as utils
    from hipporeplayimm.state_space_bin_count_validation import apply_state_space_bin_count_validation_patch, _mark_patched

    original_coerce = getattr(utils._coerce_valid_bin_mask, _ORIGINAL_ATTR, utils._coerce_valid_bin_mask)
    original_mass = getattr(utils._mass_retaining_candidate_indices, _ORIGINAL_ATTR, utils._mass_retaining_candidate_indices)

    def already_patched_coerce_valid_bin_mask(valid_bin_mask, n_bins):
        return original_coerce(valid_bin_mask, n_bins)

    _mark_patched(already_patched_coerce_valid_bin_mask, original_coerce)
    monkeypatch.setattr(utils, "_coerce_valid_bin_mask", already_patched_coerce_valid_bin_mask)
    monkeypatch.setattr(utils, "_mass_retaining_candidate_indices", original_mass)
    monkeypatch.setattr(state_space, "_mass_retaining_candidate_indices", original_mass)

    apply_state_space_bin_count_validation_patch()

    assert getattr(utils._coerce_valid_bin_mask, _PATCH_ATTR, False)
    assert getattr(utils._mass_retaining_candidate_indices, _PATCH_ATTR, False)
    assert state_space._mass_retaining_candidate_indices is utils._mass_retaining_candidate_indices

    log_emission = np.log(np.array([0.50, 0.30, 0.15, 0.05], dtype=float))
    with pytest.raises(TypeError, match="must be an integer"):
        state_space._mass_retaining_candidate_indices(log_emission, 0.95, min_k=1.5)
