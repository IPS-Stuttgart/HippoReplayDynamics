from __future__ import annotations

from functools import wraps

import numpy as np
import pytest

import hipporeplayimm

_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def _scalar_item(value: object) -> object:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return value


def _legacy_coerce_integer_count(name: str, value: object) -> int:
    del name
    return int(_scalar_item(value))


def _legacy_coerce_unit_probability(name: str, value: object) -> float:
    probability = float(_scalar_item(value))
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _legacy_top_candidate_indices(log_emission: np.ndarray, top_k: object) -> np.ndarray:
    values = np.asarray(log_emission, dtype=float)
    count = int(_scalar_item(top_k))
    if count <= 0 or count >= values.shape[0]:
        return np.arange(values.shape[0], dtype=int)
    selected = np.argpartition(values, -count)[-count:]
    return selected[np.argsort(values[selected])[::-1]]


def _legacy_mass_retaining_candidate_indices(
    log_emission: np.ndarray,
    mass_threshold: object | None = None,
    *,
    top_k: object | None = None,
    min_k: object = 1,
    max_k: object = 0,
) -> np.ndarray:
    del min_k, max_k
    if mass_threshold is not None:
        threshold = float(_scalar_item(mass_threshold))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("mass_threshold must be in [0, 1]")
    return _legacy_top_candidate_indices(
        log_emission,
        np.asarray(log_emission).shape[0] if top_k is None else top_k,
    )


def _stale_count_wrapper(original, marker: str):
    @wraps(original)
    def wrapper(name: str, value: object):
        if isinstance(value, _STRING_TYPES):
            raise TypeError(f"{name} must be an integer scalar, not string")
        return original(name, value)

    setattr(wrapper, marker, True)
    setattr(wrapper, "__hipporeplayimm_original__", original)
    return wrapper


def _stale_probability_wrapper(original, marker: str):
    @wraps(original)
    def wrapper(name: str, value: object):
        if isinstance(value, _STRING_TYPES):
            raise TypeError(f"{name} must be a numeric scalar, not string")
        return original(name, value)

    setattr(wrapper, marker, True)
    setattr(wrapper, "__hipporeplayimm_original__", original)
    return wrapper


def _stale_top_candidate_wrapper(original, marker: str):
    @wraps(original)
    def wrapper(log_emission: np.ndarray, top_k: object):
        if isinstance(top_k, _STRING_TYPES):
            raise TypeError("top_k must be an integer scalar, not string")
        return original(log_emission, top_k)

    setattr(wrapper, marker, True)
    setattr(wrapper, "__hipporeplayimm_original__", original)
    return wrapper


def _stale_mass_candidate_wrapper(original, marker: str):
    @wraps(original)
    def wrapper(
        log_emission: np.ndarray,
        mass_threshold: object | None = None,
        *,
        top_k: object | None = None,
        min_k: object = 1,
        max_k: object = 0,
    ):
        if isinstance(mass_threshold, _STRING_TYPES):
            raise TypeError("mass_threshold must be a numeric scalar, not string")
        if isinstance(top_k, _STRING_TYPES):
            raise TypeError("top_k must be an integer scalar, not string")
        return original(
            log_emission,
            mass_threshold,
            top_k=top_k,
            min_k=min_k,
            max_k=max_k,
        )

    setattr(wrapper, marker, True)
    setattr(wrapper, "__hipporeplayimm_original__", original)
    return wrapper


def test_runtime_patches_refresh_already_marked_stale_state_space_numeric_helpers(monkeypatch):
    hipporeplayimm.apply_runtime_patches()

    from hipporeplayimm import model_numeric_string_validation, state_space_model, state_space_utils

    marker = model_numeric_string_validation._STATE_SPACE_UTILS_PATCHED_FLAG
    version_attr = model_numeric_string_validation._STATE_SPACE_UTILS_PATCH_VERSION_ATTR
    version = model_numeric_string_validation._STATE_SPACE_UTILS_PATCH_VERSION

    stale_integer_count = _stale_count_wrapper(_legacy_coerce_integer_count, marker)
    stale_unit_probability = _stale_probability_wrapper(_legacy_coerce_unit_probability, marker)
    stale_top_candidates = _stale_top_candidate_wrapper(_legacy_top_candidate_indices, marker)
    stale_mass_candidates = _stale_mass_candidate_wrapper(_legacy_mass_retaining_candidate_indices, marker)

    assert stale_integer_count("n_modes", np.asarray("3")) == 3
    assert stale_unit_probability("mode_stickiness", np.asarray("0.95")) == 0.95
    assert stale_top_candidates(np.array([0.0, 1.0]), np.asarray("1")).tolist() == [1]
    assert stale_mass_candidates(np.array([0.0, -1.0]), np.asarray("0.5")).size == 2

    monkeypatch.setattr(state_space_utils, "_coerce_integer_count", stale_integer_count)
    monkeypatch.setattr(state_space_utils, "_coerce_unit_probability", stale_unit_probability)
    monkeypatch.setattr(state_space_utils, "_top_candidate_indices", stale_top_candidates)
    monkeypatch.setattr(state_space_utils, "_mass_retaining_candidate_indices", stale_mass_candidates)
    monkeypatch.setattr(state_space_model, "_top_candidate_indices", stale_top_candidates)
    monkeypatch.setattr(state_space_model, "_mass_retaining_candidate_indices", stale_mass_candidates)

    hipporeplayimm.apply_runtime_patches()

    for name in (
        "_coerce_integer_count",
        "_coerce_unit_probability",
        "_top_candidate_indices",
        "_mass_retaining_candidate_indices",
    ):
        helper = getattr(state_space_utils, name)
        assert getattr(helper, marker, False)
        assert getattr(helper, version_attr, None) == version

    assert state_space_model._top_candidate_indices is state_space_utils._top_candidate_indices
    assert state_space_model._mass_retaining_candidate_indices is state_space_utils._mass_retaining_candidate_indices

    with pytest.raises(TypeError, match="n_modes"):
        state_space_utils._coerce_integer_count("n_modes", np.asarray("3"))
    with pytest.raises(TypeError, match="mode_stickiness"):
        state_space_utils._coerce_unit_probability("mode_stickiness", np.asarray("0.95"))
    with pytest.raises(TypeError, match="top_k"):
        state_space_model._top_candidate_indices(np.array([0.0, 1.0]), np.asarray("1"))
    with pytest.raises(TypeError, match="mass_threshold"):
        state_space_model._mass_retaining_candidate_indices(np.array([0.0, -1.0]), np.asarray("0.5"))
