from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import encoding


_N_SPIKES_WRAPPER_MARKER = "_n_spikes_validation_post_init_wrapper"


def _source_post_init() -> Any:
    """Return the unpatched source ``LogEmissionTensor.__post_init__``."""

    current = encoding.LogEmissionTensor.__post_init__
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped) or wrapped is current:
            break
        current = wrapped
    return current


def _replacement_log_emission_tensor_class():
    """Mimic the class replacement performed by ``importlib.reload(encoding)``."""

    original_class = encoding.LogEmissionTensor
    source_post_init = _source_post_init()

    class RefreshedLogEmissionTensor(original_class):
        __post_init__ = source_post_init

    return RefreshedLogEmissionTensor


def _construct(
    tensor_class,
    *,
    spike_counts: np.ndarray,
    cell_ids: np.ndarray,
    n_spikes: object,
):
    return tensor_class(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=spike_counts,
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=cell_ids,
        n_spikes=n_spikes,
    )


def test_runtime_patches_restore_log_emission_count_validation_after_class_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refreshed_class = _replacement_log_emission_tensor_class()
    monkeypatch.setattr(encoding, "LogEmissionTensor", refreshed_class)

    assert not getattr(
        refreshed_class.__post_init__,
        _N_SPIKES_WRAPPER_MARKER,
        False,
    )

    hipporeplayimm.apply_runtime_patches()

    assert getattr(
        refreshed_class.__post_init__,
        _N_SPIKES_WRAPPER_MARKER,
        False,
    )

    with pytest.raises(ValueError, match="spike_counts must be integer-valued"):
        _construct(
            refreshed_class,
            spike_counts=np.array([[0.5]], dtype=float),
            cell_ids=np.array([1], dtype=int),
            n_spikes=0.5,
        )

    with pytest.raises(ValueError, match="n_spikes must equal the total spike_counts sum"):
        _construct(
            refreshed_class,
            spike_counts=np.array([[1]], dtype=int),
            cell_ids=np.array([1], dtype=int),
            n_spikes=0,
        )

    with pytest.raises(ValueError, match="cell_ids must be unique"):
        _construct(
            refreshed_class,
            spike_counts=np.array([[0, 0]], dtype=int),
            cell_ids=np.array([1, 1], dtype=int),
            n_spikes=0,
        )

    active_post_init = refreshed_class.__post_init__
    hipporeplayimm.apply_runtime_patches()
    assert refreshed_class.__post_init__ is active_post_init
