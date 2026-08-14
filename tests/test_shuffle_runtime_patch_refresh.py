from __future__ import annotations

import importlib

import hipporeplayimm
from hipporeplayimm import result_improvements
from hipporeplayimm import shuffle_controls
from hipporeplayimm import shuffle_spike_time_order as shuffle_patch


def test_runtime_patches_restore_shuffle_controls_after_reload() -> None:
    """Reloaded shuffle helpers must not trust stale module patch flags."""

    module = importlib.reload(shuffle_controls)

    assert module.shuffled_encoding is not shuffle_patch._shuffled_encoding_nonidentity
    assert module._nonnegative_integer_value is not shuffle_patch._nonnegative_integer_value
    assert module._scope_label.__module__ == module.__name__
    assert module._validate_grid_shape.__module__ == module.__name__

    hipporeplayimm.apply_runtime_patches()

    assert module.shuffled_encoding is shuffle_patch._shuffled_encoding_nonidentity
    assert module._nonnegative_integer_value is shuffle_patch._nonnegative_integer_value
    assert module._scope_label.__module__ == shuffle_patch.__name__
    assert module._validate_grid_shape.__module__ == shuffle_patch.__name__


def test_runtime_patches_restore_cell_identity_shuffle_after_reload() -> None:
    """Reloading result helpers must not disable the nonidentity cell shuffle."""

    module = importlib.reload(result_improvements)

    assert (
        module.shuffle_cell_identities_session
        is not shuffle_patch._shuffle_cell_identities_session_nonidentity
    )

    hipporeplayimm.apply_runtime_patches()

    assert (
        module.shuffle_cell_identities_session
        is shuffle_patch._shuffle_cell_identities_session_nonidentity
    )
