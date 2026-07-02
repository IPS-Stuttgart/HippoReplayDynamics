"""Runtime validation for LogEmissionTensor construction invariants."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_PATCH_ATTR = "_log_emission_tensor_validation_patch"
_ORIGINAL_ATTR = "_log_emission_tensor_validation_original"


def _validate_constructed_tensor(tensor: object) -> None:
    """Reject invalid emission tensors immediately after dataclass initialization."""

    log_likelihood = np.asarray(getattr(tensor, "log_likelihood"), dtype=float)
    if log_likelihood.ndim == 2 and log_likelihood.shape[0] == 0:
        raise ValueError("log_likelihood must contain at least one time bin")
    if np.any(np.isnan(log_likelihood)) or np.any(log_likelihood == np.inf):
        raise ValueError("log_likelihood must not contain NaN or +inf")

    spike_counts = np.asarray(getattr(tensor, "spike_counts"), dtype=float)
    if spike_counts.ndim == 2 and not np.all(
        np.isclose(spike_counts, np.rint(spike_counts), rtol=0.0, atol=0.0)
    ):
        raise ValueError("spike_counts must contain integer-valued counts")


def _wrap_post_init(post_init: Callable[[object], None]) -> Callable[[object], None]:
    if getattr(post_init, _PATCH_ATTR, False):
        return post_init

    def validated_post_init(self: object) -> None:
        post_init(self)
        _validate_constructed_tensor(self)

    validated_post_init.__name__ = getattr(post_init, "__name__", "__post_init__")
    validated_post_init.__doc__ = getattr(post_init, "__doc__", None)
    setattr(validated_post_init, _PATCH_ATTR, True)
    setattr(validated_post_init, _ORIGINAL_ATTR, post_init)
    return validated_post_init


def apply_log_emission_tensor_validation_patch() -> None:
    """Install construction-time validation for ``LogEmissionTensor``."""

    from .encoding import LogEmissionTensor

    LogEmissionTensor.__post_init__ = _wrap_post_init(LogEmissionTensor.__post_init__)


__all__ = ["apply_log_emission_tensor_validation_patch"]
