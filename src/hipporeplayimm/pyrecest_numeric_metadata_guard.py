'''Strict numeric parsing guards for PyRecEst score metadata and seeds.'''

from __future__ import annotations

import operator
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = '_pyrecest_numeric_metadata_guard_applied'
_RANDOM_SEED_PATCHED_FLAG = '_pyrecest_random_seed_validation_applied'
_RAW_FLOAT_ERROR = 'could not convert string to float'


def apply_pyrecest_numeric_metadata_guard_patch() -> None:
    '''Reject boolean and malformed values in PyRecEst numeric metadata/config.'''

    from . import pyrecest_score_metadata as metadata

    current = metadata._metadata_float_from_value
    if not getattr(current, _PATCHED_FLAG, False):

        @wraps(current)
        def metadata_float_from_value(value: Any, column: str) -> float | None:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{column} must contain finite numeric metadata values')
            try:
                return current(value, column)
            except ValueError as exc:
                if _RAW_FLOAT_ERROR in str(exc):
                    raise ValueError(f'{column} must contain finite numeric metadata values') from exc
                raise

        setattr(metadata_float_from_value, _PATCHED_FLAG, True)
        metadata._metadata_float_from_value = metadata_float_from_value

    _patch_pyrecest_random_seed_validation()


def _patch_pyrecest_random_seed_validation() -> None:
    from . import pyrecest_models

    if getattr(pyrecest_models, _RANDOM_SEED_PATCHED_FLAG, False):
        return

    current_event_seed = pyrecest_models._event_seed
    current_post_init = pyrecest_models.PyRecEstGoalParticleModel.__post_init__

    @wraps(current_event_seed)
    def event_seed(random_seed: object, emissions) -> int:
        return current_event_seed(_coerce_integer_seed(random_seed, 'random_seed'), emissions)

    @wraps(current_post_init)
    def pyrecest_goal_particle_post_init(self) -> None:
        _coerce_integer_seed(self.random_seed, 'random_seed')
        current_post_init(self)

    setattr(event_seed, _RANDOM_SEED_PATCHED_FLAG, True)
    setattr(event_seed, '__hipporeplayimm_original__', current_event_seed)
    setattr(pyrecest_goal_particle_post_init, _RANDOM_SEED_PATCHED_FLAG, True)
    setattr(pyrecest_goal_particle_post_init, '__hipporeplayimm_original__', current_post_init)
    pyrecest_models._event_seed = event_seed
    pyrecest_models.PyRecEstGoalParticleModel.__post_init__ = pyrecest_goal_particle_post_init
    pyrecest_models._coerce_integer_seed = _coerce_integer_seed
    setattr(pyrecest_models, _RANDOM_SEED_PATCHED_FLAG, True)


def _coerce_integer_seed(value: object, name: str) -> int:
    if _is_boolean_scalar(value) or _is_boolean_array(value) or not _is_scalar_value(value):
        raise ValueError(f'{name} must be an integer scalar')
    scalar = _validation_array(value).item()
    try:
        return int(operator.index(scalar))
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer scalar') from exc


def _validation_array(value: object) -> np.ndarray:
    try:
        return np.asarray(value)
    except ValueError:
        return np.asarray(value, dtype=object)


def _is_boolean_scalar(value: object) -> bool:
    array = _validation_array(value)
    if array.shape != ():
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _is_boolean_array(value: object) -> bool:
    array = _validation_array(value)
    if array.shape == ():
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in array.flat)
    return False


def _is_scalar_value(value: object) -> bool:
    return _validation_array(value).shape == ()


__all__ = ['apply_pyrecest_numeric_metadata_guard_patch']
