"""Runtime validation and compatibility patches for CLI/reporting helpers."""

from __future__ import annotations

from functools import wraps
import math

import numpy as np

_MISSING_PREDICTED_CANDIDATE_OPTION = "--state-space-momentum-predicted-candidate-top-k"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def apply_cli_float_values_validation_patch() -> None:
    """Reject invalid float grids and keep shared helper arguments complete."""

    from . import cli as _cli

    _patch_parse_float_values(_cli)
    _patch_state_space_predicted_candidate_argument(_cli)
    _patch_statistical_resampling_counts()


def _patch_parse_float_values(_cli) -> None:
    current = _cli._parse_float_values
    if getattr(current, "_hipporeplayimm_rejects_nonfinite", False):
        return

    def _parse_finite_float_values(value: str) -> tuple[float, ...]:
        parsed = current(value)
        if not all(math.isfinite(item) for item in parsed):
            raise ValueError(
                "comma-separated float value list must contain only finite values"
            )
        return parsed

    _parse_finite_float_values._hipporeplayimm_rejects_nonfinite = True  # type: ignore[attr-defined]
    _parse_finite_float_values._hipporeplayimm_original = current  # type: ignore[attr-defined]
    _cli._parse_float_values = _parse_finite_float_values


def _patch_state_space_predicted_candidate_argument(_cli) -> None:
    current = _cli._add_state_space_arguments
    if getattr(current, "_hipporeplayimm_adds_predicted_candidate_top_k", False):
        return

    def _add_state_space_arguments(parser) -> None:
        current(parser)
        if _parser_has_option(parser, _MISSING_PREDICTED_CANDIDATE_OPTION):
            return
        defaults = _cli.StateSpaceDecoderConfig()
        parser.add_argument(
            _MISSING_PREDICTED_CANDIDATE_OPTION,
            type=int,
            default=defaults.momentum_predicted_candidate_top_k,
        )

    _add_state_space_arguments.__name__ = current.__name__
    _add_state_space_arguments.__doc__ = current.__doc__
    _add_state_space_arguments._hipporeplayimm_adds_predicted_candidate_top_k = True  # type: ignore[attr-defined]
    _add_state_space_arguments._hipporeplayimm_original = current  # type: ignore[attr-defined]
    _cli._add_state_space_arguments = _add_state_space_arguments


def _patch_statistical_resampling_counts() -> None:
    from . import result_improvements

    _patch_positive_integer_kwarg(
        result_improvements,
        "hierarchical_bootstrap_ci",
        "n_bootstrap",
    )
    _patch_positive_integer_kwarg(
        result_improvements,
        "paired_sign_flip_p_value",
        "n_permutations",
    )


def _patch_positive_integer_kwarg(module, function_name: str, kwarg_name: str) -> None:
    current = getattr(module, function_name)
    patch_attr = f"_hipporeplayimm_validates_{kwarg_name}"
    if getattr(current, patch_attr, False):
        return

    @wraps(current)
    def wrapper(*args, **kwargs):
        if kwarg_name in kwargs:
            kwargs = dict(kwargs)
            kwargs[kwarg_name] = _positive_integer_count(kwarg_name, kwargs[kwarg_name])
        return current(*args, **kwargs)

    setattr(wrapper, patch_attr, True)
    wrapper._hipporeplayimm_original = current  # type: ignore[attr-defined]
    setattr(module, function_name, wrapper)


def _positive_integer_count(name: str, value: object) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar positive integer")
    item = raw.item()
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, not boolean")
    if isinstance(item, _STRING_TYPES):
        raise ValueError(f"{name} must be a positive integer, not string")
    try:
        integer = int(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    try:
        exact = bool(item == integer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer <= 0 or not exact:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _parser_has_option(parser, option: str) -> bool:
    return option in getattr(parser, "_option_string_actions", {})
