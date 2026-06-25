"""Runtime validation and compatibility patches for CLI argument helpers."""

from __future__ import annotations

import math

_MISSING_PREDICTED_CANDIDATE_OPTION = "--state-space-momentum-predicted-candidate-top-k"


def apply_cli_float_values_validation_patch() -> None:
    """Reject invalid float grids and keep shared state-space CLI options complete."""

    from . import cli as _cli

    _patch_parse_float_values(_cli)
    _patch_state_space_predicted_candidate_argument(_cli)


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


def _parser_has_option(parser, option: str) -> bool:
    return option in getattr(parser, "_option_string_actions", {})
