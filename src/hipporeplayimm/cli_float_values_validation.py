"""Runtime validation and compatibility patches for CLI/reporting helpers."""

from __future__ import annotations

from functools import wraps
import math
import operator

import numpy as np
import pandas as pd

_MISSING_PREDICTED_CANDIDATE_OPTION = "--state-space-momentum-predicted-candidate-top-k"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)
_POSTERIOR_CALIBRATION_GROUP_PATCH = "_hipporeplayimm_retains_missing_calibration_groups"
_BOOTSTRAP_DELTA_VALIDATION_PATCH = "_hipporeplayimm_filters_nonfinite_bootstrap_delta"


def apply_cli_float_values_validation_patch() -> None:
    """Reject invalid float grids and keep shared helper arguments complete."""

    from . import cli as _cli

    _patch_parse_float_values(_cli)
    _patch_state_space_predicted_candidate_argument(_cli)
    _patch_statistical_resampling_counts()
    _patch_legacy_bootstrap_delta_ci(_cli)
    _patch_posterior_calibration_missing_groups()


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


def _patch_legacy_bootstrap_delta_ci(_cli) -> None:
    """Keep the legacy flat bootstrap on finite metrics and validated counts."""

    from . import benchmarks

    current = benchmarks.bootstrap_delta_ci
    if getattr(current, _BOOTSTRAP_DELTA_VALIDATION_PATCH, False):
        _cli.bootstrap_delta_ci = current
        return

    @wraps(current)
    def bootstrap_delta_ci(
        rows: pd.DataFrame,
        model: str = "imm",
        value_column: str = "delta_vs_best_static",
        n_bootstrap: int = 1000,
        random_seed: int = 1,
    ) -> tuple[float, float]:
        validated_n_bootstrap = _positive_integer_count("n_bootstrap", n_bootstrap)
        target_mask = rows["model"].eq(model).to_numpy(dtype=bool)
        numeric_values = pd.to_numeric(rows[value_column], errors="coerce").to_numpy(dtype=float)
        keep_rows = ~target_mask | np.isfinite(numeric_values)
        filtered = rows.loc[keep_rows]
        return current(
            filtered,
            model=model,
            value_column=value_column,
            n_bootstrap=validated_n_bootstrap,
            random_seed=random_seed,
        )

    setattr(bootstrap_delta_ci, _BOOTSTRAP_DELTA_VALIDATION_PATCH, True)
    bootstrap_delta_ci._hipporeplayimm_original = current  # type: ignore[attr-defined]
    benchmarks.bootstrap_delta_ci = bootstrap_delta_ci
    _cli.bootstrap_delta_ci = bootstrap_delta_ci


def _patch_posterior_calibration_missing_groups() -> None:
    """Retain valid calibration rows whose optional session/model key is missing."""

    from . import result_improvements

    current = result_improvements.posterior_calibration_summary
    if getattr(current, _POSTERIOR_CALIBRATION_GROUP_PATCH, False):
        return

    @wraps(current)
    def posterior_calibration_summary(
        samples: pd.DataFrame,
        *,
        probability_column: str = "true_bin_probability",
        rank_column: str = "true_bin_rank",
        n_bins_column: str = "n_position_bins",
    ) -> pd.DataFrame:
        group_columns = [column for column in ("session", "model") if column in samples]
        prepared = samples.copy()
        sentinels: dict[str, str] = {}
        for column in group_columns:
            missing = prepared[column].isna()
            if not bool(missing.any()):
                continue
            sentinel = _missing_group_sentinel(prepared[column], column)
            prepared[column] = prepared[column].astype(object)
            prepared.loc[missing, column] = sentinel
            sentinels[column] = sentinel

        summary = current(
            prepared,
            probability_column=probability_column,
            rank_column=rank_column,
            n_bins_column=n_bins_column,
        )
        if summary.empty or not sentinels:
            return summary

        restored = summary.copy()
        for column, sentinel in sentinels.items():
            if column not in restored:
                continue
            missing = restored[column].astype(object).eq(sentinel)
            restored.loc[missing, column] = pd.NA
        return restored

    setattr(
        posterior_calibration_summary,
        _POSTERIOR_CALIBRATION_GROUP_PATCH,
        True,
    )
    posterior_calibration_summary._hipporeplayimm_original = current  # type: ignore[attr-defined]
    result_improvements.posterior_calibration_summary = posterior_calibration_summary


def _missing_group_sentinel(values: pd.Series, column: str) -> str:
    observed = set(values.astype(str).tolist())
    base = f"__hipporeplayimm_missing_{column}__"
    sentinel = base
    suffix = 0
    while sentinel in observed:
        suffix += 1
        sentinel = f"{base}_{suffix}"
    return sentinel


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
        exact_integer = operator.index(item)
    except TypeError:
        pass
    else:
        if exact_integer <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(exact_integer)
    try:
        integer = int(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    try:
        exactly_integral = bool(item == integer)
    except (TypeError, ValueError):
        exactly_integral = False
    if not exactly_integral or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _parser_has_option(parser, option: str) -> bool:
    return option in getattr(parser, "_option_string_actions", {})
