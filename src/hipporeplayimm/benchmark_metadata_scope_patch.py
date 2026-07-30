"""Restore benchmark scope metadata after metadata compatibility patching.

The score-table metadata compatibility layer replaces ``benchmarks._benchmark_config_metadata``
so post-hoc decoding can reconstruct older score tables. Keep that replacement
synchronized with canonical benchmark scope metadata used by relative-metric
scoping.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

_YAML_AMBIGUOUS_STRING_SCALARS = {
    "~",
    "null",
    "true",
    "false",
    "yes",
    "no",
    "on",
    "off",
    "y",
    "n",
    "nan",
    ".nan",
    "inf",
    ".inf",
    "+inf",
    "+.inf",
    "-inf",
    "-.inf",
}
_YAML_QUOTE_TRIGGER_CHARS = ":#[]{}&*!|>'\"%@`\r\n"
_YAML_DATE_SCALAR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YAML_BASE_PREFIXED_INTEGER_SCALAR = re.compile(
    r"^[+-]?0(?:[xX][0-9a-fA-F_]+|[bB][01_]+)$"
)
_YAML_SCALAR_PATCH_FLAG = "_benchmark_settings_yaml_scalar_patch_applied"
_YAML_SCALAR_WRAPPER_FLAG = "_benchmark_settings_yaml_scalar_wrapper"


def apply_benchmark_metadata_scope_patch() -> None:
    """Ensure benchmark-level split/event-subset metadata is emitted."""

    _patch_benchmark_settings_yaml_scalars()

    from . import benchmark_seed_validation
    from . import benchmarks as bench

    metadata = bench._benchmark_config_metadata
    if getattr(metadata, "_benchmark_scope_metadata_wrapped", False):
        benchmark_seed_validation.apply_benchmark_seed_validation_patch()
        return

    def benchmark_config_metadata_with_scope_fields(config: Any) -> dict[str, object]:
        out = dict(metadata(config))
        out["benchmark_n_cell_splits"] = int(getattr(config, "n_cell_splits", 1))
        out["benchmark_randomize_event_subset"] = bool(
            getattr(config, "randomize_event_subset", False)
        )
        event_subset_seed = getattr(config, "event_subset_seed", None)
        out["benchmark_event_subset_base_seed"] = (
            np.nan if event_subset_seed is None else int(event_subset_seed)
        )
        out["benchmark_event_epoch"] = _event_epoch_scope(getattr(config, "event_epoch", "run"))
        return out

    benchmark_config_metadata_with_scope_fields._benchmark_scope_metadata_wrapped = True  # type: ignore[attr-defined]
    bench._benchmark_config_metadata = benchmark_config_metadata_with_scope_fields
    benchmark_seed_validation.apply_benchmark_seed_validation_patch()


def _patch_benchmark_settings_yaml_scalars() -> None:
    """Quote YAML-ambiguous strings in every settings writer."""

    from . import observation_sweep
    from . import result_improvements
    from . import simulation_recovery

    modules = (result_improvements, observation_sweep, simulation_recovery)
    if all(
        getattr(module, _YAML_SCALAR_PATCH_FLAG, False)
        and getattr(getattr(module, "_yaml_scalar", None), _YAML_SCALAR_WRAPPER_FLAG, False)
        for module in modules
    ):
        return

    def yaml_scalar(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return str(value)
        text = str(value)
        if _yaml_string_needs_quotes(text):
            return json.dumps(text)
        return text

    setattr(yaml_scalar, _YAML_SCALAR_WRAPPER_FLAG, True)
    for module in modules:
        module._yaml_scalar = yaml_scalar
        setattr(module, _YAML_SCALAR_PATCH_FLAG, True)


def _yaml_string_needs_quotes(text: str) -> bool:
    if not text or text.strip() != text:
        return True
    if any(char in text for char in _YAML_QUOTE_TRIGGER_CHARS):
        return True
    if text.lower() in _YAML_AMBIGUOUS_STRING_SCALARS:
        return True
    if _YAML_DATE_SCALAR.fullmatch(text):
        return True
    if _YAML_BASE_PREFIXED_INTEGER_SCALAR.fullmatch(text):
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


def _event_epoch_scope(value: object) -> str:
    if value is None:
        return "run"
    label = str(value).strip()
    return label or "run"
