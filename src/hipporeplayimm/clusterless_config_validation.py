"""Runtime validation for clusterless nested encoding configuration."""

from __future__ import annotations

from functools import wraps
import sys

import numpy as np

from .encoding import EncodingConfig, _validate_encoding_config

_PATCH_MARKER = "_clusterless_encoding_config_validation_patch"


_NUMERIC_MESSAGES = {
    "mark_smoothing_sigma_bins": "mark_smoothing_sigma_bins must be finite and nonnegative",
    "mark_prior_count": "mark_prior_count must be finite and nonnegative",
    "mark_variance_floor": "mark_variance_floor must be finite and positive",
    "rate_floor_hz": "rate_floor_hz must be finite and positive",
    "mark_kde_bandwidth": "mark_kde_bandwidth must be finite and positive when provided",
    "mark_kde_spatial_sigma_bins": "mark_kde_spatial_sigma_bins must be finite and nonnegative when provided",
    "mark_kde_max_neighbors": "mark_kde_max_neighbors must be a positive integer",
}


def apply_clusterless_encoding_config_validation_patch() -> None:
    """Validate ClusterlessMarkConfig.encoding before fitting clusterless marks."""

    import hipporeplayimm.clusterless as clusterless

    current = clusterless.fit_clusterless_mark_encoding
    if getattr(current, _PATCH_MARKER, False):
        previous = getattr(current, "__wrapped__", None)
        if previous is not None:
            _synchronize_aliases(previous, current)
        return

    previous = current

    @wraps(previous)
    def fit_clusterless_mark_encoding(session, config=None):
        _validate_nested_encoding_config(config)
        _validate_clusterless_mark_config(config)
        return previous(session, config)

    setattr(fit_clusterless_mark_encoding, _PATCH_MARKER, True)
    clusterless.fit_clusterless_mark_encoding = fit_clusterless_mark_encoding
    _synchronize_aliases(previous, fit_clusterless_mark_encoding)


def _validate_nested_encoding_config(config: object | None) -> None:
    encoding_config = EncodingConfig() if config is None else getattr(config, "encoding", None)
    if encoding_config is None:
        encoding_config = EncodingConfig()
    _validate_encoding_config(encoding_config)


def _validate_clusterless_mark_config(config: object | None) -> None:
    if config is None:
        return
    _finite_config_value(config, "mark_smoothing_sigma_bins", positive=False)
    _finite_config_value(config, "mark_prior_count", positive=False)
    _finite_config_value(config, "mark_variance_floor", positive=True)
    _finite_config_value(config, "rate_floor_hz", positive=True)
    _finite_config_value(config, "mark_kde_bandwidth", positive=True, optional=True)
    _finite_config_value(config, "mark_kde_spatial_sigma_bins", positive=False, optional=True)
    _positive_integer_config_value(config, "mark_kde_max_neighbors")


def _finite_config_value(
    config: object,
    name: str,
    *,
    positive: bool,
    optional: bool = False,
) -> float | None:
    value = getattr(config, name, None)
    if value is None:
        if optional:
            return None
        raise ValueError(_NUMERIC_MESSAGES[name])
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_NUMERIC_MESSAGES[name])
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_NUMERIC_MESSAGES[name]) from exc
    if not np.isfinite(numeric) or numeric < 0.0 or (positive and numeric <= 0.0):
        raise ValueError(_NUMERIC_MESSAGES[name])
    return numeric


def _positive_integer_config_value(config: object, name: str) -> int:
    value = getattr(config, name, None)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_NUMERIC_MESSAGES[name])
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_NUMERIC_MESSAGES[name]) from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(_NUMERIC_MESSAGES[name])
    return int(numeric)


def _synchronize_aliases(previous: object, patched: object) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "fit_clusterless_mark_encoding", None) is previous:
            module.fit_clusterless_mark_encoding = patched
