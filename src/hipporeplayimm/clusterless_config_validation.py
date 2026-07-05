"""Runtime validation for clusterless nested encoding and emission configuration."""

from __future__ import annotations

from functools import wraps
import sys

import numpy as np

from .encoding import EncodingConfig, _validate_encoding_config

_PATCH_MARKER = "_clusterless_encoding_config_validation_patch"
_EMISSION_CONFIG_PATCH_MARKER = "_clusterless_emission_config_validation_patch"
_BENCHMARK_MARK_CONFIG_PATCH_MARKER = "_benchmark_clusterless_mark_config_validation_patch"

_NUMERIC_MESSAGES = {
    "mark_smoothing_sigma_bins": "mark_smoothing_sigma_bins must be finite and nonnegative",
    "mark_prior_count": "mark_prior_count must be finite and nonnegative",
    "mark_variance_floor": "mark_variance_floor must be finite and positive",
    "rate_floor_hz": "rate_floor_hz must be finite and positive",
    "mark_kde_bandwidth": "mark_kde_bandwidth must be finite and positive when provided",
    "mark_kde_spatial_sigma_bins": "mark_kde_spatial_sigma_bins must be finite and nonnegative when provided",
    "mark_kde_max_neighbors": "mark_kde_max_neighbors must be a positive integer",
    "spike_rate_scale": "spike_rate_scale must be finite and positive",
    "likelihood_temperature": "likelihood_temperature must be finite and positive",
    "negative_binomial_overdispersion": "negative_binomial_overdispersion must be finite and nonnegative",
}


def apply_clusterless_encoding_config_validation_patch() -> None:
    """Validate clusterless encoding and emission configuration before use."""

    import hipporeplayimm.clusterless as clusterless

    current = clusterless.fit_clusterless_mark_encoding
    if getattr(current, _PATCH_MARKER, False):
        previous = getattr(current, "__wrapped__", None)
        if previous is not None:
            _synchronize_aliases(previous, current, "fit_clusterless_mark_encoding")
    else:
        previous = current

        @wraps(previous)
        def fit_clusterless_mark_encoding(session, config=None):
            _validate_nested_encoding_config(config)
            _validate_clusterless_mark_config(config)
            return previous(session, config)

        setattr(fit_clusterless_mark_encoding, _PATCH_MARKER, True)
        clusterless.fit_clusterless_mark_encoding = fit_clusterless_mark_encoding
        _synchronize_aliases(previous, fit_clusterless_mark_encoding, "fit_clusterless_mark_encoding")

    _patch_clusterless_emission_config(clusterless)
    _patch_benchmark_clusterless_mark_config()


def _patch_clusterless_emission_config(clusterless) -> None:
    """Reject lossy clusterless emission calibration values before scoring."""

    current = clusterless.build_clusterless_mark_emissions
    if getattr(current, _EMISSION_CONFIG_PATCH_MARKER, False):
        previous = getattr(current, "__wrapped__", None)
        if previous is not None:
            _synchronize_aliases(previous, current, "build_clusterless_mark_emissions")
        return

    previous = current

    @wraps(previous)
    def build_clusterless_mark_emissions(session, encoding, ripple, config=None):
        _validate_clusterless_emission_config(config)
        return previous(session, encoding, ripple, config)

    setattr(build_clusterless_mark_emissions, _EMISSION_CONFIG_PATCH_MARKER, True)
    clusterless.build_clusterless_mark_emissions = build_clusterless_mark_emissions
    _synchronize_aliases(previous, build_clusterless_mark_emissions, "build_clusterless_mark_emissions")


def _patch_benchmark_clusterless_mark_config() -> None:
    """Validate raw BenchmarkConfig clusterless fields before scalar coercion."""

    benchmarks = sys.modules.get("hipporeplayimm.benchmarks")
    if benchmarks is None:
        return

    current = benchmarks._clusterless_mark_config
    if getattr(current, _BENCHMARK_MARK_CONFIG_PATCH_MARKER, False):
        return

    previous = current

    @wraps(previous)
    def _clusterless_mark_config(config):
        _validate_benchmark_clusterless_mark_config(config)
        return previous(config)

    setattr(_clusterless_mark_config, _BENCHMARK_MARK_CONFIG_PATCH_MARKER, True)
    setattr(_clusterless_mark_config, "__hipporeplayimm_original__", previous)
    benchmarks._clusterless_mark_config = _clusterless_mark_config


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


def _validate_clusterless_emission_config(config: object | None) -> None:
    if config is None:
        return
    _finite_config_value(config, "spike_rate_scale", positive=True)
    _finite_config_value(config, "likelihood_temperature", positive=True)
    _finite_config_value(config, "negative_binomial_overdispersion", positive=False)


def _validate_benchmark_clusterless_mark_config(config: object | None) -> None:
    if config is None:
        return
    _finite_config_value(
        config,
        "clusterless_mark_smoothing_sigma_bins",
        positive=False,
        message_name="mark_smoothing_sigma_bins",
    )
    _finite_config_value(
        config,
        "clusterless_mark_prior_count",
        positive=False,
        message_name="mark_prior_count",
    )
    _finite_config_value(
        config,
        "clusterless_mark_variance_floor",
        positive=True,
        message_name="mark_variance_floor",
    )
    _finite_config_value(
        config,
        "clusterless_rate_floor_hz",
        positive=True,
        message_name="rate_floor_hz",
    )
    _finite_config_value(
        config,
        "clusterless_mark_kde_bandwidth",
        positive=True,
        optional=True,
        message_name="mark_kde_bandwidth",
    )
    _finite_config_value(
        config,
        "clusterless_mark_kde_spatial_sigma_bins",
        positive=False,
        optional=True,
        message_name="mark_kde_spatial_sigma_bins",
    )
    _positive_integer_config_value(
        config,
        "clusterless_mark_kde_max_neighbors",
        message_name="mark_kde_max_neighbors",
    )


def _finite_config_value(
    config: object,
    name: str,
    *,
    positive: bool,
    optional: bool = False,
    message_name: str | None = None,
) -> float | None:
    message = _NUMERIC_MESSAGES[message_name or name]
    value = getattr(config, name, None)
    if value is None:
        if optional:
            return None
        raise ValueError(message)
    item = _scalar_config_item(value, message)
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0 or (positive and numeric <= 0.0):
        raise ValueError(message)
    return numeric


def _positive_integer_config_value(config: object, name: str, message_name: str | None = None) -> int:
    message = _NUMERIC_MESSAGES[message_name or name]
    value = getattr(config, name, None)
    item = _scalar_config_item(value, message)
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(message)
    return int(numeric)


def _scalar_config_item(value: object, message: str) -> object:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0:
        raise ValueError(message)
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(message)
    try:
        item = array.item()
    except ValueError as exc:
        raise ValueError(message) from exc
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(message)
    return item


def _synchronize_aliases(previous: object, patched: object, name: str) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, name, None) is previous:
            setattr(module, name, patched)


__all__ = ["apply_clusterless_encoding_config_validation_patch"]
