"""Runtime validation for clusterless nested encoding configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
import sys

import numpy as np

from .encoding import EncodingConfig, _validate_encoding_config
from .place_field_run_local_kinematics import (
    apply_clusterless_run_local_kinematics_patch,
)

_PATCH_MARKER = "_clusterless_encoding_config_validation_patch"
_MARK_VALUE_PATCH_MARKER = "_clusterless_mark_value_validation_patch"
_MARK_VALUE_PATCH_VERSION = 2
_BENCHMARK_MARK_CONFIG_PATCH_MARKER = "_benchmark_clusterless_mark_config_validation_patch"
_SCALED_RATE_PATCH_MARKER = "_clusterless_scaled_rate_validation_patch"
_SCALED_RATE_PATCH_VERSION = 1


_NUMERIC_MESSAGES = {
    "mark_smoothing_sigma_bins": "mark_smoothing_sigma_bins must be finite and nonnegative",
    "mark_prior_count": "mark_prior_count must be finite and nonnegative",
    "mark_variance_floor": "mark_variance_floor must be finite and positive",
    "rate_floor_hz": "rate_floor_hz must be finite and positive",
    "mark_kde_bandwidth": "mark_kde_bandwidth must be finite and positive when provided",
    "mark_kde_spatial_sigma_bins": "mark_kde_spatial_sigma_bins must be finite and nonnegative when provided",
    "mark_kde_max_neighbors": "mark_kde_max_neighbors must be a positive integer",
}
_BOOLEAN_MESSAGES = {
    "use_excitatory": "use_excitatory must be a boolean scalar",
}


def apply_clusterless_encoding_config_validation_patch() -> None:
    """Validate ClusterlessMarkConfig.encoding before fitting clusterless marks."""

    import hipporeplayimm.clusterless as clusterless

    apply_clusterless_run_local_kinematics_patch()
    _patch_clusterless_mark_value_validation(clusterless)
    _patch_clusterless_scaled_rate_validation()

    current = clusterless.fit_clusterless_mark_encoding
    if getattr(current, _PATCH_MARKER, False):
        previous = getattr(current, "__wrapped__", None)
        if previous is not None:
            _synchronize_aliases(previous, current)
    else:
        previous = current

        @wraps(previous)
        def fit_clusterless_mark_encoding(session, config=None):
            _validate_nested_encoding_config(config)
            _validate_clusterless_mark_config(config)
            return previous(session, config)

        setattr(fit_clusterless_mark_encoding, _PATCH_MARKER, True)
        clusterless.fit_clusterless_mark_encoding = fit_clusterless_mark_encoding
        _synchronize_aliases(previous, fit_clusterless_mark_encoding)

    _patch_benchmark_clusterless_mark_config()


def _contains_boolean_mark_values(values) -> bool:
    """Inspect the source representation before NumPy can promote booleans."""

    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return False
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype == object:
            return any(_contains_boolean_mark_values(value) for value in values.flat)
        return False
    if isinstance(values, Mapping):
        return any(_contains_boolean_mark_values(value) for value in values.values())
    if isinstance(values, (str, bytes, bytearray)):
        return False
    if isinstance(values, Iterable):
        return any(_contains_boolean_mark_values(value) for value in values)
    return False


def _validated_real_mark_values(marks):
    """Reject lossy mark coercions while accepting exactly real complex arrays."""

    if _contains_boolean_mark_values(marks):
        raise ValueError("marks must contain numeric values, not boolean values")

    try:
        raw = np.asarray(marks)
    except (TypeError, ValueError):
        return marks

    if np.issubdtype(raw.dtype, np.complexfloating):
        imaginary = np.imag(raw)
        if not np.all(np.isfinite(imaginary)) or np.any(imaginary != 0.0):
            raise ValueError("marks must contain real values")
        return np.real(raw)

    if raw.dtype != object:
        return marks

    converted = raw.copy()
    changed = False
    for index, value in np.ndenumerate(raw):
        if isinstance(value, (complex, np.complexfloating)):
            imaginary = float(np.imag(value))
            if not np.isfinite(imaginary) or imaginary != 0.0:
                raise ValueError("marks must contain real values")
            converted[index] = float(np.real(value))
            changed = True
    return converted if changed else marks


def _patch_clusterless_mark_value_validation(clusterless) -> None:
    """Reject malformed marks before evaluating clusterless likelihoods."""

    current = clusterless.ClusterlessMarkEncoding._coerce_marks
    if getattr(current, _MARK_VALUE_PATCH_MARKER, None) == _MARK_VALUE_PATCH_VERSION:
        return

    previous = current

    @wraps(previous)
    def _coerce_marks(self, marks):
        validated = _validated_real_mark_values(marks)
        coerced = previous(self, validated)
        if not np.all(np.isfinite(coerced)):
            raise ValueError("marks must contain finite values")
        return coerced

    setattr(_coerce_marks, _MARK_VALUE_PATCH_MARKER, _MARK_VALUE_PATCH_VERSION)
    setattr(_coerce_marks, "__hipporeplayimm_original__", previous)
    clusterless.ClusterlessMarkEncoding._coerce_marks = _coerce_marks


def _patch_clusterless_scaled_rate_validation() -> None:
    """Reject rate-scale products that overflow before likelihood evaluation."""

    from . import clusterless_mark_group_validation

    current = clusterless_mark_group_validation._scaled_log_rates
    if getattr(current, _SCALED_RATE_PATCH_MARKER, None) == _SCALED_RATE_PATCH_VERSION:
        return

    previous = current

    @wraps(previous)
    def _scaled_log_rates(values, scale, name):
        with np.errstate(over="ignore", invalid="ignore"):
            scaled, log_rates = previous(values, scale, name)
        if not np.all(np.isfinite(scaled)):
            raise ValueError(f"{name} becomes non-finite after spike-rate scaling")
        return scaled, log_rates

    setattr(_scaled_log_rates, _SCALED_RATE_PATCH_MARKER, _SCALED_RATE_PATCH_VERSION)
    setattr(_scaled_log_rates, "__hipporeplayimm_original__", previous)
    clusterless_mark_group_validation._scaled_log_rates = _scaled_log_rates


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
    _boolean_config_value(config, "use_excitatory")
    _finite_config_value(config, "mark_smoothing_sigma_bins", positive=False)
    _finite_config_value(config, "mark_prior_count", positive=False)
    _finite_config_value(config, "mark_variance_floor", positive=True)
    _finite_config_value(config, "rate_floor_hz", positive=True)
    _finite_config_value(config, "mark_kde_bandwidth", positive=True, optional=True)
    _finite_config_value(config, "mark_kde_spatial_sigma_bins", positive=False, optional=True)
    _positive_integer_config_value(config, "mark_kde_max_neighbors")


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


def _boolean_config_value(config: object, name: str) -> bool:
    """Return a strict Python/NumPy boolean scalar without truthiness coercion."""

    message = _BOOLEAN_MESSAGES[name]
    value = getattr(config, name, None)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(message)
    try:
        item = array.item()
    except ValueError as exc:
        raise ValueError(message) from exc
    if not isinstance(item, (bool, np.bool_)):
        raise ValueError(message)
    return bool(item)


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


def _synchronize_aliases(previous: object, patched: object) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "fit_clusterless_mark_encoding", None) is previous:
            module.fit_clusterless_mark_encoding = patched


__all__ = ["apply_clusterless_encoding_config_validation_patch"]
