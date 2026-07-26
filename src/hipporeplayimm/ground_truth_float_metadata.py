"""Strict numeric and boolean parsing for score-table metadata."""

from __future__ import annotations

from functools import wraps
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

_FLOAT_PATCHED_FLAG = "_ground_truth_strict_float_metadata_patch_applied"
_BOOL_PATCHED_FLAG = "_ground_truth_strict_bool_metadata_patch_applied"
_CONFIG_PATCHED_FLAG = "_ground_truth_config_numeric_validation_patch_applied"
_DIRECT_VISIT_PATCHED_FLAG = "_ground_truth_direct_visit_numeric_validation_patch_applied"
_DIRECT_WELL_WINDOW_PATCHED_FLAG = "_ground_truth_direct_well_window_numeric_validation_patch_applied"
_SCORE_BOOL_PATCHED_FLAG = "_score_metadata_strict_bool_metadata_patch_applied"
_EVIDENCE_BOOL_PATCHED_FLAG = "_evidence_reporting_strict_bool_metadata_patch_applied"


def apply_ground_truth_float_metadata_patch() -> None:
    """Reject malformed numeric/boolean metadata instead of propagating invalid configs."""

    from . import ground_truth as gt

    if not getattr(gt, _FLOAT_PATCHED_FLAG, False):

        def unique_float_from_columns(frame: Any, columns: tuple[str, ...], default: float) -> float:
            values = [
                _parse_float_metadata_value(" / ".join(columns), value)
                for value in gt._iter_present_column_values(frame, columns)
            ]
            if not values:
                return float(default)
            first = values[0]
            if any(not np.isclose(value, first, rtol=1e-05, atol=1e-08) for value in values[1:]):
                raise ValueError(f"{' / '.join(columns)} contains multiple values")
            return float(first)

        def unique_optional_float_from_column(frame: Any, column: str, default: float | None) -> float | None:
            values = [
                _parse_float_metadata_value(column, value)
                for value in gt._iter_present_column_values(frame, (column,))
            ]
            if not values:
                return default
            first = values[0]
            if any(not np.isclose(value, first, rtol=1e-05, atol=1e-08) for value in values[1:]):
                raise ValueError(f"{column} contains multiple values")
            return float(first)

        unique_float_from_columns.__name__ = gt._unique_float_from_columns.__name__
        unique_float_from_columns.__doc__ = gt._unique_float_from_columns.__doc__
        unique_optional_float_from_column.__name__ = gt._unique_optional_float_from_column.__name__
        unique_optional_float_from_column.__doc__ = gt._unique_optional_float_from_column.__doc__
        gt._unique_float_from_columns = unique_float_from_columns
        gt._unique_optional_float_from_column = unique_optional_float_from_column
        setattr(gt, _FLOAT_PATCHED_FLAG, True)

    if not getattr(gt, _BOOL_PATCHED_FLAG, False):

        def unique_bool_from_column(frame: Any, column: str, default: bool) -> bool:
            values = [
                _parse_bool_metadata_value(column, value)
                for value in gt._iter_present_column_values(frame, (column,))
            ]
            if not values:
                return bool(default)
            first = values[0]
            if any(value != first for value in values[1:]):
                raise ValueError(f"{column} contains multiple values")
            return bool(first)

        def parse_bool(value: Any) -> bool:
            return _parse_bool_metadata_value("boolean metadata", value)

        unique_bool_from_column.__name__ = gt._unique_bool_from_column.__name__
        unique_bool_from_column.__doc__ = gt._unique_bool_from_column.__doc__
        parse_bool.__name__ = gt._parse_bool.__name__
        parse_bool.__doc__ = gt._parse_bool.__doc__
        gt._unique_bool_from_column = unique_bool_from_column
        gt._parse_bool = parse_bool
        setattr(gt, _BOOL_PATCHED_FLAG, True)

    if not getattr(gt, _CONFIG_PATCHED_FLAG, False):
        original_ground_truth_config_init = gt.GroundTruthConfig.__init__
        original_sensitivity_config_init = gt.GroundTruthSensitivityConfig.__init__
        original_ground_truth_configs = gt.GroundTruthSensitivityConfig.ground_truth_configs

        @wraps(original_ground_truth_config_init)
        def ground_truth_config_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_ground_truth_config_init(self, *args, **kwargs)
            _validate_ground_truth_config(self)

        @wraps(original_sensitivity_config_init)
        def sensitivity_config_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_sensitivity_config_init(self, *args, **kwargs)
            _validate_ground_truth_sensitivity_config(self)

        @wraps(original_ground_truth_configs)
        def ground_truth_configs(self: Any):
            visit_radii_cm = _parse_positive_config_sequence(
                "visit_radii_cm",
                self.visit_radii_cm,
            )
            min_dwells_s = _parse_nonnegative_config_sequence(
                "min_dwells_s",
                self.min_dwells_s,
            )
            future_horizons_s = _parse_positive_config_sequence(
                "future_horizons_s",
                self.future_horizons_s,
            )
            well_arrival_window_s = _parse_positive_config_value(
                "well_arrival_window_s",
                self.well_arrival_window_s,
            )
            return tuple(
                gt.GroundTruthConfig(
                    well_arrival_window_s=well_arrival_window_s,
                    visit_radius_cm=float(visit_radius_cm),
                    min_dwell_s=float(min_dwell_s),
                    future_horizon_s=float(future_horizon_s),
                    event_epoch=self.event_epoch,
                )
                for visit_radius_cm, min_dwell_s, future_horizon_s in product(
                    visit_radii_cm,
                    min_dwells_s,
                    future_horizons_s,
                )
            )

        gt.GroundTruthConfig.__init__ = ground_truth_config_init
        gt.GroundTruthSensitivityConfig.__init__ = sensitivity_config_init
        gt.GroundTruthSensitivityConfig.ground_truth_configs = ground_truth_configs
        setattr(gt, _CONFIG_PATCHED_FLAG, True)

    _patch_direct_ground_truth_numeric_helpers(gt)

    from . import score_metadata as score_meta

    if not getattr(score_meta, _SCORE_BOOL_PATCHED_FLAG, False):

        def parse_score_bool(value: Any) -> bool:
            return _parse_bool_metadata_value("boolean metadata", value)

        parse_score_bool.__name__ = score_meta._parse_bool.__name__
        parse_score_bool.__doc__ = score_meta._parse_bool.__doc__
        score_meta._parse_bool = parse_score_bool
        setattr(score_meta, _SCORE_BOOL_PATCHED_FLAG, True)

    from . import evidence_reporting as evidence

    if not getattr(evidence, _EVIDENCE_BOOL_PATCHED_FLAG, False):

        def coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
            def coerce(value: object) -> bool:
                return _parse_bool_metadata_value_or_default(value, default=default)

            return values.map(coerce).astype(bool)

        coerce_bool_series.__name__ = evidence._coerce_bool_series.__name__
        coerce_bool_series.__doc__ = evidence._coerce_bool_series.__doc__
        evidence._coerce_bool_series = coerce_bool_series
        _synchronize_coerce_bool_series_aliases(coerce_bool_series)
        setattr(evidence, _EVIDENCE_BOOL_PATCHED_FLAG, True)


def _patch_direct_ground_truth_numeric_helpers(gt: Any) -> None:
    """Validate direct helper arguments that bypass dataclass constructors."""

    current_visit = gt.first_post_ripple_well_visit
    if not getattr(current_visit, _DIRECT_VISIT_PATCHED_FLAG, False):

        @wraps(current_visit)
        def first_post_ripple_well_visit(
            position: np.ndarray,
            wells: pd.DataFrame,
            ripple_peak: float,
            *,
            visit_radius_cm: float,
            min_dwell_s: float,
            future_horizon_s: float,
        ):
            peak = _parse_config_scalar("ripple_peak", ripple_peak)
            radius = _parse_positive_config_value("visit_radius_cm", visit_radius_cm)
            dwell = _parse_nonnegative_config_value("min_dwell_s", min_dwell_s)
            horizon = _parse_positive_config_value("future_horizon_s", future_horizon_s)
            return current_visit(
                position,
                wells,
                peak,
                visit_radius_cm=radius,
                min_dwell_s=dwell,
                future_horizon_s=horizon,
            )

        setattr(first_post_ripple_well_visit, _DIRECT_VISIT_PATCHED_FLAG, True)
        gt.first_post_ripple_well_visit = first_post_ripple_well_visit

    current_infer = gt.infer_well_locations_from_arrays
    if not getattr(current_infer, _DIRECT_WELL_WINDOW_PATCHED_FLAG, False):

        @wraps(current_infer)
        def infer_well_locations_from_arrays(
            position: np.ndarray,
            well_sequence: np.ndarray | None,
            well_arrival_window_s: float = 1.0,
        ):
            window = _parse_positive_config_value(
                "well_arrival_window_s",
                well_arrival_window_s,
            )
            return current_infer(
                position,
                well_sequence,
                well_arrival_window_s=window,
            )

        setattr(
            infer_well_locations_from_arrays,
            _DIRECT_WELL_WINDOW_PATCHED_FLAG,
            True,
        )
        gt.infer_well_locations_from_arrays = infer_well_locations_from_arrays


def _validate_ground_truth_config(config: Any) -> None:
    _parse_positive_config_value("well_arrival_window_s", config.well_arrival_window_s)
    _parse_positive_config_value("visit_radius_cm", config.visit_radius_cm)
    _parse_nonnegative_config_value("min_dwell_s", config.min_dwell_s)
    _parse_positive_config_value("future_horizon_s", config.future_horizon_s)


def _validate_ground_truth_sensitivity_config(config: Any) -> None:
    _parse_positive_config_sequence("visit_radii_cm", config.visit_radii_cm)
    _parse_nonnegative_config_sequence("min_dwells_s", config.min_dwells_s)
    _parse_positive_config_sequence("future_horizons_s", config.future_horizons_s)
    _parse_positive_config_value("well_arrival_window_s", config.well_arrival_window_s)


def _parse_positive_config_value(name: str, value: Any) -> float:
    numeric = _parse_config_scalar(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _parse_nonnegative_config_value(name: str, value: Any) -> float:
    numeric = _parse_config_scalar(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _parse_config_scalar(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a scalar") from exc
    if raw.ndim != 0:
        raise TypeError(f"{name} must be a scalar")
    if raw.dtype.kind in {"S", "U"}:
        raise TypeError(f"{name} must be numeric, not text")
    if raw.dtype == object and isinstance(raw.item(), (str, bytes, np.str_, np.bytes_)):
        raise TypeError(f"{name} must be numeric, not text")
    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return float(numeric)


def _parse_positive_config_sequence(name: str, values: Any) -> tuple[float, ...]:
    parsed = _parse_config_sequence(name, values)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    if any(value <= 0.0 for value in parsed):
        raise ValueError(f"{name} must contain finite positive values")
    return parsed


def _parse_nonnegative_config_sequence(name: str, values: Any) -> tuple[float, ...]:
    parsed = _parse_config_sequence(name, values)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    if any(value < 0.0 for value in parsed):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return parsed


def _parse_config_sequence(name: str, values: Any) -> tuple[float, ...]:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of numeric values") from exc
    if raw.ndim == 0:
        raise TypeError(f"{name} must be a sequence")
    if raw.size == 0:
        return ()
    return tuple(_parse_config_scalar(name, value) for value in raw.reshape(-1))


def _parse_float_metadata_value(column: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must contain finite numeric values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain finite numeric values") from exc
    if raw.ndim != 0:
        raise ValueError(f"{column} must contain finite numeric values")
    item = raw.item()
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{column} must contain finite numeric values")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{column} must contain finite numeric values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{column} must contain finite numeric values")
    return float(numeric)


def _parse_bool_metadata_value(column: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        raise ValueError(f"{column} must contain boolean values")
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{column} must contain boolean values") from exc
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "off"}:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{column} must contain boolean values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"cannot parse boolean value for {column}; {column} must contain boolean values")
    if np.isclose(numeric, 0.0, rtol=0.0, atol=0.0):
        return False
    if np.isclose(numeric, 1.0, rtol=0.0, atol=0.0):
        return True
    raise ValueError(f"cannot parse boolean value for {column}; {column} must contain boolean values")


def _parse_bool_metadata_value_or_default(value: Any, *, default: bool) -> bool:
    try:
        return _parse_bool_metadata_value("boolean metadata", value)
    except ValueError:
        return bool(default)


def _synchronize_coerce_bool_series_aliases(coerce_bool_series: Any) -> None:
    """Update package modules that imported the evidence bool helper by value."""

    import sys

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if hasattr(module, "_coerce_bool_series"):
            module._coerce_bool_series = coerce_bool_series


__all__ = ["apply_ground_truth_float_metadata_patch"]
