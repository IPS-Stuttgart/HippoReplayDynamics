"Runtime fixes for accuracy-upgrade diagnostics, emissions, and metadata."

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .evidence_status_coercion import _normalize_status_value

_STATUS_PATCHED_FLAG = "_model_probability_status_patch_applied"
_REVERSE_PATCHED_FLAG = "_accuracy_reverse_duration_patch_applied"
_REVERSE_ORIGINAL_ATTR = "_accuracy_reverse_duration_original"
_VALID_STATE_MASK_PATCHED_FLAG = "_accuracy_valid_state_mask_config_patch_applied"
_VALID_STATE_MASK_ORIGINAL_ATTR = "_accuracy_valid_state_mask_config_original"
_ENSEMBLE_PATCHED_FLAG = "_weighted_ensemble_observation_metadata_patch_applied"
_ENSEMBLE_ORIGINAL_ATTR = "_weighted_ensemble_observation_metadata_original"


def apply_model_probability_status_patch() -> None:
    """Install accuracy-upgrade runtime fixes."""

    from . import accuracy_upgrades

    _patch_model_probability_diagnostics(accuracy_upgrades)
    _patch_valid_state_mask_from_encoding(accuracy_upgrades)
    _patch_reverse_emissions(accuracy_upgrades)
    _patch_weighted_ensemble_emissions(accuracy_upgrades)


def _patch_model_probability_diagnostics(accuracy_upgrades) -> None:
    """Install input normalization for accuracy-upgrade probability summaries."""

    if getattr(accuracy_upgrades, _STATUS_PATCHED_FLAG, False):
        return

    original = accuracy_upgrades.model_probability_diagnostics

    @wraps(original)
    def model_probability_diagnostics(
        scores: pd.DataFrame,
        *,
        evidence_column: str = "log_evidence",
        group_columns=("session", "event_index"),
    ) -> pd.DataFrame:
        normalized = scores
        if not scores.empty and ("status" in scores.columns or evidence_column in scores.columns):
            normalized = scores.copy()
            if "status" in normalized.columns:
                normalized["status"] = normalized["status"].map(_normalize_status_value)
            if evidence_column in normalized.columns:
                normalized[evidence_column] = pd.to_numeric(normalized[evidence_column], errors="coerce")
                normalized = normalized.dropna(subset=[evidence_column])
        return _model_probability_diagnostics_with_nonfinite_evidence(
            accuracy_upgrades,
            normalized,
            evidence_column=evidence_column,
            group_columns=group_columns,
        )

    accuracy_upgrades.model_probability_diagnostics = model_probability_diagnostics
    setattr(accuracy_upgrades, _STATUS_PATCHED_FLAG, True)


def _model_probability_diagnostics_with_nonfinite_evidence(
    accuracy_upgrades,
    scores: pd.DataFrame,
    *,
    evidence_column: str,
    group_columns,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for key, group in scores.groupby(list(group_columns), sort=False):
        ok = group.copy()
        if "status" in ok:
            ok = ok[ok["status"].eq("success")]
        if "evidence_comparable" in ok:
            ok = ok[accuracy_upgrades._coerce_bool_series(ok["evidence_comparable"])]
        if ok.empty:
            continue

        ok = ok.dropna(subset=[evidence_column])
        values = ok[evidence_column].to_numpy(dtype=float)
        if values.size == 0:
            continue

        ordered = np.sort(values)[::-1]
        probs = _stable_log_evidence_probabilities(values)
        best_idx = _best_evidence_index(values)
        row = accuracy_upgrades._group_key_dict(group_columns, key)
        row.update(
            {
                "models": int(values.size),
                "best_model": str(ok.iloc[best_idx]["model"]),
                "best_log_evidence": float(ordered[0]),
                "evidence_margin_to_second_best": _evidence_margin_to_second_best(ordered),
                "model_probability_entropy": _probability_entropy(probs),
                "best_model_probability": float(np.max(probs)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _stable_log_evidence_probabilities(values: np.ndarray) -> np.ndarray:
    evidence = np.asarray(values, dtype=float)
    probabilities = np.zeros(evidence.shape, dtype=float)

    positive_infinite = np.isposinf(evidence)
    if np.any(positive_infinite):
        probabilities[positive_infinite] = 1.0 / float(np.sum(positive_infinite))
        return probabilities

    finite = np.isfinite(evidence)
    if np.any(finite):
        normalizer = float(logsumexp(evidence[finite]))
        probabilities[finite] = np.exp(evidence[finite] - normalizer)
        return probabilities

    negative_infinite = np.isneginf(evidence)
    if np.any(negative_infinite):
        probabilities[negative_infinite] = 1.0 / float(np.sum(negative_infinite))
        return probabilities

    return np.full(evidence.shape, np.nan, dtype=float)


def _best_evidence_index(values: np.ndarray) -> int:
    evidence = np.asarray(values, dtype=float)
    positive_infinite = np.flatnonzero(np.isposinf(evidence))
    if positive_infinite.size:
        return int(positive_infinite[0])
    finite = np.flatnonzero(np.isfinite(evidence))
    if finite.size:
        return int(finite[int(np.argmax(evidence[finite]))])
    negative_infinite = np.flatnonzero(np.isneginf(evidence))
    if negative_infinite.size:
        return int(negative_infinite[0])
    return int(np.nanargmax(evidence))


def _evidence_margin_to_second_best(ordered: np.ndarray) -> float:
    if ordered.size <= 1:
        return float("nan")
    best = float(ordered[0])
    second = float(ordered[1])
    if np.isposinf(best) and np.isposinf(second):
        return 0.0
    if np.isneginf(best) and np.isneginf(second):
        return 0.0
    return float(best - second)


def _probability_entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    positive = probs > 0.0
    if not np.any(positive):
        return float("nan")
    return float(-np.sum(probs[positive] * np.log(probs[positive])))


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _finite_float_scalar(name: str, value: object, invalid_message: str) -> float:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(invalid_message) from exc
    if array.ndim != 0:
        raise ValueError(invalid_message)
    try:
        numeric = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(invalid_message) from exc
    if not np.isfinite(numeric):
        raise ValueError(invalid_message)
    return numeric


def _validate_finite_nonnegative_numeric_parameter(name: str, value: object) -> float:
    numeric = _finite_float_scalar(
        name,
        value,
        f"{name} must be finite and nonnegative",
    )
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _validate_optional_fraction_parameter(name: str, value: object | None) -> None:
    if value is None:
        return
    numeric = _finite_float_scalar(
        name,
        value,
        f"{name} must lie in (0, 1]",
    )
    if not 0.0 < numeric <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")


def _validate_boolean_parameter(name: str, value: object) -> None:
    if not _is_boolean_scalar(value):
        raise TypeError(f"{name} must be boolean")


def _patch_valid_state_mask_from_encoding(accuracy_upgrades) -> None:
    """Reject ambiguous valid-state mask parameters before support construction."""

    current = accuracy_upgrades.valid_state_mask_from_encoding
    if getattr(current, _VALID_STATE_MASK_PATCHED_FLAG, False):
        return

    @wraps(current)
    def valid_state_mask_from_encoding(encoding, config=None):
        if config is not None:
            _validate_finite_nonnegative_numeric_parameter("min_occupancy_s", getattr(config, "min_occupancy_s"))
            _validate_optional_fraction_parameter(
                "keep_top_occupancy_fraction",
                getattr(config, "keep_top_occupancy_fraction", None),
            )
            _validate_boolean_parameter("require_finite_rates", getattr(config, "require_finite_rates"))
        return current(encoding, config)

    setattr(valid_state_mask_from_encoding, _VALID_STATE_MASK_PATCHED_FLAG, True)
    setattr(valid_state_mask_from_encoding, _VALID_STATE_MASK_ORIGINAL_ATTR, current)
    accuracy_upgrades.valid_state_mask_from_encoding = valid_state_mask_from_encoding


def _patch_reverse_emissions(accuracy_upgrades) -> None:
    """Build reversed accuracy-upgrade emissions with duration metadata attached."""

    current = accuracy_upgrades.reverse_emissions
    if getattr(current, _REVERSE_PATCHED_FLAG, False):
        return

    @wraps(current)
    def reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
        bin_durations = _reversed_duration_vector(
            getattr(emissions, "bin_durations", None),
            expected_length=emissions.n_time,
            name="bin_durations",
            fallback=float(emissions.dt),
        )
        transition_durations = _reversed_transition_durations(emissions)
        return LogEmissionTensor(
            log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[::-1].copy(),
            spike_counts=np.asarray(emissions.spike_counts)[::-1].copy(),
            times=np.asarray(emissions.times, dtype=float)[::-1].copy(),
            dt=emissions.dt,
            cell_ids=np.asarray(emissions.cell_ids).copy(),
            n_spikes=int(emissions.n_spikes),
            bin_durations=bin_durations,
            transition_durations=transition_durations,
            metadata=dict(getattr(emissions, "metadata", {}) or {}),
        )

    setattr(reverse_emissions, _REVERSE_PATCHED_FLAG, True)
    setattr(reverse_emissions, _REVERSE_ORIGINAL_ATTR, current)
    accuracy_upgrades.reverse_emissions = reverse_emissions


def _patch_weighted_ensemble_emissions(accuracy_upgrades) -> None:
    """Keep observation metadata consistent for weighted ensemble emissions."""

    current = accuracy_upgrades.weighted_ensemble_emissions
    if getattr(current, _ENSEMBLE_PATCHED_FLAG, False):
        return

    @wraps(current)
    def weighted_ensemble_emissions(
        left: LogEmissionTensor,
        right: LogEmissionTensor,
        *,
        alpha: float = 0.5,
    ) -> LogEmissionTensor:
        alpha = _finite_float_scalar(
            "alpha",
            alpha,
            "alpha must be finite and lie in [0, 1]",
        )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")

        left_likelihood = np.asarray(left.log_likelihood, dtype=float)
        right_likelihood = np.asarray(right.log_likelihood, dtype=float)
        if left_likelihood.shape != right_likelihood.shape:
            raise ValueError("emission tensors must have matching log_likelihood shapes")

        spike_counts = np.asarray(left.spike_counts).copy()
        n_spikes = int(getattr(left, "n_spikes", np.asarray(spike_counts, dtype=float).sum()))
        n_time = left_likelihood.shape[0]
        bin_durations = _copied_duration_vector(
            getattr(left, "bin_durations", None),
            expected_length=n_time,
            name="bin_durations",
        )
        transition_durations = _copied_duration_vector(
            getattr(left, "transition_durations", None),
            expected_length=max(n_time - 1, 0),
            name="transition_durations",
        )
        return LogEmissionTensor(
            log_likelihood=alpha * left_likelihood + (1.0 - alpha) * right_likelihood,
            spike_counts=spike_counts,
            times=np.asarray(left.times, dtype=float).copy(),
            dt=left.dt,
            cell_ids=np.asarray(left.cell_ids).copy(),
            n_spikes=n_spikes,
            bin_durations=bin_durations,
            transition_durations=transition_durations,
            metadata={
                "emission_model": "weighted-product-ensemble",
                "ensemble_alpha_left": alpha,
            },
        )

    setattr(weighted_ensemble_emissions, _ENSEMBLE_PATCHED_FLAG, True)
    setattr(weighted_ensemble_emissions, _ENSEMBLE_ORIGINAL_ATTR, current)
    accuracy_upgrades.weighted_ensemble_emissions = weighted_ensemble_emissions


def _reversed_transition_durations(emissions: LogEmissionTensor) -> np.ndarray:
    expected_length = max(emissions.n_time - 1, 0)
    values = getattr(emissions, "transition_durations", None)
    if values is not None:
        return _reversed_duration_vector(
            values,
            expected_length=expected_length,
            name="transition_durations",
        )

    times = np.asarray(getattr(emissions, "times", ()), dtype=float)
    if times.shape == (emissions.n_time,) and emissions.n_time > 1:
        durations = np.diff(times)
        if np.all(np.isfinite(durations)) and np.all(durations > 0.0):
            return durations[::-1].copy()

    return _reversed_duration_vector(
        None,
        expected_length=expected_length,
        name="transition_durations",
        fallback=float(emissions.dt),
    )


def _reversed_duration_vector(
    values: object,
    *,
    expected_length: int,
    name: str,
    fallback: float | None = None,
) -> np.ndarray:
    if values is None:
        if fallback is None:
            return np.empty(int(expected_length), dtype=float)
        return np.full(int(expected_length), float(fallback), dtype=float)

    array = np.asarray(values, dtype=float)
    if array.shape != (int(expected_length),):
        raise ValueError(
            f"{name} must contain {int(expected_length)} values; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive durations")
    return array[::-1].copy()


def _copied_duration_vector(
    values: object,
    *,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (int(expected_length),):
        raise ValueError(
            f"{name} must contain {int(expected_length)} values; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive durations")
    return array.copy()


__all__ = ["apply_model_probability_status_patch"]
