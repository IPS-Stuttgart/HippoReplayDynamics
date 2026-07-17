"""Validate state-space decoder score inputs before scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_bin_center_validation_patch_applied"
_SPARSE_MOMENTUM_PATCHED_FLAG = "_sparse_momentum_bin_center_validation_patch_applied"
_CORE_MODEL_PATCHED_FLAG = "_core_model_bin_center_validation_patch_applied"
_CORE_DIFFUSION_TRANSITION_PATCHED_FLAG = (
    "_core_diffusion_transition_scaling_patch_applied"
)
_CORE_GAUSSIAN_LOG_PROB_PATCHED_FLAG = "_core_gaussian_log_prob_scaling_patch_applied"
_CORE_PAIRWISE_GAUSSIAN_LOG_PROB_PATCHED_FLAG = (
    "_core_pairwise_gaussian_log_prob_scaling_patch_applied"
)
_BOOL_OR_TEXT_DTYPE_KINDS = {"b", "S", "U"}


def _validate_state_space_log_likelihood(emissions: Any) -> None:
    """Reject malformed state-space emissions before candidate selection."""

    try:
        values = np.asarray(emissions.log_likelihood, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("emissions.log_likelihood must contain numeric real values") from exc
    if values.ndim != 2:
        raise ValueError("emissions.log_likelihood must be two-dimensional")
    if values.shape[0] == 0:
        raise ValueError("emissions must contain at least one time bin")
    if values.shape[1] == 0:
        raise ValueError("emissions must contain at least one spatial bin")
    expected_shape = (int(emissions.n_time), int(emissions.n_bins))
    if values.shape != expected_shape:
        raise ValueError("emissions.log_likelihood shape must match emissions.n_time and emissions.n_bins")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("emissions.log_likelihood must not contain NaN or +inf")
    if not np.all(np.any(np.isfinite(values), axis=1)):
        raise ValueError("every emission row must contain at least one finite spatial-bin log likelihood")


def _as_numeric_real_coordinates(values: Any, name: str) -> np.ndarray:
    """Coerce coordinates without silently changing boolean, text, or complex data."""

    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric real coordinates") from exc
    if raw.dtype.kind in _BOOL_OR_TEXT_DTYPE_KINDS:
        raise ValueError(f"{name} must contain numeric real coordinates, not boolean or text")
    if raw.dtype.kind == "c":
        raise ValueError(f"{name} must contain numeric real coordinates, not complex values")
    if raw.dtype.kind == "O":
        for item in raw.ravel():
            if isinstance(item, (bool, np.bool_, str, bytes, np.str_, np.bytes_)):
                raise ValueError(f"{name} must contain numeric real coordinates, not boolean or text")
            if isinstance(item, (complex, np.complexfloating)):
                raise ValueError(f"{name} must contain numeric real coordinates, not complex values")
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric real coordinates") from exc


def _coerce_state_space_bin_centers(bin_centers: Any, n_bins: int) -> np.ndarray:
    """Return finite 2D bin centers with one row per spatial bin."""

    centers = _as_numeric_real_coordinates(bin_centers, "bin_centers")
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2 or centers.shape[1] == 0:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if centers.shape[0] != int(n_bins):
        raise ValueError("bin_centers must contain one row per emission spatial bin")
    if not np.all(np.isfinite(centers)):
        raise ValueError("bin_centers must be finite")
    return centers


def _standardized_coordinate_delta(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Compute ``abs(left - right) / sigma`` without finite subtraction overflow."""

    same_sign = np.signbit(left) == np.signbit(right)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        direct_delta = np.abs(left - right) / sigma
        split_delta = np.abs(left) / sigma + np.abs(right) / sigma
    return np.where(same_sign, direct_delta, split_delta)


def _scaled_distance(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    *,
    axis: int,
) -> np.ndarray:
    """Return Euclidean distance in sigma units without raw-square overflow."""

    standardized_delta = _standardized_coordinate_delta(left, right, sigma)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.hypot.reduce(standardized_delta, axis=axis)


def _patch_core_model_bin_center_validation() -> None:
    """Allow public core replay models to score vector-shaped 1D grids."""

    from . import models

    previous_validate = models._validate_score_inputs
    if getattr(previous_validate, _CORE_MODEL_PATCHED_FLAG, False):
        return

    def validate_score_inputs(emissions, bin_centers):
        centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
        return previous_validate(emissions, centers)

    validate_score_inputs.__name__ = getattr(previous_validate, "__name__", "_validate_score_inputs")
    validate_score_inputs.__doc__ = getattr(previous_validate, "__doc__", None)
    validate_score_inputs.__module__ = getattr(previous_validate, "__module__", __name__)
    setattr(validate_score_inputs, _CORE_MODEL_PATCHED_FLAG, True)
    setattr(validate_score_inputs, "__hipporeplayimm_original__", previous_validate)
    models._validate_score_inputs = validate_score_inputs


def _patch_core_diffusion_transition_scaling() -> None:
    """Compute core diffusion distances in sigma units to avoid overflow."""

    from . import models

    previous_transition = models._log_transition_matrix
    if getattr(previous_transition, _CORE_DIFFUSION_TRANSITION_PATCHED_FLAG, False):
        return

    def log_transition_matrix(bin_centers, sigma_cm, max_step_sigma):
        centers = np.asarray(bin_centers, dtype=float)
        sigma = float(sigma_cm)
        max_step = float(max_step_sigma)
        output: list[tuple[np.ndarray, np.ndarray]] = []
        for center in centers:
            standardized_distance = _scaled_distance(
                centers,
                center[None, :],
                sigma,
                axis=1,
            )
            keep = standardized_distance <= max_step
            if not np.any(keep):
                keep[int(np.argmin(standardized_distance))] = True
            indices = np.flatnonzero(keep)
            with np.errstate(over="ignore", invalid="ignore"):
                log_weights = -0.5 * np.square(standardized_distance[indices])
            log_weights -= models.logsumexp(log_weights)
            output.append((indices, log_weights))
        return output

    log_transition_matrix.__name__ = getattr(
        previous_transition,
        "__name__",
        "_log_transition_matrix",
    )
    log_transition_matrix.__doc__ = getattr(previous_transition, "__doc__", None)
    log_transition_matrix.__module__ = getattr(previous_transition, "__module__", __name__)
    setattr(log_transition_matrix, _CORE_DIFFUSION_TRANSITION_PATCHED_FLAG, True)
    setattr(log_transition_matrix, "__hipporeplayimm_original__", previous_transition)
    models._log_transition_matrix = log_transition_matrix


def _patch_core_gaussian_log_prob_scaling() -> None:
    """Stabilize candidate-model Gaussian kernels at large coordinate scales."""

    from . import models

    previous_gaussian = models._gaussian_log_prob
    if not getattr(previous_gaussian, _CORE_GAUSSIAN_LOG_PROB_PATCHED_FLAG, False):

        def gaussian_log_prob(predicted, observed, sigma):
            predicted_points = np.asarray(predicted, dtype=float)
            observed_points = np.asarray(observed, dtype=float)
            standardized_distance = _scaled_distance(
                observed_points,
                predicted_points,
                float(sigma),
                axis=1,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                return -0.5 * np.square(standardized_distance)

        gaussian_log_prob.__name__ = getattr(
            previous_gaussian,
            "__name__",
            "_gaussian_log_prob",
        )
        gaussian_log_prob.__doc__ = getattr(previous_gaussian, "__doc__", None)
        gaussian_log_prob.__module__ = getattr(previous_gaussian, "__module__", __name__)
        setattr(gaussian_log_prob, _CORE_GAUSSIAN_LOG_PROB_PATCHED_FLAG, True)
        setattr(gaussian_log_prob, "__hipporeplayimm_original__", previous_gaussian)
        models._gaussian_log_prob = gaussian_log_prob

    previous_pairwise = models._pairwise_gaussian_log_prob
    if not getattr(
        previous_pairwise,
        _CORE_PAIRWISE_GAUSSIAN_LOG_PROB_PATCHED_FLAG,
        False,
    ):

        def pairwise_gaussian_log_prob(predicted, observed, sigma):
            predicted_points = np.asarray(predicted, dtype=float)
            observed_points = np.asarray(observed, dtype=float)
            standardized_distance = _scaled_distance(
                predicted_points[:, None, :],
                observed_points[None, :, :],
                float(sigma),
                axis=2,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                return -0.5 * np.square(standardized_distance)

        pairwise_gaussian_log_prob.__name__ = getattr(
            previous_pairwise,
            "__name__",
            "_pairwise_gaussian_log_prob",
        )
        pairwise_gaussian_log_prob.__doc__ = getattr(previous_pairwise, "__doc__", None)
        pairwise_gaussian_log_prob.__module__ = getattr(
            previous_pairwise,
            "__module__",
            __name__,
        )
        setattr(
            pairwise_gaussian_log_prob,
            _CORE_PAIRWISE_GAUSSIAN_LOG_PROB_PATCHED_FLAG,
            True,
        )
        setattr(
            pairwise_gaussian_log_prob,
            "__hipporeplayimm_original__",
            previous_pairwise,
        )
        models._pairwise_gaussian_log_prob = pairwise_gaussian_log_prob


def _patch_sparse_momentum_bin_center_validation() -> None:
    """Install the same validation on the direct exact-sparse momentum helper."""

    from . import state_space as ss
    from . import state_space_sparse_momentum as sparse_momentum

    previous_score = sparse_momentum._score_sparse_momentum_exact
    if getattr(previous_score, _SPARSE_MOMENTUM_PATCHED_FLAG, False):
        return

    def score_sparse_momentum_exact(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory: bool = True,
    ):
        _validate_state_space_log_likelihood(emissions)
        centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
        return previous_score(
            emissions,
            centers,
            config,
            transition_durations_s,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )

    score_sparse_momentum_exact.__name__ = getattr(
        previous_score,
        "__name__",
        "_score_sparse_momentum_exact",
    )
    score_sparse_momentum_exact.__doc__ = getattr(previous_score, "__doc__", None)
    score_sparse_momentum_exact.__module__ = getattr(previous_score, "__module__", __name__)
    setattr(score_sparse_momentum_exact, _SPARSE_MOMENTUM_PATCHED_FLAG, True)
    setattr(score_sparse_momentum_exact, "__hipporeplayimm_original__", previous_score)
    sparse_momentum._score_sparse_momentum_exact = score_sparse_momentum_exact
    if getattr(ss, "_score_sparse_momentum_exact", None) is previous_score:
        ss._score_sparse_momentum_exact = score_sparse_momentum_exact


def apply_state_space_bin_center_validation_patch() -> None:
    """Install state-space score input validation for ``StateSpaceReplayModel.score``."""

    from . import state_space as ss

    _patch_core_model_bin_center_validation()
    _patch_core_diffusion_transition_scaling()
    _patch_core_gaussian_log_prob_scaling()

    if not getattr(ss.StateSpaceReplayModel.score, _PATCHED_FLAG, False):
        previous_score = ss.StateSpaceReplayModel.score

        def score(self, emissions, bin_centers, *args, **kwargs):
            _validate_state_space_log_likelihood(emissions)
            centers = _coerce_state_space_bin_centers(bin_centers, emissions.n_bins)
            return previous_score(self, emissions, centers, *args, **kwargs)

        score.__name__ = getattr(previous_score, "__name__", "score")
        score.__doc__ = getattr(previous_score, "__doc__", None)
        score.__module__ = getattr(previous_score, "__module__", __name__)
        setattr(score, _PATCHED_FLAG, True)
        if getattr(previous_score, "_native_duration_occupancy_aware", False):
            setattr(score, "_native_duration_occupancy_aware", True)
        setattr(score, "__hipporeplayimm_original__", previous_score)
        ss.StateSpaceReplayModel.score = score

    _patch_sparse_momentum_bin_center_validation()


__all__ = ["apply_state_space_bin_center_validation_patch"]
