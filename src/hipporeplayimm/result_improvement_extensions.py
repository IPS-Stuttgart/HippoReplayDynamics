"""Opt-in result-improvement helpers for replay model-evidence runs.

The utilities in this module are deliberately additive.  They expose modelling
and diagnostic upgrades without changing the default package entry points:

* replay-specific sorted-spike emission calibration;
* Gamma-Poisson / negative-binomial predictive emissions;
* spatial-bin permutation null controls;
* reverse-time and bidirectional model wrappers;
* model-averaged endpoint summaries.

They are used by ``scripts/benchmark_model_evidence_improved.py`` and can also
be imported directly by ad-hoc analysis notebooks.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .data import ReplaySession, RippleEvent
from .encoding import (
    EmissionConfig,
    EncodingModel,
    LogEmissionTensor,
    _apply_likelihood_temperature,
    _poisson_log_emissions,
    _time_bin_edges,
)
from .models import EventScore, _posterior_diagnostics


class ScorableReplayModel(Protocol):
    """Small protocol for models exposing the package ``score`` method."""

    name: str

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        ...


@dataclass(frozen=True)
class ReplayEmissionCalibration:
    """Configuration for replay-specific sorted-spike emission calibration.

    ``gain_mode`` is intentionally simple and data-local.  It adjusts the fixed
    run-period place-field rates using counts observed in the replay event with a
    pseudo-count prior.  This is not a replacement for a fully Bayesian rate
    model, but it is a useful opt-in diagnostic for separating trajectory-model
    evidence from gross replay-rate miscalibration.
    """

    gain_mode: str = "none"
    gain_prior_count: float = 10.0
    max_gain: float = 20.0
    emission_model: str = "poisson"
    negative_binomial_dispersion: float = 50.0


def build_sorted_emissions_with_replay_calibration(
    session: ReplaySession,
    encoding: EncodingModel,
    ripple: RippleEvent | int,
    config: EmissionConfig | None = None,
    calibration: ReplayEmissionCalibration | None = None,
) -> LogEmissionTensor:
    """Build sorted-spike emissions with optional replay-rate calibration.

    The default ``calibration=None`` or ``gain_mode='none', emission_model='poisson'``
    reproduces the package's ordinary Poisson emission calculation, except this
    function keeps explicit metadata describing the observation model.
    """

    config = EmissionConfig() if config is None else config
    calibration = ReplayEmissionCalibration() if calibration is None else calibration
    gain_mode = _normalize_gain_mode(calibration.gain_mode)
    emission_model = _normalize_emission_model(calibration.emission_model)
    if calibration.gain_prior_count < 0.0:
        raise ValueError("gain_prior_count must be non-negative")
    if calibration.max_gain <= 0.0:
        raise ValueError("max_gain must be positive")
    if calibration.negative_binomial_dispersion <= 0.0:
        raise ValueError("negative_binomial_dispersion must be positive")

    ripple_event = session.ripple(ripple) if isinstance(ripple, int) else ripple
    edges = _time_bin_edges(ripple_event.start, ripple_event.end, config.time_bin_s)
    bin_durations = np.diff(edges)
    times = edges[:-1] + 0.5 * bin_durations
    dt = float(np.median(bin_durations))

    counts = _sorted_spike_counts_for_edges(session, encoding, edges)
    rates_hz = np.asarray(encoding.rates_hz, dtype=float).copy()
    rates_hz = np.maximum(rates_hz * float(config.spike_rate_scale), np.finfo(float).tiny)
    rates_hz, gain_metadata = _apply_replay_gains(
        rates_hz,
        counts,
        bin_durations,
        mode=gain_mode,
        prior_count=float(calibration.gain_prior_count),
        max_gain=float(calibration.max_gain),
    )

    if emission_model == "poisson":
        log_likelihood = _poisson_log_emissions(
            counts,
            rates_hz,
            bin_durations,
            spike_rate_scale=1.0,
            cell_weights=config.cell_weights,
            negative_binomial_overdispersion=float(config.negative_binomial_overdispersion),
        )
    else:
        if config.cell_weights is not None:
            raise ValueError(
                "cell_weights are not supported by the replay-calibrated "
                "Gamma-Poisson emission; use likelihood_temperature or the "
                "base Poisson/negative-binomial EmissionConfig path instead."
            )
        log_likelihood = _negative_binomial_log_emissions(
            counts,
            rates_hz,
            bin_durations,
            dispersion=float(calibration.negative_binomial_dispersion),
        )

    log_likelihood = _apply_likelihood_temperature(log_likelihood, config.likelihood_temperature)

    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts,
        times=times,
        dt=dt,
        cell_ids=encoding.cell_ids,
        n_spikes=int(counts.sum()),
    )
    # Keep these fields explicit so improved evidence runs are auditable after aggregation.
    emissions.metadata = {
        "sorted_spike_emission_model": emission_model,
        "replay_gain_mode": gain_mode,
        "replay_gain_prior_count": float(calibration.gain_prior_count),
        "replay_gain_max_gain": float(calibration.max_gain),
        "negative_binomial_dispersion": float(calibration.negative_binomial_dispersion),
        "emission_likelihood_temperature": float(config.likelihood_temperature),
        "emission_negative_binomial_overdispersion": float(config.negative_binomial_overdispersion),
        **gain_metadata,
    }
    return emissions


def _sorted_spike_counts_for_edges(
    session: ReplaySession,
    encoding: EncodingModel,
    edges: np.ndarray,
) -> np.ndarray:
    counts = np.zeros((len(edges) - 1, encoding.n_cells), dtype=int)
    spikes = session.spikes
    if not spikes.size or not encoding.n_cells:
        return counts
    keep = (
        (spikes[:, 0] >= edges[0])
        & (spikes[:, 0] < edges[-1])
        & np.isin(spikes[:, 1].astype(int), encoding.cell_ids)
    )
    spike_times = spikes[keep, 0]
    spike_cell_ids = spikes[keep, 1].astype(int)
    time_bins = np.searchsorted(edges, spike_times, side="right") - 1
    cell_id_to_row = {
        int(cell_id): row_index
        for row_index, cell_id in enumerate(np.asarray(encoding.cell_ids, dtype=int))
    }
    rows = np.fromiter(
        (cell_id_to_row.get(int(cell_id), -1) for cell_id in spike_cell_ids),
        dtype=int,
        count=spike_cell_ids.shape[0],
    )
    valid = (time_bins >= 0) & (time_bins < counts.shape[0])
    valid &= rows >= 0
    np.add.at(counts, (time_bins[valid].astype(int), rows[valid]), 1)
    return counts


def _apply_replay_gains(
    rates_hz: np.ndarray,
    counts: np.ndarray,
    bin_durations: np.ndarray,
    *,
    mode: str,
    prior_count: float,
    max_gain: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    if mode == "none":
        return rates_hz, {
            "replay_event_gain": 1.0,
            "replay_cell_gain_geomean": 1.0,
            "replay_cell_gain_min": 1.0,
            "replay_cell_gain_max": 1.0,
        }

    calibrated = rates_hz.copy()
    total_duration = float(np.sum(bin_durations))
    mean_rate_by_cell = np.mean(calibrated, axis=1) if calibrated.size else np.zeros(0)
    expected_by_cell = np.maximum(mean_rate_by_cell * total_duration, np.finfo(float).tiny)
    observed_by_cell = counts.sum(axis=0).astype(float)

    cell_gains = np.ones(calibrated.shape[0], dtype=float)
    if mode in {"cell", "event-cell"} and calibrated.shape[0]:
        cell_gains = (observed_by_cell + prior_count) / (expected_by_cell + prior_count)
        cell_gains = np.clip(cell_gains, 1.0 / max_gain, max_gain)
        # Preserve the global event gain for the explicit event-gain term.
        positive = cell_gains > 0.0
        if np.any(positive):
            cell_gains = cell_gains / np.exp(np.mean(np.log(cell_gains[positive])))
        calibrated *= cell_gains[:, None]

    event_gain = 1.0
    if mode in {"event", "event-cell"}:
        expected_total = max(float(np.mean(np.sum(calibrated, axis=0)) * total_duration), np.finfo(float).tiny)
        observed_total = float(observed_by_cell.sum())
        event_gain = (observed_total + prior_count) / (expected_total + prior_count)
        event_gain = float(np.clip(event_gain, 1.0 / max_gain, max_gain))
        calibrated *= event_gain

    return np.maximum(calibrated, np.finfo(float).tiny), {
        "replay_event_gain": float(event_gain),
        "replay_cell_gain_geomean": float(np.exp(np.mean(np.log(np.maximum(cell_gains, np.finfo(float).tiny)))))
        if cell_gains.size
        else 1.0,
        "replay_cell_gain_min": float(np.min(cell_gains)) if cell_gains.size else 1.0,
        "replay_cell_gain_max": float(np.max(cell_gains)) if cell_gains.size else 1.0,
    }


def _negative_binomial_log_emissions(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_durations: np.ndarray,
    *,
    dispersion: float,
) -> np.ndarray:
    """Gamma-Poisson predictive log emissions.

    Uses a negative-binomial distribution parameterized by mean ``mu`` and
    dispersion ``r``.  As ``r`` grows, the model approaches the Poisson emission.
    """

    if dispersion <= 0.0:
        raise ValueError("dispersion must be positive")
    dt = np.asarray(bin_durations, dtype=float)
    if dt.ndim != 1 or dt.shape[0] != spike_counts.shape[0]:
        raise ValueError("bin_durations must contain one duration per time bin")
    if np.any(dt <= 0.0):
        raise ValueError("all bin durations must be positive")

    mean = np.maximum(dt[:, None, None] * rates_hz[None, :, :], np.finfo(float).tiny)
    counts = np.asarray(spike_counts, dtype=float)[:, :, None]
    r = float(dispersion)
    return np.sum(
        gammaln(counts + r)
        - gammaln(r)
        - gammaln(counts + 1.0)
        + r * np.log(r / (r + mean))
        + counts * np.log(mean / (r + mean)),
        axis=1,
    )


def _normalize_gain_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    aliases = {
        "none": "none",
        "off": "none",
        "event": "event",
        "cell": "cell",
        "event-cell": "event-cell",
        "cell-event": "event-cell",
    }
    if mode not in aliases:
        raise ValueError("gain_mode must be one of: none, event, cell, event-cell")
    return aliases[mode]


def _normalize_emission_model(value: str) -> str:
    model = str(value).strip().lower().replace("_", "-")
    aliases = {
        "poisson": "poisson",
        "nb": "negative-binomial",
        "negative-binomial": "negative-binomial",
        "gamma-poisson": "negative-binomial",
        "gamma_poisson": "negative-binomial",
    }
    if model not in aliases:
        raise ValueError("emission_model must be one of: poisson, negative-binomial")
    return aliases[model]


def copy_emissions_with_log_likelihood(
    emissions: LogEmissionTensor,
    log_likelihood: np.ndarray,
    *,
    reverse_time: bool = False,
) -> LogEmissionTensor:
    log_likelihood = np.asarray(log_likelihood, dtype=float)
    spike_counts = np.asarray(emissions.spike_counts)
    times = np.asarray(emissions.times)
    if reverse_time:
        log_likelihood = log_likelihood[::-1].copy()
        spike_counts = spike_counts[::-1].copy()
        times = times[::-1].copy()
    bin_durations = _copy_optional_duration_vector(
        getattr(emissions, "bin_durations", None),
        reverse_time=reverse_time,
        expected_length=emissions.n_time,
        name="bin_durations",
    )
    transition_durations = _copy_optional_duration_vector(
        getattr(emissions, "transition_durations", None),
        reverse_time=reverse_time,
        expected_length=max(emissions.n_time - 1, 0),
        name="transition_durations",
    )
    out = LogEmissionTensor(
        log_likelihood=log_likelihood.copy(),
        spike_counts=spike_counts.copy(),
        times=times.copy(),
        dt=emissions.dt,
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(emissions.n_spikes),
        bin_durations=bin_durations,
        transition_durations=transition_durations,
        metadata=dict(getattr(emissions, "metadata", {}) or {}),
    )
    return out


def _copy_optional_duration_vector(
    values: np.ndarray | None,
    *,
    reverse_time: bool,
    expected_length: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise ValueError(
            f"{name} must contain {expected_length} values; got shape {array.shape}"
        )
    return array[::-1].copy() if reverse_time else array.copy()


def score_replay_model_compat(
    model: ScorableReplayModel,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    *,
    occupancy_s: np.ndarray | None = None,
    candidate_indices: list[np.ndarray] | None = None,
) -> EventScore:
    """Score a replay model while preserving optional state-space controls.

    Several wrappers used by the improved evidence script are intentionally
    lightweight and do not inherit from ``StateSpaceReplayModel``. This helper
    centralizes the compatibility path so direct models, wrappers, and null
    controls all receive candidate supports and occupancy masks when they can
    use them, while legacy models still score normally.
    """

    candidates = candidate_indices
    if candidates is None and hasattr(model, "candidate_indices"):
        candidates = _call_candidate_indices_compat(model.candidate_indices, emissions, bin_centers)  # type: ignore[attr-defined]

    kwargs: dict[str, object] = {}
    if candidates is not None:
        kwargs["candidate_indices"] = candidates
    if occupancy_s is not None:
        kwargs["occupancy_s"] = occupancy_s
    return _call_score_with_supported_kwargs(model.score, emissions, bin_centers, kwargs)  # type: ignore[misc]


def _call_candidate_indices_compat(candidate_indices, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> list[np.ndarray]:
    """Call candidate support helpers without masking implementation TypeErrors."""

    try:
        signature = inspect.signature(candidate_indices)
    except (TypeError, ValueError):
        return candidate_indices(emissions, bin_centers)

    parameters = tuple(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return candidate_indices(emissions, bin_centers)

    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(positional) >= 2:
        return candidate_indices(emissions, bin_centers)
    if "bin_centers" in signature.parameters:
        return candidate_indices(emissions, bin_centers=bin_centers)
    if "centers" in signature.parameters:
        return candidate_indices(emissions, centers=bin_centers)
    return candidate_indices(emissions)


def _call_score_with_supported_kwargs(score, emissions: LogEmissionTensor, bin_centers: np.ndarray, optional_kwargs: dict[str, object]) -> EventScore:
    supported_kwargs = _supported_score_kwargs(score, optional_kwargs)
    if supported_kwargs is not None:
        if supported_kwargs:
            return score(emissions, bin_centers, **supported_kwargs)
        return score(emissions, bin_centers)

    try:
        if optional_kwargs:
            return score(emissions, bin_centers, **optional_kwargs)
        return score(emissions, bin_centers)
    except TypeError as exc:
        unsupported = [
            keyword
            for keyword in optional_kwargs
            if _looks_like_unexpected_keyword_type_error(exc, keyword)
        ]
        if not unsupported:
            raise
        reduced_kwargs = {key: value for key, value in optional_kwargs.items() if key not in unsupported}
        if reduced_kwargs:
            return score(emissions, bin_centers, **reduced_kwargs)
        return score(emissions, bin_centers)


def _supported_score_kwargs(score, optional_kwargs: dict[str, object]) -> dict[str, object] | None:
    if not optional_kwargs:
        return {}

    try:
        signature = inspect.signature(score)
    except (TypeError, ValueError):
        return None

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(optional_kwargs)

    supported: dict[str, object] = {}
    for keyword, value in optional_kwargs.items():
        parameter = parameters.get(keyword)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            supported[keyword] = value
    return supported


def _looks_like_unexpected_keyword_type_error(exc: TypeError, keyword: str) -> bool:
    text = str(exc)
    return keyword in text and (
        "unexpected keyword" in text
        or "got an unexpected" in text
        or "invalid keyword" in text
        or "takes no keyword" in text
    )


@dataclass
class ReverseTimeReplayModel:
    """Wrapper that scores an existing model on the time-reversed emission sequence."""

    base_model: ScorableReplayModel
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"{getattr(self.base_model, 'name', 'model')}-reverse"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        occupancy_s: np.ndarray | None = None,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        reversed_emissions = copy_emissions_with_log_likelihood(
            emissions,
            emissions.log_likelihood,
            reverse_time=True,
        )
        reversed_candidates = (
            None
            if candidate_indices is None
            else [np.asarray(curr, dtype=int).copy() for curr in candidate_indices[::-1]]
        )
        result = score_replay_model_compat(
            self.base_model,
            reversed_emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=reversed_candidates,
        )
        if result.trajectory_log_posterior is not None:
            trajectory = np.asarray(result.trajectory_log_posterior, dtype=float)[::-1].copy()
            result.trajectory_log_posterior = trajectory
            result.terminal_log_posterior = trajectory[-1].copy()
        result.model_name = str(self.name)
        result.diagnostics = dict(result.diagnostics)
        if result.terminal_log_posterior is not None:
            result.diagnostics.update(_posterior_diagnostics(result.terminal_log_posterior, bin_centers))
        result.diagnostics["direction_model"] = "reverse"
        result.diagnostics["reverse_time_base_model"] = str(getattr(self.base_model, "name", "model"))
        return result


@dataclass
class BidirectionalReplayModel:
    """Equal-prior mixture over a forward model and its time-reversed counterpart."""

    forward_model: ScorableReplayModel
    reverse_model: ScorableReplayModel
    name: str

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        *,
        occupancy_s: np.ndarray | None = None,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        forward = score_replay_model_compat(
            self.forward_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
        )
        reverse = score_replay_model_compat(
            self.reverse_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
        )
        values = np.array([forward.log_likelihood, reverse.log_likelihood], dtype=float)
        logp = float(logsumexp(values) - np.log(2.0))
        weights = np.exp(values - logsumexp(values))
        chosen = forward if weights[0] >= weights[1] else reverse
        diagnostics = dict(chosen.diagnostics)
        diagnostics.update(
            {
                "direction_model": "bidirectional",
                "direction_forward_probability": float(weights[0]),
                "direction_reverse_probability": float(weights[1]),
                "direction_forward_log_evidence": float(forward.log_likelihood),
                "direction_reverse_log_evidence": float(reverse.log_likelihood),
            }
        )
        terminal = _mixture_log_posterior(
            [forward.terminal_log_posterior, reverse.terminal_log_posterior],
            weights,
        )
        trajectory = None
        if forward.trajectory_log_posterior is not None and reverse.trajectory_log_posterior is not None:
            trajectory = _mixture_log_posterior(
                [forward.trajectory_log_posterior, reverse.trajectory_log_posterior],
                weights,
            )
        if terminal is None and trajectory is not None:
            terminal = trajectory[-1].copy()
        if terminal is not None:
            diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name,
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def _mixture_log_posterior(log_posteriors: list[np.ndarray | None], weights: np.ndarray) -> np.ndarray | None:
    valid = [(np.asarray(post, dtype=float), float(weight)) for post, weight in zip(log_posteriors, weights, strict=False) if post is not None]
    if not valid:
        return None
    stacked = np.stack([post + np.log(max(weight, np.finfo(float).tiny)) for post, weight in valid], axis=0)
    out = logsumexp(stacked, axis=0)
    return out - logsumexp(out, axis=-1, keepdims=True)


def score_spatial_shuffle_nulls(
    model: ScorableReplayModel,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    *,
    observed_log_evidence: float,
    n_shuffles: int,
    random_seed: int,
    occupancy_s: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Score spatial-bin permutation controls for one event/model pair."""

    n_shuffles = int(n_shuffles)
    if n_shuffles <= 0:
        return {}
    rng = np.random.default_rng(random_seed)
    null_values: list[float] = []
    for _ in range(n_shuffles):
        perm = rng.permutation(emissions.n_bins)
        shuffled = copy_emissions_with_log_likelihood(emissions, emissions.log_likelihood[:, perm])
        try:
            null_values.append(
                float(score_replay_model_compat(model, shuffled, bin_centers, occupancy_s=occupancy_s).log_likelihood)
            )
        except Exception:
            # Some candidate-pruned models can fail under pathological shuffles;
            # keep null diagnostics conservative and transparent.
            continue
    if not null_values:
        return {
            "null_shuffle_count": 0,
            "null_log_evidence_median": np.nan,
            "delta_vs_null_median": np.nan,
            "null_empirical_p_value": np.nan,
        }
    vals = np.asarray(null_values, dtype=float)
    return {
        "null_shuffle_count": int(vals.size),
        "null_log_evidence_mean": float(np.mean(vals)),
        "null_log_evidence_median": float(np.median(vals)),
        "null_log_evidence_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "delta_vs_null_median": float(observed_log_evidence - np.median(vals)),
        "null_empirical_p_value": float((1.0 + np.sum(vals >= observed_log_evidence)) / (vals.size + 1.0)),
    }


def add_model_averaged_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-averaged endpoint estimates to an evidence table.

    The function uses exact-comparable rows and their ``model_probability``
    values.  The same event-level model-averaged endpoint is copied to every row
    in the corresponding event group for convenient CSV joins.
    """

    if df.empty or "model_probability" not in df:
        return df
    required = {"diagnostic_decoded_endpoint_x", "diagnostic_decoded_endpoint_y"}
    if not required.issubset(df.columns):
        return df
    out = df.copy()
    out["model_averaged_endpoint_x"] = np.nan
    out["model_averaged_endpoint_y"] = np.nan
    out["model_averaged_endpoint_models"] = 0
    out["model_probability_entropy"] = np.nan
    out["model_log_evidence_margin"] = np.nan
    for _, group in out.groupby(["session", "event_index"], sort=False):
        if "evidence_comparable" in group:
            comparable = _bool_series(group["evidence_comparable"])
        else:
            # Ad-hoc CSVs produced before evidence-support columns existed are
            # still useful for endpoint averaging; renormalize the listed rows.
            comparable = pd.Series(True, index=group.index)
        exact = group[comparable].copy()
        exact = exact.dropna(subset=["model_probability", "diagnostic_decoded_endpoint_x", "diagnostic_decoded_endpoint_y"])
        if exact.empty:
            continue
        weights = exact["model_probability"].to_numpy(dtype=float, copy=True)
        total = float(np.sum(weights))
        if total <= 0.0:
            continue
        weights /= total
        x = float(np.sum(weights * exact["diagnostic_decoded_endpoint_x"].to_numpy(dtype=float)))
        y = float(np.sum(weights * exact["diagnostic_decoded_endpoint_y"].to_numpy(dtype=float)))
        positive = weights > 0.0
        entropy = float(-np.sum(weights[positive] * np.log(weights[positive])))
        logs = np.sort(exact["log_evidence"].to_numpy(dtype=float))[::-1]
        margin = float(logs[0] - logs[1]) if logs.size > 1 else np.inf
        out.loc[group.index, "model_averaged_endpoint_x"] = x
        out.loc[group.index, "model_averaged_endpoint_y"] = y
        out.loc[group.index, "model_averaged_endpoint_models"] = int(exact.shape[0])
        out.loc[group.index, "model_probability_entropy"] = entropy
        out.loc[group.index, "model_log_evidence_margin"] = margin
    return out


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)
