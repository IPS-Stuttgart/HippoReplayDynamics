"""Synthetic replay-dynamics recovery benchmark."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .data import ReplaySession, load_replay_session
from .encoding import EncodingConfig, EncodingModel, LogEmissionTensor, fit_place_field_encoding
from .models import CandidateKinematicModel, RandomModel, StationaryModel
from .position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from .sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from .state_space import StateSpaceDecoderConfig


DEFAULT_TRUE_MODELS = ("stationary", "diffusion", "momentum", "fragmented")
DEFAULT_SCORING_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-imm",
)
_TRAJECTORY = {
    "diffusion",
    "fragmented",
    "jump",
    "momentum",
    "imm",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-jump",
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-imm",
}
_NONTRAJECTORY = {
    "random",
    "stationary",
    "stationary-gaussian",
    "sorted-spike-state-space-stationary",
}


@dataclass(frozen=True)
class SimulationRecoveryConfig:
    """Configuration for the synthetic dynamics-recovery benchmark."""

    true_models: tuple[str, ...] = DEFAULT_TRUE_MODELS
    scoring_models: tuple[str, ...] = DEFAULT_SCORING_MODELS
    events: str = "run"
    max_template_events: int | None = 25
    events_per_model: int = 25
    random_seed: int = 1
    time_bin_s: float = 0.003
    spike_rate_scale: float = 1.0
    encoding: EncodingConfig = field(
        default_factory=lambda: EncodingConfig(
            bin_size_cm=VALIDATED_POSITION_BIN_SIZE_CM,
            smoothing_sigma_bins=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
            min_speed_cm_s=VALIDATED_POSITION_MIN_SPEED_CM_S,
        )
    )
    state_space: StateSpaceDecoderConfig = field(default_factory=StateSpaceDecoderConfig)
    candidate_top_k: int = 64
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm: float = 12.0
    momentum_sigma_cm: float = 12.0
    velocity_decay: float = 0.95
    mode_stickiness: float = 0.94
    continue_on_error: bool = False


@dataclass
class SimulationRecoveryResult:
    """Output tables for a simulation recovery benchmark."""

    event_scores: pd.DataFrame
    confusion_matrix: pd.DataFrame
    summary: pd.DataFrame
    settings: dict[str, object]

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_scores.to_csv(out_dir / "simulation_recovery_event_scores.csv", index=False)
        self.confusion_matrix.to_csv(out_dir / "simulation_recovery_confusion_matrix.csv", index=False)
        self.summary.to_csv(out_dir / "simulation_recovery_summary.csv", index=False)
        _write_yaml(out_dir / "simulation_recovery_settings.yml", self.settings)


def parse_model_list(spec: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(spec, str):
        values = tuple(part.strip().lower() for part in spec.replace(",", " ").split() if part.strip())
    else:
        values = tuple(str(part).strip().lower() for part in spec if str(part).strip())
    if not values:
        raise ValueError("model list must contain at least one model")
    return values


def run_session_simulation_recovery(
    dataset_root: str | Path,
    session_id: str,
    config: SimulationRecoveryConfig,
) -> SimulationRecoveryResult:
    session_dir = _session_path(dataset_root, session_id)
    session = load_replay_session(session_dir)
    encoding = fit_place_field_encoding(session, config.encoding)
    if encoding.n_cells == 0:
        raise ValueError(f"{session.session_id} has no cells available for simulation")
    template_event_ids = select_event_indices(session, config.events)
    if config.max_template_events is not None:
        template_event_ids = template_event_ids[: config.max_template_events]
    if not template_event_ids:
        raise ValueError(f"No template events selected by {config.events!r}")

    rng = np.random.default_rng(config.random_seed)
    scoring_models = build_scoring_models(config)
    template_lengths = {
        int(event_id): _event_n_time(session, int(event_id), config.time_bin_s)
        for event_id in template_event_ids
    }

    rows: list[dict[str, object]] = []
    simulation_event_index = 0
    true_models = parse_model_list(config.true_models)
    for true_model in true_models:
        for replicate in range(config.events_per_model):
            template_event_id = int(rng.choice(template_event_ids))
            n_time = int(template_lengths[template_event_id])
            emissions, path = simulate_replay_event(
                encoding,
                true_model=true_model,
                n_time=n_time,
                dt=config.time_bin_s,
                rng=rng,
                spike_rate_scale=config.spike_rate_scale,
                state_space=config.state_space,
            )
            expected_model = expected_scoring_model(true_model)
            path_unique_bins = int(np.unique(path).size)
            for requested_model, model in scoring_models.items():
                start = time.perf_counter()
                try:
                    if isinstance(model, CandidateKinematicModel):
                        candidates = model.candidate_indices(emissions)
                        score = model.score(emissions, encoding.bin_centers, candidate_indices=candidates)
                    elif isinstance(model, SortedSpikeStateSpaceReplayModel) and model.mode == "momentum":
                        candidates = model.candidate_indices(emissions)
                        score = model.score(emissions, encoding.bin_centers, candidate_indices=candidates)
                    else:
                        score = model.score(emissions, encoding.bin_centers)
                    model_name = str(score.model_name)
                    row = {
                        "status": "success",
                        "session": session.session_id,
                        "event_index": simulation_event_index,
                        "simulation_event_index": simulation_event_index,
                        "replicate": int(replicate),
                        "template_event_index": template_event_id,
                        "template_n_time": n_time,
                        "true_model": true_model,
                        "true_model_family": model_family(true_model),
                        "expected_model": expected_model,
                        "model": model_name,
                        "requested_model": requested_model,
                        "model_family": model_family(model_name),
                        "log_evidence": float(score.log_likelihood),
                        "n_time": int(score.n_time),
                        "n_spikes": int(score.n_spikes),
                        "path_start_bin": int(path[0]),
                        "path_end_bin": int(path[-1]),
                        "path_unique_bins": path_unique_bins,
                        "runtime_s": float(time.perf_counter() - start),
                        "error": "",
                        "time_bin_s": float(config.time_bin_s),
                        "spike_rate_scale": float(config.spike_rate_scale),
                        "bin_size_cm": float(config.encoding.bin_size_cm),
                        "smoothing_sigma_bins": float(config.encoding.smoothing_sigma_bins),
                        "min_speed_cm_s": float(config.encoding.min_speed_cm_s),
                        "min_occupancy_s": float(config.encoding.min_occupancy_s),
                        "rate_floor_hz": float(config.encoding.rate_floor_hz),
                    }
                    row.update({f"diagnostic_{key}": value for key, value in score.diagnostics.items()})
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        {
                            "status": "failure",
                            "session": session.session_id,
                            "event_index": simulation_event_index,
                            "simulation_event_index": simulation_event_index,
                            "replicate": int(replicate),
                            "template_event_index": template_event_id,
                            "template_n_time": n_time,
                            "true_model": true_model,
                            "true_model_family": model_family(true_model),
                            "expected_model": expected_model,
                            "model": requested_model,
                            "requested_model": requested_model,
                            "model_family": model_family(requested_model),
                            "log_evidence": np.nan,
                            "n_time": int(emissions.n_time),
                            "n_spikes": int(emissions.n_spikes),
                            "path_start_bin": int(path[0]),
                            "path_end_bin": int(path[-1]),
                            "path_unique_bins": path_unique_bins,
                            "runtime_s": float(time.perf_counter() - start),
                            "error": f"{type(exc).__name__}: {exc}",
                            "time_bin_s": float(config.time_bin_s),
                            "spike_rate_scale": float(config.spike_rate_scale),
                            "bin_size_cm": float(config.encoding.bin_size_cm),
                            "smoothing_sigma_bins": float(config.encoding.smoothing_sigma_bins),
                            "min_speed_cm_s": float(config.encoding.min_speed_cm_s),
                            "min_occupancy_s": float(config.encoding.min_occupancy_s),
                            "rate_floor_hz": float(config.encoding.rate_floor_hz),
                        }
                    )
                    if not config.continue_on_error:
                        raise
            simulation_event_index += 1

    event_scores = add_evidence_columns(pd.DataFrame(rows))
    confusion = confusion_matrix(event_scores, config.scoring_models)
    summary = recovery_summary(event_scores)
    settings = _settings(session, config, template_event_ids, encoding)
    return SimulationRecoveryResult(event_scores, confusion, summary, settings)


def simulate_replay_event(
    encoding: EncodingModel,
    *,
    true_model: str,
    n_time: int,
    dt: float,
    rng: np.random.Generator,
    spike_rate_scale: float = 1.0,
    state_space: StateSpaceDecoderConfig | None = None,
) -> tuple[LogEmissionTensor, np.ndarray]:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    state_space = StateSpaceDecoderConfig() if state_space is None else state_space
    true_model = true_model.lower()
    path = simulate_latent_path(encoding, true_model=true_model, n_time=n_time, dt=dt, rng=rng, state_space=state_space)
    counts = np.zeros((n_time, encoding.n_cells), dtype=int)
    for time_index, bin_index in enumerate(path):
        expected = encoding.rates_hz[:, int(bin_index)] * dt * spike_rate_scale
        counts[time_index] = rng.poisson(np.clip(expected, 0.0, None))
    return emissions_from_counts(encoding, counts, dt=dt, spike_rate_scale=spike_rate_scale), path


def simulate_latent_path(
    encoding: EncodingModel,
    *,
    true_model: str,
    n_time: int,
    dt: float,
    rng: np.random.Generator,
    state_space: StateSpaceDecoderConfig | None = None,
) -> np.ndarray:
    state_space = StateSpaceDecoderConfig() if state_space is None else state_space
    true_model = true_model.lower()
    allowed = {"stationary", "diffusion", "momentum", "fragmented", "jump"}
    if true_model not in allowed:
        raise ValueError(f"true_model must be one of {sorted(allowed)}")
    valid_bins, prior = _valid_bins_and_prior(encoding)
    path = np.empty(n_time, dtype=int)
    path[0] = _sample_prior(valid_bins, prior, rng)
    if true_model == "stationary":
        path[:] = path[0]
        return path
    if true_model in {"fragmented", "jump"}:
        for time_index in range(1, n_time):
            path[time_index] = _sample_prior(valid_bins, prior, rng)
        return path

    diffusion_sigma = _per_bin_sigma(state_space.diffusion_sigma_cm_sqrt_s, dt)
    momentum_sigma = _per_bin_sigma(state_space.momentum_sigma_cm_sqrt_s, dt)
    initial_sigma = _per_bin_sigma(state_space.momentum_initial_sigma_cm_sqrt_s, dt)
    if n_time >= 2:
        path[1] = _sample_gaussian_step(encoding.bin_centers, valid_bins, path[0], diffusion_sigma if true_model == "diffusion" else initial_sigma, rng)
    if true_model == "diffusion":
        for time_index in range(2, n_time):
            path[time_index] = _sample_gaussian_step(encoding.bin_centers, valid_bins, path[time_index - 1], diffusion_sigma, rng)
        return path

    for time_index in range(2, n_time):
        prev = encoding.bin_centers[path[time_index - 1]]
        prev_prev = encoding.bin_centers[path[time_index - 2]]
        predicted = prev + state_space.momentum_velocity_decay * (prev - prev_prev)
        path[time_index] = _sample_gaussian_center(encoding.bin_centers, valid_bins, predicted, momentum_sigma, rng)
    return path


def emissions_from_counts(
    encoding: EncodingModel,
    counts: np.ndarray,
    *,
    dt: float,
    spike_rate_scale: float = 1.0,
) -> LogEmissionTensor:
    spike_counts = np.asarray(counts, dtype=int)
    if spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be positive")
    if spike_counts.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    if spike_counts.shape[1] != encoding.n_cells:
        raise ValueError("counts columns must match encoding.n_cells")
    expected = encoding.rates_hz * dt * spike_rate_scale
    log_expected = np.log(np.maximum(expected, np.finfo(float).tiny))
    log_likelihood = spike_counts @ log_expected - expected.sum(axis=0)[None, :]
    log_likelihood -= gammaln(spike_counts + 1).sum(axis=1)[:, None]
    times = (np.arange(spike_counts.shape[0], dtype=float) + 0.5) * dt
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=spike_counts,
        times=times,
        dt=float(dt),
        cell_ids=encoding.cell_ids,
        n_spikes=int(spike_counts.sum()),
    )


def add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    groups = []
    for _, group in df.groupby(["session", "event_index"], sort=False):
        group = group.copy()
        ok = group["status"] == "success"
        scored = group[ok]
        if scored.empty:
            group["relative_log_evidence"] = np.nan
            group["model_probability"] = np.nan
            group["is_best_model"] = False
            group["best_model"] = ""
            groups.append(group)
            continue
        values = scored["log_evidence"].to_numpy(float)
        max_value = float(np.max(values))
        probabilities = np.exp(values - logsumexp(values))
        best = str(scored.iloc[int(np.argmax(values))]["model"])
        group["relative_log_evidence"] = np.nan
        group["model_probability"] = np.nan
        group.loc[scored.index, "relative_log_evidence"] = values - max_value
        group.loc[scored.index, "model_probability"] = probabilities
        group["is_best_model"] = group["model"] == best
        group["best_model"] = best
        group["recovered_expected_model"] = group["best_model"] == group["expected_model"]
        groups.append(group)
    return pd.concat(groups, ignore_index=True).sort_values(["event_index", "model"]).reset_index(drop=True)


def confusion_matrix(event_scores: pd.DataFrame, scoring_models: Iterable[str]) -> pd.DataFrame:
    best = _event_best_rows(event_scores)
    if best.empty:
        return pd.DataFrame()
    columns = list(parse_model_list(scoring_models))
    columns = list(dict.fromkeys([*columns, *best["best_model"].astype(str).unique()]))
    table = pd.crosstab(best["true_model"], best["best_model"]).reindex(index=parse_model_list(best["true_model"].unique()), columns=columns, fill_value=0)
    table.index.name = "true_model"
    return table.reset_index()


def recovery_summary(event_scores: pd.DataFrame) -> pd.DataFrame:
    best = _event_best_rows(event_scores)
    if best.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for true_model, group in best.groupby("true_model", sort=False):
        best_counts = group["best_model"].value_counts()
        expected = expected_scoring_model(str(true_model))
        rows.append(
            {
                "true_model": true_model,
                "expected_model": expected,
                "simulated_events": int(group["event_index"].nunique()),
                "recovered_events": int((group["best_model"] == expected).sum()),
                "recovery_accuracy": float((group["best_model"] == expected).mean()),
                "most_common_best_model": str(best_counts.index[0]),
                "most_common_best_model_events": int(best_counts.iloc[0]),
                "mean_n_time": float(group["n_time"].mean()),
                "mean_n_spikes": float(group["n_spikes"].mean()),
            }
        )
    total_events = int(best["event_index"].nunique())
    rows.append(
        {
            "true_model": "overall",
            "expected_model": "",
            "simulated_events": total_events,
            "recovered_events": int(best["recovered_expected_model"].sum()),
            "recovery_accuracy": float(best["recovered_expected_model"].mean()),
            "most_common_best_model": str(best["best_model"].value_counts().index[0]),
            "most_common_best_model_events": int(best["best_model"].value_counts().iloc[0]),
            "mean_n_time": float(best["n_time"].mean()),
            "mean_n_spikes": float(best["n_spikes"].mean()),
        }
    )
    return pd.DataFrame(rows)


def build_scoring_models(config: SimulationRecoveryConfig) -> dict[str, object]:
    available: dict[str, object] = {
        "random": RandomModel(),
        "stationary": StationaryModel(),
        "stationary-gaussian": CandidateKinematicModel(
            mode="stationary",
            top_k=config.candidate_top_k,
            stationary_sigma_cm=config.stationary_sigma_cm,
            diffusion_sigma_cm=config.diffusion_sigma_cm,
            momentum_sigma_cm=config.momentum_sigma_cm,
            velocity_decay=config.velocity_decay,
            mode_stickiness=config.mode_stickiness,
            name="stationary-gaussian",
        ),
        "diffusion": CandidateKinematicModel(
            mode="diffusion",
            top_k=config.candidate_top_k,
            stationary_sigma_cm=config.stationary_sigma_cm,
            diffusion_sigma_cm=config.diffusion_sigma_cm,
            momentum_sigma_cm=config.momentum_sigma_cm,
            velocity_decay=config.velocity_decay,
            mode_stickiness=config.mode_stickiness,
            name="diffusion",
        ),
        "momentum": CandidateKinematicModel(
            mode="momentum",
            top_k=config.candidate_top_k,
            stationary_sigma_cm=config.stationary_sigma_cm,
            diffusion_sigma_cm=config.diffusion_sigma_cm,
            momentum_sigma_cm=config.momentum_sigma_cm,
            velocity_decay=config.velocity_decay,
            mode_stickiness=config.mode_stickiness,
            name="momentum",
        ),
        "imm": CandidateKinematicModel(
            mode="imm",
            top_k=config.candidate_top_k,
            stationary_sigma_cm=config.stationary_sigma_cm,
            diffusion_sigma_cm=config.diffusion_sigma_cm,
            momentum_sigma_cm=config.momentum_sigma_cm,
            velocity_decay=config.velocity_decay,
            mode_stickiness=config.mode_stickiness,
            name="imm",
        ),
    }
    for mode in ("stationary", "diffusion", "fragmented", "jump", "momentum", "imm"):
        available[f"sorted-spike-state-space-{mode}"] = SortedSpikeStateSpaceReplayModel(
            mode=mode,
            config=replace(config.state_space, mode=mode),
        )
    names = parse_model_list(config.scoring_models)
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown scoring models: {missing}; available: {sorted(available)}")
    return {name: available[name] for name in dict.fromkeys(names)}


def select_event_indices(session: ReplaySession, spec: str) -> list[int]:
    selected = spec.strip().lower()
    if selected == "all":
        return list(range(session.ripple_count))
    if selected == "run":
        return [int(index) for index in session.ripple_indices_in_run()]
    if selected.startswith("run:"):
        run_events = [int(index) for index in session.ripple_indices_in_run()]
        output = []
        for ordinal in _ints(selected.split(":", 1)[1]):
            if ordinal < 0 or ordinal >= len(run_events):
                raise IndexError(f"run ordinal {ordinal} outside 0..{len(run_events) - 1}")
            output.append(run_events[ordinal])
        return sorted(dict.fromkeys(output))
    output = _ints(selected)
    bad = [event_id for event_id in output if event_id < 0 or event_id >= session.ripple_count]
    if bad:
        raise IndexError(f"event IDs outside 0..{session.ripple_count - 1}: {bad}")
    return output


def expected_scoring_model(true_model: str) -> str:
    name = true_model.lower()
    if name == "jump":
        name = "fragmented"
    return f"sorted-spike-state-space-{name}"


def model_family(model: str) -> str:
    name = model.lower()
    if name in _TRAJECTORY:
        return "trajectory"
    if name in _NONTRAJECTORY:
        return "nontrajectory"
    return "other"


def _event_best_rows(event_scores: pd.DataFrame) -> pd.DataFrame:
    ok = event_scores[event_scores["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    best = ok.sort_values(["session", "event_index", "log_evidence"], ascending=[True, True, False])
    return best.drop_duplicates(["session", "event_index"], keep="first").reset_index(drop=True)


def _session_path(root: str | Path, session: str) -> Path:
    parts = session.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', e.g. 'Rat1/Open1'")
    return Path(root) / parts[0] / parts[1]


def _event_n_time(session: ReplaySession, event_id: int, time_bin_s: float) -> int:
    ripple = session.ripple(event_id)
    return max(1, int(np.ceil((ripple.end - ripple.start) / time_bin_s)))


def _valid_bins_and_prior(encoding: EncodingModel) -> tuple[np.ndarray, np.ndarray]:
    valid_bins = np.flatnonzero(np.asarray(encoding.occupancy_s, dtype=float) > 0.0)
    if valid_bins.size == 0:
        valid_bins = np.arange(encoding.n_bins, dtype=int)
        weights = np.ones(valid_bins.size, dtype=float)
    else:
        weights = np.asarray(encoding.occupancy_s[valid_bins], dtype=float)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(valid_bins.size, dtype=float)
    return valid_bins.astype(int), weights / float(weights.sum())


def _sample_prior(valid_bins: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(valid_bins, p=prior))


def _sample_gaussian_step(bin_centers: np.ndarray, valid_bins: np.ndarray, previous_bin: int, sigma_cm: float, rng: np.random.Generator) -> int:
    return _sample_gaussian_center(bin_centers, valid_bins, bin_centers[int(previous_bin)], sigma_cm, rng)


def _sample_gaussian_center(bin_centers: np.ndarray, valid_bins: np.ndarray, center: np.ndarray, sigma_cm: float, rng: np.random.Generator) -> int:
    delta = bin_centers[valid_bins] - center[None, :]
    dist2 = np.sum(delta * delta, axis=1)
    weights = np.exp(-0.5 * dist2 / max(sigma_cm * sigma_cm, np.finfo(float).tiny))
    if float(weights.sum()) <= 0.0:
        return int(valid_bins[int(np.argmin(dist2))])
    weights /= float(weights.sum())
    return int(rng.choice(valid_bins, p=weights))


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    return max(float(sigma_cm_sqrt_s) * np.sqrt(max(float(dt_s), np.finfo(float).tiny)), np.finfo(float).eps)


def _ints(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(value) for value in item.split("-", 1)]
            if hi < lo:
                raise ValueError(f"descending range: {item}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("no events selected")
    return sorted(dict.fromkeys(values))


def _settings(
    session: ReplaySession,
    config: SimulationRecoveryConfig,
    template_event_ids: list[int],
    encoding: EncodingModel,
) -> dict[str, object]:
    return {
        "session": session.session_id,
        "true_models": list(parse_model_list(config.true_models)),
        "scoring_models": list(parse_model_list(config.scoring_models)),
        "events": config.events,
        "template_event_ids": [int(event_id) for event_id in template_event_ids],
        "max_template_events": config.max_template_events,
        "events_per_model": config.events_per_model,
        "random_seed": config.random_seed,
        "spike_rate_scale": config.spike_rate_scale,
        "time_bin_s": config.time_bin_s,
        "encoding": asdict(config.encoding),
        "state_space": asdict(config.state_space),
        "candidate_top_k": config.candidate_top_k,
        "stationary_sigma_cm": config.stationary_sigma_cm,
        "diffusion_sigma_cm": config.diffusion_sigma_cm,
        "momentum_sigma_cm": config.momentum_sigma_cm,
        "velocity_decay": config.velocity_decay,
        "mode_stickiness": config.mode_stickiness,
        "n_position_bins": encoding.n_bins,
        "n_cells": encoding.n_cells,
        "observation_model": "sorted-spike-poisson",
    }


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(_yaml_lines(value), encoding="utf-8")


def _yaml_lines(value: object, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(_yaml_lines(item, indent + 2).rstrip())
            elif isinstance(item, (list, tuple)):
                lines.append(f"{prefix}{key}:")
                lines.append(_yaml_lines(list(item), indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_yaml_lines(item, indent + 2).rstrip())
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.append(_yaml_lines(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    text = str(value)
    if not text or any(ch in text for ch in ":#[]{}&*!|>'\"%@`"):
        return repr(text)
    return text
