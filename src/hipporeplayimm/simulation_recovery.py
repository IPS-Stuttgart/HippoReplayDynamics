"""Synthetic replay-dynamics recovery benchmark."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .data import ReplaySession, load_replay_session
from .encoding import (
    EncodingConfig,
    EncodingModel,
    LogEmissionTensor,
    _poisson_log_emissions,
    fit_place_field_encoding,
)
from .evidence_reporting import _coerce_bool_series
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
    "sorted-spike-state-space-momentum-exact-sparse",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-displacement-momentum",
    "sorted-spike-state-space-velocity-momentum",
    "sorted-spike-state-space-displacement-imm",
    "sorted-spike-state-space-imm",
)
_MOMENTUM_EXACT_SURROGATE_MODELS = (
    "sorted-spike-state-space-momentum-exact-sparse",
    "sorted-spike-state-space-displacement-momentum",
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
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-displacement-momentum",
    "sorted-spike-state-space-velocity-momentum",
    "sorted-spike-state-space-displacement-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
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
    likelihood_temperature: float = 1.0
    negative_binomial_overdispersion: float = 0.0
    encoding: EncodingConfig = field(
        default_factory=lambda: EncodingConfig(
            bin_size_cm=VALIDATED_POSITION_BIN_SIZE_CM,
            smoothing_sigma_bins=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
            min_speed_cm_s=VALIDATED_POSITION_MIN_SPEED_CM_S,
        )
    )
    state_space: StateSpaceDecoderConfig = field(default_factory=StateSpaceDecoderConfig)
    true_state_space: StateSpaceDecoderConfig | None = None
    candidate_top_k: int = 64
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm: float = 12.0
    momentum_sigma_cm: float = 12.0
    velocity_decay: float = 0.95
    mode_stickiness: float = 0.94
    score_with_occupancy: bool = True
    oracle_candidate_support: bool = False
    continue_on_error: bool = False
    max_synthetic_events: int | None = None
    max_runtime_s: float | None = None
    checkpoint_output: str | Path | None = None
    progress_log: bool = False


@dataclass
class SimulationRecoveryResult:
    """Output tables for a simulation recovery benchmark."""

    event_scores: pd.DataFrame
    confusion_matrix: pd.DataFrame
    summary: pd.DataFrame
    settings: dict[str, object]
    certified_vs_exact_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_scores.to_csv(out_dir / "simulation_recovery_event_scores.csv", index=False)
        self.confusion_matrix.to_csv(out_dir / "simulation_recovery_confusion_matrix.csv", index=False)
        self.summary.to_csv(out_dir / "simulation_recovery_summary.csv", index=False)
        self.certified_vs_exact_summary.to_csv(
            out_dir / "simulation_recovery_certified_vs_exact_summary.csv",
            index=False,
        )
        if self.event_scores.empty:
            pd.DataFrame().to_csv(out_dir / "simulation_recovery_diagnostic_event_table.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "simulation_recovery_diagnostic_summary.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "simulation_recovery_certified_vs_exact_events.csv", index=False)
            (out_dir / "simulation_recovery_diagnostic_report.md").write_text(
                "# Simulation-recovery diagnostics\n\nNo diagnostic events were available.\n",
                encoding="utf-8",
            )
            (out_dir / "simulation_recovery_diagnostic_manifest.json").write_text(
                '{"schema_version": 1, "n_input_rows": 0, "n_diagnostic_events": 0}\n',
                encoding="utf-8",
            )
        else:
            from .recovery_diagnostics import build_recovery_diagnostic_tables
            build_recovery_diagnostic_tables(self.event_scores).write(out_dir)
        _write_yaml(out_dir / "simulation_recovery_settings.yml", self.settings)


class _SimulationRecoveryProgress:
    """Incrementally persist recovery progress for long workflow jobs."""

    def __init__(
        self,
        *,
        output: str | Path | None,
        total_events: int,
        scoring_model_count: int,
        true_state_space: StateSpaceDecoderConfig,
        scoring_state_space: StateSpaceDecoderConfig,
        progress_log: bool,
    ) -> None:
        self.output = Path(output) if output is not None else None
        self.total_events = int(total_events)
        self.scoring_model_count = int(scoring_model_count)
        self.true_state_space = true_state_space
        self.scoring_state_space = scoring_state_space
        self.progress_log = bool(progress_log)
        self.started_at = time.perf_counter()
        self.completed_events = 0
        self.failed_events = 0
        self._failed_event_indices: set[int] = set()
        self.current_true_model = ""
        self.current_template_event: int | None = None
        self.current_scoring_model = ""
        self.current_synthetic_event: int | None = None
        self.current_synthetic_events_for_model: int | None = None
        self.current_simulation_event_index: int | None = None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)

    def update_current(
        self,
        *,
        true_model: str,
        synthetic_event: int,
        synthetic_events_for_model: int,
        simulation_event_index: int,
        template_event_index: int,
        scoring_model: str,
        status: str,
    ) -> None:
        self.current_true_model = str(true_model)
        self.current_synthetic_event = int(synthetic_event)
        self.current_synthetic_events_for_model = int(synthetic_events_for_model)
        self.current_simulation_event_index = int(simulation_event_index)
        self.current_template_event = int(template_event_index)
        self.current_scoring_model = str(scoring_model)
        elapsed = self.elapsed_seconds
        if self.progress_log:
            print(
                "[recovery] "
                f"true_model={self.current_true_model} "
                f"synthetic_event={self.current_synthetic_event}/"
                f"{self.current_synthetic_events_for_model} "
                f"scoring_model={self.current_scoring_model} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        self.write_manifest(status=status)

    def complete_event(self, rows: list[dict[str, object]], *, failed: bool) -> None:
        self.completed_events += 1
        if failed:
            self.mark_current_event_failed()
        self.current_scoring_model = ""
        self.write_event_scores(rows)
        self.write_manifest(status="running")

    def mark_current_event_failed(self) -> None:
        if self.current_simulation_event_index is None:
            self.failed_events += 1
            return
        self._failed_event_indices.add(int(self.current_simulation_event_index))
        self.failed_events = len(self._failed_event_indices)

    def write_model_scores(self, rows: list[dict[str, object]]) -> None:
        self._write_scores(rows, "simulation_recovery_partial_model_scores.csv")

    def write_event_scores(self, rows: list[dict[str, object]]) -> None:
        self._write_scores(rows, "simulation_recovery_partial_event_scores.csv")

    def write_manifest(self, *, status: str) -> None:
        if self.output is None:
            return
        manifest = {
            "status": str(status),
            "completed_events": int(self.completed_events),
            "failed_events": int(self.failed_events),
            "total_events": int(self.total_events),
            "scoring_model_count": int(self.scoring_model_count),
            "current_true_model": self.current_true_model,
            "current_template_event": self.current_template_event,
            "current_simulation_event_index": self.current_simulation_event_index,
            "current_synthetic_event": self.current_synthetic_event,
            "current_synthetic_events_for_model": (
                self.current_synthetic_events_for_model
            ),
            "current_scoring_model": self.current_scoring_model,
            "elapsed_seconds": self.elapsed_seconds,
        }
        _write_json_atomic(
            self.output / "simulation_recovery_partial_manifest.json",
            manifest,
        )

    @property
    def elapsed_seconds(self) -> float:
        return float(time.perf_counter() - self.started_at)

    def _write_scores(
        self,
        rows: list[dict[str, object]],
        filename: str,
    ) -> None:
        if self.output is None or not rows:
            return
        scores = _finalize_recovery_event_scores(
            rows,
            true_state_space=self.true_state_space,
            scoring_state_space=self.scoring_state_space,
        )
        _write_dataframe_atomic(scores, self.output / filename)


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
    true_state_space = _true_state_space_config(config)
    template_lengths = {
        int(event_id): _event_n_time(session, int(event_id), config.time_bin_s)
        for event_id in template_event_ids
    }

    _validate_recovery_runtime_limits(config)
    rows: list[dict[str, object]] = []
    simulation_event_index = 0
    true_models = parse_model_list(config.true_models)
    planned_synthetic_events = len(true_models) * int(config.events_per_model)
    if config.max_synthetic_events is not None:
        planned_synthetic_events = min(
            planned_synthetic_events,
            int(config.max_synthetic_events),
        )
    checkpoint_output = Path(config.checkpoint_output) if config.checkpoint_output is not None else None
    run_started_at = time.perf_counter()
    progress = _SimulationRecoveryProgress(
        output=checkpoint_output,
        total_events=planned_synthetic_events,
        scoring_model_count=len(scoring_models),
        true_state_space=true_state_space,
        scoring_state_space=config.state_space,
        progress_log=config.progress_log,
    )
    progress.write_manifest(status="running")
    stop_reason = "completed"
    stop_requested = False
    for true_model in true_models:
        for replicate in range(config.events_per_model):
            should_stop, reason = _should_stop_recovery_before_event(
                config,
                completed_synthetic_events=simulation_event_index,
                run_started_at=run_started_at,
            )
            if should_stop:
                stop_reason = reason
                stop_requested = True
                break
            template_event_id = int(rng.choice(template_event_ids))
            n_time = int(template_lengths[template_event_id])
            rows_before_event = len(rows)
            event_start_row = rows_before_event
            event_started_at = time.perf_counter()
            if config.progress_log:
                print(
                    "[simulation-recovery] "
                    f"event {simulation_event_index + 1}/{planned_synthetic_events} "
                    f"true_model={true_model} replicate={replicate} "
                    f"template_event={template_event_id} n_time={n_time}",
                    flush=True,
                )
            emissions, path = simulate_replay_event(
                encoding,
                true_model=true_model,
                n_time=n_time,
                dt=config.time_bin_s,
                rng=rng,
                spike_rate_scale=config.spike_rate_scale,
                likelihood_temperature=config.likelihood_temperature,
                negative_binomial_overdispersion=config.negative_binomial_overdispersion,
                state_space=true_state_space,
            )
            expected_model = expected_scoring_model(true_model)
            expected_exact_surrogate = exact_surrogate_scoring_model(true_model)
            path_unique_bins = int(np.unique(path).size)
            for requested_model, model in scoring_models.items():
                start = time.perf_counter()
                progress.update_current(
                    true_model=true_model,
                    synthetic_event=replicate + 1,
                    synthetic_events_for_model=config.events_per_model,
                    simulation_event_index=simulation_event_index,
                    template_event_index=template_event_id,
                    scoring_model=requested_model,
                    status="running",
                )
                try:
                    candidates = None
                    candidate_diagnostics: dict[str, float | int] = {}
                    if _uses_candidate_support(model):
                        candidates = _candidate_indices_for_model(model, emissions, encoding.bin_centers)
                        if config.oracle_candidate_support:
                            candidates = _candidate_indices_with_path(candidates, path)
                        candidate_diagnostics = _candidate_path_support_diagnostics(candidates, path)
                    score = _score_recovery_model(
                        model,
                        emissions,
                        encoding,
                        candidate_indices=candidates,
                        score_with_occupancy=config.score_with_occupancy,
                    )
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
                        "expected_exact_surrogate_model": expected_exact_surrogate,
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
                        "likelihood_temperature": float(config.likelihood_temperature),
                        "negative_binomial_overdispersion": float(
                            config.negative_binomial_overdispersion
                        ),
                        "spike_rate_scale": float(config.spike_rate_scale),
                        "bin_size_cm": float(config.encoding.bin_size_cm),
                        "smoothing_sigma_bins": float(config.encoding.smoothing_sigma_bins),
                        "min_speed_cm_s": float(config.encoding.min_speed_cm_s),
                        "min_occupancy_s": float(config.encoding.min_occupancy_s),
                        "rate_floor_hz": float(config.encoding.rate_floor_hz),
                        "score_with_occupancy": bool(config.score_with_occupancy),
                        "oracle_candidate_support": bool(config.oracle_candidate_support),
                    }
                    row.update(candidate_diagnostics)
                    row.update({f"diagnostic_{key}": value for key, value in score.diagnostics.items()})
                    rows.append(row)
                    progress.write_model_scores(rows)
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
                            "expected_exact_surrogate_model": expected_exact_surrogate,
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
                            "likelihood_temperature": float(config.likelihood_temperature),
                            "negative_binomial_overdispersion": float(
                                config.negative_binomial_overdispersion
                            ),
                            "spike_rate_scale": float(config.spike_rate_scale),
                            "bin_size_cm": float(config.encoding.bin_size_cm),
                            "smoothing_sigma_bins": float(config.encoding.smoothing_sigma_bins),
                            "min_speed_cm_s": float(config.encoding.min_speed_cm_s),
                            "min_occupancy_s": float(config.encoding.min_occupancy_s),
                            "rate_floor_hz": float(config.encoding.rate_floor_hz),
                            "score_with_occupancy": bool(config.score_with_occupancy),
                            "oracle_candidate_support": bool(config.oracle_candidate_support),
                        }
                    )
                    progress.write_model_scores(rows)
                    progress.mark_current_event_failed()
                    progress.write_manifest(status="failed" if not config.continue_on_error else "running")
                    if not config.continue_on_error:
                        raise
            event_rows = rows[event_start_row:]
            event_failed = any(str(row.get("status")) != "success" for row in event_rows)
            progress.complete_event(rows, failed=event_failed)
            simulation_event_index += 1
            event_runtime_s = time.perf_counter() - event_started_at
            if config.progress_log:
                print(
                    "[simulation-recovery] "
                    f"event {simulation_event_index}/{planned_synthetic_events} "
                    f"done rows={len(rows) - rows_before_event} "
                    f"event_runtime_s={event_runtime_s:.3f} "
                    f"elapsed_s={time.perf_counter() - run_started_at:.3f}",
                    flush=True,
                )
            if checkpoint_output is not None:
                _write_simulation_recovery_checkpoint(
                    rows,
                    checkpoint_output,
                    session,
                    config,
                    template_event_ids,
                    encoding,
                    true_state_space,
                    run_started_at=run_started_at,
                    planned_synthetic_events=planned_synthetic_events,
                    completed_synthetic_events=simulation_event_index,
                    stop_reason=stop_reason,
                    checkpoint_status="running",
                )
        if stop_requested:
            break

    final_status = "completed" if stop_reason == "completed" else "stopped"
    progress.write_manifest(status=final_status)
    return _build_simulation_recovery_result(
        rows,
        session,
        config,
        template_event_ids,
        encoding,
        true_state_space,
        run_started_at=run_started_at,
        planned_synthetic_events=planned_synthetic_events,
        completed_synthetic_events=simulation_event_index,
        stop_reason=stop_reason,
        checkpoint_status=final_status,
    )


def _validate_recovery_runtime_limits(config: SimulationRecoveryConfig) -> None:
    if config.max_synthetic_events is not None and int(config.max_synthetic_events) <= 0:
        raise ValueError("max_synthetic_events must be positive when set")
    if config.max_runtime_s is not None and float(config.max_runtime_s) <= 0.0:
        raise ValueError("max_runtime_s must be positive when set")


def _positive_finite_scalar(name: str, value: float) -> float:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.bool_) or (
        raw.dtype == object and any(isinstance(item, (bool, np.bool_)) for item in raw.flat)
    ):
        raise TypeError(f"{name} must be finite and positive, not boolean")
    if raw.ndim != 0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _should_stop_recovery_before_event(
    config: SimulationRecoveryConfig,
    *,
    completed_synthetic_events: int,
    run_started_at: float,
) -> tuple[bool, str]:
    if (
        config.max_synthetic_events is not None
        and int(completed_synthetic_events) >= int(config.max_synthetic_events)
    ):
        return True, "max_synthetic_events"
    if (
        config.max_runtime_s is not None
        and time.perf_counter() - run_started_at >= float(config.max_runtime_s)
    ):
        return True, "max_runtime_s"
    return False, ""


def _write_simulation_recovery_checkpoint(
    rows: list[dict[str, object]],
    output: str | Path,
    session: ReplaySession,
    config: SimulationRecoveryConfig,
    template_event_ids: list[int],
    encoding: EncodingModel,
    true_state_space: StateSpaceDecoderConfig,
    *,
    run_started_at: float,
    planned_synthetic_events: int,
    completed_synthetic_events: int,
    stop_reason: str,
    checkpoint_status: str,
) -> None:
    result = _build_simulation_recovery_result(
        rows,
        session,
        config,
        template_event_ids,
        encoding,
        true_state_space,
        run_started_at=run_started_at,
        planned_synthetic_events=planned_synthetic_events,
        completed_synthetic_events=completed_synthetic_events,
        stop_reason=stop_reason,
        checkpoint_status=checkpoint_status,
    )
    result.write(output)


def _build_simulation_recovery_result(
    rows: list[dict[str, object]],
    session: ReplaySession,
    config: SimulationRecoveryConfig,
    template_event_ids: list[int],
    encoding: EncodingModel,
    true_state_space: StateSpaceDecoderConfig,
    *,
    run_started_at: float,
    planned_synthetic_events: int,
    completed_synthetic_events: int,
    stop_reason: str,
    checkpoint_status: str,
) -> SimulationRecoveryResult:
    event_scores = pd.DataFrame(rows)
    if event_scores.empty:
        event_scores = pd.DataFrame(columns=["status"])
    for key, value in _state_space_parameter_row("true", true_state_space).items():
        event_scores[key] = value
    for key, value in _state_space_parameter_row("scoring", config.state_space).items():
        event_scores[key] = value
    event_scores = add_evidence_columns(event_scores)
    if event_scores.empty:
        confusion = pd.DataFrame()
        summary = pd.DataFrame()
        certified_vs_exact = pd.DataFrame()
    else:
        confusion = confusion_matrix(event_scores, config.scoring_models)
        summary = recovery_summary(event_scores)
        certified_vs_exact = certified_vs_exact_recovery_summary(event_scores)
    settings = _settings(session, config, template_event_ids, encoding)
    settings.update(
        _simulation_recovery_checkpoint_metadata(
            config,
            run_started_at=run_started_at,
            planned_synthetic_events=planned_synthetic_events,
            completed_synthetic_events=completed_synthetic_events,
            stop_reason=stop_reason,
            checkpoint_status=checkpoint_status,
        )
    )
    return SimulationRecoveryResult(
        event_scores=event_scores,
        confusion_matrix=confusion,
        summary=summary,
        settings=settings,
        certified_vs_exact_summary=certified_vs_exact,
    )


def _simulation_recovery_checkpoint_metadata(
    config: SimulationRecoveryConfig,
    *,
    run_started_at: float,
    planned_synthetic_events: int,
    completed_synthetic_events: int,
    stop_reason: str,
    checkpoint_status: str,
) -> dict[str, object]:
    return {
        "checkpoint_status": checkpoint_status,
        "checkpoint_stop_reason": stop_reason,
        "checkpoint_completed_synthetic_events": int(completed_synthetic_events),
        "checkpoint_planned_synthetic_events": int(planned_synthetic_events),
        "checkpoint_elapsed_s": float(time.perf_counter() - run_started_at),
        "max_synthetic_events": (
            None if config.max_synthetic_events is None else int(config.max_synthetic_events)
        ),
        "max_runtime_s": None if config.max_runtime_s is None else float(config.max_runtime_s),
    }


def simulate_replay_event(
    encoding: EncodingModel,
    *,
    true_model: str,
    n_time: int,
    dt: float,
    rng: np.random.Generator,
    spike_rate_scale: float = 1.0,
    likelihood_temperature: float = 1.0,
    negative_binomial_overdispersion: float = 0.0,
    state_space: StateSpaceDecoderConfig | None = None,
) -> tuple[LogEmissionTensor, np.ndarray]:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    dt = _positive_finite_scalar("dt", dt)
    spike_rate_scale = _positive_finite_scalar("spike_rate_scale", spike_rate_scale)
    state_space = StateSpaceDecoderConfig() if state_space is None else state_space
    true_model = true_model.lower()
    path = simulate_latent_path(encoding, true_model=true_model, n_time=n_time, dt=dt, rng=rng, state_space=state_space)
    counts = np.zeros((n_time, encoding.n_cells), dtype=int)
    for time_index, bin_index in enumerate(path):
        expected = encoding.rates_hz[:, int(bin_index)] * dt * spike_rate_scale
        counts[time_index] = rng.poisson(np.clip(expected, 0.0, None))
    return (
        emissions_from_counts(
            encoding,
            counts,
            dt=dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        ),
        path,
    )


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
    dt = _positive_finite_scalar("dt", dt)
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

    momentum_velocity_decay = _momentum_velocity_decay_for_duration(state_space, dt)
    for time_index in range(2, n_time):
        prev = encoding.bin_centers[path[time_index - 1]]
        prev_prev = encoding.bin_centers[path[time_index - 2]]
        predicted = prev + momentum_velocity_decay * (prev - prev_prev)
        path[time_index] = _sample_gaussian_center(encoding.bin_centers, valid_bins, predicted, momentum_sigma, rng)
    return path


def emissions_from_counts(
    encoding: EncodingModel,
    counts: np.ndarray,
    *,
    dt: float,
    spike_rate_scale: float = 1.0,
    likelihood_temperature: float = 1.0,
    negative_binomial_overdispersion: float = 0.0,
) -> LogEmissionTensor:
    if _contains_boolean_values(counts):
        raise ValueError("counts must contain numeric integer counts, not boolean values")
    spike_counts = np.asarray(counts, dtype=int)
    dt = _positive_finite_scalar("dt", dt)
    spike_rate_scale = _positive_finite_scalar("spike_rate_scale", spike_rate_scale)
    if spike_counts.ndim != 2:
        raise ValueError("counts must be a two-dimensional array")
    if spike_counts.shape[1] != encoding.n_cells:
        raise ValueError("counts columns must match encoding.n_cells")
    log_likelihood = _poisson_log_emissions(
        spike_counts,
        encoding.rates_hz,
        dt,
        spike_rate_scale=spike_rate_scale,
        likelihood_temperature=likelihood_temperature,
        negative_binomial_overdispersion=negative_binomial_overdispersion,
    )
    times = (np.arange(spike_counts.shape[0], dtype=float) + 0.5) * dt
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=spike_counts,
        times=times,
        dt=float(dt),
        cell_ids=encoding.cell_ids,
        n_spikes=int(spike_counts.sum()),
    )


def _contains_boolean_values(values: object) -> bool:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    groups = []
    for _, group in df.groupby(["session", "event_index"], sort=False):
        group = group.copy()
        group["relative_log_evidence"] = np.nan
        group["model_probability"] = np.nan
        group["is_best_model"] = False
        group["best_model"] = ""
        group["recovered_expected_model"] = False
        group["exact_surrogate_best_model"] = ""
        group["exact_surrogate_recovered_expected_model"] = False
        group["exact_surrogate_log_evidence"] = np.nan
        group["exact_surrogate_minus_best_comparable_log_evidence"] = np.nan
        group["evidence_support"] = ""
        group["evidence_comparable"] = False
        group["best_truncated_lower_bound_model"] = ""
        group["best_truncated_lower_bound_log_evidence"] = np.nan

        ok = group["status"] == "success"
        scored = group[ok]
        if scored.empty:
            groups.append(group)
            continue

        support = scored.apply(_row_evidence_support, axis=1)
        comparable = support.map(_evidence_is_comparable).astype(bool)
        finite_log_evidence = pd.Series(
            np.isfinite(scored["log_evidence"].to_numpy(float)),
            index=scored.index,
        )
        group.loc[scored.index, "evidence_support"] = support
        group.loc[scored.index, "evidence_comparable"] = comparable & finite_log_evidence

        lower_bound_rows = scored[~comparable & finite_log_evidence]
        if not lower_bound_rows.empty:
            lower_values = lower_bound_rows["log_evidence"].to_numpy(float)
            lower_best = lower_bound_rows.iloc[int(np.argmax(lower_values))]
            group["best_truncated_lower_bound_model"] = str(lower_best["model"])
            group["best_truncated_lower_bound_log_evidence"] = float(
                lower_best["log_evidence"]
            )

        comparable_rows = scored[comparable & finite_log_evidence]
        if comparable_rows.empty:
            groups.append(group)
            continue

        values = comparable_rows["log_evidence"].to_numpy(float)
        max_value = float(np.max(values))
        probabilities = np.exp(values - logsumexp(values))
        best = str(comparable_rows.iloc[int(np.argmax(values))]["model"])
        group.loc[comparable_rows.index, "relative_log_evidence"] = values - max_value
        group.loc[comparable_rows.index, "model_probability"] = probabilities
        group.loc[comparable_rows.index, "is_best_model"] = (
            comparable_rows["model"] == best
        )
        group["best_model"] = best
        group["recovered_expected_model"] = best in _event_acceptable_recovery_models(
            group
        )
        surrogate_models = _event_expected_exact_surrogate_models(group)
        surrogate_rows = comparable_rows[
            comparable_rows["model"].astype(str).isin(surrogate_models)
        ]
        if not surrogate_rows.empty:
            surrogate = _best_log_evidence_row(surrogate_rows)
            surrogate_log_evidence = float(surrogate["log_evidence"])
            group["exact_surrogate_best_model"] = str(surrogate["model"])
            group["exact_surrogate_log_evidence"] = surrogate_log_evidence
            group["exact_surrogate_minus_best_comparable_log_evidence"] = (
                surrogate_log_evidence - max_value
            )
            group["exact_surrogate_recovered_expected_model"] = bool(
                str(surrogate["model"]) == best
            )
        groups.append(group)
    return pd.concat(groups, ignore_index=True).sort_values(["event_index", "model"]).reset_index(drop=True)


_NONCOMPARABLE_EVIDENCE_SUPPORTS = {
    "degenerate_single_bin",
    "truncated_full_grid",
}


def _row_evidence_support(row: pd.Series) -> str:
    """Return the evidence support label for one event/model row."""

    preferred = (
        "diagnostic_candidate_evidence_support",
        "diagnostic_state_space_momentum_evidence_support",
        "diagnostic_state_space_imm_evidence_support",
        "diagnostic_state_space_displacement_momentum_evidence_support",
    )
    support_columns = list(
        dict.fromkeys(
            [
                *preferred,
                *[
                    name
                    for name in row.index
                    if str(name).startswith("diagnostic_")
                    and str(name).endswith("_evidence_support")
                ],
            ]
        )
    )
    values = []
    for column in support_columns:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        label = str(value).strip()
        if label:
            values.append(label)
    if not values:
        return "exact_full_grid"
    for value in values:
        if value in _NONCOMPARABLE_EVIDENCE_SUPPORTS:
            return value
    return values[0]


def _evidence_is_comparable(support: object) -> bool:
    return str(support) not in _NONCOMPARABLE_EVIDENCE_SUPPORTS


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
        recovered = _recovered_expected_series(group, str(true_model))
        surrogate_recovered = _surrogate_recovered_series(group)
        rows.append(
            {
                "true_model": true_model,
                "expected_model": expected,
                "simulated_events": int(group["event_index"].nunique()),
                "recovered_events": int(recovered.sum()),
                "recovery_accuracy": float(recovered.mean()),
                "exact_surrogate_recovered_events": int(surrogate_recovered.sum()),
                "exact_surrogate_recovery_accuracy": float(surrogate_recovered.mean()),
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
            "exact_surrogate_recovered_events": int(_surrogate_recovered_series(best).sum()),
            "exact_surrogate_recovery_accuracy": float(_surrogate_recovered_series(best).mean()),
            "most_common_best_model": str(best["best_model"].value_counts().index[0]),
            "most_common_best_model_events": int(best["best_model"].value_counts().iloc[0]),
            "mean_n_time": float(best["n_time"].mean()),
            "mean_n_spikes": float(best["n_spikes"].mean()),
        }
    )
    return pd.DataFrame(rows)


def certified_vs_exact_event_recovery(event_scores: pd.DataFrame) -> pd.DataFrame:
    """Return event-level recovery diagnostics that respect evidence support.

    The ordinary recovery summary intentionally excludes truncated lower-bound
    rows before choosing an event winner.  That is the right strict default, but
    it makes a candidate-pruned momentum model structurally unable to recover a
    true momentum event: its own row is never allowed to be the strict best row.

    This diagnostic adds a conservative second view.  If the expected model has
    comparable evidence, recovery is the ordinary comparable-evidence decision.
    If the expected model is a truncated lower bound, it is counted as recovered
    only when that lower bound is greater than the best comparable exact row.
    Such a win is certified because the unknown full-grid evidence of the
    expected model can only be higher than its reported lower bound.  Lower-bound
    rows are never treated as exact and are not used to disqualify exact rows.
    """

    if event_scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (session, event_index), group in event_scores.groupby(
        ["session", "event_index"], sort=False
    ):
        group = group.copy()
        first = group.iloc[0]
        expected_model = str(first.get("expected_model", ""))
        base: dict[str, object] = {
            "session": session,
            "event_index": int(event_index),
            "true_model": str(first.get("true_model", "")),
            "expected_model": expected_model,
            "n_time": _event_scalar(group, "n_time"),
            "n_spikes": _event_scalar(group, "n_spikes"),
        }

        scored = group[group["status"] == "success"].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_successful_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        finite = np.isfinite(scored["log_evidence"].to_numpy(float))
        scored = scored.loc[finite].copy()
        if scored.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "no_finite_scores",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": "",
                    "best_comparable_log_evidence": np.nan,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        comparable_mask = _comparable_mask(scored)
        comparable_rows = scored.loc[comparable_mask].copy()
        best_comparable_model = ""
        best_comparable_log_evidence = np.nan
        if not comparable_rows.empty:
            best_comparable = _best_log_evidence_row(comparable_rows)
            best_comparable_model = str(best_comparable["model"])
            best_comparable_log_evidence = float(best_comparable["log_evidence"])

        acceptable_models = _event_acceptable_recovery_models(group)
        acceptable_rows = scored[
            scored["model"].astype(str).isin(acceptable_models)
        ].copy()
        expected_rows = scored[scored["model"].astype(str) == expected_model]
        if acceptable_rows.empty:
            rows.append(
                {
                    **base,
                    "certified_vs_exact_recovered_expected_model": False,
                    "certified_vs_exact_reason": "expected_model_not_scored",
                    "certified_reference_model": "",
                    "expected_model_log_evidence": np.nan,
                    "expected_model_evidence_support": "",
                    "expected_model_evidence_comparable": False,
                    "best_comparable_model": best_comparable_model,
                    "best_comparable_log_evidence": best_comparable_log_evidence,
                    "expected_minus_best_comparable_log_evidence": np.nan,
                }
            )
            continue

        if best_comparable_model in acceptable_models:
            expected = scored[scored["model"].astype(str) == best_comparable_model].iloc[0]
        elif not expected_rows.empty:
            expected = _best_log_evidence_row(expected_rows)
        else:
            expected = _best_log_evidence_row(acceptable_rows)

        certified_reference_model = str(expected["model"])
        expected_log_evidence = float(expected["log_evidence"])
        expected_support = str(expected.get("evidence_support", ""))
        raw_expected_comparable = expected.get(
            "evidence_comparable",
            _evidence_is_comparable(expected_support),
        )
        expected_comparable = (
            _evidence_is_comparable(expected_support)
            if pd.isna(raw_expected_comparable)
            else bool(_coerce_bool_series(pd.Series([raw_expected_comparable])).iloc[0])
        )
        margin = expected_log_evidence - best_comparable_log_evidence

        if best_comparable_model in acceptable_models:
            recovered = True
            reason = (
                "expected_comparable_best"
                if best_comparable_model == expected_model
                else "exact_surrogate_comparable_best"
            )
        elif expected_comparable:
            recovered = False
            reason = (
                "expected_comparable_not_best"
                if certified_reference_model == expected_model
                else "exact_surrogate_comparable_not_best"
            )
        elif expected_support != "truncated_full_grid":
            recovered = False
            reason = "expected_noncomparable_not_certified"
        elif not np.isfinite(best_comparable_log_evidence):
            recovered = False
            reason = "no_comparable_exact_reference"
        else:
            recovered = bool(margin > 0.0)
            reason = (
                "expected_lower_bound_beats_best_comparable"
                if recovered
                else "expected_lower_bound_not_above_best_comparable"
            )

        rows.append(
            {
                **base,
                "certified_vs_exact_recovered_expected_model": recovered,
                "certified_vs_exact_reason": reason,
                "certified_reference_model": certified_reference_model,
                "expected_model_log_evidence": expected_log_evidence,
                "expected_model_evidence_support": expected_support,
                "expected_model_evidence_comparable": expected_comparable,
                "best_comparable_model": best_comparable_model,
                "best_comparable_log_evidence": best_comparable_log_evidence,
                "expected_minus_best_comparable_log_evidence": float(margin),
            }
        )
    return pd.DataFrame(rows)


def certified_vs_exact_recovery_summary(event_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize conservative lower-bound-certified synthetic recovery."""

    events = certified_vs_exact_event_recovery(event_scores)
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for true_model, group in events.groupby("true_model", sort=False):
        rows.append(_certified_vs_exact_summary_row(str(true_model), group))
    rows.append(_certified_vs_exact_summary_row("overall", events))
    return pd.DataFrame(rows)


def _certified_vs_exact_summary_row(label: str, group: pd.DataFrame) -> dict[str, object]:
    recovered = _coerce_bool_series(group["certified_vs_exact_recovered_expected_model"])
    margins = pd.to_numeric(
        group["expected_minus_best_comparable_log_evidence"], errors="coerce"
    )
    expected_model = "" if label == "overall" else str(group["expected_model"].iloc[0])
    return {
        "true_model": label,
        "expected_model": expected_model,
        "simulated_events": int(group["event_index"].nunique()),
        "certified_vs_exact_recovered_events": int(recovered.sum()),
        "certified_vs_exact_recovery_accuracy": float(recovered.mean()),
        "mean_expected_minus_best_comparable_log_evidence": float(margins.mean()),
        "median_expected_minus_best_comparable_log_evidence": float(margins.median()),
        "events_without_comparable_exact_reference": int(
            (group["certified_vs_exact_reason"] == "no_comparable_exact_reference").sum()
        ),
    }


def _comparable_mask(frame: pd.DataFrame) -> pd.Series:
    if "evidence_comparable" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return _coerce_bool_series(frame["evidence_comparable"])


def _best_log_evidence_row(frame: pd.DataFrame) -> pd.Series:
    values = frame["log_evidence"].to_numpy(float)
    return frame.iloc[int(np.argmax(values))]


def _event_scalar(group: pd.DataFrame, column: str) -> object:
    return group[column].iloc[0] if column in group.columns and not group.empty else np.nan


def build_scoring_models(config: SimulationRecoveryConfig) -> dict[str, object]:
    state_space_config = _recovery_state_space_config(config)
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
    for mode in (
        "stationary",
        "diffusion",
        "fragmented",
        "jump",
        "first-order-imm",
        "momentum",
        "momentum-exact-sparse",
        "displacement-momentum",
        "displacement-imm",
        "imm",
    ):
        available[f"sorted-spike-state-space-{mode}"] = SortedSpikeStateSpaceReplayModel(
            mode=mode,
            config=replace(state_space_config, mode=mode),
        )
    available["sorted-spike-state-space-velocity-momentum"] = SortedSpikeStateSpaceReplayModel(
        mode="displacement-momentum",
        config=replace(state_space_config, mode="displacement-momentum"),
        name="sorted-spike-state-space-velocity-momentum",
    )
    available["state-space-velocity-momentum"] = SortedSpikeStateSpaceReplayModel(
        mode="displacement-momentum",
        config=replace(state_space_config, mode="displacement-momentum"),
        name="state-space-velocity-momentum",
    )
    names = parse_model_list(config.scoring_models)
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown scoring models: {missing}; available: {sorted(available)}")
    return {name: available[name] for name in dict.fromkeys(names)}


def _recovery_state_space_config(config: SimulationRecoveryConfig) -> StateSpaceDecoderConfig:
    """Return the state-space config used for synthetic recovery scoring."""

    if (
        config.score_with_occupancy
        and float(config.state_space.valid_occupancy_threshold_s) <= 0.0
    ):
        return replace(
            config.state_space,
            valid_occupancy_threshold_s=float(np.finfo(float).tiny),
        )
    return config.state_space


def _uses_candidate_support(model: object) -> bool:
    if isinstance(model, CandidateKinematicModel):
        return True
    return isinstance(model, SortedSpikeStateSpaceReplayModel) and model.mode in {
        "momentum",
        "imm",
    }


def _candidate_indices_for_model(model: object, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> list[np.ndarray]:
    """Return candidate support, passing bin centers when a model supports augmentation."""

    return _call_candidate_indices(model.candidate_indices, emissions, bin_centers)  # type: ignore[attr-defined]


def _call_candidate_indices(candidate_indices, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> list[np.ndarray]:
    """Call candidate_indices without swallowing implementation TypeErrors.

    Some scoring models expose a legacy ``candidate_indices(emissions)`` method,
    while newer models accept spatial bin centers as a second argument.  Decide
    which form to call from the signature instead of catching ``TypeError`` from
    the model body; otherwise a genuine bin-center-aware implementation failure
    can be mistaken for a legacy signature and silently retried without centers.
    """

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


def _score_recovery_model(
    model: object,
    emissions: LogEmissionTensor,
    encoding: EncodingModel,
    *,
    candidate_indices: list[np.ndarray] | None = None,
    score_with_occupancy: bool = True,
) -> object:
    """Score one synthetic event with diagnostics-friendly support handling."""

    if isinstance(model, SortedSpikeStateSpaceReplayModel):
        kwargs: dict[str, object] = {}
        if model.mode == "momentum-exact-sparse":
            kwargs["return_trajectory"] = False
        if candidate_indices is not None:
            kwargs["candidate_indices"] = candidate_indices
        if score_with_occupancy:
            kwargs["occupancy_s"] = encoding.occupancy_s
        return model.score(emissions, encoding.bin_centers, **kwargs)
    if candidate_indices is not None:
        return model.score(  # type: ignore[attr-defined]
            emissions,
            encoding.bin_centers,
            candidate_indices=candidate_indices,
        )
    return model.score(emissions, encoding.bin_centers)  # type: ignore[attr-defined]


def _candidate_indices_with_path(
    candidates: list[np.ndarray],
    path: np.ndarray,
) -> list[np.ndarray]:
    """Return candidate support augmented with the known synthetic path."""

    path = np.asarray(path, dtype=int)
    if len(candidates) != path.shape[0]:
        raise ValueError("candidate support and synthetic path lengths must match")
    augmented: list[np.ndarray] = []
    for time_index, current in enumerate(candidates):
        current = np.asarray(current, dtype=int)
        augmented.append(
            np.unique(np.concatenate([current, np.asarray([path[time_index]], dtype=int)]))
        )
    return augmented


def _candidate_path_support_diagnostics(
    candidates: list[np.ndarray],
    path: np.ndarray,
) -> dict[str, float | int]:
    """Summarize whether candidate support contains the synthetic latent path."""

    path = np.asarray(path, dtype=int)
    if len(candidates) != path.shape[0]:
        raise ValueError("candidate support and synthetic path lengths must match")
    if path.size == 0:
        return {
            "candidate_true_bin_coverage": float("nan"),
            "candidate_true_pair_coverage": float("nan"),
            "candidate_true_triplet_coverage": float("nan"),
            "candidate_true_path_fully_supported": 0,
            "candidate_true_path_missing_bins": 0,
        }
    supported = np.asarray(
        [
            bool(np.any(np.asarray(current, dtype=int) == path[time_index]))
            for time_index, current in enumerate(candidates)
        ],
        dtype=bool,
    )
    pair_supported = (
        supported[:-1] & supported[1:] if supported.size > 1 else np.empty(0, dtype=bool)
    )
    triplet_supported = (
        supported[:-2] & supported[1:-1] & supported[2:]
        if supported.size > 2
        else np.empty(0, dtype=bool)
    )
    return {
        "candidate_true_bin_coverage": _boolean_fraction(supported),
        "candidate_true_pair_coverage": _boolean_fraction(pair_supported),
        "candidate_true_triplet_coverage": _boolean_fraction(triplet_supported),
        "candidate_true_path_fully_supported": int(bool(np.all(supported))),
        "candidate_true_path_missing_bins": int(np.sum(~supported)),
    }


def _boolean_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=bool)
    return float("nan") if values.size == 0 else float(np.mean(values))


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


def exact_surrogate_scoring_model(true_model: str) -> str:
    """Return the exact-comparable surrogate expected for a true model."""

    return exact_surrogate_scoring_models(true_model)[0]


def exact_surrogate_scoring_models(true_model: str) -> tuple[str, ...]:
    """Return exact-comparable model variants that recover a true model."""

    name = true_model.lower()
    if name == "momentum":
        return _MOMENTUM_EXACT_SURROGATE_MODELS
    return (expected_scoring_model(name),)


def _event_expected_exact_surrogate_models(group: pd.DataFrame) -> tuple[str, ...]:
    models: list[str] = []
    true_model = _event_text(group, "true_model")
    if true_model:
        models.extend(exact_surrogate_scoring_models(true_model))
    models.extend(_event_model_values(group, "expected_exact_surrogate_model"))
    if not models:
        expected = _event_text(group, "expected_model")
        if expected:
            models.append(expected)
    return tuple(dict.fromkeys(models))


def _event_acceptable_recovery_models(group: pd.DataFrame) -> tuple[str, ...]:
    models: list[str] = []
    expected = _event_text(group, "expected_model")
    if expected:
        models.append(expected)
    models.extend(_event_expected_exact_surrogate_models(group))
    return tuple(dict.fromkeys(models))


def _event_text(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].dropna().astype(str)
    if values.empty:
        return ""
    return str(values.iloc[0]).strip()


def _event_model_values(group: pd.DataFrame, column: str) -> list[str]:
    values: list[str] = []
    if column not in group.columns:
        return values
    for value in group[column].dropna().astype(str):
        for model in value.replace(",", " ").split():
            if model:
                values.append(model)
    return list(dict.fromkeys(values))


def _recovered_expected_series(group: pd.DataFrame, true_model: str) -> pd.Series:
    if "recovered_expected_model" in group.columns:
        return _coerce_bool_series(group["recovered_expected_model"])
    acceptable = set(acceptable_recovery_models(true_model))
    return group["best_model"].astype(str).isin(acceptable)


def acceptable_recovery_models(true_model: str) -> tuple[str, ...]:
    """Return model names that should count as recovering a true dynamics class."""

    expected = expected_scoring_model(true_model)
    return tuple(dict.fromkeys([expected, *exact_surrogate_scoring_models(true_model)]))


def _surrogate_recovered_series(group: pd.DataFrame) -> pd.Series:
    if "exact_surrogate_recovered_expected_model" not in group.columns:
        return pd.Series(False, index=group.index)
    return _coerce_bool_series(group["exact_surrogate_recovered_expected_model"])


def model_family(model: str) -> str:
    name = model.lower()
    if name in _TRAJECTORY:
        return "trajectory"
    if name in _NONTRAJECTORY:
        return "nontrajectory"
    return "other"


def _event_best_rows(event_scores: pd.DataFrame) -> pd.DataFrame:
    ok = event_scores[event_scores["status"] == "success"]
    if "evidence_comparable" in ok.columns:
        ok = ok[_coerce_bool_series(ok["evidence_comparable"])]
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


def _momentum_velocity_decay_for_duration(
    state_space: StateSpaceDecoderConfig,
    duration_s: float,
) -> float:
    tau_s = float(getattr(state_space, "momentum_velocity_decay_tau_s", 0.0))
    if tau_s > 0.0:
        return float(np.exp(-float(duration_s) / tau_s))
    return float(state_space.momentum_velocity_decay)


def _true_state_space_config(config: SimulationRecoveryConfig) -> StateSpaceDecoderConfig:
    return config.state_space if config.true_state_space is None else config.true_state_space


def _finalize_recovery_event_scores(
    rows: list[dict[str, object]],
    *,
    true_state_space: StateSpaceDecoderConfig,
    scoring_state_space: StateSpaceDecoderConfig,
) -> pd.DataFrame:
    event_scores = pd.DataFrame(rows)
    if event_scores.empty:
        return event_scores
    for key, value in _state_space_parameter_row("true", true_state_space).items():
        event_scores[key] = value
    for key, value in _state_space_parameter_row("scoring", scoring_state_space).items():
        event_scores[key] = value
    return add_evidence_columns(event_scores)


def _state_space_parameter_row(prefix: str, config: StateSpaceDecoderConfig) -> dict[str, object]:
    return {
        f"{prefix}_state_space_{key}": value
        for key, value in asdict(config).items()
    }


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
    true_state_space = _true_state_space_config(config)
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
        "likelihood_temperature": config.likelihood_temperature,
        "negative_binomial_overdispersion": config.negative_binomial_overdispersion,
        "time_bin_s": config.time_bin_s,
        "encoding": asdict(config.encoding),
        "state_space": asdict(config.state_space),
        "scoring_state_space": asdict(config.state_space),
        "true_state_space": asdict(true_state_space),
        "true_state_space_differs_from_scoring": asdict(true_state_space) != asdict(config.state_space),
        "candidate_top_k": config.candidate_top_k,
        "stationary_sigma_cm": config.stationary_sigma_cm,
        "diffusion_sigma_cm": config.diffusion_sigma_cm,
        "momentum_sigma_cm": config.momentum_sigma_cm,
        "velocity_decay": config.velocity_decay,
        "mode_stickiness": config.mode_stickiness,
        "score_with_occupancy": bool(config.score_with_occupancy),
        "oracle_candidate_support": bool(config.oracle_candidate_support),
        "max_synthetic_events": config.max_synthetic_events,
        "max_runtime_s": config.max_runtime_s,
        "checkpoint_output": "" if config.checkpoint_output is None else str(config.checkpoint_output),
        "progress_log": bool(config.progress_log),
        "n_position_bins": encoding.n_bins,
        "n_cells": encoding.n_cells,
        "observation_model": "sorted-spike-poisson",
    }


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(_yaml_lines(value), encoding="utf-8")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


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
