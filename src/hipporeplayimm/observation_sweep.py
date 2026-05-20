"""Observation-model calibration sweeps for replay evidence experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ReplaySession, load_open_field_sessions
from .encoding import EncodingConfig
from .position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_DECODE_BIN_S,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
    PositionDecodingConfig,
    PositionDecodingResult,
    summarize_position_decoding,
    validate_session_position_decoding,
)
from .simulation_recovery import (
    DEFAULT_SCORING_MODELS,
    DEFAULT_TRUE_MODELS,
    SimulationRecoveryConfig,
    SimulationRecoveryResult,
    parse_model_list,
    run_session_simulation_recovery,
)

_DEFAULT_ENCODING = EncodingConfig(
    bin_size_cm=VALIDATED_POSITION_BIN_SIZE_CM,
    smoothing_sigma_bins=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
    min_speed_cm_s=VALIDATED_POSITION_MIN_SPEED_CM_S,
)


@dataclass(frozen=True)
class ObservationSweepConfig:
    """Configuration for joint encoder validation and recovery sweeps."""

    sessions: tuple[str, ...] | None = ("Rat1/Open1",)
    bin_sizes_cm: tuple[float, ...] = (VALIDATED_POSITION_BIN_SIZE_CM,)
    smoothing_sigmas_bins: tuple[float, ...] = (VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,)
    min_speed_cm_s: tuple[float, ...] = (VALIDATED_POSITION_MIN_SPEED_CM_S,)
    min_occupancy_s: tuple[float, ...] = (_DEFAULT_ENCODING.min_occupancy_s,)
    rate_floor_hz: tuple[float, ...] = (_DEFAULT_ENCODING.rate_floor_hz,)
    time_bin_ms: tuple[float, ...] = (3.0,)
    spike_rate_scales: tuple[float, ...] = (1.0,)
    decode_bin_s: float = VALIDATED_POSITION_DECODE_BIN_S
    n_folds: int = 5
    max_windows_per_session: int | None = None
    min_spikes_per_window: int = 0
    random_seed: int = 1
    run_simulation_recovery: bool = True
    simulation_events: str = "run"
    simulation_max_template_events: int | None = 25
    simulation_events_per_model: int = 10
    simulation_true_models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TRUE_MODELS)
    simulation_scoring_models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SCORING_MODELS)
    simulation_continue_on_error: bool = True


@dataclass
class ObservationSweepResult:
    """Tables produced by an observation-model parameter sweep."""

    summary: pd.DataFrame
    position_summary: pd.DataFrame
    position_samples: pd.DataFrame
    simulation_summary: pd.DataFrame
    simulation_event_scores: pd.DataFrame
    settings: dict[str, object]

    def write(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(output / "observation_sweep_summary.csv", index=False)
        self.position_summary.to_csv(output / "position_decoding_summary.csv", index=False)
        self.position_samples.to_csv(output / "position_decoding_samples.csv", index=False)
        self.simulation_summary.to_csv(output / "simulation_recovery_summary.csv", index=False)
        self.simulation_event_scores.to_csv(output / "simulation_recovery_event_scores.csv", index=False)
        _write_yaml(output / "observation_sweep_settings.yml", self.settings)


def write_observation_sweep_outputs(
    result: ObservationSweepResult,
    output_dir: str | Path,
) -> None:
    result.write(output_dir)


def observation_parameter_grid(config: ObservationSweepConfig) -> list[dict[str, float | int]]:
    """Return the Cartesian observation-parameter grid with stable sweep IDs."""

    _validate_config(config)
    rows: list[dict[str, float | int]] = []
    for sweep_id, values in enumerate(
        product(
            config.bin_sizes_cm,
            config.smoothing_sigmas_bins,
            config.min_speed_cm_s,
            config.min_occupancy_s,
            config.rate_floor_hz,
            config.time_bin_ms,
            config.spike_rate_scales,
        )
    ):
        (
            bin_size_cm,
            smoothing_sigma_bins,
            min_speed_cm_s,
            min_occupancy_s,
            rate_floor_hz,
            time_bin_ms,
            spike_rate_scale,
        ) = values
        rows.append(
            {
                "sweep_id": int(sweep_id),
                "bin_size_cm": float(bin_size_cm),
                "smoothing_sigma_bins": float(smoothing_sigma_bins),
                "min_speed_cm_s": float(min_speed_cm_s),
                "min_occupancy_s": float(min_occupancy_s),
                "rate_floor_hz": float(rate_floor_hz),
                "time_bin_ms": float(time_bin_ms),
                "time_bin_s": float(time_bin_ms) / 1000.0,
                "spike_rate_scale": float(spike_rate_scale),
            }
        )
    return rows


def run_observation_parameter_sweep(
    root: str | Path,
    config: ObservationSweepConfig | None = None,
) -> ObservationSweepResult:
    """Run position decoding plus optional synthetic recovery for each setting."""

    config = ObservationSweepConfig() if config is None else config
    _validate_config(config)
    sessions = _selected_sessions(root, config)
    grid = observation_parameter_grid(config)

    position_cache: dict[tuple[str, float, float, float, float, float], PositionDecodingResult] = {}
    position_summary_rows: list[pd.DataFrame] = []
    position_sample_rows: list[pd.DataFrame] = []
    simulation_summary_rows: list[pd.DataFrame] = []
    simulation_event_rows: list[pd.DataFrame] = []

    for params in grid:
        encoding = _encoding_from_params(params)
        for session in sessions:
            position_key = (
                session.session_id,
                float(params["bin_size_cm"]),
                float(params["smoothing_sigma_bins"]),
                float(params["min_speed_cm_s"]),
                float(params["min_occupancy_s"]),
                float(params["rate_floor_hz"]),
            )
            if position_key not in position_cache:
                position_cache[position_key] = _run_position_validation_for_session(
                    session,
                    encoding,
                    config,
                )
            position_result = position_cache[position_key]
            position_summary_rows.append(
                _attach_sweep_metadata(position_result.summary, params, session.session_id)
            )
            position_sample_rows.append(
                _attach_sweep_metadata(
                    position_result.samples,
                    params,
                    session.session_id,
                    include_empty_row=False,
                )
            )

            if config.run_simulation_recovery:
                simulation_result = _run_simulation_recovery_for_session(
                    root,
                    session.session_id,
                    encoding,
                    params,
                    config,
                )
                simulation_summary_rows.append(
                    _attach_sweep_metadata(simulation_result.summary, params, session.session_id)
                )
                simulation_event_rows.append(
                    _attach_sweep_metadata(
                        simulation_result.event_scores,
                        params,
                        session.session_id,
                        include_empty_row=False,
                    )
                )

    position_summary = _concat(position_summary_rows)
    position_samples = _concat(position_sample_rows)
    simulation_summary = _concat(simulation_summary_rows)
    simulation_event_scores = _concat(simulation_event_rows)
    summary = summarize_observation_sweep(position_summary, simulation_summary)
    return ObservationSweepResult(
        summary=summary,
        position_summary=position_summary,
        position_samples=position_samples,
        simulation_summary=simulation_summary,
        simulation_event_scores=simulation_event_scores,
        settings=_settings(config, grid, sessions),
    )


def summarize_observation_sweep(
    position_summary: pd.DataFrame,
    simulation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge behavioral decoding and synthetic-recovery gates into one table."""

    if position_summary.empty:
        return pd.DataFrame()
    summary = position_summary.copy()
    if not simulation_summary.empty:
        overall = simulation_summary[simulation_summary["true_model"].eq("overall")].copy()
        if not overall.empty:
            overall = overall[
                [
                    "sweep_id",
                    "session",
                    "recovery_accuracy",
                    "recovered_events",
                    "simulated_events",
                ]
            ].rename(
                columns={
                    "recovery_accuracy": "simulation_recovery_accuracy",
                    "recovered_events": "simulation_recovered_events",
                    "simulated_events": "simulation_events",
                }
            )
            summary = summary.merge(overall, on=["sweep_id", "session"], how="left")
    if "simulation_recovery_accuracy" not in summary:
        summary["simulation_recovery_accuracy"] = np.nan
    for column in ("median_posterior_mean_error_cm", "median_map_error_cm"):
        if column not in summary:
            summary[column] = np.nan
    return summary.sort_values(
        [
            "median_posterior_mean_error_cm",
            "median_map_error_cm",
            "simulation_recovery_accuracy",
            "sweep_id",
            "session",
        ],
        ascending=[True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _run_position_validation_for_session(
    session: ReplaySession,
    encoding: EncodingConfig,
    config: ObservationSweepConfig,
) -> PositionDecodingResult:
    position_config = PositionDecodingConfig(
        encoding=encoding,
        decode_bin_s=config.decode_bin_s,
        n_folds=config.n_folds,
        max_windows_per_session=config.max_windows_per_session,
        random_seed=config.random_seed,
        min_spikes_per_window=config.min_spikes_per_window,
        session=session.session_id,
    )
    samples = validate_session_position_decoding(session, position_config)
    return PositionDecodingResult(samples=samples, summary=summarize_position_decoding(samples))


def _run_simulation_recovery_for_session(
    root: str | Path,
    session_id: str,
    encoding: EncodingConfig,
    params: dict[str, float | int],
    config: ObservationSweepConfig,
) -> SimulationRecoveryResult:
    simulation_config = SimulationRecoveryConfig(
        true_models=parse_model_list(config.simulation_true_models),
        scoring_models=parse_model_list(config.simulation_scoring_models),
        events=config.simulation_events,
        max_template_events=config.simulation_max_template_events,
        events_per_model=config.simulation_events_per_model,
        random_seed=config.random_seed,
        time_bin_s=float(params["time_bin_s"]),
        spike_rate_scale=float(params["spike_rate_scale"]),
        encoding=encoding,
        continue_on_error=config.simulation_continue_on_error,
    )
    return run_session_simulation_recovery(root, session_id, simulation_config)


def _encoding_from_params(params: dict[str, float | int]) -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=float(params["bin_size_cm"]),
        smoothing_sigma_bins=float(params["smoothing_sigma_bins"]),
        min_speed_cm_s=float(params["min_speed_cm_s"]),
        min_occupancy_s=float(params["min_occupancy_s"]),
        rate_floor_hz=float(params["rate_floor_hz"]),
    )


def _attach_sweep_metadata(
    frame: pd.DataFrame,
    params: dict[str, float | int],
    session_id: str,
    *,
    include_empty_row: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        if include_empty_row:
            return pd.DataFrame([{**params, "session": session_id}])
        return pd.DataFrame()
    output = frame.copy()
    for key, value in params.items():
        output[key] = value
    if "session" not in output:
        output["session"] = session_id
    front = [*params.keys(), "session"]
    ordered = [column for column in front if column in output.columns]
    ordered.extend(column for column in output.columns if column not in ordered)
    return output.loc[:, ordered]


def _selected_sessions(root: str | Path, config: ObservationSweepConfig) -> tuple[ReplaySession, ...]:
    sessions = tuple(load_open_field_sessions(root))
    if config.sessions is None:
        return sessions
    by_id = {session.session_id: session for session in sessions}
    missing = sorted(set(config.sessions) - set(by_id))
    if missing:
        available = ", ".join(sorted(by_id))
        raise KeyError(f"unknown sessions {missing}; available: {available}")
    return tuple(by_id[session_id] for session_id in config.sessions)


def _validate_config(config: ObservationSweepConfig) -> None:
    for name in (
        "bin_sizes_cm",
        "smoothing_sigmas_bins",
        "min_speed_cm_s",
        "min_occupancy_s",
        "rate_floor_hz",
        "time_bin_ms",
        "spike_rate_scales",
    ):
        values = getattr(config, name)
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        if any(float(value) <= 0.0 for value in values):
            raise ValueError(f"{name} values must be positive")
    if config.n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if config.decode_bin_s <= 0.0:
        raise ValueError("decode_bin_s must be positive")
    if config.simulation_events_per_model <= 0:
        raise ValueError("simulation_events_per_model must be positive")


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


def _settings(
    config: ObservationSweepConfig,
    grid: list[dict[str, float | int]],
    sessions: tuple[ReplaySession, ...],
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "sessions": [session.session_id for session in sessions],
        "grid_size": len(grid),
        "grid": grid,
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
