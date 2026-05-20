"""Clusterless model support for post-hoc ground-truth decoding."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import score_metadata as _score_metadata
from .clusterless import (
    ClusterlessMarkConfig,
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from .encoding import EmissionConfig, EncodingConfig
from .state_space import StateSpaceDecoderConfig

_CLUSTERLESS_PREFIX = "clusterless-state-space-"


def apply_clusterless_ground_truth_patch() -> None:
    """Teach post-hoc decoded-endpoint comparison about clusterless models."""

    from . import benchmarks as bench
    from . import ground_truth as gt

    if getattr(gt, "_clusterless_ground_truth_patch_applied", False):
        return

    base_build_models = gt._build_models
    base_compare_scores_to_ground_truth = gt.compare_scores_to_ground_truth

    def build_models(config: object, session=None) -> dict[str, object]:
        names = tuple(getattr(config, "models"))
        output: dict[str, object] = {}
        non_clusterless = tuple(name for name in names if not _is_clusterless_model_name(name))
        if non_clusterless:
            output.update(base_build_models(_copy_config_with_models(config, non_clusterless), session=session))
        for name in names:
            if _is_clusterless_model_name(name):
                mode = str(name).removeprefix(_CLUSTERLESS_PREFIX)
                output[str(name)] = _clusterless_state_space_model(config, mode)
        return {str(name): output[str(name)] for name in names}

    def compare_scores_to_ground_truth(root, scores, **kwargs) -> pd.DataFrame:
        scores_frame = pd.read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
        if scores_frame.empty:
            return scores_frame
        scores_frame["_score_order"] = np.arange(len(scores_frame))
        clusterless_mask = _score_frame_clusterless_mask(scores_frame)
        pieces: list[pd.DataFrame] = []
        if (~clusterless_mask).any():
            pieces.append(
                base_compare_scores_to_ground_truth(
                    root,
                    scores_frame.loc[~clusterless_mask].copy(),
                    **kwargs,
                )
            )
        if clusterless_mask.any():
            pieces.append(
                _compare_clusterless_scores_to_ground_truth(
                    gt,
                    root,
                    scores_frame.loc[clusterless_mask].copy(),
                    **kwargs,
                )
            )
        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, ignore_index=True, sort=False)
        if "_score_order" in out:
            out = out.sort_values("_score_order").drop(columns=["_score_order"]).reset_index(drop=True)
        return out

    bench._build_models = build_models
    gt._build_models = build_models
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth
    gt._clusterless_ground_truth_patch_applied = True


def _compare_clusterless_scores_to_ground_truth(gt, root, scores_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ground_truth = kwargs.get("ground_truth")
    ground_truth_config = kwargs.get("ground_truth_config")
    encoding_config = _score_metadata.encoding_config_for_scores(
        scores_frame,
        EncodingConfig() if kwargs.get("encoding_config") is None else kwargs["encoding_config"],
    )
    emission_config = _score_metadata.emission_config_for_scores(
        scores_frame,
        EmissionConfig() if kwargs.get("emission_config") is None else kwargs["emission_config"],
    )
    model_names = gt._model_names_for_scores(scores_frame)
    model_config = _clusterless_model_config_for_scores(
        scores_frame,
        model_names=model_names,
        state_space_stationary_sigma_cm=kwargs.get("state_space_stationary_sigma_cm", 2.0),
        state_space_diffusion_sigma_cm_sqrt_s=kwargs.get("state_space_diffusion_sigma_cm_sqrt_s", 85.0),
        state_space_max_step_sigma=kwargs.get("state_space_max_step_sigma", 4.0),
        state_space_imm_mode_stickiness=kwargs.get("state_space_imm_mode_stickiness", 0.95),
        state_space_momentum_sigma_cm_sqrt_s=kwargs.get("state_space_momentum_sigma_cm_sqrt_s", 85.0),
        state_space_momentum_initial_sigma_cm_sqrt_s=kwargs.get(
            "state_space_momentum_initial_sigma_cm_sqrt_s", 85.0
        ),
        state_space_momentum_velocity_decay=kwargs.get("state_space_momentum_velocity_decay", 0.95),
        state_space_momentum_candidate_top_k=kwargs.get("state_space_momentum_candidate_top_k", 128),
        clusterless_mark_smoothing_sigma_bins=kwargs.get("clusterless_mark_smoothing_sigma_bins", 1.0),
        clusterless_mark_prior_count=kwargs.get("clusterless_mark_prior_count", 1.0),
        clusterless_mark_variance_floor=kwargs.get("clusterless_mark_variance_floor", 1.0),
        clusterless_rate_floor_hz=kwargs.get("clusterless_rate_floor_hz", 1e-4),
    )
    gt_frame = gt._load_or_generate_ground_truth(root, ground_truth, ground_truth_config)
    sessions = {session.session_id: session for session in gt.load_open_field_sessions(root)}
    decoded_rows: list[dict[str, object]] = []
    for session_id, session_scores in scores_frame.groupby("session", sort=False):
        session = sessions.get(str(session_id))
        if session is None:
            continue
        models = gt._build_models(_copy_config_with_models(model_config, gt._model_names_for_scores(session_scores)), session=session)
        wells = gt.infer_well_locations(session, ground_truth_config)
        clusterless_encoding = fit_clusterless_mark_encoding(
            session,
            ClusterlessMarkConfig(
                encoding=encoding_config,
                mark_smoothing_sigma_bins=model_config.clusterless_mark_smoothing_sigma_bins,
                mark_prior_count=model_config.clusterless_mark_prior_count,
                mark_variance_floor=model_config.clusterless_mark_variance_floor,
                rate_floor_hz=model_config.clusterless_rate_floor_hz,
                use_excitatory=encoding_config.use_excitatory,
            ),
        )
        for event_index, event_scores in session_scores.groupby("event_index", sort=False):
            emissions = build_clusterless_mark_emissions(session, clusterless_encoding, int(event_index), emission_config)
            if emissions.n_time == 0:
                continue
            for score_row in event_scores.itertuples(index=False):
                model_name = str(getattr(score_row, "model"))
                requested_model = gt._requested_model_name(score_row, model_name)
                model = models.get(requested_model) or models.get(model_name)
                if model is None:
                    continue
                score = model.score(emissions, clusterless_encoding.bin_centers)
                decoded_rows.append(
                    gt._decoded_row(
                        str(session_id),
                        int(event_index),
                        model_name,
                        score.terminal_log_posterior,
                        score.trajectory_log_posterior,
                        clusterless_encoding.bin_centers,
                        wells,
                    )
                )
    decoded = pd.DataFrame(decoded_rows)
    if decoded.empty:
        decoded = pd.DataFrame(columns=["session", "event_index", "model"])
    comparison = scores_frame.merge(gt_frame, on=["session", "event_index"], how="left")
    comparison = comparison.merge(decoded, on=["session", "event_index", "model"], how="left")
    return gt._add_ground_truth_metrics(comparison, decoded, gt_frame)


def _clusterless_state_space_model(config: object, mode: str) -> ClusterlessStateSpaceReplayModel:
    return ClusterlessStateSpaceReplayModel(
        mode=mode,
        config=StateSpaceDecoderConfig(
            mode=mode,
            stationary_sigma_cm=_cfg(config, "state_space_stationary_sigma_cm", 2.0),
            diffusion_sigma_cm_sqrt_s=_cfg(config, "state_space_diffusion_sigma_cm_sqrt_s", 85.0),
            max_step_sigma=_cfg(config, "state_space_max_step_sigma", 4.0),
            imm_mode_stickiness=_cfg(config, "state_space_imm_mode_stickiness", 0.95),
            momentum_sigma_cm_sqrt_s=_cfg(config, "state_space_momentum_sigma_cm_sqrt_s", 85.0),
            momentum_initial_sigma_cm_sqrt_s=_cfg(
                config,
                "state_space_momentum_initial_sigma_cm_sqrt_s",
                85.0,
            ),
            momentum_velocity_decay=_cfg(config, "state_space_momentum_velocity_decay", 0.95),
            momentum_candidate_top_k=_cfg(config, "state_space_momentum_candidate_top_k", 128),
        ),
    )


def _clusterless_model_config_for_scores(scores_frame: pd.DataFrame, *, model_names: tuple[str, ...], **defaults) -> SimpleNamespace:
    return SimpleNamespace(
        models=model_names,
        state_space_stationary_sigma_cm=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("state_space_stationary_sigma_cm", "diagnostic_state_space_stationary_sigma_cm"),
            defaults["state_space_stationary_sigma_cm"],
        ),
        state_space_diffusion_sigma_cm_sqrt_s=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("state_space_diffusion_sigma_cm_sqrt_s", "diagnostic_state_space_diffusion_sigma_cm_sqrt_s"),
            defaults["state_space_diffusion_sigma_cm_sqrt_s"],
        ),
        state_space_max_step_sigma=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("state_space_max_step_sigma", "diagnostic_state_space_max_step_sigma"),
            defaults["state_space_max_step_sigma"],
        ),
        state_space_imm_mode_stickiness=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("state_space_imm_mode_stickiness", "diagnostic_state_space_imm_mode_stickiness"),
            defaults["state_space_imm_mode_stickiness"],
        ),
        state_space_momentum_sigma_cm_sqrt_s=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("state_space_momentum_sigma_cm_sqrt_s", "diagnostic_state_space_momentum_sigma_cm_sqrt_s"),
            defaults["state_space_momentum_sigma_cm_sqrt_s"],
        ),
        state_space_momentum_initial_sigma_cm_sqrt_s=_score_metadata._unique_float_from_columns(
            scores_frame,
            (
                "state_space_momentum_initial_sigma_cm_sqrt_s",
                "diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s",
            ),
            defaults["state_space_momentum_initial_sigma_cm_sqrt_s"],
        ),
        state_space_momentum_velocity_decay=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("state_space_momentum_velocity_decay", "diagnostic_state_space_momentum_velocity_decay"),
            defaults["state_space_momentum_velocity_decay"],
        ),
        state_space_momentum_candidate_top_k=_score_metadata._unique_int_from_columns(
            scores_frame,
            (
                "state_space_momentum_candidate_top_k",
                "diagnostic_state_space_momentum_candidate_top_k",
                "diagnostic_state_space_imm_candidate_top_k",
            ),
            defaults["state_space_momentum_candidate_top_k"],
        ),
        clusterless_mark_smoothing_sigma_bins=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("clusterless_mark_smoothing_sigma_bins",),
            defaults["clusterless_mark_smoothing_sigma_bins"],
        ),
        clusterless_mark_prior_count=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("clusterless_mark_prior_count",),
            defaults["clusterless_mark_prior_count"],
        ),
        clusterless_mark_variance_floor=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("clusterless_mark_variance_floor",),
            defaults["clusterless_mark_variance_floor"],
        ),
        clusterless_rate_floor_hz=_score_metadata._unique_float_from_columns(
            scores_frame,
            ("clusterless_rate_floor_hz",),
            defaults["clusterless_rate_floor_hz"],
        ),
    )


def _score_frame_clusterless_mask(scores_frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=scores_frame.index)
    for column in ("requested_model", "model"):
        if column in scores_frame:
            mask |= scores_frame[column].fillna("").astype(str).str.startswith(_CLUSTERLESS_PREFIX)
    return mask


def _copy_config_with_models(config: object, models: tuple[str, ...]) -> SimpleNamespace:
    values = dict(getattr(config, "__dict__", {}))
    values["models"] = models
    return SimpleNamespace(**values)


def _is_clusterless_model_name(name: object) -> bool:
    return str(name).startswith(_CLUSTERLESS_PREFIX)


def _cfg(config: object, name: str, default):
    return getattr(config, name, default)
