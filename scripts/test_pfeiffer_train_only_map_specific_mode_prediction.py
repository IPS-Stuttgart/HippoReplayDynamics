#!/usr/bin/env python3
"""Leakage-free test linking train-only IMM mode content to held-out cells."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hipporeplayimm  # noqa: E402

from _provenance import build_script_provenance, file_sha256, git_metadata  # noqa: E402
from audit_pfeiffer_imm_gate_convergence import (  # noqa: E402
    partial_spearman,
    rat_cluster_bootstrap_partial,
    raw_spearman,
)
from extract_replay_commitment_composition_posteriors import (  # noqa: E402
    _successful_rows,
    decoder_configs,
)
from hipporeplayimm.benchmarks import _split_cells  # noqa: E402
from hipporeplayimm.data import load_replay_session  # noqa: E402
from hipporeplayimm.encoding import (  # noqa: E402
    EncodingModel,
    build_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.sorted_spike_state_space import (  # noqa: E402
    SortedSpikeStateSpaceReplayModel,
)

try:  # The frozen paper scorer predates this analysis-only helper module.
    from hipporeplayimm.frozen_posterior_prediction import (  # noqa: E402
        frozen_smoothed_marginal_log_score,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the pinned server run.
    LOCAL_PACKAGE_DIR = ROOT / "src" / "hipporeplayimm"
    if str(LOCAL_PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_PACKAGE_DIR))
    from frozen_posterior_prediction import (  # type: ignore[no-redef]  # noqa: E402
        frozen_smoothed_marginal_log_score,
    )


IMM = "sorted-spike-state-space-first-order-imm"
FRAGMENTED = "sorted-spike-state-space-fragmented"
PAIR_MODELS = (IMM, FRAGMENTED)
MAP_CONDITIONS = ("real_map", "population_code_permuted")

SPLIT_OUTPUT = "pfeiffer_train_only_map_specific_mode_split_scores.csv"
EVENT_OUTPUT = "pfeiffer_train_only_map_specific_mode_event_medians.csv"
ASSOCIATION_OUTPUT = "pfeiffer_train_only_map_specific_mode_associations.csv"
BY_RAT_OUTPUT = "pfeiffer_train_only_map_specific_mode_by_rat.csv"
LOO_OUTPUT = "pfeiffer_train_only_map_specific_mode_leave_one_rat_out.csv"
PERMUTATION_OUTPUT = "pfeiffer_train_only_map_specific_mode_permutation_null.csv"
GATE_OUTPUT = "pfeiffer_train_only_map_specific_mode_gate_summary.csv"
MANIFEST_OUTPUT = "pfeiffer_train_only_map_specific_mode_manifest.json"
REPORT_OUTPUT = "pfeiffer_train_only_map_specific_mode_report.md"

PRIMARY_X = "train_map_specific_nonstationary_mass_event_median"
PRIMARY_Y = "real_frozen_heldout_delta_imm_minus_fragmented_event_median"
MAP_SPECIFIC_Y = "map_specific_frozen_heldout_delta_event_median"
CORE_CONTROLS = (
    "log1p_median_train_cell_count",
    "log1p_median_test_spikes",
    "median_real_imm_train_posterior_entropy",
    "log1p_median_n_time",
)
EXTENDED_CONTROLS = (*CORE_CONTROLS, "log1p_median_train_spikes")


def _stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(int(seed)), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _format_cell_ids(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in np.asarray(list(values), dtype=int))


def _permutation_sha256(occupied: np.ndarray, permutation: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(occupied, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(permutation, dtype="<i8").tobytes())
    return digest.hexdigest()


def population_code_permuted_encoding(
    encoding: EncodingModel,
    *,
    seed: int,
) -> tuple[EncodingModel, str]:
    """Apply one occupied-bin permutation to every cell's RUN rate map."""

    occupied = np.flatnonzero(np.asarray(encoding.occupancy_s, dtype=float) > 0.0)
    if occupied.size < 2:
        raise ValueError("at least two occupied spatial bins are required")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(occupied.size)
    rates = np.asarray(encoding.rates_hz, dtype=float).copy()
    rates[:, occupied] = rates[:, occupied][:, permutation]
    return replace(encoding, rates_hz=rates), _permutation_sha256(occupied, permutation)


def _models(state_config: Any) -> dict[str, SortedSpikeStateSpaceReplayModel]:
    return {
        IMM: SortedSpikeStateSpaceReplayModel(
            mode="first-order-imm",
            config=replace(state_config, mode="first-order-imm"),
            name=IMM,
        ),
        FRAGMENTED: SortedSpikeStateSpaceReplayModel(
            mode="fragmented",
            config=replace(state_config, mode="fragmented"),
            name=FRAGMENTED,
        ),
    }


def _posterior_entropy(log_posterior: np.ndarray) -> float:
    values = np.asarray(log_posterior, dtype=float)
    values = values - np.logaddexp.reduce(values, axis=1, keepdims=True)
    probability = np.exp(values)
    return float(
        np.mean(-np.sum(probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=1))
    )


def _mode_metrics(score: Any) -> dict[str, float]:
    diagnostics = score.diagnostics
    key = "state_space_imm_mode_posterior_over_time"
    if key not in diagnostics:
        raise ValueError("first-order IMM score did not export its mode posterior")
    mode = np.asarray(json.loads(str(diagnostics[key])), dtype=float)
    if mode.ndim != 2 or mode.shape[1] != 3 or mode.shape[0] != int(score.n_time):
        raise ValueError(f"unexpected first-order IMM mode-posterior shape: {mode.shape}")
    if not np.all(np.isfinite(mode)) or np.any(mode < 0.0):
        raise ValueError("IMM mode posterior must contain finite nonnegative mass")
    row_sum = mode.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("IMM mode posterior rows must contain positive mass")
    mode = mode / row_sum
    map_mode = np.argmax(mode, axis=1)
    return {
        "mean_nonstationary_mode_probability": float(mode[:, 1:].sum(axis=1).mean()),
        "fraction_time_map_nonstationary": float(np.mean(map_mode != 0)),
        "mean_stationary_mode_probability": float(mode[:, 0].mean()),
    }


def infer_training_posteriors(
    models: dict[str, SortedSpikeStateSpaceReplayModel],
    train_emissions: Any,
    bin_centers: np.ndarray,
) -> dict[str, Any]:
    """Infer each latent posterior using training-cell replay emissions only."""

    scores: dict[str, Any] = {}
    for name in PAIR_MODELS:
        score = models[name].score(train_emissions, bin_centers)
        if score.trajectory_log_posterior is None:
            raise ValueError(f"{name} did not return a training trajectory posterior")
        posterior = np.asarray(score.trajectory_log_posterior, dtype=float)
        if posterior.shape != np.asarray(train_emissions.log_likelihood).shape:
            raise ValueError(
                f"{name} posterior shape {posterior.shape} does not match "
                f"training emissions {train_emissions.log_likelihood.shape}"
            )
        scores[name] = score
    return scores


def score_frozen_heldout(
    training_scores: dict[str, Any],
    heldout_emissions: Any,
) -> dict[str, dict[str, object]]:
    """Score held-out cells without invoking model inference again."""

    rows: dict[str, dict[str, object]] = {}
    for name in PAIR_MODELS:
        posterior = training_scores[name].trajectory_log_posterior
        if posterior is None:  # pragma: no cover - guarded during inference.
            raise ValueError(f"{name} training posterior is missing")
        score = frozen_smoothed_marginal_log_score(
            posterior,
            heldout_emissions.log_likelihood,
        )
        rows[name] = {
            "frozen_heldout_log_score": float(score.total_log_score),
            "frozen_heldout_mean_log_score_per_time_bin": float(
                score.mean_log_score_per_time_bin
            ),
            "training_posterior_sha256": score.posterior_sha256,
        }
    return rows


def _event_keys(evidence: pd.DataFrame) -> pd.DataFrame:
    pair = evidence[evidence["model"].astype(str).isin(PAIR_MODELS)].copy()
    pivot = pair.pivot_table(
        index=["session", "event_index"],
        columns="model",
        values="log_evidence",
        aggfunc="last",
    )
    pivot = pivot.dropna(subset=list(PAIR_MODELS)).reset_index()
    pivot["event_index"] = pd.to_numeric(pivot["event_index"], errors="raise").astype(int)
    pivot["rat"] = pivot["session"].astype(str).str.split("/").str[0]
    return pivot.sort_values(["session", "event_index"]).reset_index(drop=True)


def _score_event(
    *,
    session: Any,
    event_index: int,
    split_index: int,
    split_seed: int,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
    real_encoding: EncodingModel,
    wrong_encoding: EncodingModel,
    wrong_map_sha256: str,
    emission_config: Any,
    state_config: Any,
    margin_threshold: float,
) -> dict[str, object]:
    common = {
        "session": str(session.session_id),
        "rat": str(session.rat),
        "event_index": int(event_index),
        "cell_split_index": int(split_index),
        "cell_split_seed": int(split_seed),
        "actual_test_cell_fraction": float(len(test_cells) / (len(train_cells) + len(test_cells))),
        "train_cell_count": int(len(train_cells)),
        "test_cell_count": int(len(test_cells)),
        "train_cell_ids": _format_cell_ids(train_cells),
        "test_cell_ids": _format_cell_ids(test_cells),
        "cell_sets_disjoint": bool(not np.intersect1d(train_cells, test_cells).size),
        "wrong_map_permutation_sha256": wrong_map_sha256,
        "heldout_replay_spikes_used_for_latent_inference": False,
        "latent_inference_cell_scope": "training_cells_only",
        "heldout_scoring_method": "frozen_smoothed_position_marginal",
        "selection_scope": "all_160_events_no_all_cell_clean_imm_selection",
    }
    try:
        train_encodings = {
            "real_map": real_encoding.select_cells(train_cells),
            "population_code_permuted": wrong_encoding.select_cells(train_cells),
        }
        test_encodings = {
            "real_map": real_encoding.select_cells(test_cells),
            "population_code_permuted": wrong_encoding.select_cells(test_cells),
        }
        models = _models(state_config)
        training_emissions: dict[str, Any] = {}
        training_scores: dict[str, dict[str, Any]] = {}
        # All latent inference is completed before held-out emissions are built.
        for condition in MAP_CONDITIONS:
            emissions = build_emissions(
                session,
                train_encodings[condition],
                int(event_index),
                emission_config,
            )
            training_emissions[condition] = emissions
            training_scores[condition] = infer_training_posteriors(
                models,
                emissions,
                train_encodings[condition].bin_centers,
            )

        heldout_emissions: dict[str, Any] = {}
        frozen_scores: dict[str, dict[str, dict[str, object]]] = {}
        for condition in MAP_CONDITIONS:
            emissions = build_emissions(
                session,
                test_encodings[condition],
                int(event_index),
                emission_config,
            )
            heldout_emissions[condition] = emissions
            if emissions.n_time != training_emissions[condition].n_time:
                raise ValueError("training and held-out emissions have different time bins")
            frozen_scores[condition] = score_frozen_heldout(
                training_scores[condition],
                emissions,
            )

        real_imm = training_scores["real_map"][IMM]
        real_frag = training_scores["real_map"][FRAGMENTED]
        wrong_imm = training_scores["population_code_permuted"][IMM]
        wrong_frag = training_scores["population_code_permuted"][FRAGMENTED]
        real_mode = _mode_metrics(real_imm)
        wrong_mode = _mode_metrics(wrong_imm)
        real_heldout_imm = float(
            frozen_scores["real_map"][IMM]["frozen_heldout_log_score"]
        )
        real_heldout_frag = float(
            frozen_scores["real_map"][FRAGMENTED]["frozen_heldout_log_score"]
        )
        wrong_heldout_imm = float(
            frozen_scores["population_code_permuted"][IMM]["frozen_heldout_log_score"]
        )
        wrong_heldout_frag = float(
            frozen_scores["population_code_permuted"][FRAGMENTED][
                "frozen_heldout_log_score"
            ]
        )
        train_delta = float(real_imm.log_likelihood - real_frag.log_likelihood)
        event = session.ripple(int(event_index))
        return {
            **common,
            "status": "success",
            "error": "",
            "event_start_s": float(event.start),
            "event_end_s": float(event.end),
            "event_duration_s": float(event.end - event.start),
            "n_time": int(training_emissions["real_map"].n_time),
            "train_spikes": int(training_emissions["real_map"].n_spikes),
            "test_spikes": int(heldout_emissions["real_map"].n_spikes),
            "real_train_logZ_imm": float(real_imm.log_likelihood),
            "real_train_logZ_fragmented": float(real_frag.log_likelihood),
            "real_train_delta_imm_minus_fragmented": train_delta,
            "wrong_train_logZ_imm": float(wrong_imm.log_likelihood),
            "wrong_train_logZ_fragmented": float(wrong_frag.log_likelihood),
            "wrong_train_delta_imm_minus_fragmented": float(
                wrong_imm.log_likelihood - wrong_frag.log_likelihood
            ),
            "train_defined_clean_imm": bool(train_delta >= float(margin_threshold)),
            "real_train_mean_nonstationary_mode_probability": real_mode[
                "mean_nonstationary_mode_probability"
            ],
            "wrong_train_mean_nonstationary_mode_probability": wrong_mode[
                "mean_nonstationary_mode_probability"
            ],
            "train_map_specific_nonstationary_mass": float(
                real_mode["mean_nonstationary_mode_probability"]
                - wrong_mode["mean_nonstationary_mode_probability"]
            ),
            "real_train_fraction_time_map_nonstationary": real_mode[
                "fraction_time_map_nonstationary"
            ],
            "wrong_train_fraction_time_map_nonstationary": wrong_mode[
                "fraction_time_map_nonstationary"
            ],
            "train_map_specific_fraction_time_map_nonstationary": float(
                real_mode["fraction_time_map_nonstationary"]
                - wrong_mode["fraction_time_map_nonstationary"]
            ),
            "real_imm_train_posterior_entropy": _posterior_entropy(
                np.asarray(real_imm.trajectory_log_posterior)
            ),
            "wrong_imm_train_posterior_entropy": _posterior_entropy(
                np.asarray(wrong_imm.trajectory_log_posterior)
            ),
            "real_frozen_heldout_log_score_imm": real_heldout_imm,
            "real_frozen_heldout_log_score_fragmented": real_heldout_frag,
            "real_frozen_heldout_delta_imm_minus_fragmented": float(
                real_heldout_imm - real_heldout_frag
            ),
            "wrong_frozen_heldout_log_score_imm": wrong_heldout_imm,
            "wrong_frozen_heldout_log_score_fragmented": wrong_heldout_frag,
            "wrong_frozen_heldout_delta_imm_minus_fragmented": float(
                wrong_heldout_imm - wrong_heldout_frag
            ),
            "map_specific_frozen_heldout_delta": float(
                (real_heldout_imm - real_heldout_frag)
                - (wrong_heldout_imm - wrong_heldout_frag)
            ),
            "real_imm_training_posterior_sha256": frozen_scores["real_map"][IMM][
                "training_posterior_sha256"
            ],
            "real_fragmented_training_posterior_sha256": frozen_scores["real_map"][
                FRAGMENTED
            ]["training_posterior_sha256"],
            "wrong_imm_training_posterior_sha256": frozen_scores[
                "population_code_permuted"
            ][IMM]["training_posterior_sha256"],
            "wrong_fragmented_training_posterior_sha256": frozen_scores[
                "population_code_permuted"
            ][FRAGMENTED]["training_posterior_sha256"],
        }
    except Exception as exc:
        return {
            **common,
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _score_task(task: dict[str, Any]) -> list[dict[str, object]]:
    session_id = str(task["session"])
    session = load_replay_session(Path(task["dataset_root"]) / Path(session_id))
    real_encoding = fit_place_field_encoding(session, task["encoding_config"])
    wrong_encoding, wrong_hash = population_code_permuted_encoding(
        real_encoding,
        seed=_stable_seed(int(task["wrong_map_seed"]), session_id),
    )
    rows: list[dict[str, object]] = []
    for split_index in task["split_indices"]:
        split_seed = int(task["random_seed"]) + int(split_index)
        train_cells, test_cells = _split_cells(
            real_encoding.cell_ids,
            float(task["test_cell_fraction"]),
            split_seed,
        )
        for event_index in task["event_indices"]:
            rows.append(
                _score_event(
                    session=session,
                    event_index=int(event_index),
                    split_index=int(split_index),
                    split_seed=split_seed,
                    train_cells=train_cells,
                    test_cells=test_cells,
                    real_encoding=real_encoding,
                    wrong_encoding=wrong_encoding,
                    wrong_map_sha256=wrong_hash,
                    emission_config=task["emission_config"],
                    state_config=task["state_config"],
                    margin_threshold=float(task["margin_threshold"]),
                )
            )
    return rows


def _chunks(values: Sequence[int], size: int) -> list[list[int]]:
    if int(size) < 1:
        raise ValueError("splits_per_task must be at least one")
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def score_all_splits(
    *,
    dataset_root: Path,
    events: pd.DataFrame,
    evidence: pd.DataFrame,
    n_splits: int,
    test_cell_fraction: float,
    random_seed: int,
    wrong_map_seed: int,
    margin_threshold: float,
    workers: int,
    splits_per_task: int,
    partial_output: Path | None = None,
) -> pd.DataFrame:
    """Score every selected event under repeated leakage-free cell splits."""

    encoding_config, emission_config, state_config = decoder_configs(evidence)
    split_chunks = _chunks(list(range(int(n_splits))), int(splits_per_task))
    tasks: list[dict[str, object]] = []
    for session_id, group in events.groupby("session", sort=True):
        event_indices = group["event_index"].astype(int).tolist()
        for split_indices in split_chunks:
            tasks.append(
                {
                    "dataset_root": str(dataset_root),
                    "session": str(session_id),
                    "event_indices": event_indices,
                    "split_indices": split_indices,
                    "test_cell_fraction": float(test_cell_fraction),
                    "random_seed": int(random_seed),
                    "wrong_map_seed": int(wrong_map_seed),
                    "margin_threshold": float(margin_threshold),
                    "encoding_config": encoding_config,
                    "emission_config": emission_config,
                    "state_config": state_config,
                }
            )
    rows: list[dict[str, object]] = []
    if int(workers) <= 1:
        task_results = (_score_task(task) for task in tasks)
        for result in task_results:
            rows.extend(result)
            if partial_output is not None:
                pd.DataFrame(rows).to_csv(partial_output, index=False)
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as executor:
            futures = [executor.submit(_score_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())
                if partial_output is not None:
                    pd.DataFrame(rows).to_csv(partial_output, index=False)
    return pd.DataFrame(rows).sort_values(
        ["session", "event_index", "cell_split_index"]
    ).reset_index(drop=True)


def build_event_medians(split_scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated splits so every event has equal primary weight."""

    success = split_scores[split_scores["status"].astype(str).eq("success")].copy()
    numeric = [
        "train_map_specific_nonstationary_mass",
        "real_frozen_heldout_delta_imm_minus_fragmented",
        "wrong_frozen_heldout_delta_imm_minus_fragmented",
        "map_specific_frozen_heldout_delta",
        "real_train_mean_nonstationary_mode_probability",
        "wrong_train_mean_nonstationary_mode_probability",
        "real_imm_train_posterior_entropy",
        "train_cell_count",
        "test_cell_count",
        "train_spikes",
        "test_spikes",
        "n_time",
        "event_duration_s",
    ]
    rows: list[dict[str, object]] = []
    for (session, rat, event_index), group in success.groupby(
        ["session", "rat", "event_index"], sort=True
    ):
        medians = {
            column: float(pd.to_numeric(group[column], errors="coerce").median())
            for column in numeric
        }
        row = {
            "session": str(session),
            "rat": str(rat),
            "event_index": int(event_index),
            "completed_splits": int(group["cell_split_index"].nunique()),
            PRIMARY_X: medians["train_map_specific_nonstationary_mass"],
            PRIMARY_Y: medians["real_frozen_heldout_delta_imm_minus_fragmented"],
            "wrong_frozen_heldout_delta_event_median": medians[
                "wrong_frozen_heldout_delta_imm_minus_fragmented"
            ],
            MAP_SPECIFIC_Y: medians["map_specific_frozen_heldout_delta"],
            "median_real_train_nonstationary_mass": medians[
                "real_train_mean_nonstationary_mode_probability"
            ],
            "median_wrong_train_nonstationary_mass": medians[
                "wrong_train_mean_nonstationary_mode_probability"
            ],
            "median_real_imm_train_posterior_entropy": medians[
                "real_imm_train_posterior_entropy"
            ],
            "median_train_cell_count": medians["train_cell_count"],
            "median_test_cell_count": medians["test_cell_count"],
            "median_train_spikes": medians["train_spikes"],
            "median_test_spikes": medians["test_spikes"],
            "median_n_time": medians["n_time"],
            "event_duration_s": medians["event_duration_s"],
            "train_defined_clean_imm_fraction": float(
                group["train_defined_clean_imm"].astype(bool).mean()
            ),
        }
        row["train_defined_clean_imm_majority"] = bool(
            row["train_defined_clean_imm_fraction"] >= 0.5
        )
        row["log1p_median_train_cell_count"] = float(
            np.log1p(row["median_train_cell_count"])
        )
        row["log1p_median_test_spikes"] = float(np.log1p(row["median_test_spikes"]))
        row["log1p_median_train_spikes"] = float(np.log1p(row["median_train_spikes"]))
        row["log1p_median_n_time"] = float(np.log1p(row["median_n_time"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _permutation_null(
    events: pd.DataFrame,
    *,
    x: str,
    y: str,
    controls: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    observed, _, _, _ = partial_spearman(
        events,
        x,
        y,
        list(controls),
        rat_fixed_effects=True,
    )
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for replicate in range(int(replicates)):
        shuffled = events.copy()
        for _, indices in shuffled.groupby("session", sort=True).groups.items():
            index = np.asarray(list(indices))
            shuffled.loc[index, x] = rng.permutation(shuffled.loc[index, x].to_numpy())
        estimate, _, _, _ = partial_spearman(
            shuffled,
            x,
            y,
            list(controls),
            rat_fixed_effects=True,
        )
        rows.append({"replicate": int(replicate), "partial_spearman_rho": estimate})
    null = pd.DataFrame(rows)
    values = pd.to_numeric(null["partial_spearman_rho"], errors="coerce").dropna()
    p_value = float((1 + int((values >= observed).sum())) / (1 + len(values)))
    return null, p_value


def _association_row(
    events: pd.DataFrame,
    *,
    analysis_id: str,
    x: str,
    y: str,
    bootstrap_replicates: int,
    seed: int,
    permutation_replicates: int = 0,
) -> tuple[dict[str, object], pd.DataFrame]:
    raw_rho, raw_p, n_events, n_rats = raw_spearman(events, x, y)
    core_rho, core_p, _, _ = partial_spearman(
        events,
        x,
        y,
        list(CORE_CONTROLS),
        rat_fixed_effects=True,
    )
    extended_rho, extended_p, _, _ = partial_spearman(
        events,
        x,
        y,
        list(EXTENDED_CONTROLS),
        rat_fixed_effects=True,
    )
    ci_low, ci_high, positive_fraction, finite_bootstrap = rat_cluster_bootstrap_partial(
        events,
        x,
        y,
        list(CORE_CONTROLS),
        replicates=int(bootstrap_replicates),
        seed=int(seed),
    )
    permutation = pd.DataFrame(columns=["replicate", "partial_spearman_rho"])
    permutation_p = np.nan
    if int(permutation_replicates) > 0:
        permutation, permutation_p = _permutation_null(
            events,
            x=x,
            y=y,
            controls=CORE_CONTROLS,
            replicates=int(permutation_replicates),
            seed=int(seed) + 1009,
        )
        permutation["analysis_id"] = analysis_id
    return (
        {
            "analysis_id": analysis_id,
            "x_metric": x,
            "y_metric": y,
            "events": int(n_events),
            "rats": int(n_rats),
            "raw_spearman_rho": raw_rho,
            "raw_p_value_descriptive": raw_p,
            "core_adjusted_partial_spearman_rho": core_rho,
            "core_adjusted_p_value_descriptive": core_p,
            "extended_adjusted_partial_spearman_rho": extended_rho,
            "extended_adjusted_p_value_descriptive": extended_p,
            "rat_cluster_bootstrap_ci_low": ci_low,
            "rat_cluster_bootstrap_ci_high": ci_high,
            "rat_cluster_bootstrap_positive_fraction": positive_fraction,
            "finite_bootstrap_replicates": int(finite_bootstrap),
            "within_session_permutation_p_one_sided": permutation_p,
        },
        permutation,
    )


def analyze_associations(
    event_medians: pd.DataFrame,
    split_scores: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    permutations: list[pd.DataFrame] = []
    primary, primary_null = _association_row(
        event_medians,
        analysis_id="primary_all_160_events",
        x=PRIMARY_X,
        y=PRIMARY_Y,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed,
    )
    rows.append(primary)
    permutations.append(primary_null)
    map_outcome, _ = _association_row(
        event_medians,
        analysis_id="secondary_map_specific_heldout_outcome",
        x=PRIMARY_X,
        y=MAP_SPECIFIC_Y,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    rows.append(map_outcome)
    clean = event_medians[event_medians["train_defined_clean_imm_majority"].astype(bool)]
    clean_row, _ = _association_row(
        clean,
        analysis_id="secondary_training_defined_clean_imm_majority",
        x=PRIMARY_X,
        y=PRIMARY_Y,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 2,
    )
    rows.append(clean_row)

    success = split_scores[split_scores["status"].astype(str).eq("success")].copy()
    for column in (
        "train_map_specific_nonstationary_mass",
        "real_frozen_heldout_delta_imm_minus_fragmented",
    ):
        success[f"within_event_{column}"] = success[column] - success.groupby(
            ["session", "event_index"]
        )[column].transform("mean")
    split_rho = spearmanr(
        success["within_event_train_map_specific_nonstationary_mass"],
        success["within_event_real_frozen_heldout_delta_imm_minus_fragmented"],
    )
    rows.append(
        {
            "analysis_id": "secondary_split_level_within_event",
            "x_metric": "within_event_train_map_specific_nonstationary_mass",
            "y_metric": "within_event_real_frozen_heldout_delta_imm_minus_fragmented",
            "events": int(success[["session", "event_index"]].drop_duplicates().shape[0]),
            "rats": int(success["rat"].nunique()),
            "raw_spearman_rho": float(split_rho.statistic),
            "raw_p_value_descriptive": float(split_rho.pvalue),
            "core_adjusted_partial_spearman_rho": np.nan,
            "core_adjusted_p_value_descriptive": np.nan,
            "extended_adjusted_partial_spearman_rho": np.nan,
            "extended_adjusted_p_value_descriptive": np.nan,
            "rat_cluster_bootstrap_ci_low": np.nan,
            "rat_cluster_bootstrap_ci_high": np.nan,
            "rat_cluster_bootstrap_positive_fraction": np.nan,
            "finite_bootstrap_replicates": 0,
            "within_session_permutation_p_one_sided": np.nan,
        }
    )

    by_rat_rows: list[dict[str, object]] = []
    for rat, group in event_medians.groupby("rat", sort=True):
        rho, p_value, n_events, _ = raw_spearman(group, PRIMARY_X, PRIMARY_Y)
        by_rat_rows.append(
            {
                "rat": str(rat),
                "events": int(n_events),
                "raw_spearman_rho": rho,
                "p_value_descriptive": p_value,
                "median_map_specific_nonstationary_mass": float(group[PRIMARY_X].median()),
                "median_frozen_heldout_delta": float(group[PRIMARY_Y].median()),
            }
        )
    loo_rows: list[dict[str, object]] = []
    for omitted in sorted(event_medians["rat"].astype(str).unique()):
        subset = event_medians[~event_medians["rat"].astype(str).eq(omitted)]
        rho, p_value, n_events, n_rats = partial_spearman(
            subset,
            PRIMARY_X,
            PRIMARY_Y,
            list(CORE_CONTROLS),
            rat_fixed_effects=True,
        )
        loo_rows.append(
            {
                "omitted_rat": omitted,
                "events": int(n_events),
                "rats": int(n_rats),
                "core_adjusted_partial_spearman_rho": rho,
                "p_value_descriptive": p_value,
            }
        )
    permutation = pd.concat(permutations, ignore_index=True) if permutations else pd.DataFrame()
    return (
        pd.DataFrame(rows),
        pd.DataFrame(by_rat_rows),
        pd.DataFrame(loo_rows),
        permutation,
    )


def build_gate_summary(
    split_scores: pd.DataFrame,
    event_medians: pd.DataFrame,
    associations: pd.DataFrame,
    by_rat: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    *,
    expected_events: int,
    expected_splits: int,
    test_cell_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        gate_type: str,
        gate: str,
        passed: bool,
        observed: object,
        criterion: str,
    ) -> None:
        rows.append(
            {
                "gate_type": gate_type,
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
            }
        )

    expected_rows = int(expected_events) * int(expected_splits)
    success = split_scores["status"].astype(str).eq("success")
    add("technical", "all_split_event_rows_present", len(split_scores) == expected_rows, f"{len(split_scores)}/{expected_rows}", "one row per event and repeated split")
    add("technical", "all_split_event_rows_successful", bool(success.all()), f"{int(success.sum())}/{len(split_scores)}", "zero scoring failures")
    add("technical", "all_160_events_present", len(event_medians) == expected_events, f"{len(event_medians)}/{expected_events}", "all primary events have event medians")
    split_counts = event_medians["completed_splits"] if not event_medians.empty else pd.Series(dtype=int)
    add("technical", "all_repeated_splits_complete", bool(not split_counts.empty and split_counts.eq(expected_splits).all()), "" if split_counts.empty else f"min={int(split_counts.min())}; max={int(split_counts.max())}", f"{expected_splits} splits per event")
    actual_fraction = pd.to_numeric(split_scores.get("actual_test_cell_fraction"), errors="coerce")
    add("technical", "seventy_thirty_cell_split", bool(not actual_fraction.empty and np.allclose(actual_fraction, test_cell_fraction, atol=0.03)), float(actual_fraction.median()) if not actual_fraction.empty else np.nan, f"held-out fraction approximately {test_cell_fraction:g} after integer cell allocation")
    add("technical", "train_test_cells_disjoint", bool(split_scores.get("cell_sets_disjoint", pd.Series(False)).astype(bool).all()), int(split_scores.get("cell_sets_disjoint", pd.Series(False)).astype(bool).sum()), "all train/test cell sets disjoint")
    no_latent = ~split_scores.get("heldout_replay_spikes_used_for_latent_inference", pd.Series(True)).astype(bool)
    add("technical", "heldout_spikes_never_used_for_latent_inference", bool(no_latent.all()), int(no_latent.sum()), "all posteriors inferred from training cells only")
    hashes = [column for column in split_scores if column.endswith("training_posterior_sha256")]
    valid_hashes = bool(hashes) and all(split_scores[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all() for column in hashes)
    add("technical", "frozen_training_posteriors_hashed", valid_hashes, len(hashes), "four posterior hashes per split-event row")
    finite = np.isfinite(event_medians[[PRIMARY_X, PRIMARY_Y]].to_numpy(dtype=float)).all(axis=1) if not event_medians.empty else np.asarray([], dtype=bool)
    add("technical", "primary_predictor_and_outcome_finite", bool(finite.size and finite.all()), int(finite.sum()), "all event-level primary values finite")
    add("technical", "primary_uses_all_events_without_all_cell_selection", bool(split_scores.get("selection_scope", pd.Series("")).astype(str).eq("all_160_events_no_all_cell_clean_imm_selection").all()), int(event_medians.shape[0]), "all 160 events primary; no all-cell clean-IMM selection")
    technical_pass = all(row["passed"] for row in rows if row["gate_type"] == "technical")
    add("summary", "overall_technical", technical_pass, "pass" if technical_pass else "fail", "all technical leakage and coverage gates pass")

    primary = associations[associations["analysis_id"].eq("primary_all_160_events")].iloc[0]
    all_rats_positive = bool(not by_rat.empty and (pd.to_numeric(by_rat["raw_spearman_rho"], errors="coerce") > 0.0).all())
    all_loo_positive = bool(not leave_one_out.empty and (pd.to_numeric(leave_one_out["core_adjusted_partial_spearman_rho"], errors="coerce") > 0.0).all())
    scientific = [
        ("primary_core_adjusted_positive", primary["core_adjusted_partial_spearman_rho"] > 0.0, primary["core_adjusted_partial_spearman_rho"], "> 0"),
        ("primary_rat_bootstrap_ci_above_zero", primary["rat_cluster_bootstrap_ci_low"] > 0.0, f"[{primary['rat_cluster_bootstrap_ci_low']:.6g}, {primary['rat_cluster_bootstrap_ci_high']:.6g}]", "95% rat-cluster bootstrap CI lower bound > 0"),
        ("primary_within_session_permutation_significant", primary["within_session_permutation_p_one_sided"] <= 0.05, primary["within_session_permutation_p_one_sided"], "one-sided p <= 0.05"),
        ("primary_all_rats_positive", all_rats_positive, int((pd.to_numeric(by_rat["raw_spearman_rho"], errors="coerce") > 0.0).sum()), "4/4 rat raw directions positive"),
        ("primary_leave_one_rat_out_positive", all_loo_positive, int((pd.to_numeric(leave_one_out["core_adjusted_partial_spearman_rho"], errors="coerce") > 0.0).sum()), "4/4 leave-one-rat-out estimates positive"),
        ("primary_extended_controls_positive", primary["extended_adjusted_partial_spearman_rho"] > 0.0, primary["extended_adjusted_partial_spearman_rho"], "> 0 after adding training spike count"),
    ]
    for gate, passed, observed, criterion in scientific:
        add("scientific", gate, bool(passed), observed, criterion)
    supported = bool(technical_pass and all(bool(item[1]) for item in scientific))
    add("summary", "overall_population_generalizable_mode_allocation_hypothesis", supported, f"{sum(bool(item[1]) for item in scientific)}/{len(scientific)} scientific gates", "technical pass and every predeclared scientific gate passes")
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    event_medians: pd.DataFrame,
    associations: pd.DataFrame,
    by_rat: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    primary = associations.set_index("analysis_id").loc["primary_all_160_events"]
    overall = bool(
        gates.loc[
            gates["gate"].eq("overall_population_generalizable_mode_allocation_hypothesis"),
            "passed",
        ].iloc[0]
    )
    technical = bool(gates.loc[gates["gate"].eq("overall_technical"), "passed"].iloc[0])
    clean_events = int(event_medians["train_defined_clean_imm_majority"].astype(bool).sum())
    lines = [
        "# Train-only map-specific mode allocation and held-out prediction",
        "",
        f"**Decision:** `{'supported' if overall else 'not_supported'}`",
        "",
        "## Leakage boundary",
        "",
        "Every posterior was inferred from the 70% training-cell replay emissions, normalized,",
        "hashed, and then held fixed. Held-out cells were scored directly under that posterior;",
        "their spikes were never passed to a replay model or used to update the latent path.",
        "",
        "## Primary all-event result",
        "",
        f"- Events: {len(event_medians)} across {event_medians['rat'].nunique()} rats",
        f"- Raw Spearman rho: {primary['raw_spearman_rho']:.3f}",
        f"- Core adjusted partial rho: {primary['core_adjusted_partial_spearman_rho']:.3f}",
        f"- Rat-bootstrap 95% CI: [{primary['rat_cluster_bootstrap_ci_low']:.3f}, {primary['rat_cluster_bootstrap_ci_high']:.3f}]",
        f"- Within-session permutation p: {primary['within_session_permutation_p_one_sided']:.4g}",
        f"- Extended-control partial rho: {primary['extended_adjusted_partial_spearman_rho']:.3f}",
        f"- Per-rat positive directions: {int((pd.to_numeric(by_rat['raw_spearman_rho'], errors='coerce') > 0).sum())}/{len(by_rat)}",
        f"- Training-defined clean-IMM-majority sensitivity events: {clean_events}",
        "",
        "## Interpretation",
        "",
        f"Technical leakage/coverage gates: `{'PASS' if technical else 'FAIL'}`.",
        (
            "Map-specific mode allocation inferred from training cells predicts the independent "
            "held-out-cell IMM advantage under the frozen definitions."
            if overall
            else "The frozen analysis does not establish that train-only map-specific mode allocation predicts the held-out-cell IMM advantage."
        ),
        "This result does not assign a behavioral function to IMM and does not alter the standalone Gate 2, Gate 3, or Gate 4 results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence = _successful_rows(pd.read_csv(args.event_evidence))
    events = _event_keys(evidence)
    if args.sessions:
        requested = {item.strip() for item in str(args.sessions).split(",") if item.strip()}
        events = events[events["session"].astype(str).isin(requested)].copy()
    if int(args.max_events) > 0:
        events = events.head(int(args.max_events)).copy()
    if args.reuse_split_scores:
        split_scores = pd.read_csv(args.reuse_split_scores)
    else:
        partial = output / f"{SPLIT_OUTPUT}.partial"
        split_scores = score_all_splits(
            dataset_root=Path(args.dataset_root),
            events=events,
            evidence=evidence,
            n_splits=int(args.n_splits),
            test_cell_fraction=float(args.test_cell_fraction),
            random_seed=int(args.random_seed),
            wrong_map_seed=int(args.wrong_map_seed),
            margin_threshold=float(args.margin_threshold),
            workers=int(args.workers),
            splits_per_task=int(args.splits_per_task),
            partial_output=partial,
        )
    split_path = output / SPLIT_OUTPUT
    split_scores.to_csv(split_path, index=False)
    event_medians = build_event_medians(split_scores)
    associations, by_rat, leave_one_out, permutation = analyze_associations(
        event_medians,
        split_scores,
        bootstrap_replicates=int(args.bootstrap_replicates),
        permutation_replicates=int(args.permutation_replicates),
        seed=int(args.analysis_seed),
    )
    expected_events = int(len(events))
    gates = build_gate_summary(
        split_scores,
        event_medians,
        associations,
        by_rat,
        leave_one_out,
        expected_events=expected_events,
        expected_splits=int(args.n_splits),
        test_cell_fraction=float(args.test_cell_fraction),
    )
    frames = {
        EVENT_OUTPUT: event_medians,
        ASSOCIATION_OUTPUT: associations,
        BY_RAT_OUTPUT: by_rat,
        LOO_OUTPUT: leave_one_out,
        PERMUTATION_OUTPUT: permutation,
        GATE_OUTPUT: gates,
    }
    paths = {SPLIT_OUTPUT: split_path}
    for name, frame in frames.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    _write_report(output / REPORT_OUTPUT, event_medians, associations, by_rat, gates)
    paths[REPORT_OUTPUT] = output / REPORT_OUTPUT

    provenance = build_script_provenance(
        input_paths={
            "dataset_root": args.dataset_root,
            "event_evidence": args.event_evidence,
            "reuse_split_scores": args.reuse_split_scores,
        },
        cwd=ROOT,
    )
    manifest = {
        "analysis": "pfeiffer_train_only_map_specific_mode_prediction_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_scope": "all_events",
        "events": int(len(events)),
        "repeated_cell_splits": int(args.n_splits),
        "train_fraction_requested": float(1.0 - args.test_cell_fraction),
        "heldout_fraction_requested": float(args.test_cell_fraction),
        "cell_split_seed": int(args.random_seed),
        "wrong_map_seed": int(args.wrong_map_seed),
        "wrong_map": "shared occupied-bin population-code permutation per session",
        "posterior_inference": "training replay cells only",
        "heldout_scoring": "frozen smoothed position-marginal predictive log score",
        "heldout_replay_spikes_used_for_latent_inference": False,
        "event_aggregation": "median across repeated splits before primary association",
        "clean_imm_sensitivity": "training-defined within split; event majority across splits",
        "all_cell_clean_imm_selection_used": False,
        "primary_predictor": PRIMARY_X,
        "primary_outcome": PRIMARY_Y,
        "core_controls": list(CORE_CONTROLS),
        "extended_controls": list(EXTENDED_CONTROLS),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "permutation_replicates": int(args.permutation_replicates),
        "analysis_seed": int(args.analysis_seed),
        "scoring_package_file": str(Path(hipporeplayimm.__file__).resolve()),
        "scoring_source_git": git_metadata(Path(hipporeplayimm.__file__).resolve().parents[2]),
        "analysis_script_file": str(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "frozen_posterior_module_file": str(
            (ROOT / "src" / "hipporeplayimm" / "frozen_posterior_prediction.py").resolve()
        ),
        "frozen_posterior_module_sha256": file_sha256(
            ROOT / "src" / "hipporeplayimm" / "frozen_posterior_prediction.py"
        ),
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--event-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-split-scores")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--test-cell-fraction", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=20260804)
    parser.add_argument("--wrong-map-seed", type=int, default=20260805)
    parser.add_argument("--analysis-seed", type=int, default=20260806)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits-per-task", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=5000)
    parser.add_argument("--sessions", default="")
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args(argv)
    if int(args.n_splits) < 1:
        parser.error("--n-splits must be at least one")
    if not 0.0 < float(args.test_cell_fraction) < 1.0:
        parser.error("--test-cell-fraction must lie in (0, 1)")
    if int(args.workers) < 1 or int(args.splits_per_task) < 1:
        parser.error("--workers and --splits-per-task must be at least one")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run_analysis(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
