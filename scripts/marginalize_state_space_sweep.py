#!/usr/bin/env python3
"""Marginalize selected sorted-spike state-space parameter sweeps.

The state-space evidence sweep scores point-estimate configurations. This
script turns those grid scores into Bayesian model-evidence rows by integrating
over the selected parameter grid, mirroring the KD-style workflow while keeping
the sorted-spike state-space implementation separate from the KD reference
implementation.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmark_model_evidence import _add_evidence_columns, _counts, _summary
from hipporeplayimm.kd_reference import empirical_grid_prior

_EVENT_KEY = ["session", "event_index"]
_DEFAULT_MODELS = ("diffusion", "momentum")


@dataclass(frozen=True)
class _ModelSpec:
    short_name: str
    source_model: str
    output_model: str
    param_cols: tuple[str, ...]


_MODEL_SPECS = {
    "diffusion": _ModelSpec(
        short_name="diffusion",
        source_model="sorted-spike-state-space-diffusion",
        output_model="sorted-spike-state-space-diffusion-marginalized",
        param_cols=("state_space_diffusion_sigma_cm_sqrt_s",),
    ),
    "momentum": _ModelSpec(
        short_name="momentum",
        source_model="sorted-spike-state-space-momentum",
        output_model="sorted-spike-state-space-momentum-marginalized",
        param_cols=(
            "state_space_momentum_sigma_cm_sqrt_s",
            "state_space_momentum_initial_sigma_cm_sqrt_s",
            "state_space_momentum_velocity_decay",
            "state_space_momentum_candidate_top_k",
        ),
    ),
}


def marginalize_sweep(input_csv: str | Path, output: str | Path, *, models: tuple[str, ...] = _DEFAULT_MODELS, prior: str = "empirical") -> dict[str, pd.DataFrame]:
    scores = pd.read_csv(input_csv)
    if scores.empty:
        raise ValueError("state-space sweep scores are empty")
    rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []

    for model in models:
        spec = _MODEL_SPECS[_canonical_model(model)]
        grid, event_table, param_values, source = _grid_for_model(scores, spec)
        weights, prior_kind = _prior_for_grid(grid, spec, param_values, prior)
        marginalized = logsumexp(grid + np.log(np.maximum(weights, np.finfo(float).tiny))[None, ...], axis=tuple(range(1, grid.ndim)))
        rows.extend(_marginalized_rows(spec, event_table, source, param_values, marginalized, prior_kind))
        best_rows.extend(_best_parameter_rows(spec, event_table, grid, param_values))
        prior_rows.extend(_prior_rows(spec, param_values, weights, prior_kind))

    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    event_model_evidence = _add_evidence_columns(pd.DataFrame(rows))
    gridsearch_best_params = pd.DataFrame(best_rows)
    prior_weights = pd.DataFrame(prior_rows)
    event_model_evidence.to_csv(outdir / "state_space_marginalized_event_model_evidence.csv", index=False)
    _summary(event_model_evidence).to_csv(outdir / "state_space_marginalized_model_evidence_summary.csv", index=False)
    _counts(event_model_evidence).to_csv(outdir / "state_space_marginalized_best_model_counts.csv", index=False)
    gridsearch_best_params.to_csv(outdir / "state_space_marginalized_gridsearch_best_params.csv", index=False)
    prior_weights.to_csv(outdir / "state_space_marginalized_prior_weights.csv", index=False)
    for metric in ("log_evidence", "relative_log_evidence", "model_probability"):
        event_model_evidence.pivot_table(index=_EVENT_KEY, columns="model", values=metric, aggfunc="first").reset_index().to_csv(
            outdir / f"state_space_marginalized_event_model_pivot_{metric}.csv", index=False
        )
    return {
        "event_model_evidence": event_model_evidence,
        "model_evidence_summary": _summary(event_model_evidence),
        "best_model_counts": _counts(event_model_evidence),
        "gridsearch_best_params": gridsearch_best_params,
        "prior_weights": prior_weights,
    }


def _canonical_model(model: str) -> str:
    name = model.strip().lower()
    if name.startswith("sorted-spike-state-space-"):
        name = name.removeprefix("sorted-spike-state-space-")
    if name.endswith("-marginalized"):
        name = name.removesuffix("-marginalized")
    if name not in _MODEL_SPECS:
        raise ValueError(f"unsupported marginalized model {model!r}; choose from {sorted(_MODEL_SPECS)}")
    return name


def _grid_for_model(scores: pd.DataFrame, spec: _ModelSpec) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    missing = [col for col in (*_EVENT_KEY, "model", "log_evidence", *spec.param_cols) if col not in scores.columns]
    if missing:
        raise ValueError(f"sweep scores are missing columns needed for {spec.short_name}: {missing}")
    source = scores[scores["model"] == spec.source_model].copy()
    if "status" in source.columns:
        source = source[source["status"] == "success"].copy()
    if source.empty:
        raise ValueError(f"no successful rows for {spec.source_model}")

    grouped = (
        source.groupby([*_EVENT_KEY, *spec.param_cols], dropna=False, as_index=False)
        .agg(
            log_evidence=("log_evidence", _unique_log_evidence),
            n_source_rows=("log_evidence", "count"),
            n_time=("n_time", "first"),
            n_spikes=("n_spikes", "first"),
            runtime_s=("runtime_s", "sum"),
            bin_size_cm=("bin_size_cm", "first"),
            smoothing_sigma_bins=("smoothing_sigma_bins", "first"),
            min_speed_cm_s=("min_speed_cm_s", "first"),
            time_bin_s=("time_bin_s", "first"),
        )
        .sort_values([*_EVENT_KEY, *spec.param_cols])
        .reset_index(drop=True)
    )
    event_table = grouped[_EVENT_KEY].drop_duplicates().sort_values(_EVENT_KEY).reset_index(drop=True)
    param_values = {col: np.asarray(sorted(grouped[col].dropna().unique()), dtype=float) for col in spec.param_cols}
    shape = (len(event_table), *(len(values) for values in param_values.values()))
    grid = np.full(shape, np.nan, dtype=float)
    event_index = {tuple(row): idx for idx, row in enumerate(event_table[_EVENT_KEY].itertuples(index=False, name=None))}
    param_index = {col: {float(value): idx for idx, value in enumerate(values)} for col, values in param_values.items()}
    for row in grouped.itertuples(index=False):
        event = tuple(getattr(row, col) for col in _EVENT_KEY)
        params = tuple(param_index[col][float(getattr(row, col))] for col in spec.param_cols)
        grid[(event_index[event], *params)] = float(row.log_evidence)
    missing_count = int(np.isnan(grid).sum())
    if missing_count:
        raise ValueError(f"{spec.source_model} grid has {missing_count} missing event/parameter scores")
    source_lookup = grouped.drop_duplicates(_EVENT_KEY).set_index(_EVENT_KEY)
    representatives = event_table.join(source_lookup, on=_EVENT_KEY, how="left", rsuffix="_source")
    return grid, event_table, param_values, representatives


def _unique_log_evidence(values: pd.Series) -> float:
    finite = values.dropna().to_numpy(float)
    if finite.size == 0:
        return float("nan")
    if float(np.max(finite) - np.min(finite)) > 1e-7:
        raise ValueError("duplicate grid rows have conflicting log evidence")
    return float(finite[0])


def _prior_for_grid(grid: np.ndarray, spec: _ModelSpec, param_values: dict[str, np.ndarray], prior: str) -> tuple[np.ndarray, str]:
    shape = grid.shape[1:]
    if prior == "uniform":
        return np.full(shape, 1.0 / max(1, int(np.prod(shape))), dtype=float), "uniform"

    varying = [idx for idx, col in enumerate(spec.param_cols) if param_values[col].shape[0] > 1]
    squeezed = np.squeeze(grid, axis=tuple(idx + 1 for idx, col in enumerate(spec.param_cols) if param_values[col].shape[0] == 1))
    try:
        if len(varying) == 1:
            col = spec.param_cols[varying[0]]
            prior_values, _ = empirical_grid_prior({"sd_meters": param_values[col]}, squeezed)
            return _expand_prior(prior_values, spec, param_values, varying), "empirical"
        if len(varying) == 2:
            varying_cols = [spec.param_cols[idx] for idx in varying]
            sigma_col = next((col for col in varying_cols if "sigma" in col), None)
            decay_col = next((col for col in varying_cols if "decay" in col), None)
            if sigma_col is not None and decay_col is not None:
                prior_values, _ = empirical_grid_prior({"sd_meters": param_values[sigma_col], "decay": param_values[decay_col]}, squeezed)
                return _expand_prior(prior_values, spec, param_values, varying), "empirical"
    except Exception:
        pass
    return np.full(shape, 1.0 / max(1, int(np.prod(shape))), dtype=float), "uniform_fallback"


def _expand_prior(prior_values: np.ndarray, spec: _ModelSpec, param_values: dict[str, np.ndarray], varying: list[int]) -> np.ndarray:
    target_shape = tuple(param_values[col].shape[0] for col in spec.param_cols)
    reshape = [1] * len(spec.param_cols)
    for axis, size in zip(varying, prior_values.shape, strict=True):
        reshape[axis] = size
    prior = prior_values.reshape(reshape)
    return np.broadcast_to(prior, target_shape).astype(float) / float(np.sum(prior))


def _marginalized_rows(
    spec: _ModelSpec,
    event_table: pd.DataFrame,
    source: pd.DataFrame,
    param_values: dict[str, np.ndarray],
    marginalized: np.ndarray,
    prior_kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grid_points = int(np.prod([len(values) for values in param_values.values()]))
    grid_description = ";".join(f"{col}={','.join(f'{value:g}' for value in values)}" for col, values in param_values.items())
    for row_index, event in event_table.iterrows():
        representative = source.iloc[row_index]
        rows.append(
            {
                "status": "success",
                "session": event["session"],
                "event_index": int(event["event_index"]),
                "model": spec.output_model,
                "requested_model": spec.output_model,
                "model_family": "trajectory",
                "log_evidence": float(marginalized[row_index]),
                "n_time": int(representative["n_time"]),
                "n_spikes": int(representative["n_spikes"]),
                "runtime_s": float(representative["runtime_s"]),
                "error": "",
                "bin_size_cm": float(representative["bin_size_cm"]),
                "smoothing_sigma_bins": float(representative["smoothing_sigma_bins"]),
                "min_speed_cm_s": float(representative["min_speed_cm_s"]),
                "time_bin_s": float(representative["time_bin_s"]),
                "diagnostic_state_space_marginalized_source_model": spec.source_model,
                "diagnostic_state_space_marginalization_prior": prior_kind,
                "diagnostic_state_space_marginalization_grid_points": grid_points,
                "diagnostic_state_space_marginalization_grid": grid_description,
            }
        )
    return rows


def _best_parameter_rows(spec: _ModelSpec, event_table: pd.DataFrame, grid: np.ndarray, param_values: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    param_cols = list(spec.param_cols)
    for event_row, event in event_table.iterrows():
        flat_best = int(np.nanargmax(grid[event_row].reshape(-1)))
        indices = np.unravel_index(flat_best, grid.shape[1:])
        row: dict[str, object] = {
            "session": event["session"],
            "event_index": int(event["event_index"]),
            "source_model": spec.source_model,
            "marginalized_model": spec.output_model,
            "best_log_evidence": float(grid[(event_row, *indices)]),
        }
        for col, idx in zip(param_cols, indices, strict=True):
            row[f"best_{col}"] = float(param_values[col][idx])
        rows.append(row)
    return rows


def _prior_rows(spec: _ModelSpec, param_values: dict[str, np.ndarray], prior: np.ndarray, prior_kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    value_lists = [values.tolist() for values in param_values.values()]
    for indices in itertools.product(*(range(len(values)) for values in value_lists)):
        row: dict[str, object] = {
            "source_model": spec.source_model,
            "marginalized_model": spec.output_model,
            "prior": prior_kind,
            "prior_weight": float(prior[indices]),
        }
        for col, idx in zip(spec.param_cols, indices, strict=True):
            row[col] = float(param_values[col][idx])
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Marginalize selected state-space parameter-sweep evidence.")
    parser.add_argument("--input", required=True, help="state_space_evidence_sweep_event_scores.csv")
    parser.add_argument("--output", default="results/state-space-marginalized-evidence")
    parser.add_argument("--models", default=" ".join(_DEFAULT_MODELS), help="Models to marginalize: diffusion momentum")
    parser.add_argument("--prior", choices=("empirical", "uniform"), default="empirical")
    args = parser.parse_args()
    models = tuple(_canonical_model(item) for item in args.models.replace(",", " ").split() if item.strip())
    tables = marginalize_sweep(args.input, args.output, models=models, prior=args.prior)
    print(tables["model_evidence_summary"].to_string(index=False))
    print("\nBest-model counts:")
    print(tables["best_model_counts"].to_string(index=False))
    print(f"\nRows: {len(tables['event_model_evidence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
