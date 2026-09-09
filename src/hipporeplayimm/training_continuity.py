"""Training-only geometric labels for independent cross-cell validation."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

IDENTITY = ["dataset", "animal", "session"]
KEY = IDENTITY + ["event_id", "split"]
SETTINGS = ["support", "bin_filter", "min_frames"]
CRITERIA = [(False, 10), (False, 11), (True, 10), (True, 11)]
PRIMARY_CONTRASTS = [
    "imm_minus_iid",
    "imm_minus_static",
    "imm_minus_composition",
    "imm_order_advantage",
    "imm_order_map_interaction",
]
PRIMARY_GROUPS = ["rejected_with_opportunity", "lost_with_thinning"]


def nested_half(training, identity, split):
    training = np.asarray(training, int)
    if training.ndim != 1 or len(training) < 2 or len(np.unique(training)) != len(training) or (training < 0).any():
        raise ValueError("unique nonnegative training indices required")
    key = "|".join(map(str, ("training_continuity_v1", 20260910, *identity, split)))
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.sort(np.random.default_rng(seed).permutation(np.sort(training))[: len(training) // 2])


def overlapping_counts(base_counts, durations):
    counts, dt = np.asarray(base_counts), np.asarray(durations, float)
    if counts.ndim != 2 or dt.shape != (len(counts),) or not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("finite nonnegative integer counts and matching clock required")
    if not np.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("positive finite bin widths required")
    complete = np.isclose(dt, 0.005, atol=1e-9, rtol=0)
    partial = np.flatnonzero(~complete)
    if len(partial) and (len(partial) != 1 or partial[0] != len(dt) - 1 or dt[-1] >= 0.005):
        raise ValueError("only a final short partial bin is supported")
    n = int(complete.sum())
    if n < 4:
        return np.empty((0, counts.shape[1]), np.int64)
    cumulative = np.vstack([np.zeros((1, counts.shape[1]), np.int64), counts[:n].cumsum(axis=0, dtype=np.int64)])
    return cumulative[4:] - cumulative[:-4]


def poisson_map(counts, rates):
    counts, rates = np.asarray(counts), np.asarray(rates, float)
    if counts.ndim != 2 or rates.ndim != 2 or counts.shape[1] != len(rates) or min(rates.shape) < 1:
        raise ValueError("count/rate dimensions differ")
    if not np.isfinite(rates).all() or (rates <= 0).any() or not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("positive rates and finite integer counts required")
    values = csr_matrix(counts, dtype=np.float64) @ np.log(rates)
    values -= 0.020 * rates.sum(axis=0)
    return values.argmax(axis=1).astype(np.int32)


def geometry(path_indices, grid, counts, filtered=False, min_frames=10):
    path, grid, counts = np.asarray(path_indices), np.asarray(grid, float), np.asarray(counts)
    if path.ndim != 1 or counts.ndim != 2 or len(path) != len(counts) or grid.ndim != 2 or grid.shape[1] != 2 or not np.isfinite(grid).all():
        raise ValueError("aligned 2D paths and count windows required")
    if min_frames not in (10, 11) or not np.issubdtype(path.dtype, np.integer) or (path < 0).any() or (path >= len(grid)).any():
        raise ValueError("invalid path indices or frame rule")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("invalid window counts")
    edge = np.flatnonzero(counts.sum(axis=1) >= 2)
    valid = np.zeros(len(path), bool)
    if len(edge):
        valid[edge[0] : edge[-1] + 1] = True
    edge_frames = int(valid.sum())
    if filtered:
        valid &= (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)
    positions = grid[path]
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    best_length, best_start, current_start = 0, 0, 0
    for i in range(len(path)):
        if not valid[i]:
            current_start = i + 1
            continue
        if i == 0 or not valid[i - 1] or jumps[i - 1] >= 20 - 1e-9:
            current_start = i
        length = i - current_start + 1
        if length > best_length:
            best_length, best_start = length, current_start
    displacement = float(np.linalg.norm(positions[best_start + best_length - 1] - positions[best_start])) if best_length else 0.0
    passed = best_length >= min_frames and displacement >= 40 - 1e-9
    reason = (
        "pass"
        if passed
        else "no_complete_windows"
        if not len(path)
        else "no_supported_edges"
        if not edge_frames
        else "insufficient_supported_frames"
        if valid.sum() < min_frames
        else "jump_fragmentation"
        if best_length < min_frames
        else "insufficient_displacement"
    )
    return {
        "geometric_pass": bool(passed),
        "failure_reason": reason,
        "decoding_frames": len(path),
        "edge_supported_frames": edge_frames,
        "valid_frames": int(valid.sum()),
        "longest_run_frames": best_length,
        "run_start_frame": best_start if best_length else -1,
        "run_displacement_cm": displacement,
    }


def classify_training(base_counts, durations, rates, centers, training, state_mask):
    training, state_mask = np.asarray(training, int), np.asarray(state_mask, bool)
    if training.ndim != 1 or len(training) < 1 or len(np.unique(training)) != len(training) or (training < 0).any() or (training >= len(rates)).any():
        raise ValueError("invalid training partition")
    if state_mask.shape != (len(centers),) or state_mask.sum() < 2:
        raise ValueError("insufficient spatial support")
    windows = overlapping_counts(np.asarray(base_counts)[:, training], durations)
    selected_rates = np.asarray(rates)[training][:, state_mask]
    grid = np.asarray(centers)[state_mask]
    path = poisson_map(windows, selected_rates)
    results = [geometry(path, grid, windows, filtered, minimum) for filtered, minimum in CRITERIA]
    return path, windows, results


def classification_group(full, half):
    return {
        (True, True): "retained_with_thinning",
        (True, False): "lost_with_thinning",
        (False, True): "gained_with_thinning",
        (False, False): "rejected_both",
    }[bool(full), bool(half)]


def validate_labels(labels, selected):
    keys = IDENTITY + ["event_id"]
    if labels.empty or selected.empty or selected.duplicated(keys).any() or labels.duplicated(KEY + SETTINGS).any():
        raise ValueError("nonempty unique event and label keys required")
    expected = {(s, support, filtered, frames) for s in range(5) for support in ("parent", "arena_clipped") for filtered in ("edge_only", "bin_support") for frames in (10, 11)}
    if len(labels) != len(selected) * len(expected) or set(map(tuple, labels[keys].drop_duplicates().to_numpy())) != set(map(tuple, selected[keys].to_numpy())):
        raise ValueError("incomplete label cohort")
    for _, g in labels.groupby(keys, sort=False):
        if set(g[["split", *SETTINGS]].itertuples(index=False, name=None)) != expected:
            raise ValueError("incomplete within-event settings")
    if labels.heldout_used_for_classification.any():
        raise ValueError("held-out classification leakage")
    if not labels.full_training_pass.isin([True, False]).all() or not labels.nested_half_pass.isin([True, False]).all():
        raise ValueError("invalid classification flags")


def predictive_contrasts(original, factorial):
    if original.empty or original.duplicated(KEY + ["map"]).any() or factorial.empty or factorial.duplicated(KEY + ["contrast"]).any():
        raise ValueError("missing or duplicated predictions")
    real = original[original["map"].eq("real")].set_index(KEY)
    wrong = original[original["map"].eq("population_code_permuted")].set_index(KEY)
    if not real.index.equals(wrong.reindex(real.index).index) or len(real) != len(wrong) or set(real.index) != set(wrong.index):
        raise ValueError("incomplete map pair")
    result = real[["n_heldout_spikes", "n_train_spikes", "duration_s"]].copy()
    for model, prefix in [("first_order_imm", "imm"), ("diffusion", "diffusion")]:
        for comparator, short in [("iid_position", "iid"), ("static_location", "static"), ("event_global", "composition")]:
            result[f"{prefix}_minus_{short}"] = real[f"score_{model}"] - real[f"score_{comparator}"]
        for source, target in [("real_order_advantage", "order_advantage"), ("order_map_interaction", "order_map_interaction")]:
            values = factorial[factorial.contrast.eq(model + "__" + source)].set_index(KEY).delta
            if set(values.index) != set(real.index):
                raise ValueError("missing factorial event or split")
            result[f"{prefix}_{target}"] = values.reindex(real.index)
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("nonfinite predictive contrast")
    if (result.n_heldout_spikes < 0).any():
        raise ValueError("negative held-out counts")
    return result.reset_index()


def grouped_event_values(labels, predictions):
    if labels.duplicated(KEY + SETTINGS).any() or predictions.duplicated(KEY).any():
        raise ValueError("duplicate join keys")
    joined = labels.merge(predictions, on=KEY, how="left", validate="many_to_one", indicator=True)
    if not joined._merge.eq("both").all() or set(map(tuple, labels[KEY].drop_duplicates().to_numpy())) != set(map(tuple, predictions[KEY].to_numpy())):
        raise ValueError("missing classification/prediction join")
    masks = {
        "all": np.ones(len(joined), bool),
        "geometric_pass": joined.full_training_pass,
        "geometric_fail": ~joined.full_training_pass,
        "rejected_with_opportunity": ~joined.full_training_pass & (joined.full_valid_frames >= joined.min_frames),
        "short_or_unsupported": ~joined.full_training_pass & (joined.full_valid_frames < joined.min_frames),
        "lost_with_thinning": joined.group.eq("lost_with_thinning"),
        "retained_with_thinning": joined.group.eq("retained_with_thinning"),
        "gained_with_thinning": joined.group.eq("gained_with_thinning"),
        "rejected_both": joined.group.eq("rejected_both"),
    }
    contrasts = [c for c in predictions if c.startswith(("imm_", "diffusion_"))]
    frames = []
    for name, mask in masks.items():
        local = joined.loc[mask]
        if local.empty:
            continue
        for contrast in contrasts:
            values = local[IDENTITY + ["event_id", "split", *SETTINGS]].copy()
            values["delta"] = local[contrast]
            values["delta_per_heldout_spike"] = local[contrast] / local.n_heldout_spikes.replace(0, np.nan)
            keys = IDENTITY + ["event_id", *SETTINGS]
            event = values.groupby(keys, as_index=False).agg(
                delta=("delta", "median"), delta_per_heldout_spike=("delta_per_heldout_spike", "median"), qualifying_splits=("split", "nunique")
            )
            frames.append(event.assign(group=name, contrast=contrast))
    if not frames:
        raise ValueError("no group values")
    return pd.concat(frames, ignore_index=True)
