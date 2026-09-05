"""Independent RUN-half maps for paired known-path recovery experiments."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .encoding import EncodingConfig, _positions_to_flat_bins, fit_place_field_encoding
from .replay_coverage import continuity_metrics, decode_independent
from .replay_coverage_data import array_sha256
from .replay_coverage_recovery import map_interpolator, spatial_covariate
from .replay_coverage_validation import fit_training_fold, posterior_coverage, training_session


def fit_map_pair(full, direction, config, guard_s=1.):
    """A selects units/support; disjoint B supplies only a surrogate generator."""
    if direction not in (0, 1):
        raise ValueError("direction must be 0 or 1")
    lo, hi = full.run_times[:, 0].min(), full.run_times[:, 1].max()
    mid = (lo + hi) / 2
    excluded_a, excluded_b = ((mid, hi), (lo, mid)) if direction == 0 else ((lo, mid), (mid, hi))
    a, units, meta = fit_training_fold(full, *excluded_a, guard_s, config)
    b_data = training_session(full, *excluded_b, guard_s, config.maximum_position_gap_s)
    fit_config = EncodingConfig(bin_size_cm=config.bin_size_cm, smoothing_sigma_bins=config.smoothing_sigma_bins,
        min_speed_cm_s=config.min_speed_cm_s, min_occupancy_s=config.min_occupancy_s,
        rate_floor_hz=1e-4, arena_padding_cm=0., use_excitatory=False, exclude_ripple_intervals=False)
    b = fit_place_field_encoding(b_data, fit_config)
    b_rates = np.full((len(a["cell_ids"]), b.n_bins), fit_config.rate_floor_hz)
    missing = []
    for row, cell in enumerate(a["cell_ids"]):
        match = np.flatnonzero(b.cell_ids == cell)
        if len(match):
            b_rates[row] = b.rates_hz[int(match[0])]
        else:
            missing.append(int(cell))
    evaluate = map_interpolator(b_rates, b.x_edges, b.y_edges)
    b_at_a = evaluate(a["grid_cm"]).T
    b_flat = _positions_to_flat_bins(a["grid_cm"], b.x_edges, b.y_edges)
    b_supported = np.zeros(len(b_flat), bool)
    inside = b_flat >= 0
    b_supported[inside] = b.occupancy_s[b_flat[inside]] >= config.min_occupancy_s
    a_bounds = np.array([[np.mean(a[e][:2]) for e in ["x_edges_cm", "y_edges_cm"]],
                         [np.mean(a[e][-2:]) for e in ["x_edges_cm", "y_edges_cm"]]])
    b_bounds = np.array([[b.bin_centers[:, j].min() for j in range(2)], [b.bin_centers[:, j].max() for j in range(2)]])
    domain = np.array([np.maximum(a_bounds[0], b_bounds[0]), np.minimum(a_bounds[1], b_bounds[1])])
    bounds = full.metadata.get("arena_bounds_cm")
    if bounds is not None:
        domain = np.array([np.maximum(domain[0], bounds[0]), np.minimum(domain[1], bounds[1])])
    if not np.isfinite(domain).all() or np.any(domain[1] - domain[0] < 40):
        raise ValueError("insufficient common synthetic spatial extent")
    if any(max(x, u) <= min(y, v) for x, y in a["training_intervals"] for u, v in b_data.run_times):
        raise AssertionError("training/generator intervals overlap")
    correlations = [np.corrcoef(x[b_supported], y[b_supported])[0, 1]
                    for x, y in zip(a["rates_hz"], b_at_a, strict=True)
                    if b_supported.sum() > 2 and np.std(x[b_supported]) > 0 and np.std(y[b_supported]) > 0]
    model = {**a, "generator_rates_hz": b_rates, "generator_x_edges_cm": b.x_edges,
             "generator_y_edges_cm": b.y_edges, "generator_occupancy_s": b.occupancy_s,
             "generator_valid_spatial_bins": b.occupancy_s >= config.min_occupancy_s,
             "generator_at_training_states_hz": b_at_a, "generator_supported_training_states": b_supported,
             "generator_intervals": b_data.run_times, "domain_cm": domain}
    meta.update(direction=direction, split_time_s=float(mid),
        generator_positions_sha256=array_sha256(b_data.position), generator_spikes_sha256=array_sha256(b_data.spikes),
        generator_cell_ids_missing=missing, generator_grid_sha256=array_sha256(b.bin_centers),
        generator_rates_sha256=array_sha256(b_rates), shared_support_fraction=float(b_supported.mean()),
        median_half_map_correlation=float(np.median(correlations)) if correlations else None,
        generator_half_used_for_decoder_QC=False, generator_half_used_for_decoder_support=False,
        both_position_extents_used_for_synthetic_domain=True)
    return model, units, meta


def mismatch_observations(path_rates, seed, rate_scale=3., gain_cv=1.):
    """Native Poisson and independent common-gain stress observations."""
    rates = np.asarray(path_rates, float)
    if rates.ndim != 2 or not min(rates.shape) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("positive finite time-by-cell rates required")
    if not np.isfinite([rate_scale, gain_cv]).all() or rate_scale <= 0 or gain_cv < 0:
        raise ValueError("invalid observation parameters")
    children = np.random.SeedSequence(seed).spawn(3)
    native = np.random.default_rng(children[0]).poisson(rates * .001 * rate_scale)
    sigma = np.sqrt(np.log1p(gain_cv**2))
    gains = np.repeat(np.random.default_rng(children[1]).lognormal(-sigma**2 / 2, sigma,
                     int(np.ceil(len(rates) / 20))), 20)[:len(rates)]
    changed = np.random.default_rng(children[2]).poisson(rates * (.001 * rate_scale * gains[:, None]))
    return {"poisson": native, "shared_gain": changed}, gains


def decode_with_coverage(counts, rates, grid, truth_indices, likelihood):
    chunks = []
    for lo in range(0, len(counts), 512):
        part = decode_independent(counts[lo:lo + 512], rates, grid, .02, likelihood=likelihood)
        part["hpd95"] = posterior_coverage(part.pop("posterior"), truth_indices[lo:lo + 512])["hpd95"]
        chunks.append(part)
    if not chunks:
        raise ValueError("no count windows")
    return {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}


def mismatch_metrics(points, counts, truth, domain, apply_support, rms, entropy, hpd95):
    """Domain-specific moments and gap-safe speeds, including short events."""
    valid = ((counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)) if apply_support else np.ones(len(counts), bool)
    chosen = continuity_metrics(points, valid_bins=valid)
    adjacent = valid[:-1] & valid[1:]
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    errors = np.linalg.norm(points - truth["center_cm"], axis=1)
    out = {**chosen, "truth_geometric_eligible": continuity_metrics(truth["window_mean_cm"])["continuity_pass"],
           "n_decoded_bins": len(counts), "valid_bins": int(valid.sum()), "valid_adjacent_steps": int(adjacent.sum()),
           "large_jump_fraction": float(np.mean(jumps[adjacent] >= 20)) if adjacent.any() else np.nan,
           "median_position_error_cm": float(np.median(errors[valid])) if valid.any() else np.nan,
           "median_posterior_rms_cm": float(np.median(rms[valid])) if valid.any() else np.nan,
           "median_posterior_entropy_nats": float(np.median(entropy[valid])) if valid.any() else np.nan,
           "hpd95_coverage": float(np.mean(hpd95[valid])) if valid.any() else np.nan}
    idx = truth["indices"]
    pairs = np.array([valid[a:b + 1].all() for a, b in pairwise(idx)], dtype=bool)
    core = (idx >= out["continuous_start"]) & (idx < out["continuous_end_exclusive"])
    selected = pairs & core[:-1] & core[1:] & out["continuity_pass"]
    speed = np.linalg.norm(np.diff(points[idx], axis=0), axis=1) / .02
    q = spatial_covariate((points[idx[:-1]] + points[idx[1:]]) / 2, domain)
    for name, keep in [("all", pairs), ("selected", selected)]:
        out[f"{name}_steps"] = int(keep.sum())
        out[f"{name}_median_speed_cm_s"] = float(np.median(speed[keep])) if keep.any() else np.nan
        for axis, x in [("true_coordinate", truth["q"]), ("decoded_coordinate", q)]:
            for readout, values in [("decoded", speed), ("true_arclength", truth["arclength_speed"]), ("true_chord", truth["chord_speed"])]:
                finite = keep & np.isfinite(x) & np.isfinite(values)
                prefix = f"{name}__{axis}__{readout}"
                out[f"{prefix}__steps"] = int(finite.sum())
                for label, array in [("x", x), ("y", values), ("xx", x*x), ("xy", x*values)]:
                    out[f"{prefix}__mean_{label}"] = float(array[finite].mean()) if finite.any() else np.nan
        out[f"{name}_median_speed_error_vs_arclength_cm_s"] = float(np.median((speed-truth["arclength_speed"])[keep])) if keep.any() else np.nan
        out[f"{name}_median_speed_error_vs_chord_cm_s"] = float(np.median((speed-truth["chord_speed"])[keep])) if keep.any() else np.nan
    return out
