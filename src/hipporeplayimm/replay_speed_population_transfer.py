"""Retrospective animal-excluded transfer of frozen inverse-calibration baselines."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t

from hipporeplayimm.replay_speed_identifiability import conformal_radius, fit_inverse

IDENTITY = ["dataset", "animal", "session"]
READOUT = ["estimator", "bin_filter", "selection"]
PANEL_KEY = IDENTITY + ["phase", "draw_id", "generator", "observation"] + READOUT
FIT_KEY = ["dataset", "heldout_animal"] + READOUT
GROUP = READOUT + ["generator", "observation", "stratum", "method", "calibration_scope"]
METRICS = ["coverage", "finite_fraction", "finite_coverage", "median_finite_width",
           "nonzero_fraction", "equivalence_fraction"] + [f"{prefix}_equivalence_fraction_{bound:.2f}"
               for prefix in ["true", "false"] for bound in [.10, .25, .50]]


def panel_digest(frame):
    data = frame.sort_values(PANEL_KEY)[PANEL_KEY + ["statistic", "gradient", "draw_seed"]]
    return hashlib.sha256(data.to_csv(index=False, float_format="%.17g", na_rep="NaN").encode()).hexdigest()


def fit_transfers(panels):
    if panels.empty or panels.duplicated(PANEL_KEY).any() or not np.isfinite(panels.gradient).all():
        raise ValueError("nonempty unique panels with finite truth required")
    if not panels.phase.isin(["fit", "calibration", "test"]).all():
        raise ValueError("unknown panel phase")
    source = panels[panels.phase.isin(["fit", "calibration"])]
    if not source.generator.eq("A").all() or not source.observation.eq("poisson").all():
        raise ValueError("B/gain observations cannot enter calibration")
    records = []
    for (dataset, *readout), data in panels.groupby(["dataset"] + READOUT, sort=True):
        animals = sorted(data.animal.unique())
        if len(animals) < 2:
            raise ValueError("animal-excluded calibration requires other animals")
        for heldout in animals:
            others = data[~data.animal.eq(heldout)]
            training = others[others.phase.eq("fit")].sort_values(PANEL_KEY)
            calibration = others[others.phase.eq("calibration")].sort_values(PANEL_KEY)
            if training.empty or calibration.empty:
                raise ValueError("fit and calibration panels required for other animals")
            if set(training.animal) != set(calibration.animal) or set(training.animal) != set(animals) - {heldout}:
                raise ValueError("incomplete source animal membership")
            model = fit_inverse(training.statistic, training.gradient)
            radius = conformal_radius(model, calibration.statistic, calibration.gradient)
            records.append({"dataset": dataset, "heldout_animal": heldout,
                **dict(zip(READOUT, readout, strict=True)), **model, "calibration_radius": radius,
                "n_fit_scheduled": len(training), "n_calibration": len(calibration),
                "finite_calibration": int(np.isfinite(calibration.statistic).sum()),
                "source_animals": json.dumps(sorted(training.animal.unique())),
                "source_sessions": json.dumps(sorted(map(list, set(zip(training.animal, training.session, strict=True))))),
                "fit_input_sha256": panel_digest(training), "calibration_input_sha256": panel_digest(calibration)})
    return pd.DataFrame(records)


def apply_model(rows, model):
    if rows.empty or not rows.phase.eq("test").all():
        raise ValueError("only nonempty test rows can be evaluated")
    if not rows.animal.eq(model["heldout_animal"]).all() or not rows.dataset.eq(model["dataset"]).all():
        raise ValueError("wrong excluded animal or dataset")
    for name in READOUT:
        if not rows[name].eq(model[name]).all():
            raise ValueError("wrong readout for fit")
    x = rows.statistic.to_numpy(float)
    valid = np.isfinite(x) & (model["status"] == "fitted")
    point = np.full(len(x), np.nan)
    widths = {"inverse_gaussian": np.full(len(x), np.inf), "inverse_conformal": np.full(len(x), np.inf)}
    if valid.any():
        point[valid] = model["intercept"] + model["slope"] * x[valid]
        widths["inverse_gaussian"][valid] = t.ppf(.975, model["n_fit"] - 2) * model["residual_sd"] * np.sqrt(
            1 + 1 / model["n_fit"] + (x[valid] - model["x_mean"]) ** 2 / model["sxx"])
        widths["inverse_conformal"][valid] = model["calibration_radius"]
    frames = []
    for method, width in widths.items():
        frame = rows.copy()
        lo, hi = np.full(len(x), -np.inf), np.full(len(x), np.inf)
        lo[valid], hi[valid] = point[valid] - width[valid], point[valid] + width[valid]
        truth = rows.gradient.to_numpy(float)
        finite = np.isfinite(lo) & np.isfinite(hi)
        frame["method"], frame["calibration_scope"] = method, "leave_one_animal_out"
        frame["estimate"], frame["lower"], frame["upper"] = point, lo, hi
        frame["finite_interval"] = finite
        frame["covered"] = (lo <= truth) & (truth <= hi)
        frame["interval_width"] = hi - lo
        frame["nonzero_claim"] = finite & ((lo > 0) | (hi < 0))
        for bound in [.10, .25, .50]:
            frame[f"inside_{bound:.2f}"] = np.abs(truth) < bound
            frame[f"equivalence_{bound:.2f}"] = finite & (lo > -bound) & (hi < bound)
        frame["equivalence_claim"] = frame["equivalence_0.25"]
        frame["truth_inside_equivalence"] = frame["inside_0.25"]
        frame["false_equivalence"] = frame.equivalence_claim & ~frame.truth_inside_equivalence
        frame["decision"] = np.select([frame.equivalence_claim, frame.nonzero_claim, finite],
            ["equivalent_in_surrogate", "nonzero_in_surrogate", "inconclusive"], default="abstain")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_transfers(panels, fits):
    if fits.empty or fits.duplicated(FIT_KEY).any():
        raise ValueError("unique nonempty transfer fits required")
    frames = []
    test = panels[panels.phase.eq("test")]
    for record in fits.to_dict("records"):
        selected = test.dataset.eq(record["dataset"]) & test.animal.eq(record["heldout_animal"])
        for name in READOUT:
            selected &= test[name].eq(record[name])
        frames.append(apply_model(test[selected], record))
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 2 * len(test) or result.duplicated(PANEL_KEY + ["method"]).any():
        raise ValueError("incomplete or duplicated held-out test decisions")
    return result.sort_values(PANEL_KEY + ["method"]).reset_index(drop=True)


def summarize_animals(sessions, seed=20260916, bootstraps=5000):
    if sessions.empty or sessions.duplicated(IDENTITY + GROUP).any():
        raise ValueError("unique nonempty session summaries required")
    group = ["dataset", "animal"] + GROUP
    animals = sessions.groupby(group, sort=True, dropna=False)[METRICS].mean().reset_index()
    counts = sessions.groupby(group, sort=True, dropna=False).agg(sessions=("session", "size"),
        panels=("panels", "sum"), finite_panels=("finite_panels", "sum")).reset_index()
    animals = animals.merge(counts, on=group, validate="one_to_one")
    rng, records = np.random.default_rng(seed), []
    for identity, part in animals.groupby(["dataset"] + GROUP, sort=True):
        base = dict(zip(["dataset"] + GROUP, identity, strict=True))
        for metric in METRICS:
            values = part[metric].dropna().to_numpy()
            ci = np.quantile(rng.choice(values, (bootstraps, len(values)), replace=True).mean(axis=1), [.025, .975]) if len(values) else [np.nan, np.nan]
            records.append({**base, "metric": metric, "equal_animal_mean": values.mean() if len(values) else np.nan,
                "ci_low": ci[0], "ci_high": ci[1], "animals_measurable": len(values),
                "animals_total": len(part), "sessions": int(part.sessions.sum()), "panels": int(part.panels.sum())})
    return animals, pd.DataFrame(records)


def paired_comparison(sessions, seed=20260916, bootstraps=5000):
    keys = IDENTITY + [c for c in GROUP if c != "calibration_scope"]
    transfer = sessions[sessions.calibration_scope.eq("leave_one_animal_out")]
    local = sessions[sessions.calibration_scope.eq("within_session") & ~sessions.method.eq("raw_bootstrap")]
    paired = transfer.merge(local, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_transfer", "_local"))
    if not paired._merge.eq("both").all() or not paired.panels_transfer.eq(paired.panels_local).all():
        raise ValueError("reference test cohorts/denominators do not match")
    for metric in METRICS:
        paired[metric] = paired[f"{metric}_transfer"] - paired[f"{metric}_local"]
    # Finite-only rates and widths can condition on different panel subsets.
    group = ["dataset", "animal"] + [c for c in GROUP if c != "calibration_scope"]
    animal = paired.groupby(group, sort=True, dropna=False)[METRICS].mean().reset_index()
    rng, records = np.random.default_rng(seed), []
    for identity, part in animal.groupby([c for c in group if c != "animal"], sort=True):
        base = dict(zip([c for c in group if c != "animal"], identity, strict=True))
        for metric in METRICS:
            values = part[metric].dropna().to_numpy()
            ci = np.quantile(rng.choice(values, (bootstraps, len(values)), replace=True).mean(axis=1), [.025, .975]) if len(values) else [np.nan, np.nan]
            records.append({**base, "metric": metric, "transfer_minus_local": values.mean() if len(values) else np.nan,
                "ci_low": ci[0], "ci_high": ci[1], "animals_measurable": len(values),
                "animals_positive": int((values > 0).sum()), "animals_negative": int((values < 0).sum())})
    return pd.DataFrame(records)
