"""Training-only smooth Poisson nuisance rates for RUN coordination development."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(times, covariates, p):
    digest = hashlib.sha256(json.dumps(p, sort_keys=True).encode())
    for x in (times, *(covariates[k] for k in ("position", "direction", "speed", "theta"))):
        x = np.ascontiguousarray(x, dtype=np.float64)
        digest.update(str(x.shape).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


def design(train, target, p):
    """Learn transforms from training only; fixed speed bounds come from protocol."""
    axes = []
    outside = np.zeros(len(target["position"]), bool)
    outside_speed = ((target["speed"] < train["speed"].min()) |
                     (target["speed"] > train["speed"].max()))
    for k in (0, 1):
        a, b = train["position"][:, k], target["position"][:, k]
        spacing = p["glm_position_knot_cm"]
        lo, hi = np.floor(a.min() / spacing) * spacing, np.ceil(a.max() / spacing) * spacing
        if hi <= lo:
            hi = lo + spacing
        knots = np.arange(lo, hi + spacing / 2, spacing)[:, None]
        basis = SplineTransformer(knots=knots, degree=p["glm_spline_degree"], include_bias=True,
                                  extrapolation="constant", sparse_output=True)
        axes.append((basis.fit_transform(a[:, None]), basis.transform(b[:, None])))
        outside |= (b < a.min()) | (b > a.max())
    spatial_columns = axes[0][0].shape[1] * axes[1][0].shape[1]
    require(spatial_columns <= p["glm_max_spatial_features"], "Spatial basis exceeds frozen bounded capacity")
    main_penalty = p.get("glm_main_effect_l2_penalty", p["glm_l2_penalty"])
    require(np.isfinite(main_penalty) and 0 < main_penalty <= p["glm_l2_penalty"],
            "Main-effect GLM penalty must be positive and no larger than spatial penalty")
    spatial_scale = np.sqrt(main_penalty / p["glm_l2_penalty"])
    blocks = []
    for side, data in enumerate((train, target)):
        bx, by = axes[0][side].toarray(), axes[1][side].toarray()
        spatial = sparse.csr_matrix(np.einsum("ni,nj->nij", bx, by).reshape(len(bx), -1))
        direction = np.column_stack([f(h * data["direction"]) for h in range(1, p["glm_direction_harmonics"] + 1)
                                     for f in (np.sin, np.cos)])
        theta = np.column_stack([f(h * data["theta"][:, k]) for k in range(data["theta"].shape[1])
                                 for h in range(1, p["glm_theta_harmonics"] + 1) for f in (np.sin, np.cos)])
        log_speed = np.log(data["speed"])
        if p.get("glm_speed_scaling", "training_standardized_log") == "fixed_log_run_bounds":
            low, high = np.log(np.asarray(p["speed_edges_cm_s"])[[0, -1]])
            scaled = 2 * (log_speed - low) / (high - low) - 1
        else:
            train_speed = np.log(train["speed"])
            scaled = (log_speed - train_speed.mean()) / max(float(train_speed.std()), .1)
        speed = np.column_stack([scaled**k for k in range(1, p["glm_speed_degree"] + 1)])
        # Rescaling gives group-specific L2 penalties with the existing Poisson solver.
        spatial = spatial * spatial_scale
        features = [spatial, sparse.csr_matrix(direction), sparse.csr_matrix(theta), sparse.csr_matrix(speed)]
        if p["glm_spatial_direction_interaction"]:
            features.extend(spatial.multiply(direction[:, j, None]) for j in (0, 1))
        if p["glm_spatial_theta_interaction"]:
            features.extend(spatial.multiply(theta[:, k * 2 * p["glm_theta_harmonics"] + j, None])
                            for k in range(data["theta"].shape[1]) for j in (0, 1))
        blocks.append(sparse.hstack(features, format="csr"))
    return blocks[0], blocks[1], {"spatial_features": spatial_columns, "features": blocks[0].shape[1],
        "outside_training_spatial_range_fraction": float(outside.mean()),
        "outside_training_speed_range_fraction": float(outside_speed.mean()),
        "maximum_target_feature_absolute_value": float(np.max(np.abs(blocks[1].data)))}


def prepare(times, covariates, p):
    times = np.asarray(times, float)
    n = len(times)
    require(n > 0 and np.isfinite(times).all() and np.all(np.diff(times) > 0), "Nonempty chronological clock required")
    require(set(covariates) == {"position", "direction", "speed", "theta"}, "All nuisance covariates required")
    covariates = {k: np.asarray(v, float) for k, v in covariates.items()}
    require(covariates["position"].shape == (n, 2) and covariates["direction"].shape ==
            covariates["speed"].shape == (n,) and covariates["theta"].ndim == 2 and
            covariates["theta"].shape[0] == n and covariates["theta"].shape[1] > 0, "Covariate dimensions disagree")
    require(all(np.isfinite(x).all() for x in covariates.values()), "Missing covariates cannot be substituted")
    require(np.all(covariates["speed"] > p["speed_edges_cm_s"][0]) and
            np.all(covariates["speed"] <= p["speed_edges_cm_s"][-1]), "RUN speed outside frozen bounds")
    require(p["crossfit_folds"] >= 2 and p["crossfit_guard_s"] >= p["lag_max_s"] and
            p["crossfit_time_block_s"] > 2 * p["crossfit_guard_s"], "Invalid guarded crossfit")
    require(p.get("glm_speed_scaling", "training_standardized_log") in
            ("training_standardized_log", "fixed_log_run_bounds"), "Unknown GLM speed scaling")
    require(p.get("glm_group_solver", "sklearn_lbfgs") in
            ("sklearn_lbfgs", "scaled_scipy_lbfgs", "scaled_scipy_lbfgs_newton"),
            "Unknown grouped GLM solver")
    if p.get("glm_group_solver") in ("scaled_scipy_lbfgs", "scaled_scipy_lbfgs_newton"):
        require("glm_main_effect_l2_penalty" in p and p["glm_tolerance"] <= 1e-11,
                "Precise grouped GLM solver requires explicit group penalty and tight gradient tolerance")
    require(all(np.isfinite(p[k]) and p[k] > 0 for k in
                ("glm_position_knot_cm", "glm_l2_penalty", "glm_tolerance", "mean_count_floor")),
            "Finite positive GLM spacing, penalty, tolerance and count floor required")
    require(all(isinstance(p[k], int) and p[k] >= 1 for k in
                ("glm_spline_degree", "glm_max_spatial_features", "glm_direction_harmonics",
                 "glm_theta_harmonics", "glm_speed_degree", "glm_max_iter", "glm_workers")),
            "Positive integer GLM dimensions required")
    blocks = np.floor(times / p["crossfit_time_block_s"]).astype(np.int64)
    fold = blocks % p["crossfit_folds"]
    prepared = []
    for f in range(p["crossfit_folds"]):
        validation, training = fold == f, fold != f
        if not validation.any():
            continue
        for block in np.unique(blocks[validation]):
            lo = block * p["crossfit_time_block_s"] - p["crossfit_guard_s"]
            hi = (block + 1) * p["crossfit_time_block_s"] + p["crossfit_guard_s"]
            training &= (times < lo) | (times >= hi)
        item = {"fold": f, "training": training, "validation": validation}
        if training.any():
            a = {k: v[training] for k, v in covariates.items()}
            b = {k: v[validation] for k, v in covariates.items()}
            item["train_design"], item["validation_design"], item["design_qc"] = design(a, b, p)
            pseudo = sparse.csr_matrix(np.asarray(item["train_design"].mean(axis=0)))
            item["fit_design"] = sparse.vstack((item["train_design"], pseudo), format="csr")
        prepared.append(item)
    return {"identity": identity(times, covariates, p), "folds": prepared}


def crossfit(counts, times, labels, width, p, covariates, prepared=None, *, return_predictions=False):
    require(covariates is not None, "GLM requires actual position/direction/speed/LFP theta covariates")
    counts, times, labels = np.asarray(counts), np.asarray(times, float), np.asarray(labels, int)
    require(counts.ndim == 2 and len(counts) == len(times) == len(labels) and counts.shape[1] > 0,
            "Aligned counts, time and strata required")
    require(labels.shape == (len(times),) and np.all(labels >= 0), "Invalid covariate strata")
    require(np.isfinite(counts).all() and np.all(counts >= 0) and np.all(counts == np.floor(counts)), "Invalid counts")
    require(width == p["run_bin_s"] and width > 0, "Frozen positive bin width required")
    if prepared is None:
        prepared = prepare(times, covariates, p)
    require(prepared["identity"] == identity(times, covariates, p), "Prepared design identity differs")
    means, global_means = np.full(counts.shape, np.nan), np.full(counts.shape, np.nan)
    seen = np.zeros(len(times), bool)
    diagnostics = []
    pseudo_weight = p["global_rate_prior_exposure_s"] / width
    require(pseudo_weight > 0 and p["global_rate_prior_spikes"] > 0, "Positive rate prior required")
    for item in prepared["folds"]:
        f, training, validation = item["fold"], item["training"], item["validation"]
        if not training.any():
            diagnostics.append({"fold": f, "status": "no_guarded_training_bins", "validation_bins": int(validation.sum())})
            continue
        weight = np.r_[np.ones(training.sum()), pseudo_weight]
        alpha = p.get("glm_main_effect_l2_penalty", p["glm_l2_penalty"]) / weight.sum()

        def fit_cell(k):
            y = np.r_[counts[training, k], p["global_rate_prior_spikes"] / pseudo_weight]
            if p.get("glm_group_solver") in ("scaled_scipy_lbfgs", "scaled_scipy_lbfgs_newton"):
                x = item["fit_design"]
                normalized_weight = weight / weight.sum()
                initial = np.r_[np.zeros(x.shape[1]), np.log(normalized_weight @ y)]

                def objective(coefficient):
                    eta = np.asarray(x @ coefficient[:-1]).ravel() + coefficient[-1]
                    mu = np.exp(eta)
                    loss = normalized_weight @ (mu - y * eta) + alpha * (coefficient[:-1] @ coefficient[:-1]) / 2
                    error = normalized_weight * (mu - y)
                    gradient = np.r_[np.asarray(x.T @ error).ravel() + alpha * coefficient[:-1], error.sum()]
                    return loss, gradient

                # Explicit loss convergence avoids the fixed sklearn termination setting.
                fitted = minimize(objective, initial, method="L-BFGS-B", jac=True,
                    options={"maxiter": p["glm_max_iter"], "maxls": 50,
                             "gtol": p["glm_tolerance"], "ftol": 4 * np.finfo(float).eps})
                coefficient = fitted.x
                if p.get("glm_group_solver") == "scaled_scipy_lbfgs_newton" and fitted.success:
                    coefficient = precise_scaled_solution(coefficient, x, y, normalized_weight, alpha)
                prediction = np.exp(np.asarray(item["validation_design"] @ coefficient[:-1]).ravel() + coefficient[-1])
                return prediction, int(fitted.nit), fitted.success and np.isfinite(prediction).all() and fitted.nit < p["glm_max_iter"]
            model = PoissonRegressor(alpha=alpha, solver="lbfgs", fit_intercept=True,
                                     max_iter=p["glm_max_iter"], tol=p["glm_tolerance"])
            model.fit(item["fit_design"], y, sample_weight=weight)
            prediction = model.predict(item["validation_design"])
            return prediction, int(model.n_iter_), np.isfinite(prediction).all() and model.n_iter_ < p["glm_max_iter"]

        with ThreadPoolExecutor(max_workers=p["glm_workers"]) as pool:
            fitted = list(pool.map(fit_cell, range(counts.shape[1])))
        converged = all(row[2] for row in fitted)
        if converged:
            means[validation] = np.maximum(np.column_stack([row[0] for row in fitted]), p["mean_count_floor"])
            global_means[validation] = np.maximum((counts[training].sum(axis=0) + p["global_rate_prior_spikes"]) /
                (training.sum() + pseudo_weight), p["mean_count_floor"])
            seen[validation] = np.isin(labels[validation], labels[training])
        diagnostics.append({"fold": f, "status": "predicted" if converged else "glm_nonconvergence",
            "training_bins": int(training.sum()), "validation_bins": int(validation.sum()),
            "training_spikes": int(counts[training].sum()), "all_cells_converged": bool(converged),
            "max_fit_iterations": max(row[1] for row in fitted), **item["design_qc"]})
    usable = np.isfinite(means).all(axis=1)
    delta = float(np.sum(xlogy(counts[usable], means[usable] / global_means[usable])
                         - means[usable] + global_means[usable])) if usable.any() else np.nan
    residual = (counts - means) / np.sqrt(means)
    diagnostics = {"rate_model_family": "smooth_poisson_glm", "folds": diagnostics,
        "predicted_bins": int(usable.sum()), "total_bins": len(times),
        "predicted_mean_count_min": float(means[usable].min()) if usable.any() else np.nan,
        "predicted_mean_count_max": float(means[usable].max()) if usable.any() else np.nan,
        "unseen_stratum_fraction": float((~seen[usable]).mean()) if usable.any() else np.nan,
        "unseen_joint_strata_use_glm_not_global_fallback": True, "heldout_poisson_improvement_over_global": delta}
    if return_predictions:
        return residual, diagnostics, means, global_means
    return residual, diagnostics


def precise_scaled_solution(coefficient, x, y, weight, alpha):
    """Polish scaled coefficients without changing the statistical objective."""
    augmented = sparse.hstack([x, np.ones((len(y), 1))], format="csr")
    penalties = np.r_[np.full(x.shape[1], alpha), 0.]
    answer = coefficient.copy()
    for _ in range(12):
        eta = np.asarray(augmented @ answer).ravel()
        mean = np.exp(eta)
        gradient = np.asarray(augmented.T @ (weight * (mean - y))).ravel() + penalties * answer
        norm = np.max(np.abs(gradient))
        require(np.isfinite(norm), "Nonfinite producer precision gradient")
        if norm <= 1e-13:
            return answer
        curvature = augmented.T @ augmented.multiply((weight * mean)[:, None]) + sparse.diags(penalties)
        direction = spsolve(curvature.tocsc(), -gradient)
        require(np.isfinite(direction).all(), "Nonfinite producer precision direction")
        loss = weight @ (mean - y * eta) + (penalties * answer) @ answer / 2
        for fraction in (1., .5, .25, .125, .0625):
            trial = answer + fraction * direction
            trial_eta = np.asarray(augmented @ trial).ravel()
            trial_mean = np.exp(trial_eta)
            trial_gradient = np.asarray(augmented.T @ (weight * (trial_mean - y))).ravel() + penalties * trial
            trial_loss = weight @ (trial_mean - y * trial_eta) + (penalties * trial) @ trial / 2
            if (np.isfinite(trial_loss) and trial_loss <= loss + 8 * np.finfo(float).eps * max(abs(loss), 1e-20)
                    and np.max(np.abs(trial_gradient)) < norm):
                answer = trial
                break
        else:
            raise ValueError("Producer precision refinement did not reduce gradient and loss")
    raise ValueError("Producer precision failed the fixed 1e-13 gradient criterion")
