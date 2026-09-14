#!/usr/bin/env python3
"""Rat-held-out A-only prediction of B support, not of true replay content."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from scripts._provenance import build_script_provenance
except ModuleNotFoundError as exc:
    if exc.name not in ("scripts", "scripts._provenance"):
        raise
    build_script_provenance = None


LOGISTIC_MODELS = ("prevalence", "spikes", "entropy", "spikes_entropy", "full")
POLICIES = ("threshold_spikes", "threshold_entropy")
MODELS = LOGISTIC_MODELS + POLICIES
A_FIELDS = (
    "a_spikes", "a_active", "a_entropy", "a_width_cm", "a_peak",
    "a_local_run_error_cm", "a_stability_cm", "a_coverage", "n_cells",
)
B_FIELDS = ("b_mass20", "b_mass40", "b_spikes", "b_active", "b_width_cm")
TRUTH_FIELDS = ("a_truth_error_cm", "b_truth_error_cm")
DIAGNOSTICS = (
    "endpoint_separation_cm", "path_median_separation_cm", "persistent_conflict_ms",
)
IDENTITY = ("animal", "session", "window_uid", "split", "cohort", "source", "generator", "draw")
CASE = ["model", "source", "generator", "cohort", "split"]
EVENT = ["animal", "session", "window_uid"]
PRIMARY_COHORT = "all_fixed_candidates"
GENERATORS = ("matched_map", "drift_gain")
SIM_SOURCES = ("sim_test", "sim_conflict")
SUPPORTED = ("destination_supported", "coarse_region_supported")
THRESHOLD = 0.8


def validate_readouts(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(IDENTITY + A_FIELDS + B_FIELDS + TRUTH_FIELDS + DIAGNOSTICS)
    required.add("resolved_disagreement")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("No event readouts supplied")
    frame = frame.copy()
    for column in IDENTITY[:-1]:
        if frame[column].isna().any():
            raise ValueError(f"Missing identity: {column}")
    for column in ("animal", "session", "window_uid"):
        frame[column] = frame[column].astype(str)
    frame["draw"] = frame["draw"].fillna(-1)
    for column in A_FIELDS + B_FIELDS + TRUTH_FIELDS + DIAGNOSTICS + ("split",):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column]).any():
            raise ValueError(f"Infinite values in {column}")
    if not frame["split"].isin([0, 1, 2]).all():
        raise ValueError("split must be 0, 1 or 2")
    frame["split"] = frame["split"].astype(int)
    if not frame["cohort"].isin([PRIMARY_COHORT, "full_accepted_segment"]).all():
        raise ValueError("Unknown cohort")
    if not frame["source"].isin(["real", "sim_calibration", *SIM_SOURCES]).all():
        raise ValueError("Unknown source")
    real = frame["source"].eq("real")
    if not frame.loc[real, "generator"].eq("observed").all():
        raise ValueError("Real rows require generator=observed")
    if not frame.loc[~real, "generator"].isin(GENERATORS).all():
        raise ValueError("Unknown simulation generator")
    for column in B_FIELDS[:2]:
        values = frame[column].dropna()
        if not values.between(-1e-10, 1 + 1e-10).all():
            raise ValueError(f"{column} must lie in [0, 1]")
        frame[column] = frame[column].clip(0, 1)
    if (frame["b_mass40"] + 1e-10 < frame["b_mass20"]).any():
        raise ValueError("B coarse mass cannot be smaller than fine mass")
    both = frame["b_mass20"].notna() & frame["b_mass40"].notna()
    frame.loc[both, "b_mass40"] = np.maximum(frame.loc[both, "b_mass20"], frame.loc[both, "b_mass40"])
    for column in A_FIELDS + B_FIELDS[2:] + TRUTH_FIELDS + DIAGNOSTICS:
        if (frame[column] < 0).any():
            raise ValueError(f"Negative values in {column}")
    return frame


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal rats, sessions, events, splits, draws, then duplicate observations.

    Call within one model/source/generator/cohort case. Replicating a draw does
    not increase its event's weight or the number of biological observations.
    """
    if frame.empty:
        return np.empty(0)
    levels = EVENT + ["split", "draw"]
    weights = np.full(len(frame), 1.0 / frame["animal"].nunique())
    for index in range(1, len(levels)):
        counts = frame.groupby(levels[:index], dropna=False)[levels[index]].transform("nunique")
        weights /= counts.to_numpy()
    replicas = frame.groupby(levels, dropna=False)["animal"].transform("size")
    return weights / replicas.to_numpy()


def features(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    result = pd.DataFrame(index=frame.index)
    if model == "threshold_spikes":
        return frame[["a_spikes", "a_active"]].copy()
    if model == "threshold_entropy":
        return frame[["a_entropy"]].copy()
    if model in ("spikes", "spikes_entropy", "full"):
        result["log1p_a_spikes"] = np.log1p(frame["a_spikes"])
        result["log1p_a_active"] = np.log1p(frame["a_active"])
    if model in ("entropy", "spikes_entropy", "full"):
        result["a_entropy"] = frame["a_entropy"]
    if model == "full":
        for column in A_FIELDS[3:]:
            result[column] = frame[column]
    return result


def support_labels(frame: pd.DataFrame, radius: int) -> np.ndarray:
    mass = frame[f"b_mass{radius}"].to_numpy(dtype=float)
    return np.where(np.isfinite(mass), (mass >= 0.5).astype(float), np.nan)


@dataclass
class SupportPredictor:
    model: str
    status: str
    constant: float = np.nan
    pipeline: Pipeline | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            return np.full(len(frame), self.constant)
        return self.pipeline.predict_proba(features(frame, self.model))[:, 1]


def fit_predictor(train: pd.DataFrame, model: str, radius: int) -> SupportPredictor:
    if model in POLICIES:
        return SupportPredictor(model, "abstention_policy_no_probability")
    # Exact duplicated input records must not change imputation or regularization.
    train = train.drop_duplicates().copy()
    labels = support_labels(train, radius)
    train = train.loc[np.isfinite(labels)]
    labels = labels[np.isfinite(labels)]
    if not len(labels):
        return SupportPredictor(model, "no_training_labels")
    weights = balanced_weights(train)
    prevalence = float(np.dot(weights, labels) / weights.sum())
    if model == "prevalence" or len(np.unique(labels)) == 1:
        status = "prevalence" if model == "prevalence" else "single_class_fallback"
        return SupportPredictor(model, status, prevalence)
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)),
    ])
    # Keep C tied to unique events, not to accidental replication of input rows.
    fit_weights = weights * len(train[EVENT].drop_duplicates())
    pipeline.fit(
        features(train, model), labels,
        scale__sample_weight=fit_weights, logistic__sample_weight=fit_weights,
    )
    return SupportPredictor(model, "fitted", pipeline=pipeline)


def prediction_tiers(p20: np.ndarray, p40: np.ndarray) -> np.ndarray:
    return np.where(
        p20 >= THRESHOLD, SUPPORTED[0],
        np.where(p40 >= THRESHOLD, SUPPORTED[1], "content_unresolved"),
    )


def training_entropy_cutoff(train: pd.DataFrame) -> float:
    """Event-balanced training-only 25th percentile (inverse empirical CDF)."""
    train = train.drop_duplicates()
    valid = train["a_entropy"].notna()
    if not valid.any():
        return np.nan
    weights = balanced_weights(train)[valid]
    values = train.loc[valid, "a_entropy"].to_numpy()
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cumulative, 0.25)])


def loao_predictions(frame: pd.DataFrame, include_sensitivity: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validate_readouts(frame)
    primary = frame.loc[
        frame["source"].eq("real") & frame["cohort"].eq(PRIMARY_COHORT) & frame["split"].eq(0)
    ]
    evaluated = frame.loc[
        frame["source"].isin(["real", *SIM_SOURCES])
        & (frame["split"].isin([0, 1, 2]) if include_sensitivity else frame["split"].eq(0))
    ]
    if evaluated.empty:
        raise ValueError("No real or independent simulation evaluation rows")
    predictions, audit = [], []
    for animal in sorted(evaluated["animal"].unique()):
        train = primary.loc[primary["animal"].ne(animal)]
        test = evaluated.loc[evaluated["animal"].eq(animal)]
        train_animals = json.dumps(sorted(train["animal"].unique().tolist()))
        entropy_cutoff = training_entropy_cutoff(train)
        for model in MODELS:
            fitted = {radius: fit_predictor(train, model, radius) for radius in (20, 40)}
            result = test.copy()
            result["model"] = model
            result["heldout_animal"] = animal
            result["fold_id"] = f"loao:{animal}"
            result["train_animals"] = train_animals
            result["train_cohort"] = PRIMARY_COHORT
            result["train_source"] = "real"
            result["train_split"] = 0
            result["train_events"] = len(train[EVENT].drop_duplicates())
            result["entropy_cutoff"] = entropy_cutoff if model == "threshold_entropy" else np.nan
            result["p20"] = fitted[20].predict(test)
            result["p40_raw"] = fitted[40].predict(test)
            result["p40"] = result["p40_raw"]
            both = result["p20"].notna() & result["p40_raw"].notna()
            result.loc[both, "p40"] = np.maximum(result.loc[both, "p20"], result.loc[both, "p40_raw"])
            result["tier"] = prediction_tiers(result["p20"].to_numpy(), result["p40"].to_numpy())
            if model in POLICIES:
                accepted = (test["a_spikes"].ge(3) & test["a_active"].ge(2)) if model == "threshold_spikes" else test["a_entropy"].le(entropy_cutoff)
                result["tier"] = np.where(accepted, SUPPORTED[0], "content_unresolved")
            for radius in (20, 40):
                labeled = np.isfinite(support_labels(train, radius))
                fitted_animals = train_animals if model in POLICIES else json.dumps(sorted(train.loc[labeled, "animal"].unique().tolist()))
                result[f"y{radius}"] = support_labels(test, radius)
                result[f"fit_status{radius}"] = fitted[radius].status
                result[f"train_animals{radius}"] = fitted_animals
                audit.append({
                    "heldout_animal": animal, "fold_id": f"loao:{animal}", "model": model, "radius_cm": radius,
                    "train_animals": fitted_animals, "eligible_train_animals": train_animals,
                    "train_sessions": len(train[["animal", "session"]].drop_duplicates()),
                    "train_events": len(train[EVENT].drop_duplicates()),
                    "train_labeled_events": len(train.loc[np.isfinite(support_labels(train, radius)), EVENT].drop_duplicates()),
                    "train_source": "real", "train_cohort": PRIMARY_COHORT, "train_split": 0,
                    "fit_status": fitted[radius].status,
                    "entropy_cutoff": entropy_cutoff if model == "threshold_entropy" else np.nan,
                })
            radius = np.where(result["tier"].eq(SUPPORTED[0]), 20, 40)
            claimed = result["tier"].isin(SUPPORTED)
            result["truth_correct"] = np.where(
                claimed & result["source"].isin(SIM_SOURCES) & result["a_truth_error_cm"].notna(),
                (result["a_truth_error_cm"] <= radius).astype(float), np.nan,
            )
            result["claim_b_supported"] = np.where(
                result["tier"].eq(SUPPORTED[0]), result["y20"],
                np.where(result["tier"].eq(SUPPORTED[1]), result["y40"], np.nan),
            )
            predictions.append(result)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit)


def probability_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in frame.groupby(CASE + ["animal"], sort=True):
        weights = balanced_weights(group)
        claimed = group["tier"].isin(SUPPORTED)
        supported_known = claimed & group["claim_b_supported"].notna()
        denominator = weights[supported_known].sum()
        b_false = float(np.dot(weights[supported_known], 1 - group.loc[supported_known, "claim_b_supported"]) / denominator) if denominator > 0 else np.nan
        for radius in (20, 40):
            p = group[f"p{radius}"].to_numpy(dtype=float)
            y = group[f"y{radius}"].to_numpy(dtype=float)
            valid = np.isfinite(p) & np.isfinite(y)
            row = dict(zip(CASE + ["animal"], keys))
            row.update(
                radius_cm=radius, n_rows=len(group), n_events=len(group[EVENT].drop_duplicates()),
                evaluated_events=len(group.loc[valid, EVENT].drop_duplicates()),
                evaluated_weight=float(weights[valid].sum()),
                log_loss=np.nan, brier=np.nan, status="skipped", reason="no_evaluable_targets",
                claim_coverage=float(weights[claimed].sum()),
                fine_coverage=float(weights[group["tier"].eq(SUPPORTED[0])].sum()),
                coarse_coverage=float(weights[group["tier"].eq(SUPPORTED[1])].sum()),
                b_false_agreement=b_false,
                claim_status="partial" if (claimed & ~supported_known).any() else ("ok" if denominator > 0 else "skipped"),
            )
            if keys[0] in POLICIES:
                row["reason"] = "abstention_policy_has_no_probability_forecast"
            if valid.any():
                w = weights[valid] / weights[valid].sum()
                clipped = np.clip(p[valid], 1e-15, 1 - 1e-15)
                row.update(
                    log_loss=float(np.dot(w, -y[valid] * np.log(clipped) - (1 - y[valid]) * np.log1p(-clipped))),
                    brier=float(np.dot(w, (p[valid] - y[valid]) ** 2)),
                    status="ok" if valid.all() else "partial",
                    reason="" if valid.all() else "missing_labels_or_predictions",
                )
            rows.append(row)
    by_rat = pd.DataFrame(rows)
    equal_rows = []
    for keys, group in by_rat.groupby(CASE + ["radius_cm"], sort=True):
        row = dict(zip(CASE + ["radius_cm"], keys))
        defined = group["log_loss"].notna()
        row.update(
            n_rats=len(group), n_rats_defined=int(defined.sum()), n_events=int(group["n_events"].sum()),
            log_loss=group["log_loss"].mean(), brier=group["brier"].mean(),
            status="ok" if group["status"].eq("ok").all() else ("partial" if defined.any() else "skipped"),
            reason="" if group["status"].eq("ok").all() else "see_by_rat_missing_cases",
            claim_coverage=group["claim_coverage"].mean(), fine_coverage=group["fine_coverage"].mean(),
            coarse_coverage=group["coarse_coverage"].mean(), b_false_agreement=group["b_false_agreement"].mean(),
            claim_status="ok" if group["claim_status"].eq("ok").all() else "undefined_or_incomplete_rat_case",
        )
        equal_rows.append(row)
    return by_rat, pd.DataFrame(equal_rows)


def known_truth_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = CASE + [
        "animal", "aggregation", "tier", "n_events", "claimed_events", "known_claim_events",
        "coverage", "false_agreement", "correctness", "missing_truth_claims", "status", "reason",
    ]
    rows = []
    simulations = predictions.loc[predictions["source"].isin(SIM_SOURCES)]
    for keys, group in simulations.groupby(CASE + ["animal"], sort=True):
        weights = balanced_weights(group)
        for tier in ("any_supported", *SUPPORTED):
            claimed = group["tier"].isin(SUPPORTED) if tier == "any_supported" else group["tier"].eq(tier)
            known = claimed & group["truth_correct"].notna()
            missing = claimed & group["truth_correct"].isna()
            denominator = weights[known].sum()
            false = float(np.dot(weights[known], 1 - group.loc[known, "truth_correct"]) / denominator) if denominator > 0 else np.nan
            row = dict(zip(CASE + ["animal"], keys))
            row.update(
                aggregation="by_rat", tier=tier, n_events=len(group[EVENT].drop_duplicates()),
                claimed_events=len(group.loc[claimed, EVENT].drop_duplicates()),
                known_claim_events=len(group.loc[known, EVENT].drop_duplicates()),
                coverage=float(weights[claimed].sum()), false_agreement=false, correctness=1 - false,
                missing_truth_claims=len(group.loc[missing, EVENT].drop_duplicates()),
                status="partial" if missing.any() else ("ok" if denominator > 0 else "skipped"),
                reason="missing_truth_for_claims" if missing.any() else ("" if denominator > 0 else "no_supported_predictions"),
            )
            rows.append(row)
    by_rat = pd.DataFrame(rows, columns=columns)
    for keys, group in by_rat.groupby(CASE + ["tier"], sort=True):
        row = dict(zip(CASE + ["tier"], keys))
        row.update(animal="__equal_rat__", aggregation="equal_rat")
        for column in ("n_events", "claimed_events", "known_claim_events", "missing_truth_claims"):
            row[column] = int(group[column].sum())
        for column in ("coverage", "false_agreement", "correctness"):
            row[column] = group[column].mean()
        row["status"] = "ok" if group["status"].eq("ok").all() else "skipped"
        row["reason"] = "" if row["status"] == "ok" else "undefined_or_incomplete_rat_case"
        if row["status"] != "ok":
            row["false_agreement"] = np.nan
            row["correctness"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def continuation_gates(predictions: pd.DataFrame, by_rat: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, case: dict, value: float, threshold: float, defined: bool, passed: bool, reason: str = "") -> None:
        rows.append({
            "gate": name, **case, "value": value, "threshold": threshold,
            "status": ("pass" if passed else "fail") if defined else "skipped",
            "reason": reason if not defined else "",
        })

    primary = predictions.loc[
        predictions["source"].eq("real") & predictions["cohort"].eq(PRIMARY_COHORT) & predictions["split"].eq(0)
    ]
    animals = set(primary["animal"])
    for radius in (20, 40):
        group = by_rat.loc[
            by_rat["source"].eq("real") & by_rat["cohort"].eq(PRIMARY_COHORT)
            & by_rat["split"].eq(0) & by_rat["radius_cm"].eq(radius)
        ]
        full = group.loc[group["model"].eq("full")].set_index("animal")
        base = group.loc[group["model"].eq("spikes_entropy")].set_index("animal")
        valid = bool(animals) and set(full.index) == animals == set(base.index)
        valid = valid and full["status"].eq("ok").all() and base["status"].eq("ok").all()
        baseline = float(base["log_loss"].mean())
        defined = bool(valid and np.isfinite(baseline) and baseline > 1e-12)
        gain = (baseline - float(full["log_loss"].mean())) / baseline if defined else np.nan
        deltas = base["log_loss"] - full["log_loss"]
        minimum = float(deltas.min()) if valid else np.nan
        case = dict(scope="primary", source="real", generator="observed", cohort=PRIMARY_COHORT, split=0, radius_cm=radius)
        add("relative_log_loss_improvement", case, gain, 0.05, defined, gain >= 0.05, "incomplete_rat_case_or_zero_baseline_loss")
        add("every_rat_log_loss_improves", case, minimum, 0.0, bool(valid), minimum > 0, "incomplete_rat_case")

    cases = predictions[["cohort", "split"]].drop_duplicates()
    cases = pd.concat([cases, pd.DataFrame([{"cohort": PRIMARY_COHORT, "split": 0}])]).drop_duplicates()
    for cohort, split in cases.itertuples(index=False, name=None):
        expected = set(primary["animal"]) if cohort == PRIMARY_COHORT and split == 0 else set(
            predictions.loc[predictions["cohort"].eq(cohort) & predictions["split"].eq(split), "animal"]
        )
        for source in SIM_SOURCES:
            for generator in GENERATORS:
                group = truth.loc[
                    truth["model"].eq("full") & truth["source"].eq(source) & truth["generator"].eq(generator)
                    & truth["cohort"].eq(cohort) & truth["split"].eq(split)
                    & truth["tier"].eq("any_supported") & truth["aggregation"].eq("by_rat")
                ]
                complete = bool(expected) and set(group["animal"]) == expected
                coverage = float(group["coverage"].mean()) if complete else np.nan
                false = float(group["false_agreement"].mean()) if complete else np.nan
                truth_defined = complete and group["status"].eq("ok").all() and np.isfinite(false)
                if not truth_defined:
                    false = np.nan
                case = dict(
                    scope="primary" if cohort == PRIMARY_COHORT and split == 0 else "secondary",
                    source=source, generator=generator, cohort=cohort, split=split, radius_cm="tier_20_or_40",
                )
                add("known_truth_coverage", case, coverage, 0.2, complete, coverage >= 0.2, "missing_dataset_or_rat_case")
                add("known_truth_false_agreement", case, false, 0.1, bool(truth_defined), false <= 0.1, "missing_case_truth_or_supported_predictions")
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(event_readouts: Path, output_dir: Path, include_sensitivity: bool = False, protocol: Path | None = None, run_qc_note: str = "") -> dict[str, pd.DataFrame]:
    inputs = {
        "event_readouts": event_readouts.resolve(), "reporter": Path(__file__).resolve(),
        "tests": ROOT / "tests/test_pf_population_content_reliability.py",
        "producer_manifest": event_readouts.resolve().parent / "manifest.json",
        "population_confirmation": event_readouts.resolve().parent / "population_confirmation.csv",
        "regional_run_confirmation": event_readouts.resolve().parent / "regional_run_confirmation.csv",
    }
    if protocol is not None:
        inputs["protocol"] = protocol.resolve()
    if build_script_provenance is not None:
        provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    else:
        provenance = {
            "code_commit": "unavailable", "git_dirty": None,
            "git_error": "shared scripts._provenance helper unavailable",
            "input_file_paths": {name: str(path) for name, path in inputs.items()},
            "input_file_sha256": {name: sha256(path) if path.is_file() else None for name, path in inputs.items()},
        }
    provenance.update(created_utc=datetime.now(timezone.utc).isoformat(), include_sensitivity=include_sensitivity, run_qc_note=run_qc_note)
    frame = pd.read_csv(event_readouts)
    predictions, audit = loao_predictions(frame, include_sensitivity)
    by_rat, equal_rat = probability_metrics(predictions)
    truth = known_truth_metrics(predictions)
    gates = continuation_gates(predictions, by_rat, truth)
    outputs = {
        "predictions.csv.gz": predictions, "metrics_by_rat.csv": by_rat,
        "metrics_equal_rat.csv": equal_rat, "known_truth_correctness.csv": truth,
        "gates.csv": gates, "fold_audit.csv": audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)
    feature_audit = {
        "feature_columns": {model: list(features(frame, model).columns) for model in MODELS},
        "full_raw_a_fields": list(A_FIELDS),
        "transforms": {"a_spikes": "log1p in logistic models", "a_active": "log1p in logistic models"},
        "targets": {"20": "b_mass20 >= 0.5", "40": "b_mass40 >= 0.5"},
        "training": "real/all_fixed_candidates/split0; exclude all sessions of test animal",
        "preprocessing": "training-only median imputation, missing indicators, and weighted standard scaling",
        "logistic_C": 1.0, "tier_probability_threshold": THRESHOLD,
        "threshold_spikes": "fine claim when a_spikes >= 3 and a_active >= 2; otherwise abstain",
        "threshold_entropy": "fine claim when a_entropy <= event-balanced training 25th percentile; otherwise abstain",
        "policy_probability_metrics": "undefined: abstention decisions are not probability forecasts",
        "run_qc_note": run_qc_note,
    }
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    primary = gates.loc[gates["scope"].eq("primary")]
    decision = "PASS" if primary["status"].eq("pass").all() else (
        "FAIL" if primary["status"].eq("fail").any() else "UNRESOLVED"
    )
    real = predictions.loc[
        predictions["source"].eq("real") & predictions["cohort"].eq(PRIMARY_COHORT)
        & predictions["split"].eq(0) & predictions["model"].eq("full")
    ]
    lines = [
        "# PF Population Content Reliability", "",
        "This predicts independent-population B agreement, NOT true replay content or biological multiplexing.",
        "A-only inference is conditional on the pooled-cell candidate catalog. Original eligibility and support used full RUN.",
        "Predictive gates do not assess regional RUN matching. Failed local RUN gates preclude a multiplexed-biological interpretation, even if every predictive gate passes.",
        "Known-content correctness uses A's true-endpoint error: <=20 cm for a fine claim, <=40 cm for a coarse claim.",
        "B support does not excuse an incorrect A claim; resolved disagreement is retained separately.", "",
        f"Primary continuation decision: **{decision}**. Any skipped required gate prevents a pass.",
        f"Primary real cohort: {real['animal'].nunique()} rats, {len(real[['animal', 'session']].drop_duplicates())} sessions, "
        f"{len(real[EVENT].drop_duplicates())} unique events.",
        "Training: only split-0 real all_fixed_candidates, excluding every session of the held-out rat.",
        "Fixed C=1 logistic models use training-only median imputation/scaling and event-balanced sample weights.",
        "Prevalence and single-class fallbacks use training labels only. Missing training labels produce undefined predictions.",
        "Simple fine-claim/abstain baselines: A spikes >=3 AND active cells >=2; or entropy <= training-only weighted 25th percentile.",
        "These simple policies have coverage and agreement/truth error scores, but no invented probability forecasts or probability-metric passes.",
        "Both probabilities are scored after enforcing p40 >= p20; p40_raw is retained for audit. The tier threshold is fixed at 0.8.",
        "Simulation calibration rows are never used. sim_test and sim_conflict are scored separately for both generators.",
        "Paired sim_test/sim_conflict rows share exactly the same A observations; only B changes. Their A-only predictions must be identical. Different B outcomes expose an unobserved-intervention limitation, not an expectation that A can detect that intervention.",
        "All supplied conflict rows are retained; no decoded-separation or B-support filter is imposed by this reporter.",
        f"Split-1/2 prediction-only sensitivity: {'enabled' if include_sensitivity else 'disabled'}; no sensitivity retraining.",
        "Weights average draws/replicas within events, events within sessions, and sessions within rats. Rats are averaged equally.",
        "Rat-specific false agreement is conditional on supported claims. Undefined rat cases are not treated as zero errors.",
        "Individual fine/coarse truth rows are diagnostic; the continuation gate uses their combined tier policy.",
        "Partial probability summaries average defined rats only and cannot pass the improvement gates.",
        "No confidence or population-level significance claim is made from this small-rat pilot.", "",
        "## Primary Gates", "",
    ]
    if real["animal"].nunique() < 4:
        lines.insert(3, "**Fewer than four rats have primary data. Leave-one-animal-out results are a limited small-rat pilot, not four-rat validation.**")
    if run_qc_note:
        lines.insert(4, f"Supplied RUN-QC context (not recomputed): {run_qc_note}")
    for row in primary.itertuples(index=False):
        value = f"{row.value:.6g}" if pd.notna(row.value) else "undefined"
        lines.append(f"- {row.gate}, {row.source}/{row.generator}, radius={row.radius_cm}: {row.status}; value={value}; {row.reason}")
    lines += [
        "", "## Provenance", "", f"- Input: `{event_readouts.resolve()}`", f"- Input SHA256: `{sha256(event_readouts)}`",
        f"- Reporter SHA256: `{provenance['input_file_sha256']['reporter']}`",
        f"- Protocol SHA256: `{provenance['input_file_sha256'].get('protocol', 'unavailable')}`",
        f"- Reporter commit at launch: `{provenance['code_commit']}`; dirty worktree: `{provenance['git_dirty']}`.", "",
        "Tables: predictions.csv.gz, metrics_by_rat.csv, metrics_equal_rat.csv, known_truth_correctness.csv, gates.csv, fold_audit.csv. Feature audit: feature_columns.json. Provenance: manifest.json (including producer-manifest and output hashes).",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance["output_file_sha256"] = {name: sha256(output_dir / name) for name in [*outputs, "report.md", "feature_columns.json"]}
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-readouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--run-qc-note", default="", help="Supplied selection/local-RUN caveat, retained verbatim as context rather than recomputed")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "docs/pf_independent_population_content_protocol.md")
    args = parser.parse_args()
    write_report(args.event_readouts, args.output_dir, args.include_sensitivity, args.protocol, args.run_qc_note)
    print(f"Reliability report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
