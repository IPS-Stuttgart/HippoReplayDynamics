"""Synthetic-only direction-3 tests; run with python3 on gpuserver6000."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_pf_population_content_reliability.py"
SPEC = importlib.util.spec_from_file_location("pf_reliability_reporter", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def synthetic_readouts():
    rows = []
    for rat in range(4):
        for session in range(2):
            for event in range(12):
                level = event % 3
                row = dict(
                    animal=f"Rat{rat}", session=f"Open{session}", window_uid=f"event{event}",
                    split=0, cohort="all_fixed_candidates", source="real", generator="observed", draw=0,
                    a_spikes=1 + event, a_active=1 + event % 4, a_entropy=0.15 + level * 0.3,
                    a_width_cm=8 + level * 15, a_peak=0.8 - level * 0.3,
                    a_local_run_error_cm=4 + level * 5, a_stability_cm=2 + level * 7,
                    a_coverage=0.9 - level * 0.3, n_cells=40 + rat,
                    b_mass20=0.8 if level == 0 else 0.1, b_mass40=0.9 if level < 2 else 0.2,
                    b_spikes=5, b_active=3, b_width_cm=12, a_truth_error_cm=np.nan,
                    b_truth_error_cm=np.nan, endpoint_separation_cm=10,
                    path_median_separation_cm=12, persistent_conflict_ms=0, resolved_disagreement=False,
                )
                rows.append(row)
                for source in ("sim_calibration", "sim_test", "sim_conflict"):
                    for generator in ("matched_map", "drift_gain"):
                        rows.append({**row, "source": source, "generator": generator,
                                     "a_truth_error_cm": 5 if level == 0 else 30,
                                     "b_truth_error_cm": 8, "draw": 1})
    return pd.DataFrame(rows)


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_readouts()
        cls.predictions, cls.audit = reporter.loao_predictions(cls.frame)

    def test_no_b_features_and_heldout_b_perturbation(self):
        changed = self.frame.copy()
        heldout = changed["animal"].eq("Rat0")
        changed.loc[:, ["b_spikes", "b_active", "b_width_cm"]] = [1000, 100, 500]
        changed.loc[heldout, ["b_mass20", "b_mass40"]] = [0.7, 0.9]
        changed.loc[heldout, "resolved_disagreement"] = True
        for model in reporter.MODELS:
            pd.testing.assert_frame_equal(reporter.features(self.frame, model), reporter.features(changed, model))
        predictions, _ = reporter.loao_predictions(changed)
        original = self.predictions.loc[self.predictions["animal"].eq("Rat0"), ["p20", "p40", "tier"]]
        updated = predictions.loc[predictions["animal"].eq("Rat0"), ["p20", "p40", "tier"]]
        pd.testing.assert_frame_equal(original, updated)

    def test_entire_heldout_animal_excluded(self):
        for row in self.audit.itertuples():
            self.assertNotIn(row.heldout_animal, json.loads(row.train_animals))
            self.assertEqual(len(json.loads(row.train_animals)), 3)
            self.assertEqual(row.train_sessions, 6)
            self.assertEqual(row.train_events, 72)
            self.assertEqual(row.train_source, "real")
            self.assertEqual(row.train_cohort, "all_fixed_candidates")
            self.assertEqual(row.train_split, 0)
        self.assertNotIn("sim_calibration", set(self.predictions["source"]))

    def test_one_class_and_no_training_labels_fallbacks(self):
        train = self.frame.loc[self.frame["source"].eq("real")].copy()
        train["b_mass20"] = 0.8
        for model in reporter.LOGISTIC_MODELS:
            fitted = reporter.fit_predictor(train, model, 20)
            np.testing.assert_array_equal(fitted.predict(train), np.ones(len(train)))
            if model != "prevalence":
                self.assertEqual(fitted.status, "single_class_fallback")
        train["b_mass20"] = np.nan
        fitted = reporter.fit_predictor(train, "full", 20)
        self.assertEqual(fitted.status, "no_training_labels")
        self.assertTrue(np.isnan(fitted.predict(train)).all())

    def test_imputation_and_scaling_use_training_only(self):
        train = self.frame.loc[self.frame["source"].eq("real") & self.frame["animal"].ne("Rat0")].copy()
        train.loc[train.index[0], "a_width_cm"] = np.nan
        fitted = reporter.fit_predictor(train, "full", 20)
        before_impute = fitted.pipeline.named_steps["impute"].statistics_.copy()
        before_scale = fitted.pipeline.named_steps["scale"].mean_.copy()
        heldout = self.frame.loc[self.frame["animal"].eq("Rat0")].copy()
        heldout["a_width_cm"] = 1e6
        fitted.predict(heldout)
        np.testing.assert_array_equal(before_impute, fitted.pipeline.named_steps["impute"].statistics_)
        np.testing.assert_array_equal(before_scale, fitted.pipeline.named_steps["scale"].mean_)
        width_index = list(reporter.features(train, "full").columns).index("a_width_cm")
        self.assertEqual(before_impute[width_index], train["a_width_cm"].median())

    def test_known_truth_counts_wrong_agreement_as_error(self):
        frame = self.frame.copy()
        frame[["b_mass20", "b_mass40"]] = 1.0
        frame.loc[frame["source"].ne("real"), ["a_truth_error_cm", "b_truth_error_cm"]] = 100.0
        predictions, _ = reporter.loao_predictions(frame)
        truth = reporter.known_truth_metrics(predictions)
        any_tier = truth.loc[truth["tier"].eq("any_supported")]
        np.testing.assert_allclose(any_tier["false_agreement"], 1.0)
        np.testing.assert_allclose(any_tier.loc[any_tier["model"].isin(reporter.LOGISTIC_MODELS), "coverage"], 1.0)
        supported = predictions["source"].ne("real") & predictions["tier"].isin(reporter.SUPPORTED)
        self.assertTrue(predictions.loc[supported, "claim_b_supported"].eq(1).all())
        by_rat, _ = reporter.probability_metrics(predictions)
        gates = reporter.continuation_gates(predictions, by_rat, truth)
        errors = gates.loc[gates["gate"].eq("known_truth_false_agreement")]
        self.assertTrue(errors["status"].eq("fail").all())

    def test_event_draw_and_duplicate_weighting(self):
        selected = self.predictions.loc[
            self.predictions["model"].eq("full") & self.predictions["source"].eq("sim_test")
            & self.predictions["generator"].eq("matched_map")
        ].copy()
        repeated = pd.concat([selected, *[selected.iloc[[0]]] * 20], ignore_index=True)
        more_draws = selected.iloc[[0]].copy()
        more_draws["draw"] = 99
        repeated = pd.concat([repeated, more_draws], ignore_index=True)
        original_rat, original_equal = reporter.probability_metrics(selected)
        repeated_rat, repeated_equal = reporter.probability_metrics(repeated)
        for left, right in ((original_rat, repeated_rat), (original_equal, repeated_equal)):
            for column in ("log_loss", "brier", "n_events"):
                np.testing.assert_allclose(left[column], right[column])
        original_truth = reporter.known_truth_metrics(selected)
        repeated_truth = reporter.known_truth_metrics(repeated)
        for column in ("coverage", "false_agreement", "n_events"):
            np.testing.assert_allclose(original_truth[column], repeated_truth[column], equal_nan=True)
        self.assertAlmostEqual(reporter.balanced_weights(repeated).sum(), 1.0)

    def test_sensitivity_predictions_without_retraining(self):
        extra = self.frame.copy()
        extra["split"] = 1
        extra["a_width_cm"] += 10
        extra[["b_mass20", "b_mass40"]] = 1.0
        frame = pd.concat([self.frame, extra], ignore_index=True)
        predictions, audit = reporter.loao_predictions(frame, include_sensitivity=True)
        pd.testing.assert_frame_equal(self.audit, audit)
        primary = predictions.loc[predictions["split"].eq(0)].reset_index(drop=True)
        pd.testing.assert_frame_equal(self.predictions, primary)
        self.assertEqual(set(predictions["split"]), {0, 1})
        logistic = predictions.loc[predictions["model"].isin(reporter.LOGISTIC_MODELS)]
        self.assertTrue((logistic["p40"] >= logistic["p20"]).all())

    def test_tier_boundaries(self):
        actual = reporter.prediction_tiers(np.array([0.8, 0.79, 0.1, np.nan]), np.array([0.9, 0.8, 0.79, np.nan]))
        self.assertEqual(list(actual), ["destination_supported", "coarse_region_supported", "content_unresolved", "content_unresolved"])

    def test_probability_roundoff_is_clipped_but_invalid_mass_rejected(self):
        frame = self.frame.copy()
        frame.loc[frame.index[0], ["b_mass20", "b_mass40"]] = [-1e-11, 1 + 1e-11]
        valid = reporter.validate_readouts(frame)
        self.assertEqual(valid.iloc[0]["b_mass20"], 0)
        self.assertEqual(valid.iloc[0]["b_mass40"], 1)
        frame.loc[frame.index[0], "b_mass40"] = 1 + 1e-8
        with self.assertRaises(ValueError):
            reporter.validate_readouts(frame)

    def test_b_intervention_changes_targets_not_paired_a_predictions(self):
        frame = self.frame.copy()
        frame.loc[frame["source"].eq("sim_test"), ["b_mass20", "b_mass40"]] = [0.1, 0.2]
        frame.loc[frame["source"].eq("sim_conflict"), ["b_mass20", "b_mass40"]] = [0.8, 0.9]
        predictions, _ = reporter.loao_predictions(frame)
        keys = ["animal", "session", "window_uid", "split", "draw", "generator", "cohort", "model"]
        test = predictions.loc[predictions["source"].eq("sim_test")].set_index(keys).sort_index()
        conflict = predictions.loc[predictions["source"].eq("sim_conflict")].set_index(keys).sort_index()
        pd.testing.assert_frame_equal(test[["p20", "p40", "tier"]], conflict[["p20", "p40", "tier"]])
        self.assertTrue(test["y20"].eq(0).all())
        self.assertTrue(conflict["y20"].eq(1).all())

    def test_simple_thresholds_and_row_provenance(self):
        for row in self.predictions.itertuples():
            self.assertNotIn(row.animal, json.loads(row.train_animals))
            self.assertEqual(row.fold_id, f"loao:{row.animal}")
        spikes = self.predictions.loc[self.predictions["model"].eq("threshold_spikes")]
        np.testing.assert_array_equal(spikes["tier"].eq("destination_supported"), (spikes["a_spikes"] >= 3) & (spikes["a_active"] >= 2))
        self.assertTrue(spikes["p20"].isna().all())
        entropy = self.predictions.loc[self.predictions["model"].eq("threshold_entropy")]
        for animal, group in entropy.groupby("animal"):
            train = self.frame.loc[self.frame["source"].eq("real") & self.frame["animal"].ne(animal)]
            self.assertTrue(group["entropy_cutoff"].eq(reporter.training_entropy_cutoff(train)).all())
            np.testing.assert_array_equal(group["tier"].eq("destination_supported"), group["a_entropy"] <= group["entropy_cutoff"])

    def test_failed_primary_session_is_not_replaced_by_sensitivity(self):
        frame = self.frame.loc[~(self.frame["animal"].eq("Rat1") & self.frame["session"].eq("Open0"))].copy()
        sensitivity = self.frame.loc[self.frame["animal"].eq("Rat1") & self.frame["session"].eq("Open0")].copy()
        sensitivity["split"] = 1
        predictions, audit = reporter.loao_predictions(pd.concat([frame, sensitivity]), include_sensitivity=True)
        self.assertTrue(predictions.loc[predictions["animal"].eq("Rat1") & predictions["session"].eq("Open0"), "split"].eq(1).all())
        other = audit.loc[audit["heldout_animal"].ne("Rat1")]
        self.assertTrue(other["train_events"].eq(60).all())
        heldout = audit.loc[audit["heldout_animal"].eq("Rat1")]
        self.assertTrue(heldout["train_events"].eq(72).all())

    def test_missing_generator_and_zero_claims_never_pass(self):
        predictions = self.predictions.loc[self.predictions["generator"].ne("drift_gain")].copy()
        predictions["tier"] = "content_unresolved"
        predictions["truth_correct"] = np.nan
        by_rat, _ = reporter.probability_metrics(predictions)
        truth = reporter.known_truth_metrics(predictions)
        gates = reporter.continuation_gates(predictions, by_rat, truth)
        missing = gates.loc[gates["generator"].eq("drift_gain")]
        self.assertTrue(missing["status"].eq("skipped").all())
        false = gates.loc[gates["gate"].eq("known_truth_false_agreement")]
        self.assertTrue(false["status"].eq("skipped").all())
        self.assertTrue(false["value"].isna().all())
        coverage = gates.loc[gates["gate"].eq("known_truth_coverage") & gates["generator"].eq("matched_map")]
        self.assertTrue(coverage["status"].eq("fail").all())

    def test_one_rat_with_no_claims_does_not_become_zero_error(self):
        predictions = self.predictions.copy()
        predictions["tier"] = "destination_supported"
        predictions["truth_correct"] = 1.0
        missing = predictions["animal"].eq("Rat0")
        predictions.loc[missing, "tier"] = "content_unresolved"
        predictions.loc[missing, "truth_correct"] = np.nan
        by_rat, _ = reporter.probability_metrics(predictions)
        truth = reporter.known_truth_metrics(predictions)
        gates = reporter.continuation_gates(predictions, by_rat, truth)
        false = gates.loc[gates["gate"].eq("known_truth_false_agreement")]
        self.assertTrue(false["status"].eq("skipped").all())
        self.assertTrue(false["value"].isna().all())
        equal = truth.loc[truth["aggregation"].eq("equal_rat")]
        self.assertTrue(equal["false_agreement"].isna().all())

    def test_three_rat_report_is_explicitly_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "readouts.csv.gz"
            self.frame.loc[self.frame["animal"].ne("Rat3")].to_csv(source, index=False)
            reporter.write_report(source, root / "report", run_qc_note="Local matching failed in every primary session.")
            report = (root / "report/report.md").read_text()
            self.assertIn("Fewer than four rats", report)
            self.assertIn("3 rats, 6 sessions", report)
            self.assertIn("Local matching failed in every primary session.", report)

    def test_each_radius_requires_every_rat_improvement(self):
        by_rat, _ = reporter.probability_metrics(self.predictions)
        truth = reporter.known_truth_metrics(self.predictions)
        base = by_rat["source"].eq("real") & by_rat["model"].eq("spikes_entropy")
        full = by_rat["source"].eq("real") & by_rat["model"].eq("full")
        by_rat.loc[base, "log_loss"] = 1.0
        by_rat.loc[full, "log_loss"] = 0.8
        by_rat.loc[full & by_rat["animal"].eq("Rat0") & by_rat["radius_cm"].eq(20), "log_loss"] = 1.1
        gates = reporter.continuation_gates(self.predictions, by_rat, truth)
        fine = gates.loc[gates["source"].eq("real") & gates["radius_cm"].eq(20)]
        statuses = fine.set_index("gate")["status"]
        self.assertEqual(statuses["relative_log_loss_improvement"], "pass")
        self.assertEqual(statuses["every_rat_log_loss_improves"], "fail")

    def test_cli_writes_artifacts_and_explicit_claim_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "readouts.csv.gz"
            destination = root / "report"
            self.frame.to_csv(source, index=False)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--event-readouts", str(source), "--output-dir", str(destination)],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("Reliability report:", result.stdout)
            for name in ("predictions.csv.gz", "metrics_by_rat.csv", "metrics_equal_rat.csv", "known_truth_correctness.csv", "gates.csv", "fold_audit.csv", "feature_columns.json", "manifest.json"):
                self.assertTrue((destination / name).is_file(), name)
            report = (destination / "report.md").read_text()
            self.assertIn("NOT true replay content", report)
            self.assertIn("4 rats, 8 sessions, 96 unique events", report)
            self.assertIn("unobserved-intervention limitation", report)
            provenance = json.loads((destination / "manifest.json").read_text())
            self.assertIn("code_commit", provenance)
            self.assertIn("git_dirty", provenance)
            self.assertEqual(provenance["input_file_sha256"]["event_readouts"], reporter.sha256(source))
            self.assertEqual(provenance["output_file_sha256"]["report.md"], reporter.sha256(destination / "report.md"))
            feature_audit = json.loads((destination / "feature_columns.json").read_text())
            self.assertFalse(any(name.startswith("b_") for names in feature_audit["feature_columns"].values() for name in names))
            predictions = pd.read_csv(destination / "predictions.csv.gz")
            self.assertNotIn("sim_calibration", set(predictions["source"]))


if __name__ == "__main__":
    unittest.main()
