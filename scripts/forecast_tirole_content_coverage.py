"""Cross-fitted future-neuron prediction for the frozen two-track coverage experiment."""

import argparse
import importlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.independent_rejected_forecast import BASELINES, forecast_distributions, score_predictions
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.learned_assembly_prediction import fit_learned_assembly
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import nested_subsets, stable_seed
from scripts.audit_2d_count_conditioned_prediction import folds
from scripts.report_tirole_content_coverage import load_campaigns
from scripts.score_tirole_content_coverage import validate_bank

N_STATES = 50
HORIZONS = (1, 2, 4)


def relative_ids(population, units):
    population, units = np.asarray(population, int), np.asarray(units, int)
    if len(np.unique(population)) != len(population) or not np.array_equal(population, np.sort(population)) or len(np.unique(units)) != len(units):
        raise ValueError("sorted unique population and unique subset required")
    if not np.isin(units, population).all():
        raise ValueError("unknown population unit")
    return np.searchsorted(population, units)


def forecast_event(counts, train, held, fit, operator, matched):
    if set(train) & set(held) or not len(train) or not len(held):
        raise ValueError("disjoint nonempty inference and evaluation populations required")
    pred = forecast_distributions(counts[:, train], fit.probabilities[train], operator, matched, HORIZONS)
    return pred, score_predictions(pred, counts[:, held], fit.probabilities[held], fit.global_probability[held])


def run(bank, campaigns, output):
    if output.exists():
        raise ValueError("new forecast output required")
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("commit forecast protocol before scoring")
    runtime = {"python_executable": sys.executable, "python_version": sys.version, "numpy_version": np.__version__}
    for module in ["hmmlearn", "hmmlearn._hmmc"]:
        loaded = importlib.import_module(module)
        runtime[module] = {"version": getattr(loaded, "__version__", None), "path": loaded.__file__, "sha256": file_sha256(Path(loaded.__file__))}
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    data, sources = load_campaigns(campaigns)
    session = info["session"]
    data = data[data.session == session].copy()
    if data.empty:
        raise ValueError("no frozen labels for session")
    relevant = []
    for source in sources:
        m = json.loads(Path(source["manifest"]).read_text())
        if m["session"] == session:
            if m["bank_manifest_sha256"] != file_sha256(bank / "manifest.json"):
                raise ValueError("bank/score identity mismatch")
            relevant.append(source)
    labels = data[(data.arm == "real") & data.fraction.isin([1, 0.5])].set_index(["event_id", "split", "repeat", "fraction"])
    if labels.index.duplicated().any():
        raise ValueError("duplicate frozen labels")
    selected = events[events.event_id.isin(data.event_id)].copy()
    population = np.sort(np.setdiff1d(maps["common_units"], parts["detector"]))
    cache = {int(eid): counts[offsets[eid] : offsets[eid + 1]][:, population] for eid in selected.event_id}
    output.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    fit_records = []
    for fold, test, cal, excluded in folds(selected):
        if set(test.event_id) & set(cal.event_id):
            raise ValueError("test events leaked into calibration")
        fit_seed = stable_seed(20260918, session, fold, "future-neural-K50")
        fit = fit_learned_assembly([cache[int(i)] for i in cal.event_id], N_STATES, fit_seed)
        record = {
            "fold": fold,
            "test_ids": test.event_id.tolist(),
            "calibration_ids": cal.event_id.tolist(),
            "excluded_guard_ids": excluded.event_id.tolist(),
            "fit_seed": fit_seed,
            "n_calibration_bins": int(fit.n_calibration_bins),
            "converged": bool(fit.converged),
            "restart": int(fit.restart),
            "restart_objectives": list(fit.restart_objectives),
            "restart_converged": list(fit.restart_converged),
        }
        fit_records.append(record)
        (output / "folds.json").write_text(json.dumps(fit_records, indent=2) + "\n")
        if not fit.converged:
            raise ValueError("unconverged fixed K50 model; do not change K after inspecting results")
        np.savez_compressed(
            output / f"fit_{fold}.npz",
            probabilities=fit.probabilities,
            initial=fit.initial,
            transition=fit.transition,
            occupancy=fit.occupancy,
            global_probability=fit.global_probability,
            objective_trace=fit.objective_trace,
            population=population,
        )
        op = NeuralOperator(fit.initial, fit.transition, fit.occupancy)
        null = OccupancyMatchedNull.from_operator(op)
        (output / f"matched_null_{fold}.json").write_text(json.dumps(null.diagnostics(), indent=2) + "\n")
        for eid in test.event_id:
            eid = int(eid)
            c = cache[eid]
            for split, part in enumerate(parts["splits"]):
                held = relative_ids(population, part["evaluation"])
                full = relative_ids(population, part["inference"])
                _, full_scores = forecast_event(c, full, held, fit, op, null)
                for repeat in range(5):
                    subsets = nested_subsets(np.asarray(part["inference"]), session, split, repeat)
                    half = relative_ids(population, subsets[0.5])
                    _, half_scores = forecast_event(c, half, held, fit, op, null)
                    full_label = labels.loc[(eid, split, repeat, 1.0)]
                    for fraction, scores in [(1.0, full_scores), (0.5, half_scores)]:
                        label = labels.loc[(eid, split, repeat, fraction)]
                        common = {k: label[k] for k in ["animal", "cohort_stratum", "candidate_stratum", "epoch", "duration_s", "ripple_peak_z"]}
                        for score in scores:
                            rows.append(
                                {
                                    **common,
                                    "session": session,
                                    "event_id": eid,
                                    "fold": fold,
                                    "split": split,
                                    "repeat": repeat,
                                    "fraction": fraction,
                                    "horizon_ms": 20 * score["horizon"],
                                    "sequence_accepted": bool(label.sequence_accepted),
                                    "full_accepted": bool(full_label.sequence_accepted),
                                    "lost_with_thinning": bool(full_label.sequence_accepted and not labels.loc[(eid, split, repeat, 0.5)].sequence_accepted),
                                    "half_has_sequence_opportunity": bool(labels.loc[(eid, split, repeat, 0.5)].sequence_eligible),
                                    "n_inference_units": len(full) if fraction == 1 else len(half),
                                    "n_evaluation_units": len(held),
                                    **score,
                                }
                            )
        state = {"status": "running", "completed_folds": fold + 1, "expected_folds": 5, "elapsed_s": time.monotonic() - started}
        (output / "progress.json").write_text(json.dumps(state) + "\n")
        print(json.dumps(state), flush=True)
    frame = pd.DataFrame(rows)
    expected = len(selected) * 5 * 5 * 2 * len(HORIZONS)
    key = ["event_id", "split", "repeat", "fraction", "horizon_ms"]
    if len(frame) != expected or frame.duplicated(key).any():
        raise ValueError("incomplete or duplicate forecast conditions")
    scorecols = ["score_dynamic"] + ["score_" + b for b in BASELINES]
    if not np.isfinite(frame[scorecols]).all().all():
        raise ValueError("nonfinite forecast scores")
    pieces = []
    for group in ["all", "lost", "lost_with_sequence_opportunity", "full_accepted"]:
        mask = {
            "all": np.ones(len(frame), bool),
            "lost": frame.lost_with_thinning,
            "lost_with_sequence_opportunity": frame.lost_with_thinning & frame.half_has_sequence_opportunity,
            "full_accepted": frame.full_accepted,
        }[group]
        for baseline in BASELINES:
            z = frame[mask].copy()
            z["group"] = group
            z["contrast"] = "dynamic_minus_" + baseline
            z["delta_per_spike"] = (z.score_dynamic - z["score_" + baseline]) / z.n_heldout_target_spikes.replace(0, np.nan)
            by = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction", "horizon_ms", "event_id", "group", "contrast"]
            pieces.append(z.groupby(by, as_index=False).agg(delta_per_spike=("delta_per_spike", "median"), n_informative_realizations=("delta_per_spike", "count")))
    summaries = pd.concat(pieces, ignore_index=True)
    by = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction", "horizon_ms", "group", "contrast"]
    session_summary = summaries.groupby(by, as_index=False).agg(
        n_events=("event_id", "size"),
        n_informative_events=("delta_per_spike", "count"),
        mean_delta_per_spike=("delta_per_spike", "mean"),
        median_delta_per_spike=("delta_per_spike", "median"),
    )
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during execution")
    frame.to_csv(output / "forecast_by_event_split.csv", index=False)
    summaries.to_csv(output / "forecast_by_event.csv", index=False)
    session_summary.to_csv(output / "forecast_by_session.csv", index=False)
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "git_dirty": False,
        "command_line": sys.argv,
        "session": session,
        "runtime": runtime,
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "source_score_manifests": relevant,
        "n_states": N_STATES,
        "horizons_ms": [20 * h for h in HORIZONS],
        "primary_horizon_ms": 40,
        "primary_comparator": "matched_own",
        "primary_fraction": 0.5,
        "expected_rows": expected,
        "actual_rows": len(frame),
        "target_event_heldout_in_fitting": True,
        "target_heldout_spikes_in_filter": False,
        "future_spikes_in_filter": False,
        "biological_confirmation": False,
        "elapsed_s": time.monotonic() - started,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.name != "progress.json"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "progress.json").write_text(json.dumps({"status": "complete", "rows": len(frame)}) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--campaign-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.campaign_dirs, a.output_dir)
