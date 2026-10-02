"""Fixed RUN-only nuisance-model grid; never read replay order or fit associations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.special import xlogy

try:
    from scripts import _run_pair_rate_glm as glm
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    import _run_pair_rate_glm as glm
    from _provenance import build_script_provenance, file_sha256


def require(value, message):
    if not value:
        raise ValueError(message)


def candidates(base, protocol):
    require(base.get("rate_model_family") == "smooth_poisson_glm" and
            base.get("glm_speed_scaling") == "fixed_log_run_bounds", "Bounded source GLM required")
    axes = [protocol[k] for k in ("position_knot_cm", "l2_penalties", "interaction_modes")]
    require(all(len(a) > 0 and len(a) == len(set(a)) for a in axes), "Nonempty unique grid axes required")
    require(all(np.isfinite(x) and x > 0 for a in axes[:2] for x in a) and
            set(axes[2]) <= {"full", "additive"}, "Invalid fixed diagnostic grid")
    require(isinstance(protocol["cells_per_pause"], int) and protocol["cells_per_pause"] >= 2,
            "At least two frozen cells required")
    result = []
    for spacing, penalty, mode in product(*axes):
        name = f"knot{spacing:g}_l2{penalty:g}_{mode}"
        p = {**base, "glm_position_knot_cm": spacing, "glm_l2_penalty": penalty,
             "glm_spatial_direction_interaction": mode == "full",
             "glm_spatial_theta_interaction": mode == "full"}
        result.append((name, p))
    require(len(result) <= 24, "Diagnostic exceeds frozen bounded capacity")
    return result


def period_data(bank, period, cells, width):
    ids = np.asarray(bank["unit_ids"])
    require(len(ids) >= cells and len(np.unique(ids)) == len(ids), "Missing or duplicate frozen unit IDs")
    times = np.asarray(bank[f"{period}_time_s"])
    counts = np.asarray(bank[f"{period}_counts"])
    require(counts.shape == (len(times), len(ids)), "RUN counts and unit IDs disagree")
    require(len(times) > 0 and np.isfinite(times).all() and np.all(np.diff(times) > 0), "Invalid physical RUN clock")
    require(np.allclose(bank[f"{period}_bin_duration_s"], width, rtol=0, atol=1e-12) and
            np.allclose((times - times[0]) / width, np.rint((times - times[0]) / width), rtol=0, atol=1e-5),
            "Source physical RUN bin grid changed")
    covariates = {key: np.asarray(bank[f"{period}_{source}"]) for key, source in
                 (("position", "position_cm"), ("direction", "direction_rad"),
                  ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
    require(all(len(x) == len(times) for x in covariates.values()), "RUN covariate clocks disagree")
    require(covariates["theta"].ndim == 2 and covariates["theta"].shape[1] > 0, "Actual LFP theta references required")
    valid = np.isfinite(covariates["theta"]).all(axis=1)
    require(valid.any(), "No valid source LFP theta bins")
    return ids[:cells], counts[valid, :cells], times[valid], {k: x[valid] for k, x in covariates.items()}, len(times)


def evaluate(counts, times, covariates, ids, p):
    prepared = glm.prepare(times, covariates, p)
    _, qc, means, global_means = glm.crossfit(counts, times, np.zeros(len(times), int), p["run_bin_s"],
        p, covariates, prepared, return_predictions=True)
    complete = bool(np.isfinite(means).all())
    score = np.sum(xlogy(counts, means / global_means) - means + global_means, axis=0)
    rows = []
    for k, unit in enumerate(ids):
        rows.append({"unit_id": int(unit), "status": "complete" if complete else "incomplete_fit",
            "observed_bins": len(times), "observed_spikes": int(counts[:, k].sum()),
            "minimum_training_spikes": min(int(counts[item["training"], k].sum()) for item in prepared["folds"]),
            "heldout_log_score_gain": float(score[k]) if complete else None,
            "observed_mean_count": float(counts[:, k].mean()),
            "predicted_mean_count": float(means[:, k].mean()) if complete else None,
            "minimum_prediction": float(means[:, k].min()) if complete else None,
            "maximum_prediction": float(means[:, k].max()) if complete else None,
            "independent_biological_subject": False, "association_fit": False})
    return rows, qc


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--endpoint-verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.endpoint_dir
    m = json.loads((root / "manifest.json").read_text())
    v = json.loads(args.endpoint_verification.read_text())
    require(v.get("verified") and v["input_file_sha256"]["endpoint_manifest"] == file_sha256(root / "manifest.json"),
            "Independent identity-matched endpoint verification required")
    rate_protocol = Path(m["input_file_paths"]["protocol"])
    require(file_sha256(rate_protocol) == m["input_file_sha256"]["protocol"], "Source rate protocol changed")
    source = Path(m["input_file_paths"]["measurement_manifest"])
    require(file_sha256(source) == m["input_file_sha256"]["measurement_manifest"], "Source cohort changed")
    sm = json.loads(source.read_text())
    for name in ("banks.csv", "pauses.csv"):
        require(file_sha256(source.parent / name) == sm["outputs_sha256"][name], "Source inventory changed")
    base, protocol = json.loads(rate_protocol.read_text()), json.loads(args.protocol.read_text())
    grid = candidates(base, protocol)
    provenance = build_script_provenance(input_paths={"endpoint_manifest": root / "manifest.json",
        "endpoint_verification": args.endpoint_verification, "source_rate_protocol": rate_protocol,
        "measurement_manifest": source, "diagnostic_protocol": args.protocol})
    require(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed checkout required")
    banks = pd.read_csv(source.parent / "banks.csv")
    pauses = pd.read_csv(source.parent / "pauses.csv")
    require(len(banks) == len(pauses) == m["frozen_pauses"] and
            not banks.duplicated(["session", "pause_id"]).any() and
            not pauses.duplicated(["session", "pause_id"]).any(), "Frozen pause accounting differs")
    selected = banks.merge(pauses[["animal", "session", "pause_id", "theta_run_support_screen_passed"]],
        on=["animal", "session", "pause_id"], validate="one_to_one").sort_values(["animal", "session", "pause_id"])
    require(len(selected) == len(banks) and selected.theta_run_support_screen_passed.isin([True, False]).all(),
            "Missing or ambiguous source theta status")
    require(selected.theta_run_support_screen_passed.any(), "No source theta-qualified pauses")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(not (args.output_dir / "manifest.json").exists(), "Completed diagnostic cannot be overwritten")
    import fcntl
    lock = (args.output_dir / ".audit.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.output_dir / "checkpoint_index.json"
    identity = {"code_commit": provenance["code_commit"], "inputs": provenance["input_file_sha256"]}
    state = json.loads(state_path.read_text()) if state_path.exists() else {"identity": identity, "tasks": {}}
    require(state["identity"] == identity, "Checkpoint producer or input identities differ")
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    atomic_json(state_path, state)
    cell_rows, period_rows, inventory = [], [], []
    for row in selected.itertuples(index=False):
        key = {"animal": row.animal, "session": row.session, "pause_id": row.pause_id}
        require(file_sha256(row.bank_path) == row.bank_sha256, "Source RUN bank changed")
        inventory.append({**key, "bank_sha256": row.bank_sha256,
            "status": "frozen_theta_screen_failed" if not row.theta_run_support_screen_passed else "included",
            "cells_requested": protocol["cells_per_pause"]})
        if not row.theta_run_support_screen_passed:
            continue
        with np.load(row.bank_path, allow_pickle=False) as bank:
            for period in ("pre", "post"):
                ids, counts, times, covariates, source_bins = period_data(bank, period, protocol["cells_per_pause"], base["run_bin_s"])
                for name, p in grid:
                    task = {**key, "period": period, "candidate": name, "unit_ids": ids.tolist(),
                            "bank_sha256": row.bank_sha256}
                    digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
                    path = checkpoint_dir / (digest + ".json")
                    if digest in state["tasks"]:
                        require(file_sha256(path) == state["tasks"][digest], "Checkpoint contents changed")
                        result = json.loads(path.read_text())
                        require(result["task"] == task, "Checkpoint task identity changed")
                    else:
                        cells, qc = evaluate(counts, times, covariates, ids, p)
                        result = {"task": task, "cells": cells, "quality": qc}
                        # JSON checkpoints preserve incomplete fits without certifying them.
                        result = json.loads(json.dumps(result), parse_constant=lambda _: None)
                        atomic_json(path, result)
                        state["tasks"][digest] = file_sha256(path)
                        atomic_json(state_path, state)
                    cells, qc = result["cells"], result["quality"]
                    complete = all(c["status"] == "complete" for c in cells)
                    cell_rows.extend({**key, "period": period, "candidate": name, **c} for c in cells)
                    period_rows.append({**key, "period": period, "candidate": name, "source_bins": source_bins,
                        "selected_cells": len(ids), "position_knot_cm": p["glm_position_knot_cm"],
                        "l2_penalty": p["glm_l2_penalty"], "interaction_mode": "full" if p["glm_spatial_theta_interaction"] else "additive",
                        "status": "complete" if complete else "incomplete_fit",
                        "positive_cells": sum(c["heldout_log_score_gain"] is not None and c["heldout_log_score_gain"] > 0 for c in cells),
                        "observed_spikes": int(counts.sum()),
                        **{k: value for k, value in qc.items() if k != "folds"}, "fold_diagnostics": json.dumps(qc["folds"])})
                    print(f"RUN RATE {row.animal} {row.pause_id} {period} {name}: {qc['heldout_poisson_improvement_over_global']}", flush=True)
    periods = pd.DataFrame(period_rows)
    require(len(periods) == 2 * len(grid) * sum(r["status"] == "included" for r in inventory), "Incomplete diagnostic grid")
    summary = []
    for name, data in periods.groupby("candidate", sort=True):
        complete = data.status == "complete"
        summary.append({"candidate": name, "frozen_periods": len(data), "complete_periods": int(complete.sum()),
            "positive_periods": int((complete & (data.heldout_poisson_improvement_over_global > 0)).sum()),
            "median_period_log_score_gain": float(data.heldout_poisson_improvement_over_global.median()) if complete.all() else None,
            "animal_median_gains": json.dumps(data.groupby("animal").heldout_poisson_improvement_over_global.median().to_dict()),
            "selected_for_endpoint": False, "full_procedure_calibrated": False, "biological_inference": False})
    outputs = {"cell_quality.csv": pd.DataFrame(cell_rows), "period_quality.csv": periods,
               "inventory.csv": pd.DataFrame(inventory), "candidate_summary.csv": pd.DataFrame(summary)}
    for name, data in outputs.items():
        data.to_csv(args.output_dir / name, index=False)
    result = {**provenance, "protocol_id": protocol["protocol_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_versions": {"python": platform.python_version(), **{k: version(k) for k in ("numpy", "scipy", "pandas", "scikit-learn")}},
        "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in outputs},
        "checkpoints_sha256": state["tasks"], "frozen_pauses": len(selected), "diagnostic_candidates": len(grid),
        "readout": "RUN rate prediction only; no replay counts, order measures, pair endpoints or associations read",
        "cells_per_pause": protocol["cells_per_pause"], "candidate_selected": False, "association_fit": False,
        "full_procedure_calibrated": False, "goal_complete": False, "inference_unit": "animal"}
    atomic_json(args.output_dir / "manifest.json", result)
    print(f"COMPLETE RUN-only grid={len(periods)} period fits; association_fit=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
