"""Fixed-selection leave-one-evaluation-neuron-out track-content sensitivity."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed
from scripts.audit_tirole_evaluation_identity import context_odds, matched_permutations, null_summary
from scripts.report_tirole_content_coverage import load_campaigns
from scripts.report_tirole_content_uncertainty import block_weights, checked, mean_interval
from scripts.score_tirole_content_coverage import validate_bank

SESSIONS = {"RAT3_SESS2": "candidate_banks_v2", "RAT5_SESS2": "candidate_banks_v3"}
METRICS = [prefix + name for prefix in ["", "conditional_"] for name in ["observed_log_odds", "identity_excess_log_odds", "identity_z_log_odds"]]


def delete_permutation_index(permutations, omitted):
    """Delete a vertex from each permutation cycle, preserving the remaining groups."""
    p = np.asarray(permutations)
    if p.ndim != 2 or p.shape[1] < 2 or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("integer permutation matrix required")
    n = p.shape[1]
    if not 0 <= omitted < n or not np.all(np.sort(p, axis=1) == np.arange(n)):
        raise ValueError("invalid permutation or omitted index")
    keep = np.delete(np.arange(n), omitted)
    out = p[:, keep].copy()
    row, col = np.where(out == omitted)
    out[row, col] = p[row, omitted]
    if np.any(out == omitted):
        raise ValueError("cycle deletion failed")
    return np.searchsorted(keep, out)


def lost_membership(data):
    x = data[(data.arm == "real") & (data.epoch == "POST") & (data.candidate_stratum == "ripple")]
    keys = ["session", "event_id", "split", "repeat"]
    full = x[x.fraction == 1][keys + ["sequence_accepted", "inferred_track"]].rename(columns={"sequence_accepted": "full_accepted", "inferred_track": "full_track"})
    half = x[x.fraction == 0.5][keys + ["sequence_accepted", "sequence_eligible"]]
    joined = full.merge(half, on=keys, validate="one_to_one", how="outer", indicator=True)
    if not joined._merge.eq("both").all() or not joined.loc[joined.full_accepted.eq(True), "full_track"].isin([1, 2]).all():
        raise ValueError("unpaired sequence decisions or invalid labels")
    return joined[joined.full_accepted & ~joined.sequence_accepted].drop(columns="_merge")


def event_vectors(membership, readout, omitted):
    keys = ["event_id", "split"]
    base = readout[readout.omitted_index == -1].set_index(keys)[METRICS].copy()
    replacement = readout[readout.omitted_index == omitted].set_index(keys)[METRICS]
    if base.index.has_duplicates or replacement.index.has_duplicates or not replacement.index.isin(base.index).all():
        raise ValueError("duplicate or foreign readout")
    base.loc[replacement.index] = replacement
    joined = membership.merge(base.reset_index(), on=keys, validate="many_to_one", how="left", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("missing fixed-selection readout")
    sign = np.where(joined.full_track == 1, 1.0, -1.0)
    for name in METRICS:
        joined[name] *= sign
    return joined.groupby("event_id")[METRICS].median()


def run(source, output):
    if output.exists():
        raise ValueError("new influence audit directory required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze the influence protocol before scoring")
    campaigns = [source / x for x in ["campaign_RAT3_SESS2_v1", "campaign_diagnostic_stage1_v1", "campaign_remaining_stage2_v1"]]
    scores, score_sources = load_campaigns(campaigns)
    scores = scores[scores.session.isin(SESSIONS)]
    if set(scores.session) != set(SESSIONS) or not scores.cohort_stratum.eq("strict_RUN_pass").all():
        raise ValueError("both frozen primary sessions required")
    identity_dir = source / "evaluation_identity_control_v1"
    im = checked(identity_dir, "manifest.json")
    if im["status"] != "complete" or im["selection_or_thresholds_changed"]:
        raise ValueError("invalid original identity control")
    if {(x["manifest"], x["sha256"]) for x in im["scorer_manifests"]} != {(x["manifest"], x["sha256"]) for x in score_sources}:
        raise ValueError("identity and selection sources differ")
    original = pd.read_csv(identity_dir / "evaluation_identity_by_event_split.csv").set_index(["session", "event_id", "split"])
    membership = lost_membership(scores)
    if set(membership.session) != set(SESSIONS):
        raise ValueError("no lost events in a required session")
    rows, summaries, event_rows, matching, inputs = [], [], [], [], {}
    max_baseline_error = 0.0
    for session, version in SESSIONS.items():
        bank = source / version / session
        bm, parts, events, counts, offsets, maps = validate_bank(bank)
        digest = file_sha256(bank / "manifest.json")
        inputs[str(bank / "manifest.json")] = digest
        for item in score_sources:
            sm = json.loads(Path(item["manifest"]).read_text())
            if sm["session"] == session and sm["bank_manifest_sha256"] != digest:
                raise ValueError("scoring bank differs")
        local = membership[membership.session == session]
        occupancy = maps["occupancy_s"]
        weights = occupancy / occupancy.sum(axis=1, keepdims=True)
        run_rates = (maps["rates"] * weights[:, None]).sum(axis=2).mean(axis=0)
        session_rows = []
        omitted_units = set()
        for split, part in enumerate(parts["splits"]):
            evaluation = np.asarray(part["evaluation"], int)
            omitted_units.update(evaluation.tolist())
            seed = stable_seed(20260918, session, split, "rate-matched-evaluation-identity")
            perms, groups = matched_permutations(run_rates[evaluation], seed=seed)
            selected = np.sort(local.loc[local.split == split, "event_id"].unique())
            for drop in [-1, *range(len(evaluation))]:
                keep = np.arange(len(evaluation)) if drop == -1 else np.delete(np.arange(len(evaluation)), drop)
                use = evaluation[keep]
                null = perms if drop == -1 else delete_permutation_index(perms, drop)
                omitted = -1 if drop == -1 else int(evaluation[drop])
                matching.append(
                    {
                        "session": session,
                        "split": split,
                        "omitted_index": omitted,
                        "evaluation_indices": json.dumps(use.tolist()),
                        "original_rate_groups": json.dumps(groups.tolist()),
                        "null_seed": seed,
                        "n_null": len(null),
                        "n_distinct_nulls": len(np.unique(null, axis=0)),
                        "mean_absolute_log_RUN_rate_gap": float(np.abs(np.log(run_rates[use][null]) - np.log(run_rates[use])[None]).mean()),
                    }
                )
                for eid in selected:
                    c = counts[offsets[eid] : offsets[eid + 1]][:, use]
                    row = {
                        "session": session,
                        "event_id": int(eid),
                        "split": split,
                        "omitted_index": omitted,
                        "omitted_original_unit_id": int(maps["unit_ids"][omitted]) if omitted >= 0 else -1,
                        "n_evaluation_cells": len(use),
                        "n_evaluation_spikes": int(c.sum()),
                        "n_evaluation_active_cells": int((c.sum(axis=0) > 0).sum()),
                    }
                    for conditional, prefix in [(False, ""), (True, "conditional_")]:
                        values = context_odds(c, maps["rates"][:, use], maps["valid_bins"], null, conditional)
                        row.update({prefix + k: v for k, v in null_summary(values).items()})
                    if drop == -1:
                        reference = original.loc[(session, eid, split)]
                        actual = np.array([row[k] for k in METRICS])
                        expected = reference[METRICS].to_numpy(float)
                        if not np.allclose(actual, expected, atol=1e-10, equal_nan=True):
                            raise ValueError("original held-out content did not reproduce")
                        finite = np.isfinite(actual) & np.isfinite(expected)
                        if finite.any():
                            max_baseline_error = max(max_baseline_error, float(np.abs(actual[finite] - expected[finite]).max()))
                    session_rows.append(row)
            print(json.dumps({"session": session, "split": split, "lost_event_split_pairs": len(selected)}), flush=True)
        frame = pd.DataFrame(session_rows)
        rows.extend(session_rows)
        candidate_ids = events.loc[(events.epoch == "POST") & events.primary_ripple_candidate, "event_id"].to_numpy()
        clock = events.set_index("event_id").loc[candidate_ids, "start_s"]
        codes, boot = block_weights(clock, 60, stable_seed(20260918, session, "POST", "ripple_z3", 60, "occupied-time-block-bootstrap"))
        baseline = event_vectors(local, frame, -1).reindex(candidate_ids)
        for omitted in [-1, *sorted(omitted_units)]:
            values = event_vectors(local, frame, omitted).reindex(candidate_ids)
            for eid in np.sort(local.event_id.unique()):
                event_rows.append({"session": session, "event_id": int(eid), "omitted_index": omitted, **values.loc[eid].to_dict()})
            for metric in METRICS:
                v = values[metric].to_numpy()
                difference = v - baseline[metric].to_numpy()
                affected = frame[(frame.omitted_index == omitted)] if omitted >= 0 else frame.iloc[:0]
                summaries.append(
                    {
                        "session": session,
                        "animal": bm["animal"],
                        "cohort_stratum": bm["cohort_stratum"],
                        "omitted_index": omitted,
                        "omitted_original_unit_id": int(maps["unit_ids"][omitted]) if omitted >= 0 else -1,
                        "metric": metric,
                        "n_lost_events": int(local.event_id.nunique()),
                        "n_affected_event_split_pairs": len(affected),
                        "n_affected_events": int(affected.event_id.nunique()),
                        **mean_interval(v, codes, boot),
                        **{"change_" + k: v for k, v in mean_interval(difference, codes, boot).items()},
                    }
                )
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during the audit")
    output.mkdir(parents=True)
    for name, table in [
        ("evaluation_neuron_by_event_split", pd.DataFrame(rows)),
        ("evaluation_neuron_event_summary", pd.DataFrame(event_rows)),
        ("evaluation_neuron_influence_summary", pd.DataFrame(summaries)),
        ("evaluation_neuron_matching", pd.DataFrame(matching)),
        ("frozen_lost_membership", membership),
    ]:
        table.to_csv(output / (name + ".csv"), index=False)
    manifest = {
        "status": "complete",
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_line": sys.argv,
        "bank_manifest_sha256": inputs,
        "source_score_manifests": score_sources,
        "identity_manifest_sha256": file_sha256(identity_dir / "manifest.json"),
        "primary_sessions": sorted(SESSIONS),
        "selection_changed": False,
        "inference_or_detector_cells_changed": False,
        "future_prediction_rescored": False,
        "evaluation_only_deletions": True,
        "rate_groups_refitted": False,
        "n_null": 199,
        "n_rows": len(rows),
        "n_unique_lost_events": int(membership[["session", "event_id"]].drop_duplicates().shape[0]),
        "maximum_original_readout_error": max_baseline_error,
        "biological_confirmation": False,
        "scope": "Fixed-selection single-evaluation-cell influence, not a new replay test or population replication. Conditional intervals are descriptive and not simultaneous across omissions.",
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.source_dir, a.output_dir)
