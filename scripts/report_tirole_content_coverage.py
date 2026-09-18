"""Non-rescoring coverage/content report; missing cohorts never count as replication."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed


def load_campaigns(roots):
    frames = []
    sources = []
    for root in roots:
        state = json.loads((root / "campaign_status.json").read_text())
        if state["status"] != "complete" or state["failed"] or len(state["completed"]) != state["expected_jobs"]:
            raise ValueError("incomplete campaign; do not report partial runs as complete")
        for job in state["completed"]:
            path = root / job["job"]
            m = json.loads((path / "manifest.json").read_text())
            if m["technical_pilot_only"] or m["status"] != "complete" or m["selected_events"] <= 0 or m["git_dirty"]:
                raise ValueError("non-scientific or incomplete scorer manifest")
            csv = path / "event_content_coverage_scores.csv"
            if file_sha256(csv) != m["output_sha256"]:
                raise ValueError("score checksum mismatch")
            frame = pd.read_csv(csv)
            if len(frame) != m["expected_score_rows"]:
                raise ValueError("score row count mismatch")
            if set(frame.event_id) != set(m["selected_event_ids"]):
                raise ValueError("selected event identity mismatch")
            expected = {(1.0, "real"), (1.0, "cell_identity_randomized"), (0.75, "real"), (0.5, "real"), (0.5, "cell_identity_randomized"), (0.25, "real")}
            for _, g in frame.groupby("event_id"):
                if set(zip(g.fraction, g.arm, strict=True)) != expected or len(g) != 6:
                    raise ValueError("incomplete condition coverage")
            frame["candidate_stratum"] = m["candidate_stratum"]
            frames.append(frame)
            sources.append(
                {
                    "manifest": str((path / "manifest.json").resolve()),
                    "sha256": file_sha256(path / "manifest.json"),
                    "code_commit": m["code_commit"],
                    "configuration_sha256": m["configuration_sha256"],
                }
            )
    if not frames:
        raise ValueError("no scientific scores")
    x = pd.concat(frames, ignore_index=True)
    if x.duplicated(["session", "event_id", "split", "repeat", "fraction", "arm"]).any():
        raise ValueError("duplicated experimental observations")
    if len({s["configuration_sha256"] for s in sources}) != 1:
        raise ValueError("mixed scientific configurations")
    return x, sources


def event_summaries(data):
    real = data.loc[data.arm == "real"].copy()
    keys = ["session", "event_id", "split", "repeat"]
    full = real.loc[real.fraction == 1, keys + ["sequence_accepted", "inferred_track"]].rename(columns={"sequence_accepted": "full_accepted", "inferred_track": "full_track"})
    real = real.merge(full, on=keys, validate="many_to_one")
    sign = np.where(real.full_track == 1, 1, np.where(real.full_track == 2, -1, np.nan))
    real["signed_evaluation_z"] = sign * real.evaluation_z_log_odds
    real["signed_conditional_evaluation_z"] = sign * real.conditional_evaluation_z_log_odds
    masks = {
        "all": np.ones(len(real), bool),
        "full_accepted": real.full_accepted,
        "retained": real.full_accepted & real.sequence_accepted,
        "lost": real.full_accepted & ~real.sequence_accepted,
        "lost_with_sequence_opportunity": real.full_accepted & ~real.sequence_accepted & real.sequence_eligible,
        "lost_support": real.full_accepted & ~real.sequence_eligible,
        "gained": ~real.full_accepted & real.sequence_accepted,
    }
    by = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction", "event_id"]
    out = []
    for group, mask in masks.items():
        for key, z in real.loc[mask].groupby(by, dropna=False):
            out.append(
                dict(
                    zip(by, key, strict=True),
                    group=group,
                    n_contributing_splits=z.split.nunique(),
                    n_contributing_split_repeats=len(z),
                    accepted_fraction=float(z.sequence_accepted.mean()),
                    signed_evaluation_z=float(z.signed_evaluation_z.median()),
                    signed_conditional_evaluation_z=float(z.signed_conditional_evaluation_z.median()),
                    evaluation_spikes=float(z.evaluation_n_eval_spikes.median()),
                    evaluation_active_units=float(z.n_evaluation_active_units.median()),
                    duration_s=float(z.duration_s.iloc[0]),
                    ripple_peak_z=float(z.ripple_peak_z.iloc[0]),
                )
            )
    return pd.DataFrame(out)


def summarize_content(events):
    by = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction", "group"]
    rows = []
    for key, z in events.groupby(by):
        valid = z.signed_evaluation_z.dropna()
        cond = z.signed_conditional_evaluation_z.dropna()
        rows.append(
            dict(
                zip(by, key, strict=True),
                n_events=len(z),
                n_events_with_evaluation_content=len(valid),
                mean_signed_evaluation_z=float(valid.mean()),
                median_signed_evaluation_z=float(valid.median()),
                mean_signed_conditional_evaluation_z=float(cond.mean()),
                median_signed_conditional_evaluation_z=float(cond.median()),
                n_conditional_positive_events=int((cond > 0).sum()),
                median_evaluation_spikes=float(z.evaluation_spikes.median()),
            )
        )
    return pd.DataFrame(rows)


def selected_content_bias(data):
    x = data.loc[data.arm == "real"]
    keys = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "split", "repeat"]
    rows = []
    for key, g in x.groupby(keys):
        full = g[g.fraction == 1].set_index("event_id")
        for fraction, half in g[g.fraction < 1].groupby("fraction"):
            h = half.set_index("event_id")
            if not full.index.equals(h.index):
                h = h.reindex(full.index)
            if len(full) != len(h) or h.sequence_accepted.isna().any():
                raise ValueError("unpaired dose events")
            q = full.evaluation_track2_probability
            if not np.allclose(q, h.evaluation_track2_probability, equal_nan=True):
                raise ValueError("evaluation readout changed under thinning")
            a = q[full.sequence_accepted].dropna().to_numpy()
            b = q[h.sequence_accepted].dropna().to_numpy()
            status = "estimable" if len(a) > 0 and 0 < len(b) <= len(a) else "insufficient_accepted_content_or_more_half_than_full"
            delta = float(b.mean() - a.mean()) if len(a) and len(b) else np.nan
            null = np.array([])
            if status == "estimable":
                rng = np.random.default_rng(stable_seed(20260918, *key, fraction, "random-deletion"))
                null = np.array([rng.choice(a, len(b), replace=False).mean() - a.mean() for _ in range(499)])
            rows.append(
                dict(
                    zip(keys, key, strict=True),
                    fraction=fraction,
                    n_full_accepted_content=len(a),
                    n_thinned_accepted_content=len(b),
                    n_gained=int((h.sequence_accepted & ~full.sequence_accepted).sum()),
                    track2_selection_shift=delta,
                    deletion_null_p_two_sided=(1 + int((np.abs(null) >= abs(delta)).sum())) / 500 if len(null) else np.nan,
                    deletion_null_p025=float(np.quantile(null, 0.025)) if len(null) else np.nan,
                    deletion_null_p975=float(np.quantile(null, 0.975)) if len(null) else np.nan,
                    status=status,
                )
            )
    return pd.DataFrame(rows)


def run(roots, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("new report directory required")
    data, sources = load_campaigns(roots)
    events = event_summaries(data)
    content = summarize_content(events)
    bias = selected_content_bias(data)
    bias["deletion_null_estimable"] = bias.status.eq("estimable")
    by = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction", "arm"]
    dose = (
        data.groupby(by)
        .agg(
            n_unique_events=("event_id", "nunique"),
            acceptance_fraction=("sequence_accepted", "mean"),
            opportunity_fraction=("sequence_eligible", "mean"),
            mean_inference_units=("n_inference_units", "mean"),
        )
        .reset_index()
    )
    bkeys = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction"]
    bias_summary = (
        bias.groupby(bkeys)
        .agg(
            estimable_selection_shifts=("track2_selection_shift", "count"),
            deletion_null_estimable_split_repeats=("deletion_null_estimable", "sum"),
            median_track2_selection_shift=("track2_selection_shift", "median"),
            mean_track2_selection_shift=("track2_selection_shift", "mean"),
            median_full_accepted_content=("n_full_accepted_content", "median"),
            median_thinned_accepted_content=("n_thinned_accepted_content", "median"),
        )
        .reset_index()
    )
    expected = {"RAT3_SESS2", "RAT5_SESS2"}
    present = set(data.session)
    gates = [
        {"gate": "complete_primary_cohort", "passed": expected <= present, "detail": "two strict RUN-pass sessions required; absent=" + ",".join(sorted(expected - present))},
        {"gate": "scientific_jobs_complete", "passed": True, "detail": f"{len(sources)} validated terminal manifests"},
        {"gate": "independent_fixed_evaluation_readout", "passed": True, "detail": "checked across all paired inference doses"},
        {"gate": "temporal_prediction_completed", "passed": False, "detail": "this report is sequenceless context content, not a future-prediction result"},
        {"gate": "dataset_wide_biological_claim", "passed": False, "detail": "no promotion from descriptive counts or two primary animals"},
    ]
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "event_summary": events,
        "dose_summary": dose,
        "independent_content_summary": content,
        "selected_content_bias_by_split": bias,
        "selected_content_bias_summary": bias_summary,
        "gate_summary": pd.DataFrame(gates),
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"two_track_{name}.csv", index=False)
    lines = [
        "# Experience-content coverage audit",
        "",
        f"Sessions scored: {', '.join(sorted(present))}.",
        "Primary cohort complete: " + str(expected <= present) + ".",
        "",
        "This is a non-rescoring diagnostic. Common RUN-unit maps, detector cells, windows and evaluation cells are fixed as inference coverage falls.",
        "Accepted events satisfy both weighted-correlation nulls at per-track p<0.025; opportunity requires five active inference cells and five nonempty bins.",
        "Independent sequenceless track support is not itself ordered replay, replay ground truth, or future predictive information.",
        "",
        "Event summaries collapse split/repeat contributions before the session content mean. Splits/repeats are not independent biological samples.",
        "Lost-support and lost-sequence-opportunity event groups can overlap because the same event may fail differently in different cell subsets.",
        "Track 2 is the second encountered context. Random deletion is a selection null, not a neural replay null.",
        "",
        "## POST ripple candidates at half inference coverage",
        "",
    ]
    post = content[
        (content.epoch == "POST")
        & (content.candidate_stratum == "ripple")
        & (content.fraction == 0.5)
        & content.group.isin(["lost", "lost_with_sequence_opportunity", "lost_support"])
    ]
    for r in post.to_dict("records"):
        lines.append(
            f"- {r['session']} / {r['group']}: {r['n_events']} unique events; {r['n_events_with_evaluation_content']} with evaluation spikes; mean signed evaluation z {r['mean_signed_evaluation_z']:.4f}; conditional-count z {r['mean_signed_conditional_evaluation_z']:.4f}."
        )
    if post.empty:
        lines.append("No newly rejected POST ripple event groups were available; no hidden-content claim can be made.")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "No dataset-wide conclusion is justified from one or two primary animals. Report insufficient event counts explicitly. A selective content bias requires more than a fall in acceptance or surviving context information. PRE/POST comparisons still require state, rate and drift controls. Future prediction remains outstanding.",
        ]
    )
    (output / "two_track_content_report.md").write_text("\n".join(lines) + "\n")
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "non_rescoring": True,
        "input_score_manifests": sources,
        "reporter_sha256": file_sha256(Path(__file__)),
        "reporter_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "unique_events": int(data[["session", "event_id"]].drop_duplicates().shape[0]),
        "input_rows": len(data),
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dirs", nargs="+", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    run(a.campaign_dirs, a.output_dir)
