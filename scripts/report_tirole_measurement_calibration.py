"""Summarize frozen count-anchored calibration, without fitting or rescoring."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256


def check_rows(content, temporal, anchors):
    n = len(anchors)
    if not n or anchors.event_id.duplicated().any():
        raise ValueError("nonempty unique anchors required")
    ck = ["anchor_event", "split", "truth_track", "generator", "fraction"]
    tk = ["anchor_event", "split", "repeat", "generator", "horizon_ms"]
    if len(content) != n * 40 or content.duplicated(ck).any() or len(temporal) != n * 150 or temporal.duplicated(tk).any():
        raise ValueError("incomplete or duplicate calibration")
    for x in (content, temporal):
        if set(x.anchor_event) != set(anchors.event_id):
            raise ValueError("anchor identity mismatch")
        if not x.split.isin(range(5)).all():
            raise ValueError("unknown split")
        expected = anchors.set_index("event_id").loc[x.anchor_event]
        if not np.array_equal(x.epoch, expected.epoch) or not np.array_equal(x.anchor_ripple_positive, expected.primary_ripple_candidate):
            raise ValueError("changed anchor stratum")
    if set(content.truth_track) != {1, 2} or set(content.fraction) != {0.5, 1.0} or set(content.generator) != {"ordered", "whole_bin_shuffled"}:
        raise ValueError("unknown content conditions")
    if set(temporal.repeat) != set(range(5)) or set(temporal.horizon_ms) != {20, 40, 80} or set(temporal.generator) != {"dynamic", "matched_null"}:
        raise ValueError("unknown temporal conditions")
    for _, x in content.groupby(["anchor_event", "split", "truth_track"]):
        for name in ("evaluation_true_signed_log_odds", "conditional_true_signed_log_odds", "evaluation_track2_probability"):
            if x[name].nunique(dropna=False) != 1:
                raise ValueError("sequenceless readout changed with inference coverage/order")


def track_summary(x):
    keys = ["session", "animal", "cohort_stratum", "epoch", "anchor_ripple_positive", "generator", "fraction", "truth_track"]
    rows = []
    for key, g in x.groupby(keys):
        eligible = g[g.sequence_eligible]
        accepted = g[g.sequence_accepted]
        # Balanced realizations give every anchor equal weight; do not treat draws as animals.
        context = g.groupby("anchor_event")[["evaluation_true_signed_log_odds", "conditional_true_signed_log_odds", "conditional_true_signed_z"]].median()
        c = context.conditional_true_signed_log_odds.dropna()
        rows.append(
            dict(
                zip(keys, key, strict=True),
                n_anchors=g.anchor_event.nunique(),
                n_realizations=len(g),
                n_eligible=int(g.sequence_eligible.sum()),
                n_accepted=int(g.sequence_accepted.sum()),
                acceptance_fraction=float(g.sequence_accepted.mean()),
                opportunity_fraction=float(g.sequence_eligible.mean()),
                acceptance_given_opportunity=float(eligible.sequence_accepted.mean()),
                correct_track_given_opportunity=float(eligible.correct_inferred_track.mean()),
                correct_track_given_acceptance=float(accepted.correct_inferred_track.mean()),
                n_anchors_with_evaluation_content=len(c),
                evaluation_conditional_correct_fraction=float((c > 0).mean()) if len(c) else np.nan,
                evaluation_conditional_mean_signed_z=float(context.conditional_true_signed_z.mean()),
                median_inference_spikes=float(g.groupby("anchor_event").n_inference_spikes.median().median()),
            )
        )
    return pd.DataFrame(rows)


def temporal_summary(x):
    x = x.copy()
    x["dynamic_minus_null_per_spike"] = (x.score_dynamic - x.score_matched_own) / x.n_heldout_target_spikes.replace(0, np.nan)
    keys = ["session", "animal", "cohort_stratum", "epoch", "anchor_ripple_positive", "generator", "horizon_ms", "anchor_event"]
    event = (
        x.groupby(keys)
        .agg(
            event_median_gain=("dynamic_minus_null_per_spike", "median"),
            event_mean_gain=("dynamic_minus_null_per_spike", "mean"),
            n_finite_realizations=("dynamic_minus_null_per_spike", "count"),
        )
        .reset_index()
    )
    summary = (
        event.groupby(keys[:-1])
        .agg(
            n_anchors=("anchor_event", "size"),
            n_anchors_with_target_spikes=("event_median_gain", "count"),
            mean_event_median_gain=("event_median_gain", "mean"),
            median_event_median_gain=("event_median_gain", "median"),
            mean_event_mean_gain=("event_mean_gain", "mean"),
        )
        .reset_index()
    )
    pairkeys = [k for k in keys if k != "generator"]
    paired = event.pivot(index=pairkeys, columns="generator", values="event_median_gain").reset_index()
    paired["known_generator_separation"] = paired.dynamic - paired.matched_null
    separation = (
        paired.groupby(pairkeys[:-1])
        .agg(
            n_paired_anchors=("known_generator_separation", "count"),
            mean_known_generator_separation=("known_generator_separation", "mean"),
            median_known_generator_separation=("known_generator_separation", "median"),
        )
        .reset_index()
    )
    return event, summary, separation


def run(source, output):
    if output.exists():
        raise ValueError("new immutable output directory required")
    cs, ts, manifests = [], [], {}
    for p in sorted(source.glob("*/manifest.json")):
        m = json.loads(p.read_text())
        if m["status"] != "complete" or m["git_dirty"] or m["latent_paths_or_track_labels_given_to_decoder"]:
            raise ValueError("invalid calibration manifest")
        for name, digest in m["output_sha256"].items():
            if file_sha256(p.parent / name) != digest:
                raise ValueError("changed calibration input")
        c = pd.read_csv(p.parent / "known_track_calibration.csv")
        t = pd.read_csv(p.parent / "known_temporal_calibration.csv")
        check_rows(c, t, pd.read_csv(p.parent / "count_anchors.csv"))
        cs.append(c)
        ts.append(t)
        manifests[str(p.resolve())] = file_sha256(p)
    if not cs:
        raise ValueError("no completed calibrations")
    c, t = pd.concat(cs, ignore_index=True), pd.concat(ts, ignore_index=True)
    if c.duplicated(["session", "anchor_event", "split", "truth_track", "generator", "fraction"]).any():
        raise ValueError("duplicate session calibration")
    event, temporal, separation = temporal_summary(t)
    track = track_summary(c)
    output.mkdir(parents=True)
    for name, frame in {"known_track_summary": track, "known_temporal_by_anchor": event, "known_temporal_summary": temporal, "known_generator_separation": separation}.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    lines = [
        "# Count-anchored measurement calibration",
        "",
        "Non-rescoring summary of frozen known-map and known-generator simulations.",
        "",
        (
            "Known RUN maps/generating parameters are supplied to the decoder; true tracks and hidden paths are not. "
            "Actual candidate durations and population counts anchor the simulations, not their biological content. "
            "The track simulations condition on population count; they are not a validation of an exact Poisson generative model."
        ),
        "",
        (
            "Acceptance denominators include opportunity failures. Track accuracy is separately conditional on opportunity or acceptance; "
            "unclassified events are not labeled wrong-track. Evaluation content is sequenceless and invariant under time-bin shuffling. "
            "Simulated PRE denotes a PRE count anchor, not observed PRE experience replay."
        ),
        "",
        "## Primary POST ripple count anchors",
        "",
    ]
    primary = track[(track.cohort_stratum == "strict_RUN_pass") & (track.epoch == "POST") & track.anchor_ripple_positive]
    for key, g in primary.groupby(["session", "generator", "fraction"]):
        lines.append(
            f"- {key[0]}, {key[1]}, coverage {key[2]:g}: {int(g.n_accepted.sum())}/{int(g.n_realizations.sum())} accepted; "
            f"{int(g.n_eligible.sum())}/{int(g.n_realizations.sum())} opportunity-valid across equal track simulations."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        (
            "Twenty anchors per epoch are shared across simulations, cell splits and tracks; "
            "realizations are not independent biological samples. Future scores use per-anchor medians first. Means of those medians "
            "need not have zero or negative expectation under the matched null: the median is nonlinear and Monte Carlo samples are small. "
            "Both event-mean and event-median summaries are supplied. No calibrated significance threshold is inferred from a positive mean."
        ),
        "",
        (
            "Known-generator discrimination here is a best-case measurement check, not a replay label or evidence of biological selectivity. "
            "Low recovery constrains sensitivity; it does not establish that real rejected events are sequences."
        ),
    ]
    (output / "calibration_summary.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "non_rescoring": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_manifests": manifests,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.calibration_dir, a.output_dir)
