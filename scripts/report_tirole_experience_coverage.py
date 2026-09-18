"""Event-weighted independent content contrasts for RUN-only coverage interventions."""

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

ARMS = ("full", "track1_information", "track2_information", *[f"random_{r}_{side}" for r in range(5) for side in ("a", "b")])


def event_contributions(frame):
    if frame.empty:
        raise ValueError("zero selected candidates cannot pass completeness")
    expected = pd.MultiIndex.from_product([sorted(frame.event_id.unique()), range(5), ARMS], names=["event_id", "split", "arm"])
    actual = pd.MultiIndex.from_frame(frame[expected.names])
    if len(actual) != len(expected) or actual.duplicated().any() or len(expected.difference(actual)):
        raise ValueError("missing/duplicate fixed-arm scores")
    readouts = ["evaluation_conditional_track2_probability", "evaluation_conditional_track2_identity_z"]
    if (frame.groupby(["event_id", "split"])[readouts].nunique(dropna=False) > 1).any().any():
        raise ValueError("evaluation readout changes with selected inference arm")
    half = frame[frame.arm != "full"]
    if half.groupby(["event_id", "split"]).n_inference_units.nunique().gt(1).any():
        raise ValueError("unequal half-cell counts")
    x = frame[["session", "animal", "event_id", "epoch", "primary_ripple_candidate", "arm", "start_s"]].copy()
    accept = frame.sequence_accepted.to_numpy(bool)
    x["acceptance"] = accept
    x["eligible"] = frame.sequence_eligible.to_numpy(bool)
    x["label2"] = accept & frame.inferred_track.eq(2)
    for prefix, name in zip(["q", "z"], readouts, strict=True):
        value = frame[name].to_numpy()
        available = accept & np.isfinite(value)
        x[prefix + "_num"] = np.where(available, value, 0.0)
        x[prefix + "_den"] = available
    for prefix, name in [("duration", "duration_s"), ("evalspikes", "n_evaluation_spikes"), ("infspikes", "n_inference_spikes")]:
        x[prefix + "_num"] = np.where(accept, frame[name], 0.0)
    return x.groupby(["session", "animal", "event_id", "epoch", "primary_ripple_candidate", "arm", "start_s"], dropna=False).mean(numeric_only=True).reset_index()


def statistics(group, weights):
    ids = sorted(group.event_id.unique())
    weights = np.atleast_2d(np.asarray(weights, float))
    if weights.shape[1] != len(ids) or (weights < 0).any() or not (weights.sum(axis=1) > 0).all():
        raise ValueError("positive event/block weights required")
    matrices = {}
    for col in ["acceptance", "eligible", "label2", "q_num", "q_den", "z_num", "z_den", "duration_num", "evalspikes_num", "infspikes_num"]:
        a = group.pivot(index="event_id", columns="arm", values=col).reindex(index=ids, columns=ARMS).to_numpy()
        if not np.isfinite(a).all():
            raise ValueError("incomplete event contribution matrix")
        matrices[col] = weights @ a

    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    return {
        "acceptance_fraction": divide(matrices["acceptance"], weights.sum(axis=1)[:, None]),
        "eligible_fraction": divide(matrices["eligible"], weights.sum(axis=1)[:, None]),
        "selected_event_equivalents": matrices["acceptance"],
        "conditional_track2_score_mean": divide(matrices["q_num"], matrices["q_den"]),
        "conditional_track2_identity_z_mean": divide(matrices["z_num"], matrices["z_den"]),
        "classifier_track2_fraction": divide(matrices["label2"], matrices["acceptance"]),
        "mean_duration_s": divide(matrices["duration_num"], matrices["acceptance"]),
        "mean_evaluation_spikes": divide(matrices["evalspikes_num"], matrices["acceptance"]),
        "mean_inference_spikes": divide(matrices["infspikes_num"], matrices["acceptance"]),
    }


def bootstrap_weights(group, session, epoch, ripple, n_boot=2000):
    events = group[["event_id", "start_s"]].drop_duplicates().sort_values("event_id")
    if len(events) != group.event_id.nunique():
        raise ValueError("event time changes between arms")
    blocks = np.floor(events.start_s.to_numpy() / 60).astype(np.int64)
    unique, inverse = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(stable_seed(20260918, session, epoch, bool(ripple), "coverage-intervention-timeblock-bootstrap"))
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=n_boot)
    return draws[:, inverse], len(unique)


def run(source, output):
    if output.exists():
        raise ValueError("new immutable coverage report required")
    m = json.loads((source / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or not m["subset_selection_RUN_only"] or not m["evaluation_readout_reused_unchanged"]:
        raise ValueError("invalid intervention provenance")
    for name, digest in m["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed intervention products")
    raw = pd.read_csv(source / "event_coverage_scores.csv")
    if sorted(raw.event_id.unique()) != sorted(m["selected_event_ids"]) or len(raw) != m["n_score_rows"] or set(raw.session) != {m["session"]}:
        raise ValueError("score rows differ from frozen candidate manifest")
    events = event_contributions(raw)
    subsets = pd.read_csv(source / "RUN_only_subsets.csv")
    information = subsets.pivot(index="split", columns="arm", values="mean_track1_minus_track2_information")
    manipulation = bool((information.track1_information - information.track2_information > 0).all())
    arm_rows, contrasts = [], []
    for (session, animal, epoch, ripple), g in events.groupby(["session", "animal", "epoch", "primary_ripple_candidate"]):
        n = g.event_id.nunique()
        point = statistics(g, np.ones(n))
        weights, n_blocks = bootstrap_weights(g, session, epoch, ripple)
        boot = statistics(g, weights)
        meta = {"session": session, "animal": animal, "epoch": epoch, "ripple_positive": ripple, "n_candidate_events": n, "n_time_blocks": n_blocks}
        for j, arm in enumerate(ARMS):
            a = g[g.arm == arm]
            arm_rows.append(
                {
                    **meta,
                    "arm": arm,
                    **{k: v[0, j] for k, v in point.items()},
                    "unique_selected_events": int(a.acceptance.gt(0).sum()),
                    "unique_q_supported_events": int(a.q_den.gt(0).sum()),
                    "unique_z_supported_events": int(a.z_den.gt(0).sum()),
                }
            )
        for label, first, second in [("targeted", "track1_information", "track2_information"), *[(f"random_{r}", f"random_{r}_a", f"random_{r}_b") for r in range(5)]]:
            i, j = ARMS.index(first), ARMS.index(second)
            row = {**meta, "contrast": label, "first_arm": first, "second_arm": second}
            for metric in ["conditional_track2_score_mean", "conditional_track2_identity_z_mean", "acceptance_fraction", "classifier_track2_fraction"]:
                value = float(point[metric][0, j] - point[metric][0, i])
                draws = boot[metric][:, j] - boot[metric][:, i]
                good = draws[np.isfinite(draws)]
                lo, hi = np.quantile(good, [0.025, 0.975]) if len(good) >= 1900 and n_blocks >= 2 else [np.nan, np.nan]
                row.update({metric + "_delta": value, metric + "_ci025": lo, metric + "_ci975": hi, metric + "_finite_bootstrap_fraction": len(good) / 2000})
            contrasts.append(row)
    arms = pd.DataFrame(arm_rows)
    delta = pd.DataFrame(contrasts)
    primary = delta[(delta.epoch == "POST") & delta.ripple_positive & delta.contrast.eq("targeted")]
    support = arms[(arms.epoch == "POST") & arms.ripple_positive & arms.arm.isin(["track1_information", "track2_information"])]
    gates = {
        "complete_paired_scoring": True,
        "unchanged_evaluation_readout": True,
        "equal_half_cell_counts": True,
        "positive_RUN_selectivity_contrast_each_split": manipulation,
        "five_supported_events_each_targeted_arm": len(support) == 2 and bool((support.unique_q_supported_events >= 5).all() and (support.unique_z_supported_events >= 5).all()),
    }
    for label, metric in [("primary_content_contrast_positive", "conditional_track2_score_mean"), ("identity_control_contrast_positive", "conditional_track2_identity_z_mean")]:
        gates[label] = len(primary) == 1 and bool((primary[metric + "_delta"] > 0).all() and (primary[metric + "_ci025"] > 0).all())
    gates["session_mechanistic_readout"] = all(gates.values())
    output.mkdir(parents=True)
    events.to_csv(output / "event_contributions.csv", index=False)
    arms.to_csv(output / "by_stratum_arm_summary.csv", index=False)
    delta.to_csv(output / "contrast_summary.csv", index=False)
    pd.DataFrame([{"gate": k, "passed": v} for k, v in gates.items()]).to_csv(output / "gate_summary.csv", index=False)
    note = [
        "# RUN-only experience-coverage intervention",
        "",
        "Real fixed candidates; deliberately unequal RUN-defined sampling, not natural random-sampling replication.",
        "Evaluation posterior scores are not true experience proportions. No HMM or trajectory prior was introduced.",
        "Uncertainty resamples paired 60-second blocks with every event/arm/split kept together.",
        "",
        f"Session mechanistic readout gate: {gates['session_mechanistic_readout']}.",
        "",
    ]
    if len(primary) == 1:
        row = primary.iloc[0]
        for name in ["conditional_track2_score_mean", "conditional_track2_identity_z_mean"]:
            note.append(f"{name}: targeted second-minus-first {row[name + '_delta']:.6f} [{row[name + '_ci025']:.6f}, {row[name + '_ci975']:.6f}].")
    note += ["", "An association under this designed coverage intervention does not calibrate real experience fractions or establish how natural recording coverage was biased."]
    (output / "coverage_intervention.md").write_text("\n".join(note) + "\n")
    manifest = {
        "status": "complete",
        "session": m["session"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "command_line": sys.argv,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "non_rescoring": True,
        "source_manifest": str((source / "manifest.json").resolve()),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "bootstrap_block_s": 60,
        "bootstrap_origin": "absolute_clock_zero",
        "n_bootstrap": 2000,
        "real_experience_fractions_corrected": False,
        "natural_sampling_bias_confirmed": False,
        "session_mechanistic_readout_passed": gates["session_mechanistic_readout"],
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.output_dir)
