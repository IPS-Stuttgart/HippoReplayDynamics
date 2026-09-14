#!/usr/bin/env python3
"""Non-rescoring falsification report for cell-predictive temporal context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_predictive_context_content import METHODS, PRIMARY
from scripts.report_temporal_endpoint_content import gates, plot_summary, summaries


def diagnostic_summary(frame, prediction_suffix="_median_predictive_delta"):
    rows = []
    keys = ["dataset", "animal", "session", "split", "source"]
    for identity, local in frame.groupby(keys):
        baseline = local.loc[local.method.eq("independent")].set_index("event_index").sort_index()
        context = local.loc[local.method.eq("unconditional_context")].set_index("event_index").sort_index()
        if not baseline.index.equals(context.index) or baseline.empty or baseline.index.duplicated().any():
            raise ValueError("unmatched diagnostic endpoints")
        for side in ("a", "b"):
            eligible = (baseline[side + "_spikes"] >= 3) & (baseline[side + "_active"] >= 2)
            prediction = baseline[side + prediction_suffix]
            use = baseline[side + "_use_context"]
            if not np.isfinite(prediction).all() or use.isna().any() or not use.isin([True, False]).all():
                raise ValueError("invalid diagnostic values")
            gain = baseline[side + "_error"] - context[side + "_error"]
            scores, values = prediction.loc[eligible], gain.loc[eligible]
            finite = values.notna().all() and np.isfinite(values).all()
            labels = values > 1e-8
            available = len(values) > 0 and finite and labels.nunique() == 2
            auc = roc_auc_score(labels, scores) if available else np.nan
            rho = spearmanr(scores, values).statistic if finite and len(values) > 2 and scores.nunique() > 1 and values.nunique() > 1 else np.nan
            selected = gain.loc[use.astype(bool)]
            rows.append(
                dict(zip(keys, identity, strict=True))
                | dict(
                    side=side,
                    events=len(baseline),
                    eligible_events=int(eligible.sum()),
                    context_events=int(use.sum()),
                    eligible_fraction=float(eligible.mean()),
                    context_fraction=float(use.mean()),
                    gain_auroc=auc,
                    gain_spearman=rho,
                    auroc_available=bool(available),
                    selected_mean_truth_gain=float(selected.mean()) if len(selected) else np.nan,
                    selected_harm_fraction=float((selected < -1e-8).mean()) if len(selected) and selected.notna().all() else np.nan,
                )
            )
    sessions = pd.DataFrame(rows)
    animals = []
    for keys, group in sessions.groupby(["dataset", "animal", "split", "source", "side"]):
        animals.append(
            dict(zip(["dataset", "animal", "split", "source", "side"], keys, strict=True))
            | dict(
                sessions=len(group),
                events=int(group.events.sum()),
                eligible_events=int(group.eligible_events.sum()),
                context_events=int(group.context_events.sum()),
                eligible_fraction=group.eligible_fraction.mean(),
                context_fraction=group.context_fraction.mean(),
                gain_auroc=group.gain_auroc.mean() if group.auroc_available.all() else np.nan,
                gain_spearman=group.gain_spearman.mean() if group.gain_spearman.notna().all() else np.nan,
            )
        )
    return sessions, pd.DataFrame(animals)


def combined_gates(summary, frame, diagnostics, audited, primary=PRIMARY):
    base = gates(summary, frame, audited, primary=primary)
    rows = base.loc[base.gate.ne("advance_external_validation")].to_dict("records")
    for source in ("run_q4", "sim_late_jump"):
        for side in ("a", "b"):
            selected = diagnostics.loc[(diagnostics.split == 0) & (diagnostics.source == source) & (diagnostics.side == side)]
            valid = (
                len(selected) == 4
                and selected.animal.nunique() == 4
                and np.isfinite(selected.gain_auroc).all()
                and selected.gain_auroc.mean() >= 0.60
                and (selected.gain_auroc > 0.5).sum() >= 3
            )
            rows.append(dict(gate=f"{source}_{side}_predicts_known_gain", passed=bool(valid)))
    rows.append(dict(gate="advance_external_validation", passed=all(r["passed"] for r in rows)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    inputs = dict(
        producer=args.measurement_dir / "manifest.json",
        audit=args.audit_dir / "independent_audit.json",
        reporter=Path(__file__),
        common_reporter=ROOT / "scripts/report_temporal_endpoint_content.py",
        protocol=ROOT / "docs/predictive_context_content_protocol.md",
    )
    source, audit = (json.loads(inputs[k].read_text()) for k in ("producer", "audit"))
    if source["status"] != "complete" or audit["status"] != "passed" or audit["input_file_sha256"]["producer"] != file_sha256(inputs["producer"]):
        raise ValueError("completed matching independent audit required")
    frames = []
    for row in source["results"]:
        path = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
        checked = [r for r in audit["results"] if r["animal"] == row["animal"] and r["session"] == row["session"]]
        if len(checked) != 1 or checked[0]["readout_sha256"] != file_sha256(path):
            raise ValueError("readout not independently verified")
        inputs[row["session"]] = path
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    frame = pd.concat(frames, ignore_index=True)
    sessions, animals, table = summaries(frame, methods=METHODS)
    diag_sessions, diag_animals = diagnostic_summary(frame)
    checks = combined_gates(table, frame, diag_animals, audited=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    for name, value in (
        ("summary", table),
        ("by_session", sessions),
        ("by_animal", animals),
        ("diagnostic_by_session", diag_sessions),
        ("diagnostic_by_animal", diag_animals),
        ("gate_summary", checks),
    ):
        value.to_csv(args.output_dir / f"{name}.csv", index=False)
    cohort = "PF development" if frame.dataset.iloc[0] == "pfeiffer_foster" else f"{frame.dataset.iloc[0]} independent evaluation"
    plot_summary(animals, table, args.output_dir, primary_method=PRIMARY, primary_label="Own-cell predictive context choice", cohort_label=cohort)
    failed = checks.loc[~checks.passed, "gate"].tolist()
    lines = [
        "# Cell-predictive temporal-context diagnostic",
        "",
        cohort + "; fixed endpoints; no cross-population conditioning.",
        "Primary choice: median three-fold own-cell predictive gain>1e-10, at least3 endpoint spikes and2 active cells.",
        "No refit of model parameters, no event removal. Truth safeguards and external validation remain mandatory.",
        "",
        "## Primary split: equal-animal means",
        "",
        "| Source | Metric | Independent | Predictive context | Reduction |",
        "|---|---|---:|---:|---:|",
    ]
    selected = table.loc[(table.split == 0) & (table.method == PRIMARY)]
    for row in selected.itertuples(index=False):
        if row.metric in ("separation_cm", "regional_tv", "a_error", "b_error"):
            lines.append(f"| {row.source} | {row.metric} | {row.baseline:.5f} | {row.value:.5f} | {row.reduction:+.5f} |")
    lines += [
        "",
        "## Frozen decision",
        "",
        "Advancement: " + ("FAIL" if failed else "PASS"),
        "",
        *["- " + k for k in failed],
        "",
        "Undefined diagnostic AUROCs fail. Predictive-fold scores are not calibrated p-values or evidence of known replay location.",
        "The primary must improve real agreement AND preserve known-path accuracy, including jumps; better agreement alone is insufficient.",
        "Consult cohort and gates before any independent-validation claim. The original targeted Home-content remedy is not established by this benchmark alone.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if any(file_sha256(v) != provenance["input_file_sha256"][k] for k, v in inputs.items()):
        raise ValueError("report inputs changed")
    provenance.update(
        status="complete",
        non_rescoring=True,
        primary_advanced=not failed,
        primary_method=PRIMARY,
        scientific_goal_achieved=False,
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
