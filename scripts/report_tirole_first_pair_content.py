"""Non-rescoring first-pair diagnostic with the existing time-block estimands."""

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
from scripts.prepare_tirole_first_pair_content_bank import COHORT
from scripts.report_tirole_content_coverage import load_campaigns
from scripts.report_tirole_content_uncertainty import BLOCK_SECONDS, block_weights, checked, composition_interval, mean_interval
from scripts.score_tirole_content_coverage import validate_bank


def paired_arrays(local, ids, column="sequence_accepted"):
    """Require all frozen 5x5 realizations, with identical event identities."""
    expected = pd.MultiIndex.from_product([range(5), range(5)], names=["split", "repeat"])
    arrays = []
    for fraction in [1.0, 0.5]:
        frame = local[local.fraction == fraction].pivot(index="event_id", columns=["split", "repeat"], values=column)
        if set(frame.index) != set(ids) or set(frame.columns) != set(expected):
            raise ValueError("incomplete event or realization coverage")
        frame = frame.reindex(index=ids, columns=expected)
        if column in {"sequence_accepted", "sequence_eligible"} and frame.isna().any().any():
            raise ValueError("missing selection decisions")
        arrays.append(frame.to_numpy())
    return arrays


def validate_source_sets(sources, report, identity, forecast):
    expected = {(x["manifest"], x["sha256"]) for x in sources}
    for items in [report["input_score_manifests"], identity["scorer_manifests"], forecast["source_score_manifests"]]:
        observed = {(x["manifest"], x["sha256"]) for x in items}
        if observed != expected or len(items) != len(expected):
            raise ValueError("diagnostic input score provenance mismatch")


def intervals(data, events, context, identity, forecast):
    rows = []
    real = data[data.arm == "real"]
    for epoch in ["PRE", "POST"]:
        for tier in ["ripple_z3", "ripple_z5", "mua_only"]:
            local = real[(real.epoch == epoch) & (real.candidate_stratum == ("mua_only" if tier == "mua_only" else "ripple"))]
            if tier == "ripple_z5":
                local = local[local.ripple_peak_z >= 5]
            ids = np.sort(local.event_id.unique())
            if not len(ids):
                continue
            full, half = paired_arrays(local, ids)
            values = {"full_acceptance": full.mean(axis=1), "half_acceptance": half.mean(axis=1), "half_minus_full_acceptance": half.mean(axis=1) - full.mean(axis=1)}
            for label, table in [("track_swap", context), ("rate_matched_identity", identity)]:
                d = table[(table.epoch == epoch) & (table.fraction == 0.5) & (table.group == "lost")].set_index("event_id").reindex(ids)
                values[label + "_lost_context_z"] = d.signed_evaluation_z.to_numpy()
                values[label + "_lost_conditional_context_z"] = d.signed_conditional_evaluation_z.to_numpy()
            for group in ["lost", "lost_with_sequence_opportunity"]:
                d = (
                    forecast[
                        (forecast.epoch == epoch)
                        & (forecast.fraction == 0.5)
                        & (forecast.horizon_ms == 40)
                        & (forecast.group == group)
                        & (forecast.contrast == "dynamic_minus_matched_own")
                    ]
                    .set_index("event_id")
                    .reindex(ids)
                )
                values[group + "_future_matched_advantage"] = d.delta_per_spike.to_numpy()
            for width in BLOCK_SECONDS:
                codes, weights = block_weights(events.loc[ids, "start_s"], width, stable_seed(20260918, "RAT2_SESS1", epoch, tier, width, "occupied-time-block-bootstrap"))
                base = {
                    "session": "RAT2_SESS1",
                    "animal": "RAT2",
                    "cohort_stratum": COHORT,
                    "epoch": epoch,
                    "candidate_tier": tier,
                    "block_s": width,
                    "n_candidate_events": len(ids),
                    "n_candidate_blocks": weights.shape[1],
                }
                for metric, vector in values.items():
                    rows.append({**base, "metric": metric, **mean_interval(vector, codes, weights)})
                for label, column in [("poisson", "evaluation_track2_probability"), ("conditional", "conditional_evaluation_track2_probability")]:
                    q, qhalf = paired_arrays(local, ids, column)
                    if not np.allclose(q, qhalf, equal_nan=True):
                        raise ValueError("held-out population readout changed under thinning")
                    for loss_only in [True, False]:
                        rows.append(
                            {
                                **base,
                                "metric": label + ("_selective_loss_shift" if loss_only else "_total_selection_shift"),
                                **composition_interval(q, full, half, codes, weights, loss_only),
                            }
                        )
    return pd.DataFrame(rows)


def run(source, output):
    if output.exists():
        raise ValueError("new report directory required")
    bank = source / "bank"
    bm, _, events, _, _, _ = validate_bank(bank)
    if bm["cohort_stratum"] != COHORT or bm["primary_cohort_promoted"] or bm["session"] != "RAT2_SESS1":
        raise ValueError("separate first-pair diagnostic required")
    rm = checked(source / "report", "report_manifest.json")
    im = checked(source / "identity", "manifest.json")
    fm = checked(source / "forecast", "manifest.json")
    if fm["status"] != "complete" or fm["git_dirty"] or fm["actual_rows"] != fm["expected_rows"] or fm["actual_rows"] <= 0:
        raise ValueError("complete future-prediction experiment required")
    if (
        fm["bank_manifest_sha256"] != file_sha256(bank / "manifest.json")
        or not fm["target_event_heldout_in_fitting"]
        or fm["target_heldout_spikes_in_filter"]
        or fm["future_spikes_in_filter"]
    ):
        raise ValueError("forecast bank mismatch or leakage")
    if im["status"] != "complete" or im["selection_or_thresholds_changed"] or im["bank_manifests"][0]["manifest_sha256"] != file_sha256(bank / "manifest.json"):
        raise ValueError("identity-control provenance mismatch")
    data, sources = load_campaigns([source / "campaign"])
    if set(data.session) != {"RAT2_SESS1"} or set(data.cohort_stratum) != {COHORT} or set(data.event_id) != set(events.event_id):
        raise ValueError("cohort or candidate coverage mismatch")
    for item in sources:
        score_meta = json.loads(Path(item["manifest"]).read_text())
        if score_meta["bank_manifest_sha256"] != file_sha256(bank / "manifest.json"):
            raise ValueError("scorer bank mismatch")
    validate_source_sets(sources, rm, im, fm)
    context = pd.read_csv(source / "report/two_track_event_summary.csv")
    identity = pd.read_csv(source / "identity/evaluation_identity_event_summary.csv")
    forecast = pd.read_csv(source / "forecast/forecast_by_event.csv")
    for frame in [context, identity, forecast]:
        if set(frame.session) != {"RAT2_SESS1"} or not set(frame.event_id) <= set(events.event_id):
            raise ValueError("foreign session or event in diagnostic readouts")
    table = intervals(data, events.set_index("event_id"), context, identity, forecast)
    focal = table[(table.epoch == "POST") & (table.candidate_tier == "ripple_z3") & (table.block_s == 60)]
    output.mkdir(parents=True)
    table.to_csv(output / "first_pair_time_block_uncertainty.csv", index=False)
    focal.to_csv(output / "first_pair_POST_ripple_diagnostic.csv", index=False)
    lines = [
        "# First-pair sampling-content diagnostic",
        "",
        "RAT2_SESS1 remains outside the frozen primary cohort. No session-specific re-exposure flag has been recovered.",
        "Content refers to the first versus second observed RUN, with later exposure samples excluded. PRE/POST are remote immobile rest, not sleep-stage truth.",
        "",
        f"All {len(events)} candidate windows have all 25 repeated split/subset realizations and both inference controls where required. Score rows: {len(data)}.",
        "Four sampling levels were scored. The prespecified diagnostic is full versus half sampling, POST ripple-positive windows.",
        "Content and prediction use event medians across repeated realizations; acceptance uses each event's acceptance fraction. Composition recomputes paired selected means using shared block weights, averaging estimable realizations. All 30/60/120 s and PRE/ripple-z5/MUA-only sensitivities are retained.",
        "Intervals are conditional within-session descriptions, not population-level confidence or multiplicity-corrected discoveries.",
        "",
        "## POST ripple-positive diagnostic",
        "",
        "| Metric | Estimate | 95% block interval | Informative events | Status |",
        "|---|---:|---|---:|---|",
    ]
    for row in focal.to_dict("records"):
        lines.append(f"| {row['metric']} | {row['point']:.6f} | [{row['ci025']:.6f}, {row['ci975']:.6f}] | {row['n_informative_events']} | {row['interval_status']} |")
    lines += [
        "",
        "## Boundaries",
        "",
        "Lower acceptance does not prove preferential loss of one experience. Surviving context support is not replay ground truth; temporal prediction is a separate test.",
        "Track posterior probability is not the fraction of events replaying that track. Report total selection changes separately from loss-only changes and disclose empty retained sets.",
        "No new biological confirmation gate or primary-cohort promotion is asserted. Compare this diagnostic with both primary rats, including unresolved or negative results.",
    ]
    (output / "first_pair_content_diagnostic.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "status": "complete",
        "non_rescoring": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "command_line": sys.argv,
        "working_directory": str(Path.cwd()),
        "reporter_sha256": file_sha256(__file__),
        "session": "RAT2_SESS1",
        "cohort_stratum": COHORT,
        "primary_cohort_promoted": False,
        "biological_confirmation": False,
        "input_manifests": {
            name: file_sha256(source / relative)
            for name, relative in [
                ("bank", "bank/manifest.json"),
                ("content", "report/report_manifest.json"),
                ("identity", "identity/manifest.json"),
                ("forecast", "forecast/manifest.json"),
            ]
        },
        "score_manifest_count": len(sources),
        "score_rows": len(data),
        "n_bootstraps": 2000,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.source_dir, a.output_dir)
