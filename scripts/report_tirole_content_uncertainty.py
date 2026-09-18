"""Non-rescoring occupied-time-block uncertainty and frozen event-definition sensitivities."""

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
from scripts.report_tirole_content_coverage import load_campaigns

N_BOOTSTRAPS = 2000
BLOCK_SECONDS = (30, 60, 120)


def block_weights(times, width, seed, n=N_BOOTSTRAPS):
    t = np.asarray(times, float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or width <= 0 or n < 2:
        raise ValueError("finite times and positive block design required")
    labels = np.floor(t / width).astype(np.int64)
    _, codes = np.unique(labels, return_inverse=True)
    k = int(codes.max() + 1)
    weights = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n)
    return codes, weights


def summarize_bootstrap(point, draws, n_informative_blocks, n_events):
    finite = np.asarray(draws)[np.isfinite(draws)]
    ok = n_informative_blocks >= 2 and len(finite) >= 0.95 * len(draws)
    return {
        "point": float(point),
        "n_informative_events": int(n_events),
        "n_informative_blocks": int(n_informative_blocks),
        "finite_bootstrap_fraction": len(finite) / len(draws),
        "ci025": float(np.quantile(finite, 0.025)) if ok else np.nan,
        "ci975": float(np.quantile(finite, 0.975)) if ok else np.nan,
        "interval_status": "conditional_descriptive" if ok else "insufficient_blocks_or_nonempty_draws",
    }


def mean_interval(values, codes, weights):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    k = weights.shape[1]
    sums = np.bincount(codes, weights=np.where(valid, values, 0), minlength=k)
    n = np.bincount(codes, weights=valid.astype(float), minlength=k)
    den = weights @ n
    draws = np.divide(weights @ sums, den, out=np.full(len(weights), np.nan), where=den > 0)
    point = values[valid].mean() if valid.any() else np.nan
    return summarize_bootstrap(point, draws, int((n > 0).sum()), int(valid.sum()))


def finite_row_mean(x):
    count = np.isfinite(x).sum(axis=1)
    return np.divide(np.nansum(x, axis=1), count, out=np.full(len(x), np.nan), where=count > 0)


def composition_interval(q, full, thin, codes, weights, loss_only):
    q = np.asarray(q, float)
    full = np.asarray(full, bool)
    thin = np.asarray(thin, bool)
    if q.shape != full.shape or q.shape != thin.shape or q.ndim != 2 or len(q) != len(codes):
        raise ValueError("paired event-by-realization arrays required")
    support = np.isfinite(q)
    a = full & support
    b = (full & thin if loss_only else thin) & support
    k = weights.shape[1]

    def totals(mask):
        number = np.zeros((k, q.shape[1]))
        denom = np.zeros_like(number)
        np.add.at(number, codes, np.where(mask, q, 0))
        np.add.at(denom, codes, mask.astype(float))
        return number, denom

    an, ad = totals(a)
    bn, bd = totals(b)

    def difference(w):
        av = w @ ad
        bv = w @ bd
        aa = np.divide(w @ an, av, out=np.full_like(av, np.nan), where=av > 0)
        bb = np.divide(w @ bn, bv, out=np.full_like(bv, np.nan), where=bv > 0)
        return finite_row_mean(bb - aa)

    point = difference(np.ones((1, k)))[0]
    draws = difference(weights)
    result = summarize_bootstrap(point, draws, int(((ad.sum(axis=1) > 0) | (bd.sum(axis=1) > 0)).sum()), int((a.any(axis=1) | b.any(axis=1)).sum()))
    result["baseline_estimable_realizations"] = int(((ad.sum(axis=0) > 0) & (bd.sum(axis=0) > 0)).sum())
    return result


def checked(folder, name):
    manifest = json.loads((folder / name).read_text())
    for path, digest in manifest["output_sha256"].items():
        if file_sha256(folder / path) != digest:
            raise ValueError("changed report input")
    return manifest


def run(source, output):
    if output.exists():
        raise ValueError("new uncertainty report required")
    report = source / "report_six_sessions_v1"
    identity = source / "evaluation_identity_control_v1"
    forecast = source / "future_forecasts_v1"
    rm = checked(report, "report_manifest.json")
    im = checked(identity, "manifest.json")
    campaign = [source / n for n in ["campaign_RAT3_SESS2_v1", "campaign_diagnostic_stage1_v1", "campaign_remaining_stage2_v1"]]
    scores, sources = load_campaigns(campaign)
    raw = scores[scores.arm == "real"].copy()
    rows = []
    inputs = {"content_report": file_sha256(report / "report_manifest.json"), "identity_report": file_sha256(identity / "manifest.json")}
    context = pd.read_csv(report / "two_track_event_summary.csv")
    ident = pd.read_csv(identity / "evaluation_identity_event_summary.csv")
    for session in sorted(raw.session.unique()):
        fm = checked(forecast / session, "manifest.json")
        inputs[session + "_forecast"] = file_sha256(forecast / session / "manifest.json")
        bank = source / ("candidate_banks_v2" if session == "RAT3_SESS2" else "candidate_banks_v3") / session
        bm = json.loads((bank / "manifest.json").read_text())
        if fm["bank_manifest_sha256"] != file_sha256(bank / "manifest.json"):
            raise ValueError("forecast and time clock bank differ")
        if file_sha256(bank / "candidate_events.csv") != bm["outputs_sha256"]["candidate_events.csv"]:
            raise ValueError("candidate clock changed")
        events = pd.read_csv(bank / "candidate_events.csv").set_index("event_id")
        pred = pd.read_csv(forecast / session / "forecast_by_event.csv")
        for epoch in ["PRE", "POST"]:
            for tier in ["ripple_z3", "ripple_z5", "mua_only"]:
                local = raw[(raw.session == session) & (raw.epoch == epoch) & (raw.candidate_stratum == ("mua_only" if tier == "mua_only" else "ripple"))]
                if tier == "ripple_z5":
                    local = local[local.ripple_peak_z >= 5]
                ids = np.sort(local.event_id.unique())
                if not len(ids):
                    continue
                full = local[local.fraction == 1].pivot(index="event_id", columns=["split", "repeat"], values="sequence_accepted").reindex(ids)
                thin = local[local.fraction == 0.5].pivot(index="event_id", columns=["split", "repeat"], values="sequence_accepted").reindex(ids)
                if full.shape[1] != 25 or thin.isna().any().any() or full.isna().any().any():
                    raise ValueError("incomplete paired cell realizations")
                values = {}
                values["half_minus_full_acceptance"] = thin.mean(axis=1).to_numpy() - full.mean(axis=1).to_numpy()
                for label, table in [("track_swap", context), ("rate_matched_identity", ident)]:
                    d = table[(table.session == session) & (table.epoch == epoch) & (table.fraction == 0.5) & (table.group == "lost")].set_index("event_id").reindex(ids)
                    values[label + "_lost_context_z"] = d.signed_evaluation_z.to_numpy()
                    values[label + "_lost_conditional_context_z"] = d.signed_conditional_evaluation_z.to_numpy()
                for group in ["lost", "lost_with_sequence_opportunity"]:
                    d = (
                        pred[
                            (pred.session == session)
                            & (pred.epoch == epoch)
                            & (pred.fraction == 0.5)
                            & (pred.horizon_ms == 40)
                            & (pred.group == group)
                            & (pred.contrast == "dynamic_minus_matched_own")
                        ]
                        .set_index("event_id")
                        .reindex(ids)
                    )
                    values[group + "_future_matched_advantage"] = d.delta_per_spike.to_numpy()
                for width in BLOCK_SECONDS:
                    codes, weights = block_weights(events.loc[ids, "start_s"], width, stable_seed(20260918, session, epoch, tier, width, "occupied-time-block-bootstrap"))
                    base = {
                        "animal": bm["animal"],
                        "session": session,
                        "cohort_stratum": bm["cohort_stratum"],
                        "epoch": epoch,
                        "candidate_tier": tier,
                        "block_s": width,
                        "n_candidate_events": len(ids),
                        "n_candidate_blocks": weights.shape[1],
                    }
                    for metric, v in values.items():
                        rows.append({**base, "metric": metric, **mean_interval(v, codes, weights)})
                    for readout, column in [("poisson", "evaluation_track2_probability"), ("conditional", "conditional_evaluation_track2_probability")]:
                        q = local[local.fraction == 1].pivot(index="event_id", columns=["split", "repeat"], values=column).reindex(ids).to_numpy()
                        for loss in [True, False]:
                            rows.append(
                                {
                                    **base,
                                    "metric": readout + ("_selective_loss_shift" if loss else "_total_selection_shift"),
                                    **composition_interval(q, full.to_numpy(), thin.to_numpy(), codes, weights, loss),
                                }
                            )
    results = pd.DataFrame(rows)
    output.mkdir(parents=True)
    results.to_csv(output / "time_block_uncertainty.csv", index=False)
    primary = results[(results.cohort_stratum == "strict_RUN_pass") & (results.epoch == "POST") & (results.candidate_tier == "ripple_z3") & (results.block_s == 60)]
    primary.to_csv(output / "primary_time_block_uncertainty.csv", index=False)
    note = """# Conditional uncertainty and predeclared event-definition sensitivities

No scoring, thresholds, units, event labels or selection rules are changed.
Resample occupied candidate-time blocks jointly for every repeated cell split.
Primary block length is 60 s; 30/120 s assess dependence on that choice. PRE and
POST are separate, as are ripple z>=3, the planned z>=5 sensitivity, and MUA-only.
Event content/prediction medians collapse repeated realizations before resampling.
For composition, the same block weights apply to every cell realization; selected
means are recomputed. Sparse realizations can become empty and remain missing.
Intervals require two informative blocks and >=95% finite bootstrap draws.

These are within-session intervals conditional on fitted maps, frozen cell
partitions and observed candidates, not biological-population confidence bounds.
Only two primary animals passed the frozen RUN gate. No animal-level significance,
equivalence, corrected multiplicity claim, or absence-of-bias claim follows.
Selection composition can be nonregular at very small retained counts; all
denominators and finite-draw fractions remain visible. This addresses temporal
clustering, not per-cell firing drift or acquisition uncertainty.
"""
    (output / "uncertainty_scope.md").write_text(note)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "non_rescoring": True,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_manifests": inputs,
        "source_score_manifests": sources,
        "content_source_commit": rm["reporter_commit"],
        "identity_source_commit": im["code_commit"],
        "n_bootstraps": N_BOOTSTRAPS,
        "primary_block_s": 60,
        "population_level_claim": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.source_dir, a.output_dir)
