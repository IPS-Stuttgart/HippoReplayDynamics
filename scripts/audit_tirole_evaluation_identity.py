"""Rate-matched evaluation cell-identity control with unchanged sequence selections."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_tirole_content_coverage import event_summaries, load_campaigns, summarize_content
from score_tirole_content_coverage import validate_bank

from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed


def matched_permutations(run_rate, n_null=199, seed=20260918):
    """Exchange maps within small ordered RUN-rate groups, never using rest spikes."""
    run_rate = np.asarray(run_rate, float)
    if run_rate.ndim != 1 or len(run_rate) < 4 or not np.isfinite(run_rate).all() or (run_rate <= 0).any() or n_null < 39:
        raise ValueError("four finite positive RUN rates and at least 39 nulls required")
    groups = np.array_split(np.argsort(run_rate, kind="stable"), max(1, len(run_rate) // 3))
    rng = np.random.default_rng(seed)
    permutations = np.tile(np.arange(len(run_rate)), (n_null, 1))
    membership = np.empty(len(run_rate), int)
    for k, group in enumerate(groups):
        membership[group] = k
        for row in permutations:
            row[group] = rng.permutation(group)
    return permutations, membership


def context_odds(counts, rates, valid, permutations, conditional=False):
    """Observed and identity-null sequenceless log odds, positive toward track 1."""
    counts, rates = np.asarray(counts), np.asarray(rates, float)
    valid, permutations = np.asarray(valid, bool), np.asarray(permutations, int)
    if counts.ndim != 2 or rates.ndim != 3 or rates.shape[:2] != (2, counts.shape[1]) or valid.shape != (2, rates.shape[2]):
        raise ValueError("count/map/track dimensions do not match")
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(rates).all() or (rates < 0).any() or not valid.any(axis=1).all():
        raise ValueError("invalid count, rate or support")
    if permutations.ndim != 2 or permutations.shape[1] != counts.shape[1] or not np.all(np.sort(permutations, axis=1) == np.arange(counts.shape[1])):
        raise ValueError("null must permute all evaluation identities")
    nonempty = counts.sum(axis=1) > 0
    if not nonempty.any():
        return np.full(len(permutations) + 1, np.nan)
    indices = np.vstack([np.arange(counts.shape[1]), permutations])
    lam = np.maximum(np.moveaxis(rates[:, indices, :], 0, 1), 1e-4)
    if conditional:
        lam = lam / lam.sum(axis=2, keepdims=True)
    ll = np.einsum("tu,skup->stkp", counts[nonempty], np.log(lam))
    if not conditional:
        ll -= 0.02 * lam.sum(axis=2)[:, None, :, :]
    ll -= np.log(valid.sum(axis=1))[None, None, :, None]
    ll[:, :, ~valid] = -np.inf
    posterior = np.exp(ll - logsumexp(ll, axis=(-2, -1), keepdims=True))
    mass = posterior.sum(axis=(1, 3))
    return np.log(np.maximum(mass[:, 0], 1e-300)) - np.log(np.maximum(mass[:, 1], 1e-300))


def null_summary(values):
    sd, mean = float(np.std(values[1:], ddof=1)), float(np.mean(values[1:]))
    return {
        "observed_log_odds": float(values[0]),
        "null_mean_log_odds": mean,
        "null_sd_log_odds": sd,
        "identity_z_log_odds": float((values[0] - mean) / sd) if np.isfinite(sd) and sd > 1e-12 else np.nan,
        "identity_excess_log_odds": float(values[0] - mean),
    }


def run(campaigns, banks, output):
    if output.exists():
        raise ValueError("new control output directory required")
    repo = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("commit the frozen control before running")
    data, sources = load_campaigns(campaigns)
    bank_by_session = {}
    for bank in banks:
        session = json.loads((bank / "manifest.json").read_text())["session"]
        if session in bank_by_session:
            raise ValueError("duplicate session bank")
        bank_by_session[session] = bank
    if set(bank_by_session) != set(data.session):
        raise ValueError("one exact bank is required per scored session")
    rows, matching, provenance = [], [], []
    for session, bank in bank_by_session.items():
        _info, parts, _events, counts, offsets, maps = validate_bank(bank)
        digest = file_sha256(bank / "manifest.json")
        for source in sources:
            m = json.loads(Path(source["manifest"]).read_text())
            if m["session"] == session and m["bank_manifest_sha256"] != digest:
                raise ValueError("scorer used a different bank")
        provenance.append({"session": session, "bank": str(bank.resolve()), "manifest_sha256": digest})
        occupancy = maps["occupancy_s"]
        if not (occupancy.sum(axis=1) > 0).all():
            raise ValueError("missing RUN occupancy")
        weights = occupancy / occupancy.sum(axis=1, keepdims=True)
        run_rate = (maps["rates"] * weights[:, None, :]).sum(axis=2).mean(axis=0)
        for split, partition in enumerate(parts["splits"]):
            evaluation = np.asarray(partition["evaluation"], int)
            seed = stable_seed(20260918, session, split, "rate-matched-evaluation-identity")
            permutations, groups = matched_permutations(run_rate[evaluation], seed=seed)
            matching.append(
                {
                    "session": session,
                    "split": split,
                    "n_evaluation_units": len(evaluation),
                    "n_groups": int(groups.max() + 1),
                    "n_null": len(permutations),
                    "n_distinct_permutations": len(np.unique(permutations, axis=0)),
                    "mean_absolute_log_rate_difference": float(np.abs(np.log(run_rate[evaluation][permutations]) - np.log(run_rate[evaluation])[None, :]).mean()),
                    "evaluation_indices": json.dumps(evaluation.tolist()),
                    "rate_group": json.dumps(groups.tolist()),
                    "permutation_seed": seed,
                }
            )
            selected = data.loc[(data.session == session) & (data.split == split), "event_id"].unique()
            for eid in selected:
                c = counts[offsets[eid] : offsets[eid + 1], :][:, evaluation]
                row = {"session": session, "split": split, "event_id": int(eid)}
                for conditional in (False, True):
                    values = context_odds(c, maps["rates"][:, evaluation], maps["valid_bins"], permutations, conditional)
                    prefix = "conditional_" if conditional else ""
                    row.update({prefix + k: v for k, v in null_summary(values).items()})
                rows.append(row)
            print(json.dumps({"session": session, "split": split, "events": len(selected)}), flush=True)
    readout = pd.DataFrame(rows)
    joined = data.merge(readout, on=["session", "split", "event_id"], validate="many_to_one")
    # Every original inference label, event group and selection stays fixed.
    joined["evaluation_z_log_odds"] = joined.identity_z_log_odds
    joined["conditional_evaluation_z_log_odds"] = joined.conditional_identity_z_log_odds
    events = event_summaries(joined)
    summary = summarize_content(events)
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("code changed during control")
    output.mkdir(parents=True)
    readout.to_csv(output / "evaluation_identity_by_event_split.csv", index=False)
    pd.DataFrame(matching).to_csv(output / "evaluation_RUN_rate_matching.csv", index=False)
    events.to_csv(output / "evaluation_identity_event_summary.csv", index=False)
    summary.to_csv(output / "evaluation_identity_content_summary.csv", index=False)
    note = """# Evaluation cell-identity control

Existing windows, inference labels and accepted/rejected groups are unchanged.
Only evaluation map-to-cell identity is permuted, within small sorted RUN-rate
groups of at least three cells. Each split uses 199 fixed permutations across
all events. RUN rate is the occupancy-weighted fitted-map rate, averaged equally
across tracks. Rest spikes never choose the groups. Identity draws and repeated
draws are retained.

This tests sequenceless cross-population context support against a rate-matched
cell-identity null. It is not future prediction or replay truth. Event medians
collapse repeat/split contributions; overlapping split-specific populations do
not become independent animals or exact population-level tests. Finite positive
means alone are insufficient for a biological claim. Inspect the actual RUN-rate
mismatch and null diversity before interpretation.
"""
    (output / "evaluation_identity_control.md").write_text(note)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "code_commit": commit,
        "git_dirty": False,
        "command_line": sys.argv,
        "scorer_manifests": sources,
        "bank_manifests": provenance,
        "selection_or_thresholds_changed": False,
        "future_prediction": False,
        "biological_confirmation": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--bank-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.campaign_dirs, a.bank_dirs, a.output_dir)
