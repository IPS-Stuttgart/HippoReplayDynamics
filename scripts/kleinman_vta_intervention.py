"""Design checks and animal-level inference for the bounded VTA experiment."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

ANIMALS = {"Con_1": "control", "Con_2": "control", "Con_3": "control", "Exp_1": "experimental", "Exp_3": "experimental", "Exp_4": "experimental"}
KEY = ["animal", "session"]
PACKET_KEY = KEY + ["packet_id"]
AUTHOR_FIELDS = [
    "physical_track_id",
    "pretraining_track",
    "experience_number_on_track",
    "cno_dose_mg_kg",
    "injection_to_recording_minutes",
    "same_track_used_under_both_drugs",
    "confirmed_group",
    "treatment_allocation",
    "author_confirmed",
    "source_reference",
]
NULLS = ("unchanged", "drug_recruitment_exposure", "scalar_gain", "cell_expression", "bursting", "elapsed_time_drift", "shared_baseline")
EFFECTS = (-0.7, -0.35, -0.175, 0.175, 0.35, 0.7)
SCORE_COLUMNS = [f"{lag}_{kind}" for lag in ("preceding", "prospective") for kind in ("score", "information")]


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("missing_columns:" + ",".join(sorted(missing)))


def unique(frame, keys):
    require_columns(frame, keys)
    if frame[keys].isna().any().any() or frame[keys].astype(str).eq("").any().any() or frame.duplicated(keys).any():
        raise ValueError("missing_or_duplicate_identity:" + ",".join(keys))


def boolean(value):
    if str(value).strip().lower() in ("true", "1", "yes"):
        return True
    if str(value).strip().lower() in ("false", "0", "no"):
        return False
    raise ValueError("invalid_boolean:" + str(value))


def request_status(state, now=None):
    """The author deadline starts on actual sending, never on drafting."""
    now = now or datetime.now(UTC)
    rounds = state.get("clarification_rounds", 0)
    if not isinstance(rounds, int) or rounds not in (0, 1):
        raise ValueError("only_one_clarification_round_allowed")
    days = state.get("feasibility_working_days_used", 0)
    if not isinstance(days, (int, float)) or not 0 <= days <= 5:
        return "stopped_feasibility_budget"
    sent = state.get("sent_at_utc")
    if sent is None:
        return "awaiting_author_request_approval"
    sent = datetime.fromisoformat(sent)
    if sent.tzinfo is None or sent > now:
        raise ValueError("invalid_request_sent_time")
    return "blocked_metadata" if now >= sent + timedelta(days=14) else "awaiting_author_metadata"


def import_metadata(request, verified=None):
    """Preserve the release inventory; author confirmation is never inferred."""
    unique(request, KEY)
    require_columns(request, ["released_drug", "released_novel"])
    if not set(request.animal).issubset(ANIMALS):
        raise ValueError("unexpected_animal")
    base = request.copy().fillna("")
    if verified is not None:
        unique(verified, KEY)
        if not set(map(tuple, verified[KEY].to_numpy())).issubset(set(map(tuple, base[KEY].to_numpy()))):
            raise ValueError("unknown_author_session")
        supplied = verified.fillna("").set_index(KEY)
        base = base.set_index(KEY)
        for key, row in supplied.iterrows():
            for name in ("released_drug", "released_novel"):
                if name in row and str(row[name]) != str(base.loc[key, name]):
                    raise ValueError("author_release_conflict:" + str(key) + ":" + name)
            for name in AUTHOR_FIELDS:
                if name in row:
                    base.loc[key, name] = row[name]
        base = base.reset_index()
    for name in AUTHOR_FIELDS:
        if name not in base:
            base[name] = ""
    base = base.fillna("")
    reasons = []
    for r in base.to_dict("records"):
        failures = ["missing_" + n for n in AUTHOR_FIELDS if str(r[n]).strip() == ""]
        if r["released_drug"] not in ("saline", "CNO") or r["released_novel"] not in ("familiar", "novel"):
            raise ValueError("invalid_release_labels")
        if not failures:
            if not boolean(r["author_confirmed"]):
                failures.append("not_author_confirmed")
            if r["confirmed_group"] != ANIMALS[r["animal"]]:
                raise ValueError("experimental_control_conflict")
            for field in ("pretraining_track", "same_track_used_under_both_drugs"):
                boolean(r[field])
            for field in ("experience_number_on_track", "cno_dose_mg_kg", "injection_to_recording_minutes"):
                value = float(r[field])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("invalid_author_numeric:" + field)
            experience = float(r["experience_number_on_track"])
            if experience < 1 or experience != int(experience):
                raise ValueError("invalid_experience_order")
            if (r["released_drug"] == "saline" and float(r["cno_dose_mg_kg"]) != 0) or (r["released_drug"] == "CNO" and float(r["cno_dose_mg_kg"]) <= 0):
                raise ValueError("drug_dose_conflict")
        reasons.append(";".join(failures))
    base["metadata_exclusion"] = reasons
    base["metadata_verified"] = base.metadata_exclusion.eq("")
    base["drug"] = base.released_drug.map({"saline": 0, "CNO": 1})
    base["group"] = base.animal.map(ANIMALS)
    good = base.loc[base.metadata_verified]
    for _, g in good.groupby(["animal", "physical_track_id"]):
        if g.pretraining_track.map(boolean).nunique() != 1 or g.same_track_used_under_both_drugs.map(boolean).nunique() != 1:
            raise ValueError("track_metadata_conflict")
        if g.experience_number_on_track.astype(float).duplicated().any():
            raise ValueError("duplicate_track_experience_order")
    return base


def prior_readouts(registry):
    require_columns(registry, KEY + ["traversal", "source_artifact", "role"])
    if registry.empty or registry.source_artifact.isna().any() or not registry.role.isin(["past", "baseline", "target"]).all():
        raise ValueError("audited_previous_readouts_required")
    ids = pd.to_numeric(registry.traversal, errors="raise")
    if not np.isfinite(ids).all() or (ids != np.floor(ids)).any():
        raise ValueError("invalid_previous_traversal")
    return set(zip(registry.animal, registry.session, ids.astype(int), strict=True))


def validate_packets(packets):
    unique(packets, PACKET_KEY)
    require_columns(packets, ["reference_traversals", "span_start_s", "span_end_s", "drug", "novel", "selected", "availability_descriptor"])
    selected = packets.loc[packets.selected.map(boolean)]
    for r in selected.itertuples():
        refs = json.loads(r.reference_traversals)
        ids = refs + [r.past_traversal, r.baseline_traversal, r.target_traversal]
        if len(refs) != 3 or len(set(ids)) != 6:
            raise ValueError("six_distinct_traversals_required")
        times = [r.span_start_s, r.past_start_s, r.past_end_s, r.baseline_start_s, r.baseline_end_s, r.target_start_s, r.target_end_s]
        if not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or r.span_end_s != r.target_end_s:
            raise ValueError("packet_temporal_leakage")
    for _, g in selected.groupby(KEY):
        g = g.sort_values("span_start_s")
        if np.any(g.span_start_s.to_numpy()[1:] < g.span_end_s.to_numpy()[:-1]):
            raise ValueError("overlapping_packets_across_directions")


def cohort_tables(metadata, packets, previous):
    validate_packets(packets)
    prior = prior_readouts(previous)
    if not set(map(tuple, packets[KEY].to_numpy())).issubset(set(map(tuple, metadata[KEY].to_numpy()))):
        raise ValueError("packet_session_not_in_inventory")
    fields = KEY + [
        "physical_track_id",
        "experience_number_on_track",
        "cno_dose_mg_kg",
        "injection_to_recording_minutes",
        "metadata_verified",
        "released_novel",
        "group",
        "drug",
        "same_track_used_under_both_drugs",
    ]
    ledger = packets.merge(metadata[fields], on=KEY, how="left", validate="many_to_one", suffixes=("", "_author"))
    if not ledger.drug.eq(ledger.drug_author).all() or not ledger.novel.eq(ledger.released_novel.map({"familiar": 0, "novel": 1})).all():
        raise ValueError("packet_release_metadata_conflict")
    ledger["previous_readout_overlap"] = [any((r.animal, r.session, getattr(r, n + "_traversal")) in prior for n in ("past", "baseline", "target")) for r in ledger.itertuples()]
    ledger["exclusion_reason"] = [
        ";".join(
            reason
            for fail, reason in (
                (not boolean(r.selected), "not_chronologically_selected"),
                (not r.metadata_verified, "unverified_metadata"),
                (r.released_novel != "familiar", "novel_exploratory_only"),
                (r.previous_readout_overlap, "previously_scored_readout_exploratory_only"),
                (not boolean(r.availability_descriptor), "recruitment_or_reference_unavailable"),
            )
            if fail
        )
        for r in ledger.itertuples()
    ]
    audits = []
    for (animal, track), g in ledger.loc[ledger.exclusion_reason.eq("")].groupby(["animal", "physical_track_id"]):
        drugs = set(g.drug)
        overlap = False
        if drugs == {0, 1}:
            ranges = [g.loc[g.drug.eq(d), "experience_number_on_track"].astype(float) for d in (0, 1)]
            overlap = max(x.min() for x in ranges) <= min(x.max() for x in ranges)
        declared_shared = g.same_track_used_under_both_drugs.map(boolean).all()
        reason = "" if drugs == {0, 1} and overlap and declared_shared else "unsupported_track_drug_or_experience_comparison"
        if reason:
            ledger.loc[g.index, "exclusion_reason"] = reason
        audit = {
            "animal": animal,
            "physical_track_id": track,
            "both_drugs": drugs == {0, 1},
            "experience_overlap": bool(overlap),
            "author_shared_track": bool(declared_shared),
            "supported": not reason,
        }
        for d, label in ((0, "saline"), (1, "CNO")):
            sub = g.loc[g.drug.eq(d)]
            audit.update({label + "_sessions": sub.session.nunique(), label + "_days": sub.session.str[:8].nunique(), label + "_packets": len(sub)})
            for field in ("cno_dose_mg_kg", "injection_to_recording_minutes", "experience_number_on_track"):
                audit[label + "_" + field] = json.dumps(sorted(sub[field].astype(float).unique().tolist()))
        audits.append(audit)
    ledger["primary_eligible"] = ledger.exclusion_reason.eq("")
    cohort = ledger.loc[ledger.primary_eligible].copy()
    coverage = []
    for animal, group in ANIMALS.items():
        a = cohort.loc[cohort.animal.eq(animal)]
        coverage.append({"animal": animal, "group": group, "n_packets": len(a), "n_tracks": a.physical_track_id.nunique(), "paired_drugs": set(a.drug) == {0, 1}})
    return ledger, pd.DataFrame(audits), pd.DataFrame(coverage)


def animal_contrasts(frame, require_all=True):
    """Pool U/V within track/drug, pair tracks, then weight tracks/animals equally."""
    unique(frame, PACKET_KEY)
    require_columns(frame, SCORE_COLUMNS + ["physical_track_id", "drug"])
    if not set(frame.animal).issubset(ANIMALS) or not frame.drug.isin([0, 1]).all():
        raise ValueError("invalid_analysis_labels")
    for name in SCORE_COLUMNS:
        if not np.isfinite(frame[name]).all():
            raise ValueError("nonfinite_score_or_information")
        if name.endswith("information") and frame[name].lt(0).any():
            raise ValueError("negative_information")
    paired = frame.loc[frame.preceding_information.gt(1e-10) & frame.prospective_information.gt(1e-10)]
    tracks = []
    for (animal, track), original in frame.groupby(["animal", "physical_track_id"]):
        g = paired.loc[paired.animal.eq(animal) & paired.physical_track_id.eq(track)]
        row = {"animal": animal, "physical_track_id": track, "n_frozen_packets": len(original), "n_informative_packets": len(g), "status": "missing_paired_information"}
        if set(g.drug) == {0, 1}:
            for drug, label in ((0, "saline"), (1, "CNO")):
                sub = g.loc[g.drug.eq(drug)]
                for lag in ("preceding", "prospective"):
                    row[label + "_" + lag] = sub[lag + "_score"].sum() / sub[lag + "_information"].sum()
                row["A_" + label] = row[label + "_prospective"] - row[label + "_preceding"]
            row.update(drug_difference=row["A_CNO"] - row["A_saline"], status="scored")
        tracks.append(row)
    tf = pd.DataFrame(tracks)
    animals = []
    for animal in ANIMALS if require_all else sorted(frame.animal.unique()):
        g = tf.loc[tf.animal.eq(animal)] if len(tf) else tf
        good = len(g) > 0 and g.status.eq("scored").all()
        animals.append(
            {
                "animal": animal,
                "group": ANIMALS[animal],
                "n_tracks": len(g),
                "status": "scored" if good else "missing_paired_information",
                "A_saline": g.A_saline.mean() if good else np.nan,
                "A_CNO": g.A_CNO.mean() if good else np.nan,
                "drug_difference": g.drug_difference.mean() if good else np.nan,
            }
        )
    return tf, pd.DataFrame(animals)


def contrast_inference(animals, require_all=True):
    unique(animals, ["animal"])
    if not set(animals.animal).issubset(ANIMALS) or any(r.group != ANIMALS[r.animal] for r in animals.itertuples()):
        raise ValueError("invalid_group_identity")
    result = {"status": "missing_paired_information", "D": None, "ci_low": None, "ci_high": None, "p_two_sided": None, "flag": False, "n_animals": len(animals)}
    if (require_all and set(animals.animal) != set(ANIMALS)) or not animals.status.eq("scored").all():
        return result
    values = [animals.loc[animals.group.eq(g), "drug_difference"].to_numpy(float) for g in ("experimental", "control")]
    if min(map(len, values)) < 2 or not all(np.isfinite(x).all() for x in values):
        return result
    a, b = values
    d = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb <= 1e-20:
        return {**result, "status": "degenerate_animal_variance", "D": d}
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    se = np.sqrt(va + vb)
    half = float(student_t.ppf(0.975, df) * se)
    p = float(2 * student_t.sf(abs(d) / se, df))
    return {
        **result,
        "status": "scored",
        "D": d,
        "ci_low": d - half,
        "ci_high": d + half,
        "p_two_sided": p,
        "flag": p < 0.05,
        "welch_df": float(df),
        "inference": "two_sided_animal_level_Welch_working_interval_not_randomization",
    }


def calibration_summary(estimates, validation=True):
    require_columns(estimates, ["scenario", "effect", "replicate", "status", "D", "flag"])
    unique(estimates, ["scenario", "effect", "replicate"])
    expected = {(s, 0.0) for s in NULLS} | {("intervention", e) for e in EFFECTS}
    if set(zip(estimates.scenario, estimates.effect, strict=True)) != expected:
        raise ValueError("incomplete_calibration_scenarios")
    rows = []
    for (scenario, effect), g in estimates.groupby(["scenario", "effect"]):
        n = len(g)
        complete = g.status.eq("scored").all() and np.isfinite(g.D).all() and (not validation or set(g.replicate) == set(range(1000)))
        flags = g.flag.map(boolean)
        correct = flags if effect == 0 else flags & g.D.mul(effect).gt(0)
        hits = int(correct.sum())
        # Exact two-sided 95% binomial interval; report its upper endpoint for nulls.
        low = float(beta_distribution.ppf(0.025, hits, n - hits + 1)) if hits else 0.0
        high = float(beta_distribution.ppf(0.975, hits + 1, n - hits)) if hits < n else 1.0
        rows.append(
            {
                "scenario": scenario,
                "effect": effect,
                "n_replicates": n,
                "n_flags_correct_direction": hits,
                "frequency": hits / n,
                "mc95_low": low,
                "mc95_high": high,
                "mean_D": g.D.mean(),
                "complete": bool(complete),
                "null_pass": bool(complete and high <= 0.075) if effect == 0 else None,
                "engineering_benchmark_035": abs(effect) == 0.35,
            }
        )
    return pd.DataFrame(rows)
