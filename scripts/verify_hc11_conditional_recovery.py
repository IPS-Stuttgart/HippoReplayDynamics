#!/usr/bin/env python3
"""Independent count, draw, predictive and bootstrap audit of hc-11 recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _provenance import build_script_provenance, file_sha256
from verify_hc11_count_conditioned_prediction import dense_kernels, direct_parts, direct_predict

MODELS = ("iid_position", "static_location", "diffusion", "first_order_imm")
COMPARISONS = {
    "imm_minus_iid_position": ("first_order_imm", "iid_position"),
    "imm_minus_static_location": ("first_order_imm", "static_location"),
    "imm_minus_diffusion": ("first_order_imm", "diffusion"),
    "diffusion_minus_iid": ("diffusion", "iid_position"),
    "static_minus_iid": ("static_location", "iid_position"),
    "oracle_minus_iid": ("oracle", "iid_position"),
}


def independent_draw(data, session, phase, event_id, generator, replicate):
    uid = f"{phase}_{event_id}"
    seed = int.from_bytes(hashlib.sha256(f"20260908|hc11_count_recovery|{session}|{replicate}|{generator}|{phase}|{event_id}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    direction = int(rng.integers(2))
    edges = data[f"edges_{uid}"]
    totals = data[f"counts_{uid}"].sum(axis=1)
    n = len(data["centers"])
    kernels = dense_kernels(data["centers"], edges, str(data["topology"]), float(data["track_length"]))
    if generator == "iid_position":
        path = rng.integers(n, size=len(totals))
        modes = np.full(len(totals), -1)
    elif generator == "static_location":
        path = np.full(len(totals), rng.integers(n))
        modes = np.full(len(totals), -1)
    else:
        nstate = 3 * n if generator == "first_order_imm" else n
        current = int(rng.integers(nstate))
        states = [current]
        for transition in kernels[generator]:
            current = int(rng.choice(nstate, p=transition[:, current]))
            states.append(current)
        path = np.array(states) % n
        modes = np.array(states) // n if generator == "first_order_imm" else np.full(len(totals), -1)
    rate = data[f"rates_direction_mixture_{direction}"]
    counts = np.array([rng.multinomial(int(total), rate[:, position] / rate[:, position].sum()) for total, position in zip(totals, path, strict=True)], dtype=np.int64)
    return counts, path, modes, direction, seed, kernels


def independent_weights(cohort, draws=2000, seed=20260908):
    frame = cohort.reset_index(drop=True)
    names = sorted(frame.rat.unique())
    sessions = {rat: [part.index.to_numpy() for _, part in frame[frame.rat == rat].groupby("session")] for rat in names}
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(draws):
        weights = np.zeros(len(frame))
        for rat in rng.choice(names, len(names), replace=True):
            choices = rng.integers(0, len(sessions[rat]), len(sessions[rat]))
            draws_for_rat = np.concatenate([rng.choice(sessions[rat][i], len(sessions[rat][i]), replace=True) for i in choices])
            weights += np.bincount(draws_for_rat, minlength=len(frame)) / (len(names) * len(draws_for_rat))
        result.append(weights)
    return np.array(result)


def reconstruct_events(scores):
    keys = ["session", "rat", "phase", "event_index", "generator", "replicate", "encoding_variant"]
    paired = scores[keys + ["split"]].copy()
    for contrast, (a, b) in COMPARISONS.items():
        paired[contrast] = scores[f"score_{a}"] - scores[f"score_{b}"]
    for model in MODELS:
        paired[f"gain_{model}"] = scores[f"score_{model}"] - scores.score_iid_position
    events = paired.groupby(keys, as_index=False).median().drop(columns="split")
    gain = events[[f"gain_{m}" for m in MODELS]].to_numpy()
    order = np.argsort(gain, axis=1)
    gap = gain[np.arange(len(gain)), order[:, -1]] - gain[np.arange(len(gain)), order[:, -2]]
    events["raw_predictive_winner"] = np.where(gap <= 1e-9, "ambiguous", np.array(MODELS)[order[:, -1]])
    return events


def verify_score_scope(scores):
    keys = ["session", "phase", "event_index", "generator", "replicate", "encoding_variant", "split"]
    if scores.empty or scores.duplicated(keys).any():
        raise ValueError("empty or duplicate score conditions")
    expected = pd.MultiIndex.from_product(
        [MODELS, range(50), ("direction_mixture", "pooled"), range(5)],
        names=["generator", "replicate", "encoding_variant", "split"],
    ).sort_values()
    for _, group in scores.groupby(["session", "phase", "event_index"]):
        actual = pd.MultiIndex.from_frame(group[list(expected.names)]).sort_values()
        if not actual.equals(expected):
            raise ValueError("missing or unexpected score conditions")
    columns = [f"score_{model}" for model in (*MODELS, "oracle")]
    values = scores[columns].to_numpy()
    if not scores.status.eq("success").all() or not np.isfinite(values).all() or np.any(values > 1e-8):
        raise ValueError("failed or invalid normalized predictive scores")


def verify_aggregates(scores, root):
    events = reconstruct_events(scores)
    keys = ["session", "rat", "phase", "event_index", "generator", "replicate", "encoding_variant"]
    stored = pd.read_csv(root / "recovery_event_contrasts.csv")
    joined = events.merge(stored, on=keys, validate="one_to_one", suffixes=("_audit", ""))
    if len(joined) != len(stored) or len(events) != len(stored):
        raise ValueError("missing event contrasts")
    for contrast in COMPARISONS:
        if not np.allclose(joined[f"{contrast}_audit"], joined[contrast], atol=1e-10, rtol=1e-10):
            raise ValueError("event contrast mismatch")
    if not joined.raw_predictive_winner.eq(joined.raw_predictive_winner_audit).all():
        raise ValueError("predictive rank mismatch")
    panels = pd.read_csv(root / "recovery_simulation_panels.csv")
    checked = []
    max_ci_error = 0.0
    for phase, subset in events.groupby("phase"):
        cohort = subset[["session", "rat", "event_index"]].drop_duplicates().sort_values(["rat", "session", "event_index"]).reset_index(drop=True)
        weights = independent_weights(cohort)
        rat_indices = [group.index.to_numpy() for _, group in cohort.groupby("rat")]
        for (generator, replicate, variant), group in subset.groupby(["generator", "replicate", "encoding_variant"]):
            ordered = cohort.merge(group, on=["session", "rat", "event_index"], validate="one_to_one")
            values = ordered[list(COMPARISONS)].to_numpy()
            rat_means = np.array([values[idx].mean(axis=0) for idx in rat_indices])
            ci = np.quantile(weights @ values, [0.025, 0.975], axis=0)
            rows = panels[(panels.phase == phase) & (panels.generator == generator) & (panels.replicate == replicate) & (panels.encoding_variant == variant)].set_index("contrast")
            if len(rows) != len(COMPARISONS):
                raise ValueError("missing simulation panel")
            for j, contrast in enumerate(COMPARISONS):
                row = rows.loc[contrast]
                mean = float(rat_means[:, j].mean())
                positive = int((rat_means[:, j] > 0).sum())
                err = max(abs(mean - row.equal_animal_mean), abs(ci[0, j] - row.ci_low), abs(ci[1, j] - row.ci_high))
                max_ci_error = max(max_ci_error, err)
                if err > 1e-8 or positive != row.positive_animals:
                    raise ValueError("independent point/CI reconstruction mismatch")
                checked.append(
                    {
                        "phase": phase,
                        "generator": generator,
                        "replicate": replicate,
                        "encoding_variant": variant,
                        "contrast": contrast,
                        "mean": mean,
                        "ci_low": ci[0, j],
                        "positive": positive == len(rat_indices) and mean > 0 and ci[0, j] > 0,
                    }
                )
    verified = pd.DataFrame(checked)
    patterns = pd.read_csv(root / "recovery_positive_pattern_summary.csv")
    for row in patterns.itertuples(index=False):
        primary = verified[
            (verified.phase == row.phase)
            & (verified.generator == row.generator)
            & (verified.encoding_variant == row.encoding_variant)
            & verified.contrast.isin(["imm_minus_iid_position", "imm_minus_static_location"])
        ]
        flags = primary.groupby("replicate").positive.all()
        if len(flags) != 50 or int(flags.sum()) != row.positive_pattern_count:
            raise ValueError("positive-pattern summary mismatch")
    return {"event_rows": len(events), "independent_bootstrap_panels": len(checked), "max_point_ci_error": max_ci_error, "positive_pattern_conditions": len(patterns)}


def check_path(path, modes, data, uid, generator):
    if len(path) != len(data[f"counts_{uid}"]) or np.any(path < 0) or np.any(path >= len(data["centers"])):
        raise ValueError("invalid stored path")
    if generator == "static_location" and not np.all(path == path[0]):
        raise ValueError("nonstatic static generator")
    if generator != "first_order_imm" and not np.all(modes == -1):
        raise ValueError("invalid non-IMM modes")
    if generator == "first_order_imm" and (np.any(modes < 0) or np.any(modes >= 3)):
        raise ValueError("invalid IMM modes")
    if generator not in ("diffusion", "first_order_imm"):
        return
    distance = np.abs(np.diff(data["centers"][path]))
    if str(data["topology"]) == "circular":
        distance = np.minimum(distance, float(data["track_length"]) - distance)
    edge = data[f"edges_{uid}"]
    limit = 4 * 85 * np.sqrt(np.diff((edge[:-1] + edge[1:]) / 2))
    if generator == "first_order_imm":
        limit = np.where(modes[1:] == 0, 8.0, np.where(modes[1:] == 2, np.inf, limit))
    if np.any(distance > limit + 1e-8):
        raise ValueError("path exceeds generator transition support")


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite audit")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "recovery_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["n_replicates"] != 50 or manifest["n_splits"] != 5 or manifest["n_bootstraps"] != 2000:
        raise ValueError("recovery incomplete or changed specification")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"output changed: {name}")
    for key, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][key]:
            raise ValueError("parent provenance changed")
    source = Path(manifest["source_dir"])
    parent = json.loads((source / "hc11_conditional_manifest.json").read_text())
    selection = pd.read_csv(source / "frozen_selection.csv")
    draws_checked, regenerated, prediction_rows = [], [], []
    frames = []
    for session, selected in selection.groupby("session"):
        cache_name = f"{session}_cache.npz"
        if file_sha256(source / cache_name) != parent["output_sha256"][cache_name]:
            raise ValueError("source cache changed")
        with np.load(source / cache_name) as archive:
            data = {key: archive[key] for key in archive.files}
        inspect = set(selected.groupby("phase").event_id.min().items())
        for item in [s for s in manifest["shards"] if s["session"] == session]:
            table = pd.read_csv(root / f"{item['tag']}_split_scores.csv")
            frames.append(table)
            indexed = table.set_index(["generator", "replicate", "phase", "event_index"]).sort_index()
            with np.load(root / f"{item['tag']}_draws.npz") as saved:
                if len(saved.files) != item["draws"] * 4:
                    raise ValueError("draw archive incomplete")
                for e in selected.itertuples(index=False):
                    uid = f"{e.phase}_{e.event_id}"
                    for replicate in range(item["replicate_start"], item["replicate_stop"]):
                        for generator in MODELS:
                            name = f"{generator}_{replicate}_{uid}"
                            counts, path, modes = (saved[f"{prefix}_{name}"] for prefix in ("counts", "path", "modes"))
                            direction = int(saved[f"direction_{name}"])
                            if (
                                direction not in (0, 1)
                                or not np.issubdtype(counts.dtype, np.integer)
                                or np.any(counts < 0)
                                or not np.array_equal(counts.sum(axis=1), data[f"counts_{uid}"].sum(axis=1))
                            ):
                                raise ValueError("invalid counts or changed bin totals")
                            check_path(path, modes, data, uid, generator)
                            digest = hashlib.sha256(str(counts.shape).encode() + counts.dtype.str.encode() + np.ascontiguousarray(counts).tobytes()).hexdigest()
                            rows = indexed.loc[(generator, replicate, e.phase, e.event_id)].reset_index()
                            if len(rows) != 10 or not rows.counts_sha256.eq(digest).all() or not rows.n_spikes.eq(int(counts.sum())).all():
                                raise ValueError("shared population draw/score mismatch")
                            for split in range(5):
                                train, held = data[f"train_{split}"], data[f"held_{split}"]
                                part = rows[rows.split == split]
                                if len(part) != 2 or not part.n_train_spikes.eq(counts[:, train].sum()).all() or not part.n_heldout_spikes.eq(counts[:, held].sum()).all():
                                    raise ValueError("split count metadata mismatch")
                            draws_checked.append(
                                {"session": session, "phase": e.phase, "event_id": e.event_id, "replicate": replicate, "generator": generator, "n_spikes": int(counts.sum())}
                            )
                            if replicate not in (0, 24, 49) or (e.phase, e.event_id) not in inspect:
                                continue
                            c, p, m, d, seed, kernels = independent_draw(data, session, e.phase, e.event_id, generator, replicate)
                            if not np.array_equal(c, counts) or not np.array_equal(p, path) or not np.array_equal(m, modes) or d != direction or not rows.seed.eq(seed).all():
                                raise ValueError("independent draw regeneration mismatch")
                            regenerated.append({"session": session, "phase": e.phase, "event_id": e.event_id, "replicate": replicate, "generator": generator})
                            train, held = data["train_0"], data["held_0"]
                            edges = data[f"edges_{uid}"]
                            for variant, part in rows[rows.split == 0].groupby("encoding_variant"):
                                row = part.iloc[0]
                                rates = [data[f"rates_{variant}_{i}"] for i in range(1 if variant == "pooled" else 2)]
                                tr = [direct_parts(counts[:, train], r[train], np.diff(edges)) for r in rates]
                                he = [direct_parts(counts[:, held], r[held], np.diff(edges)) for r in rates]
                                for model in MODELS:
                                    prediction = direct_predict(tr, he, model, "count_conditioned", 1.0, kernels)["count_conditioned"]
                                    error = abs(prediction - row[f"score_{model}"])
                                    if error > 1e-8 or not np.isfinite(error):
                                        raise ValueError("independent simulation predictive mismatch")
                                    prediction_rows.append(
                                        {
                                            "session": session,
                                            "phase": e.phase,
                                            "event_id": e.event_id,
                                            "replicate": replicate,
                                            "generator": generator,
                                            "encoding_variant": variant,
                                            "model": model,
                                            "error": error,
                                        }
                                    )
                                oracle_parts = direct_parts(counts[:, held], data[f"rates_direction_mixture_{direction}"][held], np.diff(edges))
                                oracle = oracle_parts["count_conditioned"][np.arange(len(path)), path].sum()
                                if abs(oracle - row.score_oracle) > 1e-8:
                                    raise ValueError("oracle mismatch")
        print(f"verified draws for {session}", flush=True)
    scores = pd.concat(frames, ignore_index=True)
    if len(draws_checked) != 64000 or len(regenerated) != 192 or len(prediction_rows) != 1536 or len(scores) != 640000:
        raise ValueError("audit scope incomplete")
    verify_score_scope(scores)
    summary = verify_aggregates(scores, root)
    pd.DataFrame(draws_checked).to_csv(output / "draw_count_audit.csv", index=False)
    pd.DataFrame(regenerated).to_csv(output / "regeneration_audit.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictive_audit.csv", index=False)
    report = build_script_provenance(input_paths={"recovery_manifest": root / "recovery_manifest.json"})
    report.update(
        status="passed",
        draws_checked=len(draws_checked),
        regenerated_draws=len(regenerated),
        independent_predictions=len(prediction_rows),
        maximum_predictive_error=max(row["error"] for row in prediction_rows),
        **summary,
        scope="All bin totals/valid paths/split metadata, 192 independently regenerated population draws, 1536 separate log-domain predictions, all event contrasts and hierarchical bootstrap intervals/positive-pattern counts. Simulation verification is not biological validation.",
    )
    (output / "audit_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir)
