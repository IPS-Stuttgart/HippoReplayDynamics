#!/usr/bin/env python3
"""Raw-count and separate log-domain verification of the hc-11 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _hc11_native_encoding as native
from _provenance import build_script_provenance, file_sha256


def direct_parts(counts, rates, dt):
    total = counts.sum(axis=1)
    constants = gammaln(counts + 1).sum(axis=1)
    proportions = rates / rates.sum(axis=0)
    identity = counts @ np.log(proportions) + gammaln(total + 1)[:, None] - constants[:, None]
    full = counts @ np.log(rates) + total[:, None] * np.log(dt[:, None]) - dt[:, None] * rates.sum(axis=0) - constants[:, None]
    return {"count_conditioned": identity, "full_poisson": full, "total_rate_only": full - identity}


def dense_kernels(centers, edges, topology, length):
    distances = np.abs(centers[:, None] - centers[None, :])
    if topology == "circular":
        distances = np.minimum(distances, length - distances)

    def gaussian(sigma):
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights[distances > 4 * sigma] = 0
        return weights / weights.sum(axis=0)

    stationary = gaussian(2.0)
    fragmented = np.ones_like(stationary) / len(centers)
    diffusion = [gaussian(85 * np.sqrt(dt)) for dt in np.diff((edges[:-1] + edges[1:]) / 2)]
    imm = [np.block([[(0.95 if src == dst else 0.025) * kernel for src in range(3)] for dst, kernel in enumerate([stationary, d, fragmented])]) for d in diffusion]
    return {"diffusion": diffusion, "first_order_imm": imm}


def direct_hmm(ll, matrices):
    """Independent dense log-domain forward/backward, no repository engine."""
    ntime, nstate = ll.shape
    with np.errstate(divide="ignore"):
        logs = [np.log(matrix) for matrix in matrices]
    alpha = np.empty_like(ll)
    alpha[0] = ll[0] - np.log(nstate)
    for t in range(1, ntime):
        alpha[t] = ll[t] + logsumexp(logs[t - 1] + alpha[t - 1][None, :], axis=1)
    z = logsumexp(alpha[-1])
    beta = np.zeros_like(ll)
    for t in range(ntime - 2, -1, -1):
        beta[t] = logsumexp(logs[t] + (ll[t + 1] + beta[t + 1])[:, None], axis=0)
    return z, alpha + beta - z


def direct_predict(train_parts, held_parts, model, observation, temperature, kernels):
    direction_z, direction_posterior = [], []
    for part in train_parts:
        ll = part[observation] / temperature
        n = ll.shape[1]
        if model == "iid_position":
            z = np.sum(logsumexp(ll, axis=1) - np.log(n))
            posterior = ll - logsumexp(ll, axis=1, keepdims=True)
        elif model == "static_location":
            summed = ll.sum(axis=0)
            z = logsumexp(summed) - np.log(n)
            posterior = np.tile(summed - logsumexp(summed), (len(ll), 1))
        elif model == "diffusion":
            z, posterior = direct_hmm(ll, kernels[model])
        elif model == "first_order_imm":
            z, joint = direct_hmm(np.concatenate([ll, ll, ll], axis=1), kernels[model])
            posterior = logsumexp(joint.reshape(len(ll), 3, n), axis=1)
        else:
            raise ValueError("unknown model")
        direction_z.append(z)
        direction_posterior.append(posterior)
    weights = np.array(direction_z) - logsumexp(direction_z)
    result = {}
    for target in ("count_conditioned", "full_poisson"):
        prediction = np.stack([q + h[target] + w for q, h, w in zip(direction_posterior, held_parts, weights, strict=True)])
        result[target] = float(logsumexp(prediction, axis=(0, 2)).sum())
    return result


def required_contrasts():
    def r(model, obs="count_conditioned"):
        return ("real", obs, model)

    return {
        "imm_minus_iid_position": (r("first_order_imm"), r("iid_position")),
        "imm_minus_static_location": (r("first_order_imm"), r("static_location")),
        "imm_minus_diffusion": (r("first_order_imm"), r("diffusion")),
        "real_minus_wrong_imm": (r("first_order_imm"), ("population_code_permuted", "count_conditioned", "first_order_imm")),
        "identity_minus_rate_imm": (r("first_order_imm"), r("first_order_imm", "total_rate_only")),
        "identity_minus_full_imm": (r("first_order_imm"), r("first_order_imm", "full_poisson")),
        "diffusion_minus_iid": (r("diffusion"), r("iid_position")),
        "static_minus_iid": (r("static_location"), r("iid_position")),
    }


def verify_tables(scores, root):
    keys = ["session", "rat", "phase", "event_index", "match_pair_id", "split", "encoding_variant", "inference_temperature"]
    wide = scores.pivot(index=keys, columns=["map", "observation", "model"], values="conditional_heldout_log_score")
    components = []
    for name, (a, b) in required_contrasts().items():
        part = (wide[a] - wide[b]).rename("delta").reset_index()
        part["contrast"] = name
        components.append(part)
    reconstructed = pd.concat(components, ignore_index=True)
    stored = pd.read_csv(root / "hc11_conditional_split_contrasts.csv")
    joined = reconstructed.merge(stored, on=keys + ["contrast"], validate="one_to_one", suffixes=("_audit", ""))
    if len(joined) != len(stored) or not np.allclose(joined.delta_audit, joined.delta, atol=1e-10, rtol=1e-10):
        raise ValueError("split contrast mismatch")
    event_keys = [key for key in keys if key != "split"] + ["contrast"]
    events = reconstructed.groupby(event_keys, as_index=False).delta.median()
    stored_events = pd.read_csv(root / "hc11_conditional_event_contrasts.csv")
    joined_events = events.merge(stored_events, on=event_keys, validate="one_to_one", suffixes=("_audit", ""))
    if len(joined_events) != len(stored_events) or not np.allclose(joined_events.delta_audit, joined_events.delta, atol=1e-10, rtol=1e-10):
        raise ValueError("event median mismatch")
    summary = pd.read_csv(root / "hc11_conditional_summary.csv")
    aggregate_keys = ["phase", "encoding_variant", "inference_temperature", "contrast"]
    paired = events.pivot(index=["session", "rat", "match_pair_id", "encoding_variant", "inference_temperature", "contrast"], columns="phase", values="delta").reset_index()
    paired["delta"] = paired.POST - paired.PRE
    paired["phase"] = "POST_minus_PRE"
    all_events = pd.concat([events, paired], ignore_index=True)
    rats = all_events.groupby(aggregate_keys + ["rat"], as_index=False).delta.mean()
    avg = rats.groupby(aggregate_keys, as_index=False).delta.mean().rename(columns={"delta": "audited_estimate"})
    joined_summary = avg.merge(summary, on=aggregate_keys, validate="one_to_one")
    if len(joined_summary) != len(summary) or not np.allclose(joined_summary.audited_estimate, joined_summary.equal_animal_mean_event_median_delta, atol=1e-10, rtol=1e-10):
        raise ValueError("equal-animal summary mismatch")
    return len(stored), len(stored_events), len(summary)


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite verification")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "hc11_conditional_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["code_commit"] in (None, "unavailable"):
        raise ValueError("run incomplete or missing commit")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"output hash mismatch: {name}")
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError(f"source changed: {name}")
    scores = pd.read_csv(root / "hc11_conditional_event_scores.csv")
    selection = pd.read_csv(root / "frozen_selection.csv")
    if len(scores) != 153600 or len(selection) != 320 or selection.session.nunique() != 8 or selection.animal.nunique() != 4:
        raise ValueError("incomplete cohort/rows")
    if not scores.heldout_temperature.eq(1).all() or scores.heldout_used_for_inference.any() or not scores.posterior_unchanged.all():
        raise ValueError("invalid held-out prediction contract")
    expected_settings = {
        "time_bin_s": 0.02,
        "diffusion_sigma_cm_sqrt_s": 85.0,
        "stationary_sigma_cm": 2.0,
        "max_step_sigma": 4.0,
        "mode_stickiness": 0.95,
        "n_splits": 5,
        "heldout_fraction": 0.3,
        "rate_scale": 1.0,
    }
    if any(manifest.get(k) != v for k, v in expected_settings.items()):
        raise ValueError("frozen model settings changed")
    checks = []
    audited = []
    for item in manifest["sessions"]:
        session = item["session"]
        for path, digest in item["source_hashes"].items():
            if file_sha256(path) != digest:
                raise ValueError(f"raw input changed: {path}")
        spike_path = next(Path(p) for p in item["source_hashes"] if p.endswith("spikes.cellinfo.mat"))
        raw = loadmat(spike_path, squeeze_me=True, struct_as_record=False)["spikes"]
        raw_units = np.asarray(raw.UID, dtype=int).ravel()
        by_unit = {int(unit): np.sort(np.asarray(times, dtype=float).ravel()) for unit, times in zip(raw_units, np.asarray(raw.times, dtype=object).ravel(), strict=True)}
        with np.load(root / f"{session}_cache.npz") as data:
            units = data["unit_ids"]
            seed = int.from_bytes(hashlib.sha256(f"20260908|{session}".encode()).digest()[:4], "little")
            if not np.array_equal(data["permutation"], np.random.default_rng(seed).permutation(len(data["centers"]))):
                raise ValueError("wrong-map permutation mismatch")
            track = native.load_track_samples(spike_path.parent)
            if str(data["topology"]) != track.topology or not np.isclose(float(data["track_length"]), track.track_length_cm):
                raise ValueError("track geometry cache mismatch")
            spikes = native.load_spikes(spike_path.parent)
            refit, _ = native.build_session_encodings(track, spikes, **manifest["encoding_parameters"])
            for variant, maps in refit.items():
                for d, encoding in enumerate(maps):
                    if (
                        not np.array_equal(encoding.unit_ids, units)
                        or not np.array_equal(encoding.bin_centers_cm, data["centers"])
                        or not np.allclose(data[f"rates_{variant}_{d}"], encoding.rates_hz, atol=1e-12, rtol=1e-12)
                    ):
                        raise ValueError("RUN map cache mismatch")
            for split in range(5):
                shuffled = units.copy()
                np.random.default_rng(20260804 + split).shuffle(shuffled)
                n_test = max(1, min(round(len(units) * 0.30), len(units) - 1))
                if not np.array_equal(np.sort(shuffled[:n_test]), units[data[f"held_{split}"]]) or not np.array_equal(np.sort(shuffled[n_test:]), units[data[f"train_{split}"]]):
                    raise ValueError("split reconstruction mismatch")
            selected = selection[selection.session == session].sort_values(["phase", "event_id"])
            inspect = set(selected.groupby("phase", sort=True).event_id.first().items())
            for e in selected.itertuples(index=False):
                uid = f"{e.phase}_{e.event_id}"
                edges = data[f"edges_{uid}"]
                if abs(edges[0] - e.start_time_s) > 1e-9 or abs(edges[-1] - e.end_time_s) > 1e-9 or not np.allclose(np.diff(edges)[:-1], 0.02, atol=1e-10, rtol=0):
                    raise ValueError("time bin mismatch")
                raw_counts = np.column_stack([np.histogram(t[(t >= edges[0]) & (t < edges[-1])], bins=edges)[0] for t in (by_unit[int(u)] for u in units)])
                if not np.array_equal(raw_counts, data[f"counts_{uid}"]):
                    raise ValueError("raw spike recount mismatch")
                checks.append({"session": session, "phase": e.phase, "event_id": e.event_id, "n_spikes": int(raw_counts.sum()), "n_units": len(units)})
                if (e.phase, e.event_id) not in inspect:
                    continue
                train, held = data["train_0"], data["held_0"]
                kernels = dense_kernels(data["centers"], edges, str(data["topology"]), float(data["track_length"]))
                rows = scores[(scores.session == session) & (scores.phase == e.phase) & (scores.event_index == e.event_id) & (scores.split == 0)]
                for (variant, map_name), g in rows.groupby(["encoding_variant", "map"]):
                    order = np.arange(len(data["centers"])) if map_name == "real" else data["permutation"]
                    rates = [data[f"rates_{variant}_{d}"][:, order] for d in range(1 if variant == "pooled" else 2)]
                    tr = [direct_parts(raw_counts[:, train], r[train], np.diff(edges)) for r in rates]
                    he = [direct_parts(raw_counts[:, held], r[held], np.diff(edges)) for r in rates]
                    for row in g.itertuples(index=False):
                        result = direct_predict(tr, he, row.model, row.observation, row.inference_temperature, kernels)
                        err = max(abs(result["count_conditioned"] - row.conditional_heldout_log_score), abs(result["full_poisson"] - row.poisson_heldout_log_score))
                        if not np.isfinite(err) or err > 1e-8:
                            raise ValueError(f"independent predictive mismatch: {session}, {uid}, {row.model}, {err}")
                        audited.append(
                            {
                                "session": session,
                                "phase": e.phase,
                                "event_id": e.event_id,
                                "encoding_variant": variant,
                                "map": map_name,
                                "model": row.model,
                                "observation": row.observation,
                                "inference_temperature": row.inference_temperature,
                                "max_error": err,
                            }
                        )
        print(f"audited {session}", flush=True)
    dimensions = verify_tables(scores, root)
    pd.DataFrame(checks).to_csv(output / "raw_count_audit.csv", index=False)
    pd.DataFrame(audited).to_csv(output / "predictive_score_audit.csv", index=False)
    report = build_script_provenance(input_paths={"run_manifest": root / "hc11_conditional_manifest.json"})
    report.update(
        status="passed",
        raw_events=len(checks),
        raw_files=sum(len(s["source_hashes"]) for s in manifest["sessions"]),
        refitted_session_maps=len(manifest["sessions"]),
        independently_scored_rows=len(audited),
        maximum_predictive_error=max(r["max_error"] for r in audited),
        split_contrasts=dimensions[0],
        event_contrasts=dimensions[1],
        aggregate_point_estimates=dimensions[2],
        scope="all raw event counts/splits/map caches and paired point estimates; independent predictions for lowest event ID per phase/session in split 0; same native RUN-fitting helper; bootstrap CIs not independently recomputed",
    )
    (output / "audit_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir)
