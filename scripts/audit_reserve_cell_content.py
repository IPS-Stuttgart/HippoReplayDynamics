#!/usr/bin/env python3
"""Independent full-likelihood, cell-allocation and table reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
METHODS = ("baseline", "targeted") + tuple(f"random_{j:02}" for j in range(20))
METRICS = ("home_gap", "high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy") + tuple(
    f"balanced_{s}_{m}" for s in ("high", "low") for m in ("error", "brier")
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as obj:
        return dict(obj)


def posterior(counts, rates):
    if np.any(counts < 0) or not np.isfinite(counts).all() or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("invalid Poisson observations")
    expected = rates * 0.02
    ll = np.einsum("nc,cb->nb", counts, np.log(expected), optimize=True) - expected.sum(axis=0)
    ll -= gammaln(counts + 1).sum(axis=1)[:, None]
    return np.exp(ll - logsumexp(ll, axis=1)[:, None])


def check_assignment(enc, bank, assignment):
    originals = {s: list(map(int, enc[f"{s}_indices"])) for s in ("high", "low")}
    reserve = sorted(set(range(len(enc["cell_ids"]))) - set(originals["high"]) - set(originals["low"]))
    quota = len(reserve) // 2
    _, index = np.unique(bank["parent_ids"], return_index=True)
    index = np.sort(index)
    labels = bank["labels"][index]
    assert min(np.sum(labels == k) for k in (0, 1)) >= 10
    assert assignment["calibration_rows"] == index.tolist()
    assert assignment["reserve"] == reserve and assignment["quota"] == quota
    counts, rates = bank["counts"][index], enc["early_run"]

    def loss(ix):
        prob = posterior(counts[:, ix], rates[ix])[:, enc["near"]].sum(axis=1)
        return float(sum(np.mean((prob[labels == k] - k) ** 2) for k in (0, 1)) / 2)

    chosen = {s: [] for s in originals}
    left = reserve.copy()
    for step, trace in enumerate(assignment["trace"]):
        options = []
        for s in ("high", "low"):
            if len(chosen[s]) >= quota:
                continue
            before = loss(originals[s] + chosen[s])
            for cell in left:
                after = loss(originals[s] + chosen[s] + [cell])
                options.append((before - after, s, cell, before, after))
        gain = max(x[0] for x in options)
        winner = next(x for x in options if gain - x[0] <= 1e-12)
        assert (trace["step"], trace["side"], trace["cell_index"]) == (step, winner[1], winner[2])
        np.testing.assert_allclose([trace["gain"], trace["before"], trace["after"]], [winner[0], winner[3], winner[4]], atol=1e-10, rtol=1e-9)
        chosen[winner[1]].append(winner[2])
        left.remove(winner[2])
    assert len(assignment["trace"]) == 2 * quota and assignment["methods"]["targeted"] == chosen
    assert set(assignment["methods"]) == set(METHODS)
    for name, pair in assignment["methods"].items():
        a, b = set(pair["high"]), set(pair["low"])
        size = 0 if name == "baseline" else quota
        assert len(pair["high"]) == len(a) == len(pair["low"]) == len(b) == size
        assert not a & b and (a | b).issubset(reserve)
        if name.startswith("random_"):
            draw = int(name.split("_")[1])
            seed = int.from_bytes(hashlib.sha256(f"20260915|reserve|{assignment['session']}|{draw}".encode()).digest()[:8], "little")
            perm = np.random.default_rng(seed).permutation(reserve)
            assert pair == dict(high=perm[:quota].tolist(), low=perm[quota : 2 * quota].tolist())


def values_for(bank, enc, encoding, pair):
    means, tiles, value = [], [], {}
    grid = enc["grid_cm"]
    labels = bank.get("labels", np.full(len(bank["counts"]), np.nan)).astype(float)
    scale = np.maximum(grid.max(axis=0) - grid.min(axis=0), 1e-12)
    tile_ids = np.clip(np.floor(3 * (grid - grid.min(axis=0)) / scale).astype(int), 0, 2)
    tile_ids = tile_ids[:, 0] + 3 * tile_ids[:, 1]
    for s in ("high", "low"):
        ix = list(enc[f"{s}_indices"]) + pair[s]
        counts = bank["counts"][:, ix]
        p = posterior(counts, enc[encoding][ix])
        mean = np.einsum("nb,bd->nd", p, grid)
        value[f"{s}_home"] = p[:, enc["near"]].sum(axis=1)
        value[f"{s}_entropy"] = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=1) / np.log(len(grid))
        value[f"{s}_error"] = np.sqrt(np.sum((mean - bank["truth_cm"]) ** 2, axis=1))
        value[f"{s}_brier"] = (value[f"{s}_home"] - labels) ** 2
        value[f"{s}_spikes"] = counts.sum(axis=1)
        value[f"{s}_active"] = (counts != 0).sum(axis=1)
        means.append(mean)
        tiles.append(np.stack([p[:, tile_ids == k].sum(axis=1) for k in range(9)], axis=1))
    value["true_home"] = labels
    value["separation"] = np.sqrt(np.sum((means[0] - means[1]) ** 2, axis=1))
    value["regional_tv"] = np.abs(tiles[0] - tiles[1]).sum(axis=1) * 0.5
    return value


def mean_row(value, session, source, encoding, method):
    row = dict(session=session, animal=session.split("/")[0], source=source, encoding=encoding, method=method, events=len(value["true_home"]))
    for key in ("high_home", "low_home", "separation", "regional_tv", "high_entropy", "low_entropy", "high_spikes", "low_spikes", "high_active", "low_active"):
        row[key] = np.mean(value[key])
    row["home_gap"] = abs(row["high_home"] - row["low_home"])
    for side in ("high", "low"):
        for m in ("error", "brier"):
            row[f"balanced_{side}_{m}"] = sum(np.mean(value[f"{side}_{m}"][value["true_home"] == k]) for k in (0, 1)) / 2 if source in TRUTH else np.nan
    return row


def rebuild_tables(rows):
    session = pd.DataFrame(rows)
    combined = []
    for key, group in session.groupby(["animal", "session", "source", "encoding"], sort=True):
        for method in ("baseline", "targeted", "random_mean"):
            selected = group[group.method.str.startswith("random_")] if method == "random_mean" else group[group.method == method]
            row = dict(zip(("animal", "session", "source", "encoding"), key, strict=True), method=method)
            row.update({m: selected[m].mean() for m in METRICS})
            if method != "random_mean":
                row.update({k: selected.iloc[0][k] for k in ("events", "high_spikes", "low_spikes", "high_active", "low_active")})
            combined.append(row)
    combined = pd.DataFrame(combined)
    animal = combined.groupby(["animal", "source", "encoding", "method"], as_index=False)[list(METRICS)].mean()
    summary = animal.groupby(["source", "encoding", "method"], as_index=False)[list(METRICS)].mean()
    flags = {}
    expected = {(s, r, e, m) for s in SESSIONS for r in REAL + TRUTH for e in (("early_run", "full_run") if r in REAL else ("early_run",)) for m in METHODS}
    flags["source_and_random_coverage"] = len(session) == len(expected) and set(session[["session", "source", "encoding", "method"]].itertuples(index=False, name=None)) == expected
    for src, total in ((REAL[0], 1836), (REAL[1], 513)):
        subset = session[(session.source == src) & (session.encoding == "early_run")]
        flags[f"fixed_{src}"] = len(subset) == 4 * 22 and subset.groupby("method").events.sum().eq(total).all()
    for src in REAL + TRUTH:
        for enc in ("early_run", "full_run") if src in REAL else ("early_run",):
            sums = summary[(summary.source == src) & (summary.encoding == enc)].set_index("method")
            rats = animal[(animal.source == src) & (animal.encoding == enc)]

            def no_rat_worse(metric):
                wide = rats.pivot(index="animal", columns="method", values=metric)
                return len(wide) == 3 and np.isfinite(wide).all().all() and (wide.targeted <= wide.baseline + 1e-10).all()

            if src in REAL:
                factor = 0.8 if src == REAL[0] else 1
                flags[f"home_{src}_{enc}"] = no_rat_worse("home_gap") and sums.loc["targeted", "home_gap"] <= factor * sums.loc["baseline", "home_gap"] + 1e-10
                if src == REAL[0] and enc == "early_run":
                    for m in ("separation", "regional_tv"):
                        wide = rats.pivot(index="animal", columns="method", values=m)
                        flags[f"real_{m}"] = (
                            len(wide) == 3 and np.isfinite(wide).all().all() and (wide.targeted < wide.baseline).all() and sums.loc["targeted", m] <= 0.9 * sums.loc["baseline", m]
                        )
                    for m in ("high_entropy", "low_entropy"):
                        flags[f"real_{m}"] = np.isfinite(sums[m]).all() and len(rats.animal.unique()) == 3 and sums.loc["targeted", m] <= sums.loc["baseline", m] + 1e-10
                    for m in ("home_gap", "separation", "regional_tv"):
                        flags[f"beats_equal_budget_random_{m}"] = (
                            np.isfinite(sums[m]).all() and len(rats.animal.unique()) == 3 and sums.loc["targeted", m] < sums.loc["random_mean", m]
                        )
            else:
                for m in (x for x in METRICS if x.startswith("balanced_")):
                    flags[f"truth_{src}_{m}"] = no_rat_worse(m)
    flags["development_numerical_screen"] = all(flags.values())
    return dict(session_draws=session, session_summary=combined, animal_summary=animal, summary=summary), {k: bool(v) for k, v in flags.items()}


def compare_frame(actual, expected, keys):
    assert set(actual.columns) == set(expected.columns)
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True)[sorted(actual.columns)],
        expected.sort_values(keys).reset_index(drop=True)[sorted(expected.columns)],
        check_dtype=False,
        atol=1e-9,
        rtol=1e-9,
    )


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    for path, h in manifest["input_file_sha256"].items():
        assert sha(path) == h, path
    for name, h in manifest["output_sha256"].items():
        assert sha(root / name) == h, name
    assert sha(root / "pre_scoring.json") == manifest["assignments_sha256"]
    frozen = json.loads((root / "pre_scoring.json").read_text())
    assert frozen["input_file_sha256"] == manifest["input_file_sha256"]
    assert frozen["created_at_utc"] < manifest["created_at_utc"]
    assert set(frozen["assignments"]) == set(SESSIONS)
    rows, count = [], 0
    for session in SESSIONS:
        folder = Path(manifest["source_dir"]) / session.replace("/", "_")
        enc = load(folder / "encoding.npz")
        assignment = frozen["assignments"][session]
        check_assignment(enc, load(folder / "run_q3.npz"), assignment)
        actual = pd.read_csv(root / f"{session.replace('/', '_')}_events.csv.gz", dtype={"event_id": str})
        reconstructed = []
        for source in REAL + TRUTH:
            bank = load(folder / f"{source}.npz")
            for encoding in ("early_run", "full_run") if source in REAL else ("early_run",):
                for method, pair in assignment["methods"].items():
                    v = values_for(bank, enc, encoding, pair)
                    rows.append(mean_row(v, session, source, encoding, method))
                    count += len(bank["counts"])
                    if method == "baseline":
                        np.testing.assert_allclose(np.array([v["high_home"], v["low_home"]]).T, bank[f"{encoding}_scores"], atol=1e-9, rtol=1e-9)
                    if method in ("baseline", "targeted"):
                        f = pd.DataFrame(v).assign(
                            observation_index=np.arange(len(bank["counts"])),
                            event_id=bank.get("event_ids", np.arange(len(bank["counts"]))).astype(str),
                            session=session,
                            source=source,
                            encoding=encoding,
                            method=method,
                        )
                        reconstructed.append(f)
        compare_frame(actual, pd.concat(reconstructed, ignore_index=True), ["session", "source", "encoding", "method", "observation_index"])
        print(f"independently reconstructed {session}", flush=True)
    tables, gates = rebuild_tables(rows)
    for name, table in tables.items():
        compare_frame(pd.read_csv(root / f"{name}.csv"), table, [k for k in ("animal", "session", "source", "encoding", "method") if k in table])
    actual_gates = pd.read_csv(root / "gates.csv")
    assert not actual_gates.gate.duplicated().any()
    assert dict(zip(actual_gates.gate, actual_gates.passed, strict=True)) == gates
    assert count == manifest["evaluated_event_method_rows"]
    result = dict(
        status="pass",
        manifest_sha256=sha(root / "manifest.json"),
        reconstructed_event_method_rows=count,
        reconstructed_population_posteriors=2 * count,
        gates=gates,
        scope="all RUN-only greedy allocations, budget-matched random draws, full count likelihoods, target rows, truth, summaries and gates; no biological ground truth",
        audit_script_sha256=sha(__file__),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.output)
