#!/usr/bin/env python3
"""Independently rebuild every cell-predictive context decision and posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz, recount
from scripts.audit_encoding_uncertainty_content import check_metrics, h, scipy_posterior
from scripts.audit_temporal_endpoint_content import dense_gaussian, native_context_intervals

METHODS = ("independent", "unconditional_context", "predictive_context", "entropy_matched")


def dense_log_posteriors(events, rates, matrix):
    output = np.empty((len(events), 2, len(matrix)))
    for length in sorted({len(x) for x in events}):
        index = np.array([i for i, x in enumerate(events) if len(x) == length])
        counts = np.stack([events[i] for i in index])
        _, ll = scipy_posterior(counts.reshape(-1, len(rates)), rates)
        ll = ll.reshape(len(index), length, len(matrix))
        current = ll[:, 0] - logsumexp(ll[:, 0], axis=1, keepdims=True)
        for t in range(1, length):
            prediction = 0.75 * (np.exp(current) @ matrix.T) + 0.25 / len(matrix)
            weights = np.log(prediction) + ll[:, t]
            current = weights - logsumexp(weights, axis=1, keepdims=True)
        output[index, 0] = ll[:, -1] - logsumexp(ll[:, -1], axis=1, keepdims=True)
        output[index, 1] = current
    return output


def rebuild_population(events, rates, matrix, fold):
    scores = np.empty((len(events), 3, 2))
    sizes = np.array([(fold == k).sum() for k in range(3)])
    endpoint = np.stack([x[-1] for x in events])
    for k in range(3):
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        lp = dense_log_posteriors([e[:, train] for e in events], rates[train], matrix)
        _, held = scipy_posterior(endpoint[:, test], rates[test])
        scores[:, k] = logsumexp(lp + held[:, None, :], axis=2)
    differences = (scores[:, :, 1] - scores[:, :, 0]) / sizes[None, :]
    choice = (np.median(differences, axis=1) > 1e-10) & (endpoint.sum(axis=1) >= 3) & (np.count_nonzero(endpoint, axis=1) >= 2)
    full = dense_log_posteriors(events, rates, matrix)
    return np.exp(full), scores, differences, choice


def verify_one(row):
    folder = Path(row.artifact_dir)
    outputs = json.loads((folder / "outputs.json").read_text())
    if any(file_sha256(folder / k) != v for k, v in outputs.items()):
        raise ValueError("changed measurement")
    frozen = json.loads((folder / "frozen_input.json").read_text())
    prior, freeze = Path(frozen["edge_source"]), frozen["freeze"]
    if file_sha256(prior / "outputs.json") != frozen["source_outputs_sha256"] or any(file_sha256(prior / k) != v for k, v in frozen["source_outputs"].items()):
        raise ValueError("changed source")
    if file_sha256(freeze["encoding_path"]) != freeze["encoding_sha256"]:
        raise ValueError("changed native encoding")
    training_path = Path(freeze["encoding_path"]).parent / "encoding_manifest.json"
    if file_sha256(training_path) != frozen["training_manifest_sha256"]:
        raise ValueError("changed training provenance")
    native = load_npz(freeze["encoding_path"])
    frame = pd.read_csv(folder / "event_readouts.csv.gz", float_precision="round_trip")
    if (
        len(frame) != row.rows
        or frame.duplicated(["source", "split", "method", "event_index"]).any()
        or set(frame.source) != set(SOURCES)
        or set(frame.method) != set(METHODS)
        or set(frame.split) != {0, 1, 2}
    ):
        raise ValueError("incomplete or duplicated measurements")
    for key in ("dataset", "animal", "session"):
        if not frame[key].eq(getattr(row, key)).all():
            raise ValueError("identity mismatch")
    verified = posterior_rows = predictive_scores = native_blocks = 0
    for source in SOURCES:
        arrays = load_npz(prior / f"{source}_audit.npz")
        rates, grid, ids = (arrays[k] for k in ("rates_hz", "grid_cm", "cell_ids"))
        matrix = dense_gaussian(grid)
        blocks, truth, starts, context, raw_start, raw_end = [], [], [], [], [], []
        for j, (lo, hi) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)):
            n = min(10, int((hi - lo) // 4))
            if n < 1:
                raise ValueError("missing endpoint")
            blocks.append(np.stack([arrays["counts"][k : k + 4].sum(axis=0) for k in range(hi - 4 * n, hi, 4)]))
            truth.append(arrays["truth_base_cm"][hi - 4 : hi].mean(axis=0))
            starts.append(arrays["starts_s"][j] + 0.005 * (hi - lo - 4))
            context.append(20 * n)
            if source in ("real", "run_q4"):
                left, right = native_context_intervals(arrays, j, native, source)
                raw_start.append(left)
                raw_end.append(right)
                native_blocks += n
        if raw_start:
            np.testing.assert_array_equal(np.concatenate(blocks), recount(native["spikes"], ids, np.concatenate(raw_start), np.concatenate(raw_end)))
        counts, truth, starts = np.stack([e[-1] for e in blocks]), np.asarray(truth), np.asarray(starts)
        for part in freeze["groups"]:
            split = part["split"]
            saved = load_npz(folder / f"{source}_split{split}_audit.npz")
            for key, expected in (("counts", counts), ("truth_cm", truth), ("starts_s", starts), ("context_ms", context), ("event_ids", arrays["event_ids"])):
                np.testing.assert_allclose(saved[key], expected, atol=1e-10, rtol=0, equal_nan=True)
            groups = []
            for side in ("a", "b"):
                lookup = {int(k): i for i, k in enumerate(ids)}
                groups.append(np.array([lookup[int(k)] for k in part[side + "_ids"]]))
            if len(groups[0]) != len(groups[1]) or np.intersect1d(*groups).size:
                raise ValueError("invalid population split")
            view = frame.loc[(frame.source == source) & (frame.split == split)]
            reference = view.loc[view.method.eq("independent")].set_index("event_index").loc[arrays["event_ids"]]
            banks = []
            for side, group in zip(("a", "b"), groups, strict=True):
                key = f"20260914|predictive_context|{row.session}|{split}|{side}"
                number = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
                order = np.random.default_rng(number).permutation(len(group))
                fold = np.empty(len(group), int)
                fold[order] = np.arange(len(group)) % 3
                if len(group) < 6:
                    raise ValueError("too few inner-fold cells")
                np.testing.assert_array_equal(saved[side + "_indices"], group)
                np.testing.assert_array_equal(saved[side + "_fold_ids"], fold)
                bank, scores, delta, choose = rebuild_population([b[:, group] for b in blocks], rates[group], matrix, fold)
                np.testing.assert_allclose(saved[side + "_fold_scores"], scores, atol=2e-8, rtol=2e-9)
                np.testing.assert_allclose(saved[side + "_fold_delta"], delta, atol=2e-9, rtol=2e-9)
                sizes = np.tile([(fold == k).sum() for k in range(3)], (len(blocks), 1))
                np.testing.assert_array_equal(saved[side + "_fold_sizes"], sizes)
                np.testing.assert_array_equal(reference[side + "_use_context"], choose)
                np.testing.assert_allclose(reference[side + "_median_predictive_delta"], np.median(delta, axis=1), atol=2e-9, rtol=2e-9)
                np.testing.assert_array_equal(reference[side + "_spikes"], counts[:, group].sum(axis=1))
                np.testing.assert_array_equal(reference[side + "_active"], np.count_nonzero(counts[:, group], axis=1))
                probabilities = dict(independent=bank[:, 0], unconditional_context=bank[:, 1], predictive_context=bank[np.arange(len(bank)), choose.astype(int)])
                _, last_ll = scipy_posterior(counts[:, group], rates[group])
                temperature = reference[side + "_matched_log_temperature"].to_numpy()
                if not ((temperature >= -20) & (temperature <= 20)).all():
                    raise ValueError("entropy temperature outside bracket")
                tempered = (last_ll - last_ll.max(axis=1, keepdims=True)) / np.exp(temperature)[:, None]
                probabilities["entropy_matched"] = np.exp(tempered - logsumexp(tempered, axis=1, keepdims=True))
                good = np.abs(h(probabilities["entropy_matched"]) - h(probabilities["predictive_context"])) <= 1e-7
                np.testing.assert_array_equal(reference[side + "_entropy_control_available"], good)
                for method, prob in probabilities.items():
                    np.testing.assert_allclose(saved[f"{side}_{method}_posterior"], prob, atol=5e-9, rtol=2e-8, err_msg=method)
                    posterior_rows += len(prob)
                predictive_scores += scores.size
                banks.append(probabilities)
            for method in METHODS:
                selected = view.loc[view.method.eq(method)].set_index("event_index")
                np.testing.assert_array_equal(sorted(selected.index), sorted(arrays["event_ids"]))
                selected = selected.loc[arrays["event_ids"]]
                for key in ("original_start_s", "original_end_s", "context_ms", "n_cells_per_group"):
                    expected = dict(original_start_s=starts, original_end_s=starts + 0.02, context_ms=context, n_cells_per_group=np.full(len(starts), len(groups[0])))[key]
                    np.testing.assert_allclose(selected[key], expected, atol=1e-10, rtol=0)
                for side in ("a", "b"):
                    for key in ("spikes", "active", "use_context", "median_predictive_delta", "matched_log_temperature", "entropy_control_available"):
                        np.testing.assert_array_equal(selected[side + "_" + key], reference[side + "_" + key])
                check_metrics(selected, banks[0][method], banks[1][method], grid, truth)
                verified += len(selected)
        print(json.dumps(dict(session=row.session, source=source, status="verified")), flush=True)
    if verified != len(frame):
        raise ValueError("unaudited rows")
    return dict(
        dataset=row.dataset,
        animal=row.animal,
        session=row.session,
        status="passed",
        rows=verified,
        posterior_rows=posterior_rows,
        predictive_scores=predictive_scores,
        native_context_blocks=native_blocks,
        readout_sha256=file_sha256(folder / "event_readouts.csv.gz"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    producer_path = args.measurement_dir / "manifest.json"
    producer = json.loads(producer_path.read_text())
    if producer["status"] != "complete" or not producer["inputs_unchanged"]:
        raise ValueError("producer incomplete")
    for k, path in producer["input_file_paths"].items():
        if file_sha256(path) != producer["input_file_sha256"][k]:
            raise ValueError("changed producer input")
    inputs = dict(
        producer=producer_path,
        sessions=args.measurement_dir / "measurement_sessions.csv",
        auditor=Path(__file__),
        metrics=ROOT / "scripts/audit_encoding_uncertainty_content.py",
        transition=ROOT / "scripts/audit_temporal_endpoint_content.py",
    )
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    table = pd.read_csv(inputs["sessions"])
    if len(table) != 8 or table.animal.nunique() != 4 or table.duplicated(["animal", "session"]).any() or not table.status.eq("complete").all():
        raise ValueError("incomplete cohort")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for row in table.itertuples(index=False):
        results.append(verify_one(row))
        pd.DataFrame(results).to_csv(args.output_dir / "independent_audit_sessions.csv", index=False)
        print(json.dumps(results[-1]), flush=True)
    if any(file_sha256(v) != manifest["input_file_sha256"][k] for k, v in inputs.items()):
        raise ValueError("changed audit input")
    manifest.update(status="passed", results=results, all_fold_scores_and_posteriors_independently_rebuilt=True)
    (args.output_dir / "independent_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
