#!/usr/bin/env python3
"""Exhaustively inspect single-swap calibration feasibility; never rescore replay."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts import exchange_content as producer
from scripts import audit_exchange_content as independent
from scripts._provenance import build_script_provenance, file_sha256

RISK_NAMES = tuple(f"{s}_class{k}_{metric}" for s in ("high", "low") for k in (0, 1) for metric in ("error_cm", "home_brier"))
TOL = 1e-10


def calibration(enc, q3):
    producer.base.reserve_indices(enc)
    index = producer.base.calibration_rows(q3)
    bank = {key: q3[key][index] for key in ("counts", "truth_cm", "labels")}
    if not np.isfinite(bank["truth_cm"]).all() or not np.isin(bank["labels"], (0, 1)).all():
        raise ValueError("invalid calibration truth")
    pair = {s: sorted(map(int, enc[f"{s}_indices"])) for s in ("high", "low")}
    options = [(i, j) for i in sorted(set(pair["high"]) - set(pair["low"])) for j in sorted(set(pair["low"]) - set(pair["high"]))]
    return bank, pair, options


def feasibility(objective, risks, original_objective, original_risks):
    improve = bool(objective < original_objective - TOL)
    failure = np.asarray(risks) > np.asarray(original_risks) + TOL
    return improve, failure, bool(improve and not failure.any())


def inspect_session(enc, q3, frozen):
    bank, pair, options = calibration(enc, q3)
    if frozen["methods"]["baseline"] != pair or frozen["accepted_steps"] != 0 or frozen["net_exchanged"] != 0:
        raise ValueError("this diagnostic requires the original stopped zero-swap baseline")
    rates, counts = enc["early_run"], bank["counts"]
    ll = {s: producer.likelihood(counts, rates, pair[s]) for s in pair}
    kwargs = (enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])
    original = producer.readout(ll, *kwargs)
    gradient = producer.derivative(original, counts, rates, enc["near"], bank["labels"])
    ranked = sorted(options, key=lambda ij: (round(float(gradient[ij[1]] - gradient[ij[0]]), 10), *ij))
    rows = []
    for rank, (i, j) in enumerate(ranked, 1):
        difference = counts[:, j, None] * np.log(rates[j]) - 0.02 * rates[j] - counts[:, i, None] * np.log(rates[i]) + 0.02 * rates[i]
        result = producer.readout(dict(high=ll["high"] + difference, low=ll["low"] - difference), *kwargs)
        improve, failure, allowed = feasibility(result["objective"], result["risks"], original["objective"], original["risks"])
        row = dict(
            high=i,
            low=j,
            derivative_rank=rank,
            previously_evaluated=rank <= 32,
            gradient=float(gradient[j] - gradient[i]),
            objective=result["objective"],
            objective_improvement=original["objective"] - result["objective"],
            improves_objective=improve,
            admissible=allowed,
            failed_guards=int(failure.sum()),
        )
        for n, name in enumerate(RISK_NAMES):
            row[name] = result["risks"][n]
            row[f"change_{name}"] = result["risks"][n] - original["risks"][n]
            row[f"failed_{name}"] = bool(failure[n])
        rows.append(row)
    columns = ["high", "low", "derivative_rank", "previously_evaluated", "gradient", "objective", "objective_improvement", "improves_objective", "admissible", "failed_guards"] + [
        k for name in RISK_NAMES for k in (name, f"change_{name}", f"failed_{name}")
    ]
    table = pd.DataFrame(rows, columns=columns)
    return table


def verify_session(enc, q3, frozen, table):
    bank, pair, options = calibration(enc, q3)
    assert not table.duplicated(["high", "low"]).any()
    assert set(zip(table.high, table.low, strict=True)) == set(options), "missing or unexpected swaps"
    assert sorted(table.derivative_rank.tolist()) == list(range(1, len(options) + 1))
    p, h, initial_risks, delta, initial_objective = independent.state_for(pair, enc, bank)
    gradient = independent.direct_derivative(p, h, delta, enc, bank)
    ordered = sorted(options, key=lambda ij: (round(float(gradient[ij[1]] - gradient[ij[0]]), 10), *ij))
    assert list(zip(table.high, table.low, strict=True)) == ordered
    np.testing.assert_allclose(initial_risks, frozen["baseline_risks"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(initial_objective, frozen["baseline_objective"], atol=1e-10, rtol=1e-9)
    traces = {(row["high"], row["low"]): row for row in frozen["trace"][0]["proposals"]}
    assert set(traces) == set(ordered[:32])
    max_error = 0.0
    for rank, row in enumerate(table.to_dict("records"), 1):
        i, j = int(row["high"]), int(row["low"])
        _, _, risks, _, objective = independent.state_for(independent.exchanged(pair, i, j), enc, bank)
        values = np.array([row[key] for key in RISK_NAMES])
        max_error = max(max_error, float(np.max(np.abs(values - risks))), abs(row["objective"] - objective))
        np.testing.assert_allclose(values, risks, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(
            [row["objective"], row["objective_improvement"], row["gradient"]], [objective, initial_objective - objective, gradient[j] - gradient[i]], atol=1e-10, rtol=1e-9
        )
        improves = objective < initial_objective - 1e-10
        failures = risks > initial_risks + 1e-10
        assert row["improves_objective"] == improves
        assert row["admissible"] == bool(improves and not failures.any())
        assert row["previously_evaluated"] == (rank <= 32) and row["failed_guards"] == int(failures.sum())
        for n, key in enumerate(RISK_NAMES):
            assert row[f"failed_{key}"] == failures[n]
            np.testing.assert_allclose(row[f"change_{key}"], risks[n] - initial_risks[n], atol=1e-9, rtol=1e-9)
        if (i, j) in traces:
            old = traces[(i, j)]
            assert old["admissible"] == row["admissible"]
            np.testing.assert_allclose(old["risks"], risks, atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(old["objective"], objective, atol=1e-10, rtol=1e-9)
    return dict(verified_pairs=len(options), max_absolute_reconstruction_error=max_error)


def summarize(table, session):
    valid = table.loc[table.admissible.astype(bool)]
    improving = table.loc[table.improves_objective.astype(bool)]
    summary = dict(
        session=session,
        legal_single_swaps=len(table),
        objective_improving=len(improving),
        admissible=len(valid),
        admissible_outside_original32=int((valid.derivative_rank > 32).sum()),
        first_admissible_rank=None if valid.empty else int(valid.derivative_rank.min()),
        best_admissible_objective=None if valid.empty else float(valid.objective.min()),
        minimum_failed_guards_among_improving=None if improving.empty else int(improving.failed_guards.min()),
        no_legal_swap=table.empty,
        validated_remedy=False,
    )
    rejections = [dict(session=session, safeguard=name, objective_improving_swaps=len(improving), rejected=int(improving[f"failed_{name}"].sum())) for name in RISK_NAMES]
    return summary, rejections


def run(result_dir, audit_path, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("source audit does not verify this manifest")
    frozen_path = result_dir / "pre_scoring.json"
    if file_sha256(frozen_path) != manifest["assignments_sha256"]:
        raise ValueError("source assignments changed")
    frozen = json.loads(frozen_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        str(p): file_sha256(p)
        for p in (
            manifest_path,
            audit_path,
            frozen_path,
            Path(__file__),
            ROOT / "scripts/exchange_content.py",
            ROOT / "scripts/audit_exchange_content.py",
            ROOT / "scripts/audit_reserve_cell_content.py",
            ROOT / "scripts/reserve_cell_content.py",
            ROOT / "docs/exchange_search_protocol.md",
        )
    }
    summaries, rejections, reconstruction = [], [], {}
    for session in producer.base.SESSIONS:
        folder = Path(manifest["source_dir"]) / session.replace("/", "_")
        paths = (folder / "encoding.npz", folder / "run_q3.npz")
        for path in paths:
            if file_sha256(path) != manifest["input_file_sha256"].get(str(path)):
                raise ValueError(f"calibration source changed: {path}")
            inputs[str(path)] = file_sha256(path)
        enc, q3 = (producer.base.read_npz(p) for p in paths)
        assignment = frozen["assignments"][session]
        table = inspect_session(enc, q3, assignment)
        print("measured", session, len(table), "admissible", int(table.admissible.sum()), flush=True)
        reconstruction[session] = verify_session(enc, q3, assignment, table)
        table.to_csv(output_dir / f"{session.replace('/', '_')}_proposals.csv", index=False)
        # Verify the serialized artifact too; avoids silently truncated or changed tables.
        saved = pd.read_csv(output_dir / f"{session.replace('/', '_')}_proposals.csv")
        pd.testing.assert_frame_equal(saved, table, check_dtype=False, atol=1e-9, rtol=1e-9)
        summary, rejects = summarize(table, session)
        summaries.append(summary)
        rejections.extend(rejects)
        print("independently verified", session, flush=True)
    summary, rejects = pd.DataFrame(summaries), pd.DataFrame(rejections)
    summary.to_csv(output_dir / "summary.csv", index=False)
    rejects.to_csv(output_dir / "rejections.csv", index=False)
    missed = int(summary.admissible_outside_original32.sum())
    verdict = "bounded_search_missed_admissible_swaps" if missed else "no_admissible_single_swap_in_complete_neighborhood"
    from scripts.report_exchange_content import markdown

    text = "# Exhaustive single-swap RUN calibration diagnostic\n\n"
    text += f"Verdict: {verdict}. Original bounded experiment remains FAILED.\n\n"
    text += markdown(summary) + "\n\n## Failed regional safeguards among objective-improving swaps\n\n" + markdown(rejects)
    text += "\n\nNo replay, Q4 or simulated test observations were read. This is not independent-data validation or a remedy. "
    text += "Completeness concerns the original single-cell neighborhood only, not joint/multiple exchanges. "
    text += "Rejection counts overlap: a proposal can violate several safeguards.\n"
    (output_dir / "report.md").write_text(text)
    for path, expected in inputs.items():
        if file_sha256(path) != expected:
            raise ValueError(f"input changed during audit: {path}")
    record = build_script_provenance()
    record.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        verdict=verdict,
        input_file_sha256=inputs,
        independent_reconstruction=reconstruction,
        validated_remedy=False,
        external_validation=False,
        replay_scored=False,
        output_sha256={p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    )
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (output_dir / "independent_audit.json").write_text(
        json.dumps(
            dict(
                status="pass",
                manifest_sha256=file_sha256(output_dir / "manifest.json"),
                reconstructed_pairs=sum(r["verified_pairs"] for r in reconstruction.values()),
                validated_remedy=False,
                external_validation=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "audit", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(args.result_dir, args.audit, args.output_dir)
