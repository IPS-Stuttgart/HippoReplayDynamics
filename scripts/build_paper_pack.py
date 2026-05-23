#!/usr/bin/env python3
"""Build a compact paper-ready result pack from benchmark artifacts.

This script is intentionally a light-weight orchestrator.  It does not rerun the
full benchmark; it consumes already generated CSV artifacts and writes a single
directory with paired model-effect claims, simulation-recovery diagnostics, and
a manifest that records which inputs were used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from hipporeplayimm.recovery_diagnostics import (
    build_recovery_diagnostic_tables,
    load_recovery_score_tables,
)

try:  # Allows both `python scripts/build_paper_pack.py` and test imports.
    from scripts.make_paper_claims import (
        PaperClaimConfig,
        build_paper_claim_tables,
        load_score_tables,
    )
except ModuleNotFoundError:  # pragma: no cover
    from make_paper_claims import (  # type: ignore[no-redef]
        PaperClaimConfig,
        build_paper_claim_tables,
        load_score_tables,
    )


def build_paper_pack(
    *,
    output: str | Path,
    scores: Sequence[str | Path] | None = None,
    simulation_recovery_scores: Sequence[str | Path] | None = None,
    primary_model: str,
    baseline_model: str,
    value_column: str,
    n_bootstrap: int,
    random_seed: int,
    allow_missing_evidence_support: bool = False,
) -> dict[str, object]:
    """Write a paper pack and return its manifest."""

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "output": str(out_dir),
        "inputs": {
            "scores": [] if scores is None else [str(path) for path in scores],
            "simulation_recovery_scores": [] if simulation_recovery_scores is None else [str(path) for path in simulation_recovery_scores],
        },
        "outputs": {},
        "claim_config": {
            "primary_model": primary_model,
            "baseline_model": baseline_model,
            "value_column": value_column,
            "n_bootstrap": int(n_bootstrap),
            "random_seed": int(random_seed),
            "allow_missing_evidence_support": bool(allow_missing_evidence_support),
        },
    }

    if scores:
        claim_scores = load_score_tables(scores)
        claim_tables = build_paper_claim_tables(
            claim_scores,
            PaperClaimConfig(
                primary_model=primary_model,
                baseline_model=baseline_model,
                value_column=value_column,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
                require_evidence_support=not allow_missing_evidence_support,
            ),
        )
        claim_dir = out_dir / "model-claims"
        claim_tables.write(claim_dir)
        manifest["outputs"]["model_claims"] = str(claim_dir)
        manifest["model_claim_summary"] = claim_tables.summary.to_dict(orient="records")

    if simulation_recovery_scores:
        recovery_scores = load_recovery_score_tables(simulation_recovery_scores)
        recovery_tables = build_recovery_diagnostic_tables(recovery_scores, source_paths=simulation_recovery_scores)
        recovery_dir = out_dir / "simulation-recovery-diagnostics"
        recovery_tables.write(recovery_dir)
        manifest["outputs"]["simulation_recovery_diagnostics"] = str(recovery_dir)
        manifest["simulation_recovery_summary"] = recovery_tables.summary.to_dict(orient="records")

    (out_dir / "paper_pack_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    return manifest


def _render_readme(manifest: dict[str, object]) -> str:
    outputs = manifest.get("outputs", {})
    lines = [
        "# HippoReplayIMM paper pack",
        "",
        "This directory was generated from existing benchmark artifacts.",
        "",
        "## Outputs",
        "",
    ]
    if isinstance(outputs, dict) and outputs:
        for name, path in sorted(outputs.items()):
            lines.append(f"- `{name}`: `{path}`")
    else:
        lines.append("No score inputs were supplied, so only the manifest was written.")
    lines.extend(["", "See `paper_pack_manifest.json` for exact inputs and configuration.", ""])
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, float):
        return None if not math.isfinite(scalar) else float(scalar)
    if isinstance(scalar, (int, bool, str)) or scalar is None:
        return scalar
    return str(scalar)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-ready result pack from HippoReplayIMM artifacts.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores", nargs="+", help="Benchmark score CSVs/directories for paired paper claims.")
    parser.add_argument("--simulation-recovery-scores", nargs="+", help="Simulation recovery score CSVs/directories for recovery diagnostics.")
    parser.add_argument("--primary-model", default="sorted-spike-state-space-momentum-exact-sparse")
    parser.add_argument("--baseline-model", default="sorted-spike-state-space-diffusion")
    parser.add_argument("--value-column", default="heldout_log_likelihood")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--allow-missing-evidence-support", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_paper_pack(
        output=args.output,
        scores=args.scores,
        simulation_recovery_scores=args.simulation_recovery_scores,
        primary_model=args.primary_model,
        baseline_model=args.baseline_model,
        value_column=args.value_column,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        allow_missing_evidence_support=args.allow_missing_evidence_support,
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
