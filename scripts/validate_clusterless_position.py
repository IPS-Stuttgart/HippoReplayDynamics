#!/usr/bin/env python3
"""Run clusterless marked-point-process behavioral position validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from hipporeplayimm.clusterless import ClusterlessMarkConfig
from hipporeplayimm.clusterless_position_validation import (
    ClusterlessPositionValidationConfig,
    run_clusterless_position_validation,
)
from hipporeplayimm.encoding import EncodingConfig


def _optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--session")
    parser.add_argument("--decode-bin-s", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--mark-smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--mark-prior-count", type=float, default=1.0)
    parser.add_argument("--mark-variance-floor", type=float, default=1.0)
    parser.add_argument("--rate-floor-hz", type=float, default=1e-4)
    parser.add_argument("--mark-likelihood", choices=("local-kde", "diagonal-gaussian"), default="local-kde")
    parser.add_argument("--mark-kde-bandwidth", type=_optional_float, default=None)
    parser.add_argument("--mark-kde-spatial-sigma-bins", type=_optional_float, default=None)
    parser.add_argument("--mark-kde-max-neighbors", type=int, default=256)
    args = parser.parse_args()

    config = ClusterlessPositionValidationConfig(
        clusterless=ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=args.bin_size_cm,
                smoothing_sigma_bins=args.smoothing_sigma_bins,
                min_speed_cm_s=args.min_speed_cm_s,
            ),
            mark_smoothing_sigma_bins=args.mark_smoothing_sigma_bins,
            mark_prior_count=args.mark_prior_count,
            mark_variance_floor=args.mark_variance_floor,
            rate_floor_hz=args.rate_floor_hz,
            mark_likelihood=args.mark_likelihood,
            mark_kde_bandwidth=args.mark_kde_bandwidth,
            mark_kde_spatial_sigma_bins=args.mark_kde_spatial_sigma_bins,
            mark_kde_max_neighbors=args.mark_kde_max_neighbors,
        ),
        decode_bin_s=args.decode_bin_s,
        max_windows_per_session=args.max_windows,
        random_seed=args.random_seed,
        session=args.session,
    )
    samples, summary = run_clusterless_position_validation(args.root, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output / "clusterless_position_decoding_samples.csv", index=False)
    summary.to_csv(output / "clusterless_position_decoding_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
