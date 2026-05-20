#!/usr/bin/env python3
"""Write place-field quality diagnostics and stable-cell recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

from hipporeplayimm.advanced_result_diagnostics import place_field_quality, stable_cell_ids
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True, help="Session label such as Rat1/Open1")
    parser.add_argument("--output", default="results/place-field-quality")
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--min-spatial-information-bits", type=float, default=0.25)
    parser.add_argument("--min-peak-rate-hz", type=float, default=1.0)
    args = parser.parse_args()

    session_path = Path(args.dataset_root) / args.session.replace("\\", "/")
    session = load_replay_session(session_path)
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    quality = place_field_quality(encoding)
    stable = stable_cell_ids(
        quality,
        min_spatial_information_bits=args.min_spatial_information_bits,
        min_peak_rate_hz=args.min_peak_rate_hz,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    quality.to_csv(out / "place_field_quality.csv", index=False)
    (out / "stable_cell_ids.txt").write_text("\n".join(str(int(x)) for x in stable) + "\n", encoding="utf-8")
    print(quality.sort_values("spatial_information_bits_per_spike", ascending=False).head(20).to_string(index=False))
    print(f"Stable cells: {len(stable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
