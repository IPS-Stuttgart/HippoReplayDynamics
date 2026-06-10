# Off-SWR ripple-negativity validation

`scripts/off_swr_ripple_negativity_validation.py` evaluates whether promoted
off-SWR candidates are physiologically ripple-band negative, rather than merely
distant from detected SWR annotations.

The intended inputs are:

```bash
python scripts/off_swr_ripple_negativity_validation.py \
  --promoted-decisions path/to/promoted_off_swr_candidate_exact_core_decisions.csv \
  --off-swr-window-table path/to/off_swr_trajectory_discovery_decisions.csv \
  --promoted-covariates path/to/off_swr_candidate_table.csv \
  --swr-off-swr-dynamics path/to/swr_off_swr_dynamics_comparison.csv \
  --output results/off-swr-ripple-negativity \
  --ripple-z-threshold 3.0
```

The script writes:

- `off_swr_candidate_lfp_ripple_power.csv`
- `off_swr_candidate_lfp_gate_summary.csv`
- `off_swr_candidate_ripple_power_matched_null.csv`
- `swr_off_swr_lfp_dynamics_comparison.csv`

Recognized ripple/LFP columns include aliases for peak and mean ripple-band
power z-scores, sharp-wave-band power, candidate-window ripple-threshold
crossing, and threshold crossings in +/-50 ms, +/-100 ms, and +/-250 ms buffers.

The overall gate passes only when promoted candidates have:

- full LFP/ripple-power or threshold-crossing coverage,
- buffer crossing fields for +/-50 ms, +/-100 ms, and +/-250 ms,
- no candidate-window ripple-threshold crossing,
- no buffer ripple-threshold crossing,
- peak ripple-band power below the configured threshold, and
- preserved full-core trajectory-family support after the LFP filter.

Current artifact interpretation:

```text
The 27242588406 promoted-candidate artifact and the associated 27237703414
off-SWR discovery artifact do not contain ripple_power, lfp_power, or
ripple_band_power fields. The LFP gate therefore reports unsupported rather
than promoting a physiologically ripple-negative replay claim.
```

Paper-safe wording when the gate is unsupported:

```text
The current off-SWR result is detected-SWR-independent and trajectory-family
validated, but it is not yet physiologically ripple-negative because continuous
LFP/ripple-band power covariates are absent from the available artifact.
```
