# Off-SWR Promotion FDR Calibration

`scripts/calibrate_off_swr_promotion_fdr.py` calibrates the strict off-SWR
promotion rule against observed controls and shuffled nulls.

Example:

```bash
python scripts/calibrate_off_swr_promotion_fdr.py \
  --discovery-dir path/to/off-swr-discovery-artifact \
  --validation-dir path/to/promoted-off-swr-validation-artifact \
  --output results/off-swr-promotion-fdr-calibration \
  --n-permutations 10000 \
  --random-seed 1
```

Outputs:

- `off_swr_promotion_null_calibration.csv`
- `off_swr_promotion_empirical_fdr_summary.csv`
- `off_swr_promotion_threshold_sensitivity.csv`
- `off_swr_promotion_null_gate_summary.csv`

The observed-control rows test whether running high-specificity candidates or
ordinary movement/spiking-like high-specificity candidates pass the final
promotion rule. The permutation-null rows keep candidate margins, SWR distance,
and availability fixed while shuffling the specificity-label and/or immobility
ingredients. This asks whether the real alignment of high margin, immobility,
and interesting-candidate phenotype produces more promoted windows than a
shuffled alignment.

Interpretation should stay disciplined:

- A pass supports the claim that the strict promotion rule is empirically
  enriched beyond running/ordinary controls and label/immobility shuffles.
- It does not prove a universal false-discovery rate for all possible off-SWR
  windows or replace future time-shifted, wrong-map, LFP, or independent-dataset
  controls.
