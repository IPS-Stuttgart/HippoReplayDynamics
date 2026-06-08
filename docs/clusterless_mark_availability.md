# Clusterless mark availability audit

Clusterless state-space evidence requires spike-mark or waveform-feature data.
Some Pfeiffer/Foster dataset copies contain sorted spikes but no spike marks.
In that case clusterless rows should be recorded as unavailable or unsupported,
not treated as a failed scientific comparison.

Run the audit with:

```bash
python scripts/clusterless_mark_availability.py \
  --dataset-root data/DataSetFromPfeifferFoster \
  --output results/clusterless-mark-availability
```

or through GitHub Actions:

```bash
gh workflow run clusterless-mark-availability.yml \
  -R IPS-Stuttgart/HippoReplayIMM \
  --ref main \
  -f dataset_root="data/DataSetFromPfeifferFoster"
```

The audit writes:

- `clusterless_mark_availability.csv`
- `clusterless_mark_gate_summary.csv`
- `clusterless_mark_availability_manifest.json`

Interpretation:

- `marks_detected`: at least one mark-like path or file key was detected for the session.
- `no_marks_detected`: the session was found, but no mark-like files or keys were found.
- `session_missing`: the audit could not find session-specific data under the supplied root.

The paper-facing rule is:

```text
If all audited sessions have marks:
    run true clusterless consistency.
If some audited sessions have marks:
    run a partial clusterless consistency screen and label coverage explicitly.
If no audited sessions have marks:
    record Goal 3 as data-limited on this artifact set.
```

Do not claim clusterless observation-model generalization from a dataset copy
that only produces unsupported clusterless rows.
