"""Independent table checks and enumerated-path check for the simulation pilot."""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import hmmlearn
import numpy as np
import pandas as pd

repo, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(repo))
from scripts.simulate_hc11_phase_order_identifiability import predictive_gain, replicate

manifest = json.loads((root / "manifest.json").read_text())
assert manifest["status"] == "complete" and manifest["git_dirty"] is False
assert manifest["latent_trajectory_supplied"] is False
assert manifest["real_data_analysis_authorized"] is False
for name, digest in manifest["output_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
rows = pd.read_csv(root / "simulation_phase_scores.csv")
assert len(rows) == 64 * 5 * 2 * 2
assert not rows.duplicated(["replicate_seed", "scenario", "emission_mode", "phase"]).any()
assert np.isfinite(rows.select_dtypes(include="number").to_numpy()).all()
assert rows.n_events.eq(40).all()
summary = pd.read_csv(root / "simulation_recovery_summary.csv")
for result in summary.itertuples(index=False):
    paired = rows[rows.emission_mode.eq(result.emission_mode)]
    paired = paired.groupby(["scenario", "replicate_seed"]).mean(numeric_only=True).reset_index()
    calibration = paired[paired.scenario.eq("unchanged") & paired.replicate_seed.mod(2).eq(0)]
    q = np.percentile(calibration.mean_event_gain_per_spike, 95)
    test = paired[paired.scenario.eq(result.scenario) & paired.replicate_seed.mod(2).eq(1)]
    assert len(test) == 32
    values = test.mean_event_gain_per_spike
    assert np.isclose(q, result.null_calibration_p95, atol=1e-14, rtol=0)
    assert np.isclose(values.median(), result.median_gain_per_spike, atol=1e-14, rtol=0)
    assert np.isclose((values > q).mean(), result.above_null_p95_fraction, atol=1e-14, rtol=0)

# Enumerate every hidden path independently for a tiny four-frame event.
e = np.array([[.50, .20, .20, .10], [.15, .55, .05, .25], [.25, .15, .45, .15]])
own = np.array([[.5, .4, .1], [.1, .5, .4], [.4, .1, .5]])
other = own.T
x = np.array([[3, 0, 1, 0], [0, 2, 0, 2], [2, 1, 2, 1], [1, 3, 0, 2]])
train, held = np.array([0, 1]), np.array([2, 3])
scores = []
for a in (own, other):
    total = 0.
    for origin in range(2):
        numerator = denominator = 0.
        for path in itertools.product(range(3), repeat=origin + 3):
            probability = 1 / 3
            for t in range(1, len(path)):
                probability *= a[path[t-1], path[t]]
            for t in range(origin + 1):
                p = e[path[t], train] / e[path[t], train].sum()
                probability *= np.prod(p ** x[t, train])
            p = e[path[-1], held] / e[path[-1], held].sum()
            denominator += probability
            numerator += probability * np.prod(p ** x[origin + 2, held])
        total += np.log(numerator / denominator)
    scores.append(total)
gain, _, _ = predictive_gain(x, e, own, other, train, held)
assert np.isclose(gain, scores[0] - scores[1], atol=1e-12, rtol=0)

# Deterministic integration check: one seed across all five predeclared cases.
checked = 0
for scenario in rows.scenario.unique():
    repeat = pd.DataFrame(replicate((20260924, scenario, 80, 40, 20)))
    original = rows[rows.scenario.eq(scenario) & rows.replicate_seed.eq(20260924)]
    joined = original.merge(repeat, on=["replicate_seed", "scenario", "emission_mode", "phase"], validate="one_to_one", suffixes=("_saved", "_repeat"))
    assert len(joined) == 4
    np.testing.assert_allclose(joined.mean_event_gain_per_spike_saved, joined.mean_event_gain_per_spike_repeat, atol=1e-12, rtol=0)
    checked += len(joined)
result = {"status": "passed", "independent_path_enumeration": True,
          "rows_schema_checked": len(rows), "summary_rows_recomputed": len(summary),
          "phase_rows_deterministically_reproduced": checked,
          "output_hashes_checked": len(manifest["output_sha256"]),
          "hmmlearn_version": hmmlearn.__version__, "numpy_version": np.__version__,
          "producer_commit": manifest["code_commit"], "scientific_validity_established": False}
(root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
