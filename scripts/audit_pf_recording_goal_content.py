#!/usr/bin/env python3
"""Independent dense reconstruction and accepted-segment endpoint diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def posterior(counts, rates):
    score = (np.asarray(counts)[:, None] * np.log(rates) - .02*rates).sum(axis=0)
    weights = np.exp(score-score.max())
    return weights/weights.sum()


def segment_endpoint(path, counts, grid):
    supported = np.flatnonzero(counts.sum(axis=1) >= 2)
    if not len(supported):
        return None
    start, end = supported[0], supported[-1]
    jumps = np.linalg.norm(np.diff(grid[path[start:end+1]], axis=0), axis=1)
    segments = np.split(np.arange(start, end+1), np.flatnonzero(jumps >= 20-1e-9)+1)
    return int(max(segments, key=len)[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    manifest = json.loads((root/"manifest.json").read_text())
    benchmark_path = manifest["input_file_paths"]["benchmark"]
    assert digest(benchmark_path) == manifest["input_file_sha256"]["benchmark"]
    benchmark = json.loads(Path(benchmark_path).read_text())
    events = pd.read_csv(root/"event_content.csv.gz")
    metadata = pd.read_csv(root/"home_metadata_qc.csv").set_index("session")
    sample_rows, trajectory_rows, gates = [], [], []
    for rec in benchmark["results"]:
        if rec["dataset"] != "pfeiffer_foster":
            continue
        session = rec["session"]
        checkpoint = json.loads((root/session.replace("/", "_")/"checkpoint.json").read_text())
        for path, sha in checkpoint["inputs"].items():
            assert digest(path) == sha, path
        with np.load(rec["input_arrays_path"]) as z:
            arrays = {k: z[k] for k in z.files}
        grid, counts = arrays["grid_cm"], arrays["frame_counts"]
        rates = arrays["rates_hz"][:, arrays["support"]]
        home = metadata.loc[session, ["home_x_cm", "home_y_cm"]].to_numpy(float)
        near = np.linalg.norm(grid-home, axis=1) <= 20
        lookup = {key: j for j, key in enumerate(arrays["window_uids"])}
        with np.load(rec["shuffle_files"][0]["path"]) as z:
            full_path = z["original_path"]
        full_specs = rec["population_specs"]
        selected = events[events.session.eq(session)]
        sample_ids = set(sorted(selected.window_uid.unique())[::max(1, selected.window_uid.nunique()//10)])
        for row in selected.itertuples(index=False):
            if row.window_uid not in sample_ids and not row.accepted_full:
                continue
            i = lookup[row.window_uid]
            a, b = arrays["frame_offsets"][2*i:2*i+2]
            c = counts[a:b]
            ix = full_specs[int(row.population_replicate)+1]["indices"]
            edge = int(np.flatnonzero(c.sum(axis=1) >= 2)[-1])
            end = segment_endpoint(full_path[a:b], c, grid)
            assert edge == row.fixed_endpoint_frame
            assert end == row.full_segment_endpoint_frame
            pf, ph = posterior(c[edge], rates), posterior(c[edge, ix], rates[ix])
            errors = [abs(np.linalg.norm(ph@grid-pf@grid)-row.endpoint_mean_shift_cm),
                abs(pf[near].sum()-row.home_mass_r20_full),
                abs(ph[near].sum()-row.home_mass_r20_half),
                abs(np.abs(pf-ph).sum()/2-row.endpoint_posterior_tv)]
            assert max(errors) < 1e-9
            if row.window_uid in sample_ids:
                sample_rows.append(dict(session=session, window_uid=row.window_uid,
                    population_replicate=row.population_replicate, max_absolute_error=max(errors)))
            if row.accepted_full:
                pf, ph = posterior(c[end], rates), posterior(c[end, ix], rates[ix])
                assert abs(pf[near].sum()-row.trajectory_home_mass_full) < 1e-9
                assert abs(ph[near].sum()-row.trajectory_home_mass_half) < 1e-9
                trajectory_rows.append(dict(animal=rec["animal"], session=session,
                    window_uid=row.window_uid, population_replicate=row.population_replicate,
                    half_accepted=row.accepted_half, fixed_full_segment_endpoint_frame=end,
                    endpoint_mean_shift_cm=np.linalg.norm(ph@grid-pf@grid),
                    endpoint_map_shift_cm=np.linalg.norm(grid[ph.argmax()]-grid[pf.argmax()]),
                    home_mass_full=pf[near].sum(), home_mass_half=ph[near].sum(),
                    full_endpoint_spikes=c[end].sum(), full_endpoint_active_cells=(c[end]>0).sum(),
                    half_endpoint_spikes=c[end, ix].sum(), half_endpoint_active_cells=(c[end, ix]>0).sum()))
        gates.append(dict(gate=session+":inputs_and_dense_reconstruction", passed=True))
    assert events.window_uid.nunique() == 4001 and len(events) == 12003
    assert not events.duplicated(["window_uid", "population_replicate"]).any()
    population = pd.read_csv(root/"population_summary.csv")
    decomposition = population[population.cohort.eq("reselection_decomposition")]
    assert len(decomposition) == 24
    assert np.allclose(decomposition.decoding+decomposition.composition+decomposition.timing, decomposition.total)
    expected = events.groupby(["animal", "session", "population_replicate"]).endpoint_mean_shift_cm.median().groupby(["animal", "session"]).mean().groupby("animal").mean().mean()
    summary = pd.read_csv(root/"summary.csv")
    reported = summary[summary.cohort.eq("all_fixed_candidates") & summary.metric.eq("endpoint_mean_shift_cm")].equal_rat_mean.iloc[0]
    assert np.isclose(expected, reported)
    simulations = pd.read_csv(root/"simulation_content.csv.gz")
    for _, group in simulations.groupby(["session", "simulation_event"]):
        assert len(group) == 4
        for name in ["true_home", "true_endpoint_x_cm", "true_endpoint_y_cm", "full_population_spikes"]:
            assert group[name].nunique() == 1
    assert simulations[simulations.cell_fraction.eq(1)].groupby("session").true_home.mean().eq(.5).all()
    gates.extend([dict(gate=k, passed=True) for k in ["fixed_denominators", "decomposition_identity",
                  "independent_equal_rat_summary", "simulation_truth_unchanged"]])
    pd.DataFrame(gates).to_csv(root/"independent_audit_gates.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(root/"dense_reconstruction_audit.csv", index=False)
    tf = pd.DataFrame(trajectory_rows)
    tf.to_csv(root/"accepted_segment_endpoint_diagnostic.csv.gz", index=False)
    numerical = ["endpoint_mean_shift_cm", "endpoint_map_shift_cm", "home_mass_full", "home_mass_half",
                 "full_endpoint_spikes", "full_endpoint_active_cells", "half_endpoint_spikes", "half_endpoint_active_cells"]
    ts = tf.groupby(["animal", "session", "population_replicate"])[numerical].median().groupby(["animal", "session"]).mean().groupby("animal").mean()
    ts.to_csv(root/"accepted_segment_endpoint_by_rat.csv")
    supported = (tf.half_endpoint_spikes >= 3) & (tf.half_endpoint_active_cells >= 2)
    support_summary = tf[supported].groupby(["animal", "session", "population_replicate"]).endpoint_mean_shift_cm.median().groupby(["animal", "session"]).mean().groupby("animal").mean()
    support_summary.to_csv(root/"posthoc_supported_endpoint_by_rat.csv")
    audit = dict(status="pass", dense_sample_rows=len(sample_rows),
        accepted_segment_pair_rows=len(trajectory_rows), accepted_unique_events=tf.window_uid.nunique(),
        max_dense_error=float(pd.DataFrame(sample_rows).max_absolute_error.max()),
        accepted_segment_equal_rat_mean_shift_cm=ts.endpoint_mean_shift_cm.mean(),
        posthoc_supported_endpoint_pair_rows=int(supported.sum()),
        posthoc_supported_endpoint_equal_rat_mean_shift_cm=support_summary.mean(),
        code_file_sha256=digest(__file__), source_manifest_sha256=digest(root/"manifest.json"),
        claim_boundary="post-run endpoint-anchor diagnostic; original primary metrics unchanged")
    (root/"independent_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
