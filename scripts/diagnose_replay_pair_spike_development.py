"""Non-rescoring localization of a failed synthetic order-update recovery.

Oracle endpoints and known copying architecture are supplied-truth diagnostics,
not replacements for measured replay features or primary calibration results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._provenance import build_script_provenance, file_sha256  # noqa: E402
from scripts._replay_pair_prediction import fit, hierarchical_weights, nuisance_features, require  # noqa: E402
from scripts.check_replay_pair_prediction_development import write_csv  # noqa: E402


def known_copying_feature(bank, unit_a, unit_b, generator):
    feature = np.zeros((len(bank["unit_ids"]), len(bank["unit_ids"])))
    if generator in {"existing_coordination_only", "order_specific_update"}:
        order = bank["generating_event_order"]
        for source, target in zip(order[:-1], order[1:], strict=True):
            feature[source, target] = 1
            feature[target, source] = -1
    return feature[unit_a, unit_b]


def diagnostic_gain(fields, pre, post, feature, penalty=1):
    baseline = nuisance_features(pre, fields["pre_rate_a"], fields["pre_rate_b"],
                                fields["event_spikes_a"], fields["event_spikes_b"], fields["participation"])
    outcome = post - pre
    gain, slopes = [], []
    for animal in sorted(set(fields["animal"])):
        train, test = fields["animal"] != animal, fields["animal"] == animal
        w = hierarchical_weights(fields["animal"][train], fields["session"][train], fields["pause"][train])
        base = fit(baseline[train], outcome[train], w)
        augmented = np.column_stack((baseline, feature))
        model = fit(augmented[train], outcome[train], w, order_column=augmented.shape[1] - 1,
                    order_penalty=penalty)
        target_w = hierarchical_weights(fields["animal"][test], fields["session"][test], fields["pause"][test])
        gain.append(target_w @ ((outcome[test] - base.predict(baseline[test])) ** 2 - (
            outcome[test] - model.predict(augmented[test])) ** 2))
        slopes.append(model.coefficient[-1] / model.scale[-1])
    return {"animal_balanced_gain": float(np.mean(gain)), "animals_positive": int(np.sum(np.array(gain) > 0)),
            "mean_order_coefficient": float(np.mean(slopes)), "animals": len(gain)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.results / "manifest.json"
    m, v = json.loads(manifest_path.read_text()), json.loads(args.verification.read_text())
    require(v.get("verified") and v["input_file_sha256"]["manifest"] == file_sha256(manifest_path),
            "Successful identity-matched independent count verification required")
    require(m["real_recordings_used"] is False and m["biological_calibration_complete"] is False,
            "Only the unchanged development simulation may be diagnosed")
    for name, digest in m["outputs_sha256"].items():
        require(file_sha256(args.results / name) == digest, "Changed development output")
    with (args.results / "inventory.csv").open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    with (args.results / "development_summary.csv").open(newline="") as stream:
        original_gains = {r["generator"]: float(r["original_mean_animal_gain"]) for r in csv.DictReader(stream)}
    protocol_path = Path(m["input_file_paths"]["spike_protocol"])
    require(file_sha256(protocol_path) == m["input_file_sha256"]["spike_protocol"],
            "Development protocol changed after simulation")
    p = json.loads(protocol_path.read_text())
    provenance = build_script_provenance(input_paths={"development_manifest": manifest_path,
                                                     "verification": args.verification,
                                                     "spike_protocol": protocol_path})
    require(provenance["git_dirty"] is False, "Clean committed diagnostic checkout required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for generator in m["generators"]:
        with np.load(args.results / f"{generator}_predictions.npz", allow_pickle=False) as fields:
            architecture = np.zeros(len(fields["pre"]))
            for row in [r for r in inventory if r["generator"] == generator]:
                selected = fields["pause"] == row["pause"]
                with np.load(args.results / row["bank"], allow_pickle=False) as bank:
                    architecture[selected] = known_copying_feature(bank, fields["unit_a"][selected],
                                                                   fields["unit_b"][selected], generator)
            for endpoint in ("fitted", "oracle"):
                pre = fields["pre"] if endpoint == "fitted" else fields["oracle_pre"]
                post = fields["post"] if endpoint == "fitted" else fields["oracle_post"]
                for name, feature in (("measured_event_order", fields["order"][:, 0]),
                                      ("known_copying_architecture", architecture)):
                    result = diagnostic_gain(fields, pre, post, feature, p["order_penalty"])
                    if endpoint == "fitted" and name == "measured_event_order":
                        require(np.isclose(result["animal_balanced_gain"], original_gains[generator],
                                           rtol=1e-10, atol=1e-15), "Primary development calculation changed")
                    rows.append({"generator": generator, "endpoint": endpoint, "order_feature": name,
                        "truth_supplied": endpoint == "oracle" or name == "known_copying_architecture",
                        **result, "new_spikes_generated": False, "primary_calibration_result": False})
    write_csv(args.output_dir / "supplied_truth_diagnostic.csv", rows)
    result = {**provenance, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_sha256": {"supplied_truth_diagnostic.csv": file_sha256(args.output_dir / "supplied_truth_diagnostic.csv")},
        "scope": "Post-hoc supplied-truth diagnostic of the unchanged count-development bank; no primary endpoint, generator, threshold or result changed.",
        "new_spikes_generated": False, "primary_calibration_changed": False,
        "real_association_fit": False, "biological_inference_authorized": False, "goal_complete": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("COMPLETE supplied-truth localization; primary calibration remains unchanged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
