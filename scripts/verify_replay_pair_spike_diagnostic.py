"""Independently verify supplied-truth prediction and training-support diagnostics."""
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
from scripts.verify_replay_pair_prediction_development import (  # noqa: E402
    close, reference_predictions, reference_weights,
)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def reference_architecture(order, a, b):
    position = np.argsort(order)
    difference = position[b] - position[a]
    return np.where(np.abs(difference) == 1, np.sign(difference), 0).astype(float)


def reference_support(training, target, weights):
    scale = np.sqrt(np.einsum("i,ij,ij->j", weights / weights.sum(), training, training))
    scale[scale == 0] = 1
    design = training / scale * np.sqrt(weights[:, None] / weights.sum())
    projection = np.linalg.pinv(design, rcond=max(design.shape) * np.finfo(float).eps) @ design
    target_scaled = target / scale
    loading = np.linalg.norm(target_scaled - target_scaled @ projection, axis=1)
    guard = 100 * np.finfo(float).eps * max(design.shape) * np.maximum(1, np.linalg.norm(target_scaled, axis=1))
    return round(float(np.trace(projection))), float(loading.max(initial=0)), int(np.sum(loading > guard))


def verify(args):
    diagnostic_path = args.diagnostic / "manifest.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    original_path = args.results / "manifest.json"
    original = json.loads(original_path.read_text())
    if diagnostic["input_file_sha256"]["development_manifest"] != file_sha256(original_path):
        raise ValueError("Diagnostic input identity changed")
    for key in ("new_spikes_generated", "primary_calibration_changed", "real_association_fit",
                "biological_inference_authorized", "goal_complete"):
        if diagnostic.get(key) is not False:
            raise ValueError("Supplied-truth diagnostic claim boundary changed")
    for directory, record in ((args.results, original), (args.diagnostic, diagnostic)):
        for name, digest in record["outputs_sha256"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("Changed input/output bank")
    for name, digest in diagnostic["input_file_sha256"].items():
        if file_sha256(diagnostic["input_file_paths"][name]) != digest:
            raise ValueError("Changed diagnostic provenance input")
    protocol = json.loads(Path(diagnostic["input_file_paths"]["spike_protocol"]).read_text())
    inventory = rows(args.results / "inventory.csv")
    summary = rows(args.diagnostic / "supplied_truth_diagnostic.csv")
    animals = rows(args.diagnostic / "animal_support_diagnostic.csv")
    if len(summary) != 24 or len(animals) != 96:
        raise ValueError("Incomplete supplied-truth/animal denominator")
    entries = 0
    for generator in original["generators"]:
        with np.load(args.results / f"{generator}_predictions.npz", allow_pickle=False) as saved:
            architecture = np.zeros(len(saved["pre"]))
            if generator in {"existing_coordination_only", "order_specific_update"}:
                for row in [r for r in inventory if r["generator"] == generator]:
                    selected = saved["pause"] == row["pause"]
                    with np.load(args.results / row["bank"], allow_pickle=False) as bank:
                        architecture[selected] = reference_architecture(bank["generating_event_order"],
                            saved["unit_a"][selected], saved["unit_b"][selected])
            for endpoint in ("fitted", "oracle"):
                pre = saved["pre"] if endpoint == "fitted" else saved["oracle_pre"]
                post = saved["post"] if endpoint == "fitted" else saved["oracle_post"]
                baseline = np.column_stack((pre,
                    np.log1p(saved["pre_rate_a"]) - np.log1p(saved["pre_rate_b"]),
                    pre * np.log1p(saved["pre_rate_a"] + saved["pre_rate_b"]),
                    np.log1p(saved["event_spikes_a"]) - np.log1p(saved["event_spikes_b"]),
                    pre * saved["participation"]))
                for feature_name, feature in (("measured_event_order", saved["order"][:, 0]),
                                              ("known_copying_architecture", architecture)):
                    fields = dict(saved)
                    fields.update(pre=pre, post=post, baseline=baseline, order=feature[:, None])
                    prediction, base = reference_predictions(fields, protocol["order_penalty"])
                    gains = ((post - pre)[:, None] - base) ** 2 - ((post - pre)[:, None] - prediction) ** 2
                    recorded = [r for r in summary if (r["generator"], r["endpoint"], r["order_feature"])
                                == (generator, endpoint, feature_name)]
                    if len(recorded) != 1:
                        raise ValueError("Missing or duplicate diagnostic condition")
                    values = []
                    for animal in sorted(set(saved["animal"])):
                        test = saved["animal"] == animal
                        train = ~test
                        weight = reference_weights(saved["animal"][test], saved["session"][test], saved["pause"][test])
                        value = float(weight @ gains[test, 0])
                        values.append(value)
                        recorded_animal = [r for r in animals if
                            (r["generator"], r["endpoint"], r["order_feature"], r["heldout_animal"])
                            == (generator, endpoint, feature_name, animal)]
                        if len(recorded_animal) != 1:
                            raise ValueError("Missing or duplicate animal diagnostic")
                        row = recorded_animal[0]
                        close(value, float(row["heldout_gain"]), "independent diagnostic gain")
                        training_weight = reference_weights(saved["animal"][train], saved["session"][train], saved["pause"][train])
                        rank, loading, unsupported = reference_support(baseline[train], baseline[test], training_weight)
                        if rank != int(row["baseline_training_rank"]) or int(row["heldout_pair_rows"]) != test.sum():
                            raise ValueError("Support or animal denominator differs")
                        if int(row["heldout_zero_event_rows"]) != np.sum(test & (saved["participation"] == 0)):
                            raise ValueError("Zero-event observations were omitted")
                        close(loading, float(row["baseline_max_heldout_nullspace_loading"]), "independent support projection")
                        if int(row["baseline_unsupported_heldout_rows"]) != unsupported or (
                                row["baseline_prediction_support_complete"] != str(unsupported == 0)):
                            raise ValueError("Training support-completeness flag differs")
                    close(np.mean(values), float(recorded[0]["animal_balanced_gain"]), "equal-animal diagnostic aggregate")
                    if recorded[0]["feature_has_nonzero_values"] != str(bool(np.any(feature))):
                        raise ValueError("Uninformative known-feature flag differs")
                    entries += prediction.size
    return {**build_script_provenance(input_paths={"diagnostic_manifest": diagnostic_path,
                                                  "development_manifest": original_path}),
        "verified": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_conditions": len(summary), "animal_support_rows": len(animals),
        "independent_prediction_entries": entries,
        "scope": "Independent weighted projection/scalar ridge for 24 supplied-truth conditions and pseudoinverse row-space checks for 96 animal diagnostics; existing count-level verification supplies the upstream endpoint audit.",
        "primary_calibration_verified": False, "real_association_verified": False, "goal_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Do not overwrite independent verification")
    args.output.write_text(json.dumps(verify(args), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
