#!/usr/bin/env python3
"""Frozen non-rescoring within-animal arena-scale diagnostic."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from report_2d_geometry_predictive_dissociation import validate

RUN_HASH = "8861c60bf36b98ce179237276e8d16cb6294c1d71077c9100b7f4776c36237c0"
SOURCE_HASH = "2ed1c4d910a3ddc4027eefb398d48a55bae6829c3bf16e8f57955beb9ee55613"
KEY = ["dataset", "animal", "session", "event_id", "split"]
PRIMARY = ("geometric_fraction", "imm_order_advantage_per_spike", "imm_order_map_interaction_per_spike")
METRICS = (*PRIMARY, "imm_order_advantage", "imm_order_map_interaction", "imm_minus_event_global", "imm_minus_event_global_per_spike", "diffusion_order_advantage_per_spike")
ANIMALS = {"R2470", "R2474", "R2478", "R2481", "R2482"}


def area_design(metadata):
    m = metadata.copy().sort_values(["animal", "session"])
    if m.duplicated(["animal", "session"]).any() or set(m.animal) != ANIMALS:
        raise ValueError("unique complete five-animal metadata required")
    area = m.arena_area_m2.to_numpy(float)
    if not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError("finite positive native arena area required")
    log_area = np.log2(area / (87.5 * 125 / 10000))
    if not np.allclose(log_area, np.round(log_area), atol=0.03, rtol=0):
        raise ValueError("unexpected arena size")
    m["area_level"] = np.round(log_area).astype(int)
    m["visit_index"] = m.groupby("animal").cumcount()
    for _, g in m.groupby("animal"):
        if len(g) != 5 or sorted(g.area_level) != [0, 0, 1, 2, 3] or g.iloc[0].area_level != 0 or g.iloc[-1].area_level != 0:
            raise ValueError("expected A, randomized B/C/D, A-return design")
    m["environment"] = m.area_level.map(dict(enumerate("ABCD")))
    m["visit_role"] = np.where(m.visit_index.eq(4), "A_return", m.environment)
    return m


def holm(probabilities):
    p = np.asarray(probabilities, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("finite probabilities required")
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def slopes_and_reference(sessions, metric, include_a=False):
    rows, references = [], []
    for animal, g in sessions.groupby("animal", sort=True):
        g = g if include_a else g[g.area_level.gt(0)]
        y = g.groupby("area_level")[metric].mean().sort_index()
        expected = [0, 1, 2, 3] if include_a else [1, 2, 3]
        if list(y.index) != expected or not np.isfinite(y).all():
            raise ValueError("complete finite per-area endpoints required")
        x = y.index.to_numpy(float)
        xc = x - x.mean()
        value = float(xc @ y.to_numpy() / (xc @ xc))
        rows.append({"animal": animal, "metric": metric, "slope": value})
        if not include_a:
            references.append([float((np.array(p) - 2) @ y.to_numpy() / 2) for p in itertools.permutations((1, 2, 3))])
    values = np.array([r["slope"] for r in rows])
    mean = float(values.mean())
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    reference_p = np.nan
    if not include_a:
        null = np.array(list(itertools.product(*references))).mean(axis=1)
        reference_p = float(np.mean(np.abs(null) >= abs(mean) - 1e-12))
    return pd.DataFrame(rows), {
        "metric": metric,
        "mean_slope": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "positive_animals": int((values > 0).sum()),
        "negative_animals": int((values < 0).sum()),
        "animals": len(values),
        "exact_label_reference_p": reference_p,
        "label_reference_draws": 6 ** len(values) if not include_a else 0,
        "leave_one_animal_out_min": float(min((values.sum() - v) / (len(values) - 1) for v in values)),
        "leave_one_animal_out_max": float(max((values.sum() - v) / (len(values) - 1) for v in values)),
    }


def adjusted_slope(sessions, metric, entropy=False):
    g = sessions[sessions.area_level.gt(0)].copy()
    y = g[metric].to_numpy(float)
    area = g.area_level.to_numpy(float)
    cov = [np.log(g.n_train_cells.to_numpy(float)), np.log1p(g.median_train_spikes.to_numpy(float)), np.log(g.median_duration_s.to_numpy(float))]
    if entropy:
        cov.append(g.mean_normalized_training_entropy.to_numpy(float))
    if not np.isfinite(np.column_stack([y, area, *cov])).all():
        return {"status": "missing_covariates", "coefficient": np.nan}
    controls = np.column_stack([np.ones(len(g)), pd.get_dummies(g.animal, drop_first=True).to_numpy(float), *cov])
    scales = controls.std(axis=0)
    standardized = controls.copy()
    vary = scales > 1e-12
    standardized[:, vary] = (controls[:, vary] - controls[:, vary].mean(axis=0)) / scales[vary]
    design = np.column_stack([standardized, area])
    rank = np.linalg.matrix_rank(design)
    residual = area - standardized @ np.linalg.lstsq(standardized, area, rcond=None)[0]
    within_area = area - g.groupby("animal").area_level.transform("mean").to_numpy()
    result = {
        "status": "ok",
        "rows": len(g),
        "parameters": design.shape[1],
        "rank": int(rank),
        "condition_number": float(np.linalg.cond(design)),
        "area_residual_fraction": float(residual @ residual / (within_area @ within_area)),
        "coefficient": np.nan,
    }
    if rank != design.shape[1] or len(g) <= rank:
        result["status"] = "rank_or_degrees_of_freedom_failure"
    else:
        result["coefficient"] = float(np.linalg.lstsq(design, y, rcond=None)[0][-1])
    return result


def join_and_aggregate(labels, predictions, metadata):
    labels = labels[labels.dataset.eq("tanni2022") & labels.criterion.eq("edge10")].copy()
    predictions = predictions[predictions.dataset.eq("tanni2022")].copy()
    if labels.duplicated(KEY).any() or predictions.duplicated(KEY).any():
        raise ValueError("duplicate event/split")
    lk, pk = set(labels[KEY].itertuples(index=False, name=None)), set(predictions[KEY].itertuples(index=False, name=None))
    if lk != pk or not lk:
        raise ValueError("unmatched label/prediction coverage")
    if not labels.heldout_used_for_label.eq(False).all() or not labels.geometric_pass.isin([True, False]).all():
        raise ValueError("leaking or malformed geometry labels")
    e = predictions.merge(labels[KEY + ["geometric_pass", "mean_training_entropy_nats"]], on=KEY, validate="one_to_one")
    e = e.merge(metadata, on=["dataset", "animal", "session"], validate="many_to_one")
    if len(e) != len(predictions):
        raise ValueError("missing metadata")
    e["geometric_fraction"] = e.geometric_pass.astype(float)
    e["normalized_training_entropy"] = e.mean_training_entropy_nats / np.log(e.n_valid_bins)
    for raw in ("imm_order_advantage", "imm_order_map_interaction", "imm_minus_event_global", "diffusion_order_advantage"):
        expected = e[raw] / e.n_heldout_spikes.replace(0, np.nan)
        if not np.allclose(expected, e[raw + "_per_spike"], equal_nan=True, atol=1e-10, rtol=1e-10):
            raise ValueError("incorrect per-spike ratio")
    rows = []
    for keys, g in e.groupby(["dataset", "animal", "session", "split"], sort=True):
        first = g.iloc[0]
        row = dict(zip(["dataset", "animal", "session", "split"], keys, strict=True))
        for name in ("area_level", "arena_area_m2", "visit_index", "visit_role", "environment", "native_duration_s", "n_valid_bins"):
            row[name] = first[name]
        for name in ("n_train_cells", "n_heldout_cells"):
            if g[name].nunique() != 1:
                raise ValueError("cell partition changed within session")
            row[name] = int(first[name])
        row.update(
            events=len(g),
            zero_heldout_events=int(g.n_heldout_spikes.eq(0).sum()),
            median_train_spikes=float(g.n_train_spikes.median()),
            median_heldout_spikes=float(g.n_heldout_spikes.median()),
            median_duration_s=float(g.duration_s.median()),
            mean_normalized_training_entropy=float(g.normalized_training_entropy.mean()),
        )
        for metric in METRICS:
            row[metric] = float(g[metric].mean())
            row[metric + "_events"] = int(g[metric].notna().sum())
        rows.append(row)
    return e, pd.DataFrame(rows)


def native_metadata(source):
    table = pd.read_csv(source / "coverage_input_sessions.csv")
    rows = []
    for r in table[table.dataset.eq("tanni2022")].itertuples(index=False):
        artifact = Path(r.artifact_path)
        record = json.loads(artifact.with_suffix(".json").read_text())
        with h5py.File(record["source_path"], "r") as f:
            size = np.asarray(f["/general/data_collection/Settings/General/arena_size"][()], float)
            animal = f["/general/data_collection/Settings/General/animal"][()]
        if isinstance(animal, bytes):
            animal = animal.decode()
        bounds = np.asarray(record["arena_bounds_cm"], float)
        if str(animal) != r.animal or size.shape != (2,) or not np.allclose(size, bounds[1] - bounds[0], rtol=0, atol=1e-8):
            raise ValueError("native arena identity/size differs from source cache")
        with np.load(artifact) as z:
            valid = np.asarray(z["valid_spatial_bins"], bool)
            n_valid = int(valid.sum())
        area = float(np.prod(size) / 10000)
        if not np.isclose(area, r.arena_area_m2) or n_valid < 2:
            raise ValueError("invalid source spatial metadata")
        rows.append(
            {
                "dataset": r.dataset,
                "animal": r.animal,
                "session": r.session,
                "arena_area_m2": area,
                "native_width_cm": float(size[0]),
                "native_height_cm": float(size[1]),
                "native_duration_s": float(record["position_end_s"] - record["position_start_s"]),
                "n_valid_bins": n_valid,
                "native_nwb": record["source_path"],
                "source_record_sha256": file_sha256(artifact.with_suffix(".json")),
            }
        )
    return area_design(pd.DataFrame(rows))


def figure(sessions, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    labels = ("Geometry accepted (%)", "Original-order benefit\n(nats per held-out spike)", "Order x map interaction\n(nats per held-out spike)")
    colors = ("#157f79", "#bb465c", "#4675ad", "#9c7836", "#7b598f")
    primary = sessions[sessions.split.eq(0)]
    for ax, metric, title in zip(axes, PRIMARY, labels, strict=True):
        for color, (animal, g) in zip(colors, primary.groupby("animal", sort=True), strict=True):
            g = g.groupby("area_level")[metric].mean()
            scale = 100 if metric == "geometric_fraction" else 1
            ax.plot(g.index, g * scale, marker="o", ms=4, color=color, label=animal)
        ax.set_xticks(range(4), ["A\n1.09", "B\n2.19", "C\n4.38", "D\n8.75"])
        ax.set_xlabel("Environment / area (m2)")
        ax.set_ylabel(title)
        ax.axhline(0, lw=0.7, color="gray")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Tanni: environment scale versus geometry and held-out temporal prediction", fontsize=13)
    fig.supxlabel(
        "One line per animal; sessions average events equally. A averages first and return visits.\nPrimary inference uses randomized B/C/D visits only. These are not decoded physical-speed measurements.",
        fontsize=9,
    )
    fig.savefig(out / "tanni_environment_scale_structure.png", dpi=180)
    plt.close(fig)


def run(args):
    root, audit, source, out = (Path(p).resolve() for p in (args.run_dir, args.audit_manifest, args.source_dir, args.output_dir))
    if file_sha256(root / "geometry_predictive_manifest.json") != RUN_HASH or file_sha256(source / "coverage_input_manifest.json") != SOURCE_HASH:
        raise ValueError("wrong frozen inputs")
    validate(root, audit)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite analysis")
    metadata = native_metadata(source)
    labels, prediction = (pd.read_csv(root / f"geometry_predictive_{name}.csv") for name in ("labels", "predictions"))
    events, sessions = join_and_aggregate(labels, prediction, metadata)
    if len(events) != 5224 * 5 or len(sessions) != 25 * 5 or set(sessions.split) != set(range(5)):
        raise ValueError("incomplete Tanni cohort")
    summaries, animal_rows, adjusted, repeats = [], [], [], []
    for split, subset in sessions.groupby("split", sort=True):
        for include_a in (False, True):
            for metric in METRICS:
                animals, summary = slopes_and_reference(subset, metric, include_a)
                scope = "ABCD_A_averaged" if include_a else "BCD_primary"
                summaries.append(summary | {"split": split, "scope": scope, "primary": split == 0 and not include_a and metric in PRIMARY})
                animal_rows.append(animals.assign(split=split, scope=scope))
        if split == 0:
            for metric in METRICS:
                for entropy in (False, True):
                    for omitted in ("none", *sorted(ANIMALS)):
                        part = subset if omitted == "none" else subset[~subset.animal.eq(omitted)]
                        adjusted.append(adjusted_slope(part, metric, entropy) | {"metric": metric, "entropy_control": entropy, "omitted_animal": omitted})
                for animal, g in subset.groupby("animal", sort=True):
                    a = g[g.area_level.eq(0)].sort_values("visit_index")
                    repeats.append(
                        {"animal": animal, "metric": metric, "first": a.iloc[0][metric], "return": a.iloc[-1][metric], "return_minus_first": a.iloc[-1][metric] - a.iloc[0][metric]}
                    )
    summary = pd.DataFrame(summaries)
    summary["primary_holm_reference_p"] = np.nan
    mask = summary.primary
    summary.loc[mask, "primary_holm_reference_p"] = holm(summary.loc[mask, "exact_label_reference_p"])
    adjusted = pd.DataFrame(adjusted)
    out.mkdir(parents=True)
    frames = {
        "metadata": metadata,
        "events": events,
        "sessions": sessions,
        "summary": summary,
        "animal_slopes": pd.concat(animal_rows, ignore_index=True),
        "adjusted": adjusted,
        "A_repeat": pd.DataFrame(repeats),
    }
    for name, frame in frames.items():
        frame.to_csv(out / f"tanni_environment_scale_{name}.csv", index=False)
    figure(sessions, out)
    lines = [
        "# Tanni environment scale and predictive structure",
        "",
        "Exploratory, non-rescoring analysis of frozen events and proper held-out scores.",
        "",
        "## Primary B/C/D results",
        "",
        "Slopes per doubling of arena area. Equal-animal means and t95% intervals (five animals); exact within-animal label-permutation references, Holm over three endpoints.",
        "",
        "|Metric|Slope|95% interval|Positive/negative animals|Holm reference p|",
        "|---|---:|---|---|---:|",
    ]
    for r in summary[summary.primary].itertuples(index=False):
        lines.append(f"|{r.metric}|{r.mean_slope:+.6f}|[{r.ci_low:+.6f}, {r.ci_high:+.6f}]|{r.positive_animals}/{r.negative_animals}|{r.primary_holm_reference_p:.6f}|")
    lines += [
        "",
        "## Support-adjusted diagnostics",
        "",
        "Session regressions condition on animal, training cell count, median training spike count and event duration. A separate sensitivity adds normalized training entropy; these are not causal mediation estimates. All coefficients and leave-one-animal-out/rank diagnostics are in the adjusted table.",
        "",
    ]
    for r in adjusted[adjusted.omitted_animal.eq("none") & adjusted.metric.isin(PRIMARY)].itertuples(index=False):
        lines.append(f"- {r.metric}; entropy={r.entropy_control}: coefficient {r.coefficient:+.6f}, area residual fraction {r.area_residual_fraction:.4f}, status {r.status}.")
    lines += [
        "",
        "## Boundaries",
        "",
        "All5,224 events/25recordings/fiveanimals remain included; no geometry or evidence-positive selection. Primary split0 only; other cell splits stay separate. Ratios with zero held-out spikes are missing and counted, not set to zero. Each session has equal weight within the animal size contrast, so the large environment's longer recording does not dominate by event count.",
        "",
        "B/C/D visit order was randomized in the source design, but environment identity and recording/exposure duration change with area. The comparison cannot isolate physical size causally. A-first versus A-return is a separate time/exposure diagnostic; A is not part of the primary randomized-visit comparison.",
        "",
        "The targets are geometry acceptance and independently predictive order, NOT physical replay speed. A nonsignificant area slope does not establish invariance or equivalence. Positive order sensitivity does not rescue the parent's failed Tanni other-event-composition baseline. Reused data, multiple prior hypotheses and five animals make this an exploratory lead, not independent biological confirmation.",
        "",
    ]
    (out / "tanni_environment_scale_report.md").write_text("\n".join(lines))
    p = build_script_provenance(
        input_paths={
            "parent_manifest": root / "geometry_predictive_manifest.json",
            "parent_audit": audit,
            "source_manifest": source / "coverage_input_manifest.json",
            "source_sessions": source / "coverage_input_sessions.csv",
            "protocol": ROOT / "docs/tanni_environment_scale_protocol.md",
        }
    )
    p.update(status="complete", non_rescoring=True, events=5224, sessions=25, animals=5, biological_speed_claim=False, independent_confirmation=False)
    p["output_sha256"] = {f.name: file_sha256(f) for f in out.iterdir()}
    (out / "tanni_environment_scale_manifest.json").write_text(json.dumps(p, indent=2) + "\n")
    print(summary[summary.primary].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
