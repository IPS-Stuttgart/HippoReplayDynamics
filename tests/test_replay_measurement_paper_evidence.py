"""Source extraction must not silently drop datasets or change estimands."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.report_replay_measurement_paper_evidence import checked, collect_panels, one


def sources():
    tables = {k: [] for k in ["shuffle_baseline", "prediction", "map_mismatch", "geometry"]}
    for dataset, n in [("pfeiffer_foster", 4), ("tanni2022", 5)]:
        tables["shuffle_baseline"].append(
            {
                "dataset": dataset,
                "detector": "source_high_mua",
                "observation": "original_order",
                "animals_with_events": n,
                "eligible_events": 50,
                "significant_full_percent": 20,
                "significant_half_percent": 8,
                "half_minus_full_pp": -12,
                "delta_ci_low_pp": -15,
                "delta_ci_high_pp": -9,
            }
        )
        for contrast in ["imm_minus_iid", "imm_minus_static", "imm_minus_composition", "imm_order_advantage", "imm_order_map_interaction"]:
            tables["prediction"].append(
                {
                    "dataset": dataset,
                    "support": "parent",
                    "bin_filter": "edge_only",
                    "min_frames": 10,
                    "group": "rejected_with_opportunity",
                    "contrast": contrast,
                    "animals": n,
                    "events": 40,
                    "mean": 1.0,
                    "ci_low": -0.1 if dataset == "tanni2022" and contrast == "imm_minus_composition" else 0.2,
                    "ci_high": 2.0,
                }
            )
        for decoder_map, readout in [("generator_known", "true_arclength"), ("generator_known", "true_chord"), ("generator_known", "decoded"), ("independent_RUN_half", "decoded")]:
            tables["map_mismatch"].append(
                {
                    "dataset": dataset,
                    "observation": "poisson",
                    "cell_fraction": 1.0,
                    "likelihood": "poisson",
                    "estimator": "posterior_mean",
                    "bin_filter": "unfiltered",
                    "selection": "all",
                    "coordinate": "true_coordinate",
                    "metric": "gradient_response",
                    "decoder_map": decoder_map,
                    "readout": readout,
                    "animals_available": n,
                    "animals_expected": n,
                    "mean": 0.3,
                    "ci95_low": 0.1,
                    "ci95_high": 0.5,
                }
            )
    for w in [20, 40]:
        for kind, metric in [("continuous", "eligible_recovery_fraction"), ("shuffled_continuous", "acceptance_fraction")]:
            tables["geometry"].append(
                {
                    "area_m2": 8.75,
                    "aspect": 1.4,
                    "sigma_cm": 30.0,
                    "n_cells": 128,
                    "support_domain": "full_arena",
                    "grid_cm": 8.0,
                    "stride_ms": 5.0,
                    "observation": "native_poisson",
                    "likelihood": "poisson",
                    "estimator": "map",
                    "bin_filter": "unfiltered",
                    "continuity_rule": "literal_20cm_10frames",
                    "gradient": 0.0,
                    "window_ms": w,
                    "truth_kind": kind,
                    "metric": metric,
                    "populations_available": 8,
                    "populations_expected": 8,
                    "mean": 0.1,
                    "ci95_low": 0.05,
                    "ci95_high": 0.2,
                }
            )
    return {k: pd.DataFrame(v) for k, v in tables.items()}


def test_extracts_all_panels_and_retains_nonpositive_interval():
    tables = sources()
    out = collect_panels(tables)
    assert len(out) == 28
    assert set(out.panel) == set("ABCD")
    failed = one(out, {"panel": "B", "dataset": "tanni2022", "condition": "imm_minus_composition"})
    assert failed.ci_low < 0
    assert out[out.panel.eq("C")].value.eq(10).all()
    assert out[out.condition.eq("full")].value.eq(20).all()
    for r in out.itertuples(index=False):
        source = one(tables[r.source_table], json.loads(r.source_filter))
        assert r.value == float(source[r.source_field]) * r.scale


@pytest.mark.parametrize("table", ["prediction", "geometry", "shuffle_baseline", "map_mismatch"])
def test_missing_or_duplicate_row_fails(table):
    tables = sources()
    original = tables[table]
    tables[table] = original.iloc[1:]
    with pytest.raises(ValueError, match="exactly one"):
        collect_panels(tables)
    tables[table] = pd.concat([original, original.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        collect_panels(tables)


def test_incomplete_animals_and_invalid_values_fail():
    tables = sources()
    tables["prediction"].loc[0, "animals"] = 3
    with pytest.raises(ValueError, match="incomplete"):
        collect_panels(tables)
    tables = sources()
    tables["geometry"].loc[0, "mean"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        collect_panels(tables)


def test_changed_source_hash_fails(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        checked(source, "0" * 64)
