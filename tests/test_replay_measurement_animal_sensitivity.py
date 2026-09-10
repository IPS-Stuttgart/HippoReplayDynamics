from itertools import product

import numpy as np
import pandas as pd
import pytest

from scripts.report_replay_measurement_animal_sensitivity import (
    CONTRASTS,
    DATASETS,
    animal_rows,
    checked,
    collect_values,
    file_sha256,
    sign_sensitivity,
    summarize,
)


@pytest.mark.parametrize("n,one,two", [(4, 0.0625, 0.125), (5, 0.03125, 0.0625)])
def test_exact_small_sample_limits(n, one, two):
    result = sign_sensitivity(np.arange(1, n + 1), 1)
    assert result["sign_p_one_sided"] == one
    assert result["sign_p_two_sided"] == two
    assert result["minimum_attainable_one_sided_p"] == one
    assert result["minimum_attainable_two_sided_p"] == two
    assert sign_sensitivity(-np.arange(1, n + 1), -1) == result | {"mean": -(n + 1) / 2}


def test_ties_and_all_ties_are_explicit():
    result = sign_sensitivity([0, 1, 2, 0], 1)
    assert result["zero_ties"] == 2
    assert result["sign_p_one_sided"] == 0.25
    result = sign_sensitivity([0, 0, 0, 0], 1)
    assert result["sign_p_one_sided"] == 1
    assert result["animals_supporting_direction"] == 0


@pytest.mark.parametrize("values,direction", [([], 1), ([1], 1), ([1, np.nan], 1), ([1, np.inf], 1), ([[1, 2]], 1), ([1, 2], 0), ([1, 2], 0.5)])
def test_invalid_sign_inputs_fail(values, direction):
    with pytest.raises(ValueError):
        sign_sensitivity(values, direction)


def endpoint(values):
    return pd.DataFrame(
        [
            {"dataset": "pfeiffer_foster", "family": "prediction", "endpoint": "test", "units": "nats", "animal": f"Rat{i}", "value": v, "direction": 1, "events": 10**i}
            for i, v in enumerate(values)
        ]
    )


def test_animal_influence_not_event_weighting():
    data = endpoint([-1, -1, -1, 7])
    summary, loo = summarize(data)
    assert summary.iloc[0]["mean"] == 1
    assert summary.iloc[0].loo_mean_min == -1
    assert summary.iloc[0].loo_mean_max == pytest.approx(5 / 3)
    assert not summary.iloc[0].all_loo_means_support_direction
    assert loo.loc[loo.excluded_animal.eq("Rat3"), "mean"].iloc[0] == -1
    data["events"] = [1000000, 0, 5, 8]
    pd.testing.assert_frame_equal(summary, summarize(data)[0])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "null_identity"])
def test_invalid_animal_selection(mutation):
    frame = endpoint([1, 2, 3, 4])
    if mutation == "missing":
        frame = frame.iloc[:-1]
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, "animal"] = None
    with pytest.raises(ValueError):
        animal_rows(frame, {"dataset": "pfeiffer_foster"}, 4)


def test_summary_rejects_duplicate_and_incomplete():
    frame = endpoint([1, 2, 3, 4])
    for invalid in [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])]:
        with pytest.raises(ValueError):
            summarize(invalid)


def test_changed_source_hash_fails(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("value\n1\n")
    digest = file_sha256(path)
    assert checked(path, digest) == path
    path.write_text("value\n2\n")
    with pytest.raises(ValueError, match="checksum"):
        checked(path, digest)


def sources():
    coverage, prediction, gradients = [], [], []
    references = {k: [] for k in ["shuffle_baseline", "prediction", "map_mismatch"]}
    for dataset, (_, count) in DATASETS.items():
        ripple = "native_ripple_table" if dataset == "pfeiffer_foster" else "lfp_ripple_detected"
        for detector in ["source_high_mua", ripple]:
            f = {"dataset": dataset, "detector": detector, "observation": "original_order", "bin_filter": "edge_only", "min_frames": 10, "alpha": 0.02}
            coverage.extend(f | {"animal": f"Rat{i}", "accepted_fraction_delta": -0.1} for i in range(count))
            references["shuffle_baseline"].append(f | {"half_minus_full_pp": -10})
        for group, contrast in product(["rejected_with_opportunity", "lost_with_thinning"], CONTRASTS):
            f = {"dataset": dataset, "support": "parent", "bin_filter": "edge_only", "min_frames": 10, "group": group, "contrast": contrast}
            values = np.ones(count)
            if dataset == "tanni2022" and contrast == "imm_minus_composition":
                values[:2] = -2
            prediction.extend(f | {"animal": f"Rat{i}", "delta": v, "delta_per_heldout_spike": v / 10} for i, v in enumerate(values))
            references["prediction"].append(f | {"mean": values.mean(), "mean_per_heldout_spike": values.mean() / 10})
        f = {
            "dataset": dataset,
            "observation": "poisson",
            "cell_fraction": 1.0,
            "likelihood": "poisson",
            "estimator": "posterior_mean",
            "bin_filter": "unfiltered",
            "selection": "all",
            "coordinate": "true_coordinate",
        }
        for decoder, readout, value in [
            ("independent_RUN_half", "decoded", 0.3),
            ("generator_known", "decoded", 0.5),
            ("generator_known", "true_chord", 0.8),
            ("generator_known", "true_arclength", 1.0),
        ]:
            selected = f | {"decoder_map": decoder, "readout": readout}
            gradients.extend(selected | {"animal": f"Rat{i}", "gradient_response": value} for i in range(count))
            references["map_mismatch"].append(selected | {"metric": "gradient_response", "mean": value})
    return tuple(pd.DataFrame(x) for x in [coverage, prediction, gradients]) + ({k: pd.DataFrame(v) for k, v in references.items()},)


def test_all_endpoints_preserve_adverse_contrast_and_scales():
    values = collect_values(*sources())
    summary, loo = summarize(values)
    assert len(values) == 225
    assert len(loo) == 225
    assert len(summary) == 50
    adverse = summary[summary.dataset.eq("tanni2022") & summary.endpoint.str.contains("imm_minus_composition")]
    assert len(adverse) == 4
    assert (adverse["mean"] < 0).all()
    assert (adverse.animals_supporting_direction == 3).all()
    assert (adverse.sign_p_one_sided == 0.5).all()
    assert np.allclose(summary.loc[summary.family.eq("continuity"), "mean"], -10)


def test_point_reconstruction_and_identity_checks():
    coverage, prediction, gradients, reference = sources()
    prediction.loc[0, "delta"] += 1
    with pytest.raises(ValueError, match="reconstruct"):
        collect_values(coverage, prediction, gradients, reference)
    coverage, prediction, gradients, reference = sources()
    prediction["animal"] = prediction.animal + "_unmatched"
    with pytest.raises(ValueError, match="identities"):
        collect_values(coverage, prediction, gradients, reference)
