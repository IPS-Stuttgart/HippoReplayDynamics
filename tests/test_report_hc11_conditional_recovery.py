import numpy as np
import pandas as pd
import pytest

from scripts import report_hc11_conditional_recovery as report


def fixture_tables():
    panels = pd.MultiIndex.from_product(
        [("POST", "PRE"), ("direction_mixture", "pooled"), report.GENERATORS, report.PRIMARY, range(50)], names=["phase", "encoding_variant", "generator", "contrast", "replicate"]
    ).to_frame(index=False)
    panels["equal_animal_mean"] = panels.replicate / 50
    patterns = (
        panels[["phase", "encoding_variant", "generator"]]
        .drop_duplicates()
        .assign(replicates=50, positive_pattern_count=40, positive_pattern_fraction=0.8, binomial_ci_low=0.65, binomial_ci_high=0.90)
    )
    real = (
        panels[["phase", "encoding_variant", "contrast"]].drop_duplicates().assign(inference_temperature=1.0, equal_animal_mean_event_median_delta=0.02, ci_low=-0.2, ci_high=0.3)
    )
    return panels, patterns, real


def test_all_conditions_reported():
    result = report.build_readout(*fixture_tables())
    assert len(result) == 32
    assert np.allclose(result.simulated_median, 0.49)
    assert result.real_effect.eq(0.02).all()


@pytest.mark.parametrize("kind", ["empty", "missing", "duplicate_real", "missing_generator", "nonfinite"])
def test_incomplete_readout_rejected(kind):
    panels, patterns, real = fixture_tables()
    if kind == "empty":
        panels = panels.iloc[:0]
    elif kind == "missing":
        panels = panels.iloc[1:]
    elif kind == "duplicate_real":
        real = pd.concat([real, real.iloc[:1]])
    elif kind == "missing_generator":
        panels = panels[panels.generator != "diffusion"]
    else:
        panels.loc[0, "equal_animal_mean"] = np.nan
    with pytest.raises(ValueError):
        report.build_readout(panels, patterns, real)


def test_information_is_not_claimed_fully_matched():
    observed = pd.DataFrame({"phase": ["POST"] * 3, "n_spikes": [20] * 3, "n_train_spikes": [14] * 3, "n_heldout_spikes": [6] * 3, "n_active_units": [8] * 3})
    simulated = pd.DataFrame(
        {
            "phase": ["POST"] * 50,
            "generator": ["diffusion"] * 50,
            "replicate": range(50),
            "n_spikes": [20] * 50,
            "n_train_spikes": [13] * 50,
            "n_heldout_spikes": [7] * 50,
            "n_active_units": [11] * 50,
        }
    )
    result = report.information_comparison(simulated, observed).set_index("metric")
    assert result.loc["n_spikes", "real_split_median"] == result.loc["n_spikes", "median_simulated_dataset_median"]
    assert result.loc["n_active_units", "real_split_median"] != result.loc["n_active_units", "median_simulated_dataset_median"]
