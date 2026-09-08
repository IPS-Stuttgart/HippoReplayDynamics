import numpy as np
import pandas as pd
import pytest

from scripts.report_hc11_sleep_rate_transfer import AXES, decision, figure


def example():
    return pd.DataFrame(
        [
            {"phase": phase, "encoding_variant": variant, "regime": regime, "contrast": contrast, "mean": 0.2, "ci_low": -0.1, "ci_high": 0.5, "positive_animals": 2, "animals": 4}
            for phase in ("PRE", "POST")
            for variant in ("pooled", "direction_mixture")
            for regime in ("run_original", "sleep_alpha100", "sleep_alpha1000")
            for contrast in AXES
        ]
    )


def test_no_positive_headline_from_static_win():
    table = example()
    static = table.contrast.eq("imm_minus_static")
    table.loc[static, ["mean", "ci_low", "ci_high", "positive_animals"]] = [2, 1, 3, 4]
    assert decision(table) == "external_temporal_advantage_remains_unsupported"


def test_positive_remains_diagnostic():
    table = example().assign(ci_low=0.1, positive_animals=4)
    assert decision(table) == "diagnostic_transfer_lead_requires_replication"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nan", "animal"])
def test_invalid_primary(kind):
    table = example()
    chosen = table.phase.eq("POST") & table.regime.eq("sleep_alpha100") & table.encoding_variant.eq("direction_mixture")
    if kind == "missing":
        table = table[~chosen]
    elif kind == "duplicate":
        table = pd.concat([table, table[chosen]])
    elif kind == "nan":
        table.loc[chosen, "mean"] = np.nan
    else:
        table.loc[chosen, "animals"] = 3
    with pytest.raises(ValueError):
        decision(table)


def test_plot(tmp_path):
    path = tmp_path / "summary.png"
    figure(example(), path)
    assert path.stat().st_size > 10000
