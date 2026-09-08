import pandas as pd
import pytest

from scripts.report_hc11_native_maze_transfer import CONTRASTS, compact_table, write_figure


def fixture():
    return pd.DataFrame(
        [
            {"unit_regime": u, "count_regime": c, "encoding_variant": "direction_mixture", "contrast": k, "equal_animal_mean": 1.0, "ci_low": -0.2, "ci_high": 2.0}
            for u in ("train_only_qc", "frozen_parent_units")
            for c in ("native", "sleep_total_cap")
            for k in CONTRASTS
        ]
    )


def test_keeps_negative_interval_sensitivity():
    table = compact_table(fixture())
    assert len(table) == 16
    assert (table.ci_low < 0).all()


@pytest.mark.parametrize("corruption", ["empty", "duplicate", "missing", "nan"])
def test_incomplete(corruption):
    table = fixture()
    if corruption == "empty":
        table = table.iloc[:0]
    elif corruption == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif corruption == "missing":
        table = table.iloc[1:]
    else:
        table.loc[0, "equal_animal_mean"] = float("nan")
    with pytest.raises(ValueError):
        compact_table(table)


def test_figure(tmp_path):
    table = fixture()
    animal = pd.concat([table.assign(rat=rat, delta=i / 4) for i, rat in enumerate(("Achilles", "Buddy", "Cicero", "Gatsby"))])
    path = tmp_path / "plot.png"
    write_figure(table, animal, path)
    assert path.stat().st_size > 10000
