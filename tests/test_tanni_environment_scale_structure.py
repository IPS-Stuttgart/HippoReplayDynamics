from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_tanni_environment_scale_structure import (
    ANIMALS,
    METRICS,
    adjusted_slope,
    area_design,
    holm,
    join_and_aggregate,
    slopes_and_reference,
)


def metadata():
    return pd.DataFrame(
        [
            {
                "dataset": "tanni2022",
                "animal": animal,
                "session": f"visit{i}",
                "arena_area_m2": 1.09375 * 2**level,
                "native_duration_s": 100 * 2**level,
                "n_valid_bins": 10 * 2**level,
            }
            for animal in sorted(ANIMALS)
            for i, level in enumerate((0, 2, 1, 3, 0))
        ]
    )


def session_fixture():
    s = area_design(metadata())
    rng = np.random.default_rng(123)
    s["y"] = s.area_level * 2.0 + s.animal.map(dict(zip(sorted(ANIMALS), range(5), strict=True)))
    s["n_train_cells"] = rng.integers(30, 100, len(s))
    s["median_train_spikes"] = rng.integers(4, 90, len(s))
    s["median_duration_s"] = rng.uniform(0.06, 0.3, len(s))
    s["mean_normalized_training_entropy"] = rng.uniform(0.1, 0.9, len(s))
    return s


def test_native_area_assignment_and_repeat():
    m = area_design(metadata().sample(frac=1, random_state=4))
    assert list(m[m.animal.eq("R2470")].environment) == list("ACBDA")
    assert m.visit_role.eq("A_return").sum() == 5
    assert m.environment.eq("A").sum() == 10


@pytest.mark.parametrize("damage", ["missing", "duplicate", "area", "order"])
def test_bad_area_design_fails(damage):
    m = metadata()
    if damage == "missing":
        m = m.iloc[1:]
    elif damage == "duplicate":
        m = pd.concat([m, m.iloc[:1]])
    elif damage == "area":
        m.loc[0, "arena_area_m2"] = 0
    else:
        m.loc[0, "arena_area_m2"] = 4.375
    with pytest.raises(ValueError):
        area_design(m)


def test_known_slope_and_complete_exact_reference():
    a, s = slopes_and_reference(session_fixture(), "y")
    assert np.allclose(a.slope, 2)
    assert s["mean_slope"] == pytest.approx(2)
    assert s["ci_low"] == pytest.approx(2)
    assert s["label_reference_draws"] == 7776
    assert s["exact_label_reference_p"] == pytest.approx(2 / 7776)


def test_constant_outcome_reference_is_one():
    d = session_fixture().assign(y=7)
    _, s = slopes_and_reference(d, "y")
    assert s["mean_slope"] == 0
    assert s["exact_label_reference_p"] == 1


def test_primary_excludes_nonrandomized_A_visits():
    d = session_fixture()
    d.loc[d.area_level.eq(0), "y"] = 10000
    _, primary = slopes_and_reference(d, "y")
    _, sensitivity = slopes_and_reference(d, "y", include_a=True)
    assert primary["mean_slope"] == pytest.approx(2)
    assert sensitivity["mean_slope"] < 0


def test_A_replicates_average_before_slope():
    d = session_fixture()
    d.loc[d.visit_index.eq(0), "y"] -= 10
    d.loc[d.visit_index.eq(4), "y"] += 10
    _, s = slopes_and_reference(d, "y", include_a=True)
    assert s["mean_slope"] == pytest.approx(2)


def test_holm_in_original_order():
    assert np.allclose(holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
    with pytest.raises(ValueError):
        holm([np.nan])


def test_adjustment_recovers_known_coefficient():
    d = session_fixture()
    d["y"] += 3 * np.log(d.n_train_cells) - np.log1p(d.median_train_spikes)
    r = adjusted_slope(d, "y")
    assert r["status"] == "ok"
    assert r["coefficient"] == pytest.approx(2)
    assert 0 < r["area_residual_fraction"] <= 1


def test_adjustment_reports_rank_failure_instead_of_false_effect():
    d = session_fixture().assign(n_train_cells=20)
    assert adjusted_slope(d, "y")["status"] == "rank_or_degrees_of_freedom_failure"


def event_fixture():
    labels, predictions = [], []
    for m in area_design(metadata()).itertuples(index=False):
        for eid, n_held in enumerate((0, 2, 4)):
            base = {"dataset": m.dataset, "animal": m.animal, "session": m.session, "event_id": eid, "split": 0}
            labels.append(base | {"criterion": "edge10", "geometric_pass": eid == 0, "heldout_used_for_label": False, "mean_training_entropy_nats": 0.5})
            p = base | {"n_train_cells": 10, "n_heldout_cells": 5, "n_train_spikes": 5, "n_heldout_spikes": n_held, "duration_s": 0.1}
            for metric in METRICS:
                if metric == "geometric_fraction":
                    continue
                raw = metric.removesuffix("_per_spike")
                p[raw] = 1.0 if n_held else 0.0
                p[raw + "_per_spike"] = 1 / n_held if n_held else np.nan
            predictions.append(p)
    return pd.DataFrame(labels), pd.DataFrame(predictions)


def test_zero_heldout_ratio_not_zero_or_hidden():
    l, p = event_fixture()
    e, s = join_and_aggregate(l, p, area_design(metadata()))
    assert len(e) == 75 and len(s) == 25
    assert s.zero_heldout_events.eq(1).all()
    assert s.imm_order_advantage_per_spike_events.eq(2).all()
    assert np.allclose(s.imm_order_advantage_per_spike, 0.375)
    assert np.allclose(s.geometric_fraction, 1 / 3)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "leak", "ratio", "metadata"])
def test_unsafe_joins_fail(damage):
    l, p = event_fixture()
    m = area_design(metadata())
    if damage == "duplicate":
        p = pd.concat([p, p.iloc[:1]])
    elif damage == "missing":
        l = l.iloc[1:]
    elif damage == "leak":
        l.loc[0, "heldout_used_for_label"] = True
    elif damage == "ratio":
        p.loc[1, "imm_order_advantage_per_spike"] = 99
    else:
        m = m.iloc[1:]
    with pytest.raises(ValueError):
        join_and_aggregate(l, p, m)
