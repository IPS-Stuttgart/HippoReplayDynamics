import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scripts.audit_temporal_endpoint_content import dense_bank
from scripts.measure_temporal_endpoint_content import METHODS, endpoint_bank
from scripts.report_temporal_endpoint_content import SOURCES, TRUTH, gates, summaries


def fixture():
    records = []
    for animal in range(4):
        for session in range(2):
            for split in range(3):
                for source in SOURCES:
                    for method in METHODS:
                        for event in range(2):
                            scale = 0.7 if method == "diffusion_reset" else 1.0
                            value = dict(
                                dataset="pfeiffer_foster",
                                animal=str(animal),
                                session=f"{animal}/{session}",
                                split=split,
                                source=source,
                                method=method,
                                event_index=event,
                                original_start_s=float(event),
                                original_end_s=event + 0.02,
                                context_ms=200,
                                separation_cm=30 * scale,
                                regional_tv=0.5 * scale,
                                a_entropy=0.8 * scale,
                                b_entropy=0.8 * scale,
                                a_entropy_control_available=True,
                                b_entropy_control_available=True,
                            )
                            value.update({key: 20 * scale if source != "real" else np.nan for key in TRUTH})
                            records.append(value)
    return pd.DataFrame(records)


def test_success_requires_truth_and_independent_audit():
    data = fixture()
    _, _, table = summaries(data)
    assert gates(table, data, audited=True).passed.all()
    assert not gates(table, data, audited=False).set_index("gate").loc["advance_external_validation", "passed"]
    selected = data.source.eq("sim_late_jump") & data.method.eq("diffusion_reset")
    data.loc[selected, "a_error"] = 25
    _, _, table = summaries(data)
    result = gates(table, data, audited=True).set_index("gate")
    assert result.loc["real_reduction_separation_cm", "passed"]
    assert not result.loc["sim_late_jump_no_worse_a_error", "passed"]
    assert not result.loc["advance_external_validation", "passed"]


@pytest.mark.parametrize("kind", ["empty", "missing", "time", "nan", "duplicate"])
def test_incomplete_or_shifted_inputs_fail(kind):
    data = fixture()
    if kind == "empty":
        data = data.iloc[:0]
    elif kind == "missing":
        data = data.loc[data.method.ne("diffusion_reset")]
    elif kind == "time":
        data.loc[data.method.eq("diffusion_reset"), "original_end_s"] += 0.005
    elif kind == "nan":
        data.loc[data.method.eq("diffusion_reset"), "separation_cm"] = np.nan
    else:
        data = pd.concat([data, data.iloc[:1]])
    with pytest.raises((ValueError, AssertionError)):
        summaries(data)


def test_equal_animal_weight_and_no_vacuous_entropy_match():
    data = fixture()
    data.loc[data.animal.eq("3"), "separation_cm"] *= 2
    _, _, first = summaries(data)
    extra = data.loc[data.animal.eq("3")].copy()
    extra.event_index += 20
    extra.original_start_s += 20
    extra.original_end_s += 20
    _, _, second = summaries(pd.concat([data, extra]))
    np.testing.assert_allclose(first.value, second.value)
    data.loc[data.animal.eq("3"), "a_entropy_control_available"] = False
    assert not gates(first, data, audited=True).set_index("gate").loc["advance_external_validation", "passed"]


def test_independent_dense_auditor_matches_sparse_producer():
    grid = np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 8.0]])
    rates = np.array([[15.0, 0.2, 0.4], [0.2, 0.6, 20.0]])
    distance2 = ((grid[:, None] - grid[None]) ** 2).sum(axis=2)
    g = np.exp(-distance2 / 800) * (distance2 <= 6400)
    g /= g.sum(axis=0)
    events = [np.array([[0, 1]]), np.array([[3, 0], [0, 3]]), np.array([[1, 0], [0, 0], [2, 1]])]
    bank, _ = dense_bank(events, rates, grid)
    for i, event in enumerate(events):
        expected, _ = endpoint_bank(event, rates, csr_matrix(g))
        for method in expected:
            np.testing.assert_allclose(bank[method][i], expected[method], atol=1e-12)
