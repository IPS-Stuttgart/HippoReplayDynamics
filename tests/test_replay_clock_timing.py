import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.replay_clock_timing import log_scores, probabilities, rng, sample
from scripts.run_replay_clock_timing_recovery import summarize


def test_identity_and_envelope_information_are_separable():
    a = probabilities(np.array([[[9.0, 1.0], [1.0, 9.0]]]))
    b = probabilities(np.array([[[1.0, 9.0], [9.0, 1.0]]]))
    x = 100 * a["fine_joint"]
    sa, sb = log_scores(x, a), log_scores(x, b)
    assert sa["coarse_identity"] == pytest.approx(sb["coarse_identity"])
    assert sa["fine_timing"] == pytest.approx(sb["fine_timing"])
    assert sa["fine_identity"] > sb["fine_identity"]
    a = probabilities(np.array([[[9.0, 9.0], [1.0, 1.0]]]))
    b = probabilities(np.array([[[1.0, 1.0], [9.0, 9.0]]]))
    sa, sb = log_scores(100 * a["fine_joint"], a), log_scores(100 * a["fine_joint"], b)
    assert sa["fine_identity"] == pytest.approx(sb["fine_identity"])
    assert sa["fine_timing"] > sb["fine_timing"]


@pytest.mark.parametrize("seed", range(5))
def test_data_processing_and_score_decomposition(seed):
    gen = np.random.default_rng(seed)
    a = probabilities(np.exp(gen.normal(size=(3, 20, 6))))
    b = probabilities(np.exp(gen.normal(size=(3, 20, 6))))
    x = np.array([3, 10, 0])[:, None, None] * a["fine_joint"]
    sa, sb = log_scores(x, a), log_scores(x, b)
    assert sa["fine_joint"] == pytest.approx(sa["fine_identity"] + sa["fine_timing"])
    assert sa["fine_joint"] - sb["fine_joint"] >= sa["coarse_identity"] - sb["coarse_identity"] - 1e-9
    observed = sample(a["fine_joint"], [3, 10, 0], gen)
    np.testing.assert_array_equal(observed.sum(axis=(1, 2)), [3, 10, 0])


def test_incomplete_summary_and_invalid_counts_fail():
    p = probabilities(np.ones((2, 20, 3)))
    with pytest.raises(ValueError):
        sample(p["fine_joint"], [1.5, 3], rng("bad"))
    with pytest.raises(ValueError):
        probabilities(np.zeros((1, 20, 3)))
    with pytest.raises(ValueError):
        summarize(pd.DataFrame())


def test_end_to_end_fresh_counts_independent_audit(tmp_path):
    from hipporeplayimm.literal_replay_clock import bin_average, clock_coordinates
    from scripts.run_replay_clock_timing_recovery import record_task
    from scripts.verify_replay_clock_timing_recovery import record_audit

    parent = tmp_path / "parent"
    parent.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    x = np.linspace(0, 80, 801)
    points = np.c_[x, np.zeros(len(x))]
    rates = np.column_stack([0.2 + 10 * np.exp(-((x - center) ** 2) / 200) for center in [5, 20, 60, 75]])
    _, _, clocks = clock_coordinates(points, rates)
    totals = np.array([10, 0, 20])
    arrays = {"totals": totals, "gains": np.array([1.1, 0.8, 1.2, 0.9]), "path_rates": rates}
    for name, t in clocks.items():
        arrays[f"clock_{name}"] = t
        arrays[f"bin_rates_{name}"] = bin_average(t, rates, 3)
    np.savez_compressed(parent / "test_p000.npz", **arrays)
    item = {"dataset": "test", "animal": "one", "session": "run", "tag": "test"}
    result = record_task(item, parent, out, n_paths=1, repeats=2)
    audit = record_audit(item, parent, out, n_paths=1, repeats=2)
    assert result["rows"] == audit["scores_checked"] == 32
    assert audit["max_probability_error"] < 1e-4
    table = pd.read_csv(out / "test_scores.csv.gz")
    assert len(summarize(table)[-1]) == 2
    with pytest.raises(ValueError):
        summarize(table.iloc[1:])
    table.loc[0, "log_score_neural"] += 1
    table.to_csv(out / "test_scores.csv.gz", index=False)
    with pytest.raises(AssertionError):
        record_audit(item, parent, out, n_paths=1, repeats=2)
