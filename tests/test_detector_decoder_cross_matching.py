import pandas as pd
import pytest

from scripts.report_detector_decoder_cross import matched_timing


def test_unique_matching_separates_window_and_population_effects(tmp_path):
    meta = {"dataset": "test", "animal": "a", "session": "s", "source": "real", "generator": "real", "peak_gain": 0, "replicate": 0}
    matches = [
        dict(**meta, population="subset", event_id=j, full_event_id=f, endpoint_shift_ms=shift)
        for j, f, shift in [(0, 0, 5), (1, 1, 10), (2, 1, 20), (3, -1, float("nan")), (4, 2, 30)]
    ]
    pd.DataFrame(matches).to_csv(tmp_path / "event_matches.csv", index=False)
    event_rows = []
    for name, windows in [("full", [(0, 0.1), (1, 1.1), (2, 2.1), (2.2, 2.3)]), ("subset", [(0, 0.105), (1, 1.07), (1.08, 1.12), (9, 9.1), (2, 2.3)])]:
        event_rows.extend({"detector": name, "event_id": j, "event_start_s": a, "event_end_s": b} for j, (a, b) in enumerate(windows))
    pd.DataFrame(event_rows).to_csv(tmp_path / "detector_events.csv", index=False)
    rows = []
    for likelihood in ("poisson", "conditional_multinomial"):
        for detector, decoder, values in [
            ("full", "full", [0.2, 0.9, 0.1, 0.2]),
            ("full", "subset", [0.3, 0.4, 0.8, 0.5]),
            ("subset", "full", [0.25, 0.8, 0.7, 0.99, 0.9]),
            ("subset", "subset", [0.4, 0.6, 0.5, 0.99, 0.9]),
        ]:
            rows.extend({"detector": detector, "decoder": decoder, "likelihood": likelihood, "event_id": j, "mass_4": value} for j, value in enumerate(values))
    pd.DataFrame(rows).to_csv(tmp_path / "window_readouts.csv.gz", index=False)
    result = matched_timing(tmp_path)
    assert len(result) == 2
    for row in result.itertuples():
        assert row.matched_unique_pairs == 1
        assert row.same_full_window_population_delta == pytest.approx(0.1)
        assert row.full_decoder_boundary_delta == pytest.approx(0.05)
        assert row.subset_decoder_boundary_delta == pytest.approx(0.1)
        assert row.median_endpoint_shift_ms == 5
