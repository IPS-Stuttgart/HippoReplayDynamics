from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = repo_root / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    module_path = scripts_path / "select_olafsdottir_sleeppost_pilot_events.py"
    spec = importlib.util.spec_from_file_location("select_olafsdottir_sleeppost_pilot_events", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pilot_event_selection_writes_balanced_deterministic_tiers(tmp_path: Path) -> None:
    module = _load_module()
    events = _candidate_events()
    decoder = _decoder_qc()
    events_csv = tmp_path / "events.csv"
    decoder_csv = tmp_path / "decoder.csv"
    events.to_csv(events_csv, index=False)
    decoder.to_csv(decoder_csv, index=False)

    tables = module.run_pilot_event_selection(
        candidate_events_csv=events_csv,
        decoder_qc_csv=decoder_csv,
        output_dir=tmp_path / "selection-a",
        seed=123,
        pilot20_events_per_pair=2,
        pilot50_events_per_pair=3,
        pilot100_events_per_pair=4,
        min_pilot20_animals_fraction=1.0,
    )
    repeat = module.run_pilot_event_selection(
        candidate_events_csv=events_csv,
        decoder_qc_csv=decoder_csv,
        output_dir=tmp_path / "selection-b",
        seed=123,
        pilot20_events_per_pair=2,
        pilot50_events_per_pair=3,
        pilot100_events_per_pair=4,
        min_pilot20_animals_fraction=1.0,
    )

    out = tmp_path / "selection-a"
    assert (out / module.SELECTION_OUTPUT).is_file()
    assert (out / module.ANIMAL_OUTPUT).is_file()
    assert (out / module.GATE_OUTPUT).is_file()
    assert (out / module.SUMMARY_OUTPUT).is_file()
    selection = tables["selection"]
    gates = tables["gates"].set_index("gate")
    assert list(selection.columns) == module.SELECTION_COLUMNS
    assert module.tier_count(selection, "pilot_20_balanced") == 4
    assert module.tier_count(selection, "pilot_50_balanced") == 6
    assert module.tier_count(selection, "pilot_100_balanced") == 8
    assert module.tier_count(selection, "all_immobile_qc_valid") == 10
    assert selection["selection_seed"].dropna().eq(123).all()
    assert selection["selection_rule_version"].eq("pre_evidence_v1").all()
    assert selection["event_qc_status"].eq("pass").all()
    assert (selection["mean_speed_cm_s"] <= 5.0).all()
    assert gates["passed"].map(bool).all()
    first = selection[selection["selection_tier"].eq("pilot_20_balanced")][["animal", "session", "event_id"]].reset_index(drop=True)
    second = repeat["selection"][repeat["selection"]["selection_tier"].eq("pilot_20_balanced")][["animal", "session", "event_id"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    summary = (out / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "seed | 123" in summary
    assert "pilot_20_balanced events | 4" in summary
    assert "does not use replay model scores" in summary


def test_pilot_event_selection_ignores_decoder_failed_pairs(tmp_path: Path) -> None:
    module = _load_module()
    events = _candidate_events()
    decoder = _decoder_qc()
    decoder.loc[decoder["animal"].eq("R2335"), "decoder_status"] = "fail"
    events_csv = tmp_path / "events.csv"
    decoder_csv = tmp_path / "decoder.csv"
    events.to_csv(events_csv, index=False)
    decoder.to_csv(decoder_csv, index=False)

    tables = module.run_pilot_event_selection(
        candidate_events_csv=events_csv,
        decoder_qc_csv=decoder_csv,
        output_dir=tmp_path / "selection",
        seed=456,
        pilot20_events_per_pair=2,
        pilot50_events_per_pair=3,
        pilot100_events_per_pair=4,
    )

    selection = tables["selection"]
    assert set(selection["animal"]) == {"R2142"}
    assert module.tier_count(selection, "pilot_20_balanced") == 2
    gates = tables["gates"].set_index("gate")
    assert bool(gates.loc["pilot_20_balanced_complete", "passed"])
    assert bool(gates.loc["selection_is_pre_evidence_only", "passed"])


def test_pilot_event_selection_normalizes_pass_status_tokens(tmp_path: Path) -> None:
    module = _load_module()
    events = _candidate_events()
    decoder = _decoder_qc()
    events.loc[events["event_qc_status"].eq("pass"), "event_qc_status"] = " PASS "
    decoder.loc[decoder["decoder_status"].eq("pass"), "decoder_status"] = " Pass "
    events_csv = tmp_path / "events.csv"
    decoder_csv = tmp_path / "decoder.csv"
    events.to_csv(events_csv, index=False)
    decoder.to_csv(decoder_csv, index=False)

    tables = module.run_pilot_event_selection(
        candidate_events_csv=events_csv,
        decoder_qc_csv=decoder_csv,
        output_dir=tmp_path / "selection",
        seed=789,
        pilot20_events_per_pair=2,
        pilot50_events_per_pair=3,
        pilot100_events_per_pair=4,
        min_pilot20_animals_fraction=1.0,
    )

    selection = tables["selection"]
    assert module.tier_count(selection, "pilot_20_balanced") == 4
    assert selection["event_qc_status"].astype(str).str.strip().str.lower().eq("pass").all()
    gates = tables["gates"].set_index("gate")
    assert bool(gates.loc["decoder_pass_pairs_present", "passed"])
    assert bool(gates.loc["eligible_events_for_decoder_pairs", "passed"])
    assert bool(gates.loc["pilot_20_balanced_complete", "passed"])


def test_pilot_event_selection_uses_explicit_scoring_available_debug_tiers(tmp_path: Path) -> None:
    module = _load_module()
    events = _candidate_events()
    decoder = _decoder_qc()
    decoder["decoder_status"] = "fail"
    decoder["decoder_qc_paper_ready"] = False
    decoder["decoder_qc_scoring_available"] = True
    events_csv = tmp_path / "events.csv"
    decoder_csv = tmp_path / "decoder.csv"
    events.to_csv(events_csv, index=False)
    decoder.to_csv(decoder_csv, index=False)

    strict = module.run_pilot_event_selection(
        candidate_events_csv=events_csv,
        decoder_qc_csv=decoder_csv,
        output_dir=tmp_path / "strict-selection",
        seed=123,
        pilot20_events_per_pair=2,
        pilot50_events_per_pair=3,
        pilot100_events_per_pair=4,
        min_pilot20_animals_fraction=1.0,
    )
    debug = module.run_pilot_event_selection(
        candidate_events_csv=events_csv,
        decoder_qc_csv=decoder_csv,
        output_dir=tmp_path / "debug-selection",
        seed=123,
        pilot20_events_per_pair=2,
        pilot50_events_per_pair=3,
        pilot100_events_per_pair=4,
        decoder_filter="scoring_available",
        min_pilot20_animals_fraction=1.0,
    )

    assert strict["selection"].empty
    debug_selection = debug["selection"]
    assert module.tier_count(debug_selection, "pilot_20_decoder_available_debug") == 4
    assert module.tier_count(debug_selection, "pilot_50_decoder_available_debug") == 6
    assert module.tier_count(debug_selection, "pilot_100_decoder_available_debug") == 8
    assert module.tier_count(debug_selection, "pilot_20_balanced") == 0
    assert debug_selection["decoder_filter"].dropna().eq("scoring_available").all()
    assert debug_selection["decoder_qc_paper_ready"].map(bool).eq(False).all()
    assert debug_selection["decoder_qc_scoring_available"].map(bool).all()
    gates = debug["gates"].set_index("gate")
    assert bool(gates.loc["pilot_20_decoder_available_debug_complete", "passed"])
    assert bool(gates.loc["pilot_20_decoder_available_debug_spans_decoder_animals", "passed"])
    assert bool(gates.loc["pilot_20_decoder_available_debug_spans_decoder_pairs", "passed"])
    summary = (tmp_path / "debug-selection" / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "decoder_filter | scoring_available" in summary
    assert "pilot_20_decoder_available_debug events | 4" in summary


def test_pilot_event_selection_rejects_missing_inputs(tmp_path: Path) -> None:
    module = _load_module()
    events_csv = tmp_path / "events.csv"
    decoder_csv = tmp_path / "decoder.csv"
    pd.DataFrame([{"animal": "R2142"}]).to_csv(events_csv, index=False)
    pd.DataFrame([{"animal": "R2142"}]).to_csv(decoder_csv, index=False)

    try:
        module.load_candidate_events(events_csv)
    except ValueError as exc:
        assert "event_qc_status" in str(exc)
    else:
        raise AssertionError("load_candidate_events should reject incomplete event tables")

    try:
        module.load_decoder_qc(decoder_csv)
    except ValueError as exc:
        assert "decoder_status" in str(exc)
    else:
        raise AssertionError("load_decoder_qc should reject incomplete decoder tables")


def _candidate_events() -> pd.DataFrame:
    rows = []
    for animal, date, session in [
        ("R2142", "2014-08-06", "20140806_R2142_sleepPOST"),
        ("R2335", "2015-10-26", "20151026_R2335_sleepPOST"),
    ]:
        for event_id in range(7):
            rows.append(
                {
                    "animal": animal,
                    "date": date,
                    "session": session,
                    "event_id": event_id,
                    "start_time_s": 10.0 + event_id,
                    "end_time_s": 10.04 + event_id,
                    "duration_ms": 40.0,
                    "n_spikes": 10 + event_id,
                    "n_active_units": 4 + (event_id % 3),
                    "mean_mua_rate_hz": 100.0 + event_id,
                    "peak_mua_rate_hz": 200.0 + event_id,
                    "mean_speed_cm_s": 1.0 if event_id < 5 else 10.0,
                    "event_detection_score": 5.0 + event_id,
                    "candidate_tier": "strong",
                    "event_qc_status": "pass" if event_id != 6 else "artifact",
                    "event_qc_reason": "" if event_id != 6 else "test_artifact",
                }
            )
    return pd.DataFrame(rows)


def _decoder_qc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "animal": "R2142",
                "date": "2014-08-06",
                "track1_session": "20140806_R2142_track1",
                "sleeppost_session": "20140806_R2142_sleepPOST",
                "decoder_status": "pass",
                "posterior_mean_error_cm_median": 20.0,
                "map_error_cm_median": 25.0,
                "posterior_coverage_fraction": 0.95,
            },
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "track1_session": "20151026_R2335_track1",
                "sleeppost_session": "20151026_R2335_sleepPOST",
                "decoder_status": "pass",
                "posterior_mean_error_cm_median": 22.0,
                "map_error_cm_median": 28.0,
                "posterior_coverage_fraction": 0.90,
            },
        ]
    )
