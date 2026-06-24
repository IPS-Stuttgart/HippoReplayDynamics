from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = scripts_path / "score_olafsdottir_1d_sleeppost_evidence.py"
    spec = importlib.util.spec_from_file_location("score_olafsdottir_1d_sleeppost_evidence", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sleeppost_evidence_smoke_writes_required_outputs_and_gates(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    linearization_root = tmp_path / "linearization-qc"
    rows = [
        ("R2142", "2014-08-06", "20140806_R2142_track1", "20140806_R2142_sleepPOST", 1, "1"),
        ("R2335", "2015-10-26", "20151026_R2335_track1", "20151026_R2335_sleepPOST", 9, "9"),
    ]
    pairs_rows = []
    linearization_rows = []
    decoder_rows = []
    pilot_rows = []
    for animal, date, track, sleep, tetrode, tetrode_list in rows:
        day_dir = dataset_root / animal.lower() / date
        _write_linearized_position(linearization_root / "sessions" / animal / date / "linearized_position.csv")
        _write_track_spike_session(day_dir, track, tetrode)
        _write_sleep_spike_session(day_dir, sleep, tetrode)
        pairs_rows.append(
            {
                "animal": animal,
                "date": date,
                "track_session": track,
                "sleepPOST_session": sleep,
                "hippocampal_tetrodes": tetrode_list,
                "usable_pair": True,
            }
        )
        linearization_rows.append(
            {
                "animal": animal,
                "date": date,
                "track_session": track,
                "sleeppost_session": sleep,
                "linearization_status": "pass",
            }
        )
        decoder_rows.append(
            {
                "animal": animal,
                "date": date,
                "track1_session": track,
                "sleeppost_session": sleep,
                "decoder_status": "pass",
            }
        )
        for event_id, start in enumerate([0.50, 1.20]):
            pilot_rows.append(
                {
                    "selection_tier": "pilot_20_balanced",
                    "animal": animal,
                    "date": date,
                    "track1_session": track,
                    "sleeppost_session": sleep,
                    "event_id": event_id,
                    "start_time_s": start,
                    "end_time_s": start + 0.06,
                    "duration_ms": 60.0,
                    "n_spikes": 8,
                    "n_active_units": 4,
                    "mean_speed_cm_s": 0.5,
                }
            )

    pairs_csv = tmp_path / "pairs.csv"
    linearization_csv = linearization_root / "olafsdottir_track1_linearization_qc.csv"
    decoder_csv = tmp_path / "decoder.csv"
    pilot_csv = tmp_path / "pilot.csv"
    pd.DataFrame(pairs_rows).to_csv(pairs_csv, index=False)
    pd.DataFrame(linearization_rows).to_csv(linearization_csv, index=False)
    pd.DataFrame(decoder_rows).to_csv(decoder_csv, index=False)
    pd.DataFrame(pilot_rows).to_csv(pilot_csv, index=False)

    tables = module.run_sleep_evidence(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=linearization_csv,
        decoder_qc=decoder_csv,
        pilot_selection=pilot_csv,
        pilot_tier="pilot_20_balanced",
        output_dir=tmp_path / "evidence",
        position_bin_size_cm=10.0,
        time_bin_s=0.02,
        min_unit_spikes=3,
        min_encoding_units=4,
        smoothing_bins=1,
    )

    out = tmp_path / "evidence"
    assert (out / module.EVENT_MODEL_OUTPUT).is_file()
    assert (out / module.DECISION_OUTPUT).is_file()
    assert (out / module.TRAJECTORY_SUMMARY_OUTPUT).is_file()
    assert (out / module.IMM_FRAGMENTED_OUTPUT).is_file()
    assert (out / module.PAIR_OUTPUT).is_file()
    assert (out / module.ANIMAL_OUTPUT).is_file()
    assert (out / module.GATE_OUTPUT).is_file()
    assert (out / module.MANIFEST_OUTPUT).is_file()
    assert (out / module.SUMMARY_OUTPUT).is_file()

    evidence = tables["event_model_evidence"]
    decisions = tables["model_claim_decisions"]
    gates = tables["gate_summary"].set_index("gate")
    assert list(evidence.columns) == module.EVENT_MODEL_COLUMNS
    assert list(decisions.columns) == module.DECISION_COLUMNS
    assert len(decisions) == 4
    assert len(evidence) == 4 * len(module.REQUIRED_MODELS)
    assert evidence["status"].eq("success").all()
    assert set(evidence["model"]) == set(module.REQUIRED_MODELS)
    assert np.isfinite(evidence["log_evidence"]).all()
    assert np.isfinite(evidence["runtime_s"]).all()
    assert (evidence["runtime_s"] >= 0.0).all()
    assert np.isfinite(decisions["logZ_stationary"]).all()
    assert np.isfinite(decisions["logZ_diffusion"]).all()
    assert np.isfinite(decisions["logZ_fragmented"]).all()
    assert np.isfinite(decisions["logZ_first_order_imm"]).all()
    assert np.isfinite(decisions["delta_imm_minus_fragmented"]).all()
    assert gates["passed"].map(bool).all()
    assert bool(gates.loc["overall", "passed"])
    summary = (out / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "scoring/readiness smoke only" in summary
    assert "pilot_20_balanced" in summary


def test_sleeppost_evidence_smoke_rejects_missing_inputs(tmp_path: Path) -> None:
    module = _load_module()
    pairs_csv = tmp_path / "pairs.csv"
    linearization_csv = tmp_path / "linearization.csv"
    decoder_csv = tmp_path / "decoder.csv"
    pilot_csv = tmp_path / "pilot.csv"
    pd.DataFrame([{"animal": "R2142"}]).to_csv(pairs_csv, index=False)
    pd.DataFrame([{"animal": "R2142"}]).to_csv(linearization_csv, index=False)
    pd.DataFrame([{"animal": "R2142"}]).to_csv(decoder_csv, index=False)
    pd.DataFrame([{"animal": "R2142"}]).to_csv(pilot_csv, index=False)

    try:
        module.load_pairs(pairs_csv)
    except ValueError as exc:
        assert "sleepPOST_session" in str(exc)
    else:
        raise AssertionError("load_pairs should reject incomplete pair tables")

    try:
        module.load_linearization_qc(linearization_csv)
    except ValueError as exc:
        assert "linearization_status" in str(exc)
    else:
        raise AssertionError("load_linearization_qc should reject incomplete linearization tables")

    try:
        module.load_decoder_qc(decoder_csv)
    except ValueError as exc:
        assert "decoder_status" in str(exc)
    else:
        raise AssertionError("load_decoder_qc should reject incomplete decoder tables")

    try:
        module.load_pilot_selection(pilot_csv)
    except ValueError as exc:
        assert "selection_tier" in str(exc)
    else:
        raise AssertionError("load_pilot_selection should reject incomplete pilot tables")


def test_sleeppost_evidence_smoke_marks_decoder_failures(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    linearization_root = tmp_path / "linearization-qc"
    animal = "R2335"
    date = "2015-10-26"
    track = "20151026_R2335_track1"
    sleep = "20151026_R2335_sleepPOST"
    _write_linearized_position(linearization_root / "sessions" / animal / date / "linearized_position.csv")
    _write_track_spike_session(dataset_root / animal.lower() / date, track, 9)
    _write_sleep_spike_session(dataset_root / animal.lower() / date, sleep, 9)
    pairs_csv = tmp_path / "pairs.csv"
    linearization_csv = linearization_root / "olafsdottir_track1_linearization_qc.csv"
    decoder_csv = tmp_path / "decoder.csv"
    pilot_csv = tmp_path / "pilot.csv"
    pd.DataFrame(
        [
            {
                "animal": animal,
                "date": date,
                "track_session": track,
                "sleepPOST_session": sleep,
                "hippocampal_tetrodes": "9",
                "usable_pair": True,
            }
        ]
    ).to_csv(pairs_csv, index=False)
    pd.DataFrame(
        [{"animal": animal, "date": date, "track_session": track, "sleeppost_session": sleep, "linearization_status": "pass"}]
    ).to_csv(linearization_csv, index=False)
    pd.DataFrame(
        [{"animal": animal, "date": date, "track1_session": track, "sleeppost_session": sleep, "decoder_status": "fail"}]
    ).to_csv(decoder_csv, index=False)
    pd.DataFrame(
        [
            {
                "selection_tier": "pilot_20_balanced",
                "animal": animal,
                "date": date,
                "track1_session": track,
                "sleeppost_session": sleep,
                "event_id": 0,
                "start_time_s": 0.50,
                "end_time_s": 0.56,
                "duration_ms": 60.0,
                "n_spikes": 8,
                "n_active_units": 4,
                "mean_speed_cm_s": 0.5,
            }
        ]
    ).to_csv(pilot_csv, index=False)

    tables = module.run_sleep_evidence(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=linearization_csv,
        decoder_qc=decoder_csv,
        pilot_selection=pilot_csv,
        pilot_tier="pilot_20_balanced",
        output_dir=tmp_path / "evidence",
        min_encoding_units=1,
    )

    evidence = tables["event_model_evidence"]
    assert evidence["status"].eq("fail").all()
    assert evidence["failure_reason"].eq("decoder_qc_not_passed").all()
    gates = tables["gate_summary"].set_index("gate")
    assert not bool(gates.loc["no_model_scoring_failures", "passed"])
    assert not bool(gates.loc["overall", "passed"])


def _write_linearized_position(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(0.0, 20.0, 0.05)
    phase = (times % 5.0) / 5.0
    linear = 100.0 * phase
    pd.DataFrame(
        {
            "time_s": times,
            "x_cm": linear,
            "y_cm": np.zeros_like(linear),
            "linear_position_cm": linear,
            "speed_cm_s": np.full_like(linear, 20.0),
            "valid_position": True,
        }
    ).to_csv(path, index=False)


def _write_track_spike_session(day_dir: Path, stem: str, tetrode: int) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    spike_times: list[float] = []
    labels: list[int] = []
    centers = [10, 30, 50, 70, 90]
    times = np.arange(0.0, 20.0, 0.05)
    linear = 100.0 * ((times % 5.0) / 5.0)
    for time_s, pos in zip(times, linear):
        for label, center in enumerate(centers, start=1):
            if abs(pos - center) <= 7.5:
                spike_times.append(float(time_s + 0.001 * label))
                labels.append(label)
    order = np.argsort(spike_times)
    spike_times = [spike_times[index] for index in order]
    labels = [labels[index] for index in order]
    _write_tetrode(day_dir / f"{stem}.{tetrode}", spike_times)
    _write_cut(day_dir / f"{stem}_{tetrode}.cut", labels)


def _write_sleep_spike_session(day_dir: Path, stem: str, tetrode: int) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    spike_times = [
        0.505,
        0.510,
        0.515,
        0.520,
        0.530,
        0.540,
        1.205,
        1.210,
        1.215,
        1.220,
        1.230,
        1.240,
    ]
    labels = [1, 2, 3, 4, 1, 2, 2, 3, 4, 5, 3, 4]
    _write_tetrode(day_dir / f"{stem}.{tetrode}", spike_times)
    _write_cut(day_dir / f"{stem}_{tetrode}.cut", labels)


def _write_tetrode(path: Path, spike_times_s: list[float]) -> None:
    header = (
        f"num_spikes {len(spike_times_s)}\n"
        "timebase 96000 hz\n"
        "samples_per_spike 2\n"
        "data_start"
    ).encode("ascii")
    payload = b"".join(
        struct.pack(">I", int(round(time_s * 96000.0))) + b"\x00" * 8
        for time_s in spike_times_s
    )
    path.write_bytes(header + payload)


def _write_cut(path: Path, labels: list[int]) -> None:
    path.write_text(
        f"Exact_cut_for: {path.name} spikes: {len(labels)}\n"
        + " ".join(str(label) for label in labels)
        + "\n",
        encoding="ascii",
    )
