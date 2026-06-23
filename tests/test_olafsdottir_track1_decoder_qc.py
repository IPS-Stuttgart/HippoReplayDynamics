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
    module_path = scripts_path / "summarize_olafsdottir_track1_decoder_qc.py"
    spec = importlib.util.spec_from_file_location("summarize_olafsdottir_track1_decoder_qc", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_track1_decoder_qc_writes_outputs_and_passes_gates(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    linearization_root = tmp_path / "linearization"
    rows = [
        ("R2142", "2014-08-06", "20140806_R2142_track1", "20140806_R2142_sleepPOST", 1, "1,2,3,4,5,6,7,8", True),
        ("R2335", "2015-10-26", "20151026_R2335_track1", "20151026_R2335_sleepPOST", 9, "9,10,11,12,13,14,15,16", False),
    ]
    pairs_rows = []
    lin_rows = []
    for animal, date, track, sleep, tetrode, tetrode_list, reversal in rows:
        _write_track_spike_session(dataset_root / animal.lower() / date, track, tetrode)
        _write_linearized_position(linearization_root / "sessions" / animal / date / "linearized_position.csv")
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
        lin_rows.append(
            {
                "animal": animal,
                "date": date,
                "track_session": track,
                "sleeppost_session": sleep,
                "valid_position_fraction": 1.0,
                "linearized_position_span_cm": 100.0,
                "occupancy_nonzero_bins": 20,
                "orientation_rule": "inferred_occupied_bin_diameter",
                "reversal_applied": reversal,
                "linearization_status": "pass",
            }
        )
    pairs_csv = tmp_path / "pairs.csv"
    linearization_csv = linearization_root / "olafsdottir_track1_linearization_qc.csv"
    pd.DataFrame(pairs_rows).to_csv(pairs_csv, index=False)
    pd.DataFrame(lin_rows).to_csv(linearization_csv, index=False)

    tables = module.run_decoder_qc(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=linearization_csv,
        output_dir=tmp_path / "decoder-qc",
        crossval_folds=2,
        min_encoding_units=5,
        min_unit_spikes=3,
        max_posterior_median_error_cm=40.0,
        max_map_median_error_cm=50.0,
        max_decoder_animal_fraction=0.75,
        max_decoder_session_fraction=0.75,
    )

    out = tmp_path / "decoder-qc"
    assert (out / module.UNIT_OUTPUT).is_file()
    assert (out / module.DECODER_OUTPUT).is_file()
    assert (out / module.ANIMAL_OUTPUT).is_file()
    assert (out / module.GATE_OUTPUT).is_file()
    assert (out / module.SUMMARY_OUTPUT).is_file()
    assert (out / module.FIGURE_OUTPUT).is_file()
    units = tables["units"]
    decoder = tables["decoder"]
    gates = tables["gates"].set_index("gate")
    figures = tables["figures"]
    assert list(units.columns) == module.UNIT_COLUMNS
    assert list(decoder.columns) == module.DECODER_COLUMNS
    assert len(decoder) == 2
    assert decoder["decoder_status"].eq("pass").all()
    assert decoder["encoding_units_passing_qc"].min() >= 5
    assert decoder["posterior_coverage_fraction"].min() >= 0.80
    assert np.isfinite(decoder["posterior_mean_error_cm_median"]).all()
    assert np.isfinite(decoder["map_error_cm_median"]).all()
    assert units["unit_qc_passed"].map(bool).sum() >= 10
    assert gates["passed"].map(bool).all()
    assert set(figures["figure_type"]) == {
        "encoding_place_fields",
        "decoder_error_histogram",
        "decoder_predicted_vs_true",
    }
    assert all(Path(path).is_file() for path in figures["figure_path"])
    summary = (out / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "minimum encoding units per session | 5" in summary
    assert "Track1 sessions passing decoder QC | 2" in summary


def test_track1_decoder_qc_marks_missing_linearization_failure(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    pairs = pd.DataFrame(
        [
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "track_session": "20151026_R2335_track1",
                "sleepPOST_session": "20151026_R2335_sleepPOST",
                "hippocampal_tetrodes": "9",
                "usable_pair": True,
            }
        ]
    )
    linearization = pd.DataFrame(
        [
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "track_session": "20151026_R2335_track1",
                "sleeppost_session": "20151026_R2335_sleepPOST",
                "valid_position_fraction": 0.0,
                "linearized_position_span_cm": 0.0,
                "occupancy_nonzero_bins": 0,
                "orientation_rule": "inferred_occupied_bin_diameter",
                "reversal_applied": False,
                "linearization_status": "fail",
            }
        ]
    )
    pairs_csv = tmp_path / "pairs.csv"
    linearization_csv = tmp_path / "linearization.csv"
    pairs.to_csv(pairs_csv, index=False)
    linearization.to_csv(linearization_csv, index=False)

    tables = module.run_decoder_qc(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=linearization_csv,
        output_dir=tmp_path / "decoder-qc",
        min_encoding_units=1,
        min_posterior_coverage_fraction=0.0,
    )

    row = tables["decoder"].iloc[0]
    assert row["decoder_status"] == "fail"
    assert "linearization_qc_not_passed" in row["exclusion_reason"]
    gates = tables["gates"].set_index("gate")
    assert bool(gates.loc["track1_decoder_outputs_present", "passed"])
    assert not bool(gates.loc["animals_retained_after_decoder_qc", "passed"])
    assert not bool(gates.loc["finite_crossval_errors", "passed"])


def test_track1_decoder_qc_rejects_missing_inputs(tmp_path: Path) -> None:
    module = _load_module()
    pairs = tmp_path / "pairs.csv"
    linearization = tmp_path / "linearization.csv"
    pd.DataFrame([{"animal": "R2142"}]).to_csv(pairs, index=False)
    pd.DataFrame([{"animal": "R2142"}]).to_csv(linearization, index=False)

    try:
        module.load_pairs(pairs)
    except ValueError as exc:
        assert "track_session" in str(exc)
    else:
        raise AssertionError("load_pairs should reject incomplete pair tables")

    try:
        module.load_linearization_qc(linearization)
    except ValueError as exc:
        assert "linearization_status" in str(exc)
    else:
        raise AssertionError("load_linearization_qc should reject incomplete linearization tables")


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
    day_dir.mkdir(parents=True)
    spike_times: list[float] = []
    labels: list[int] = []
    centers = [8, 22, 36, 50, 64, 78, 92]
    times = np.arange(0.0, 20.0, 0.05)
    linear = 100.0 * ((times % 5.0) / 5.0)
    for time_s, pos in zip(times, linear):
        for label, center in enumerate(centers, start=1):
            if abs(pos - center) <= 7.0:
                spike_times.append(float(time_s + 0.001 * label))
                labels.append(label)
    order = np.argsort(spike_times)
    spike_times = [spike_times[index] for index in order]
    labels = [labels[index] for index in order]
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
