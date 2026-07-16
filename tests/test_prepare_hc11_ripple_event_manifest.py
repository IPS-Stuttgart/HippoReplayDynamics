from __future__ import annotations

from pathlib import Path
import sys

import h5py
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_hc11_ripple_event_manifest as prepare  # noqa: E402


def _event_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_group("ripplesNREM")


def test_manifest_combines_native_and_method_validated_generated_events(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    native_session = root / "RatA" / "RatA_day1"
    generated_session = root / "RatB" / "RatB_day1"
    native_session.mkdir(parents=True)
    generated_session.mkdir(parents=True)
    native = native_session / "RatA_day1.ripplesNREM.event.mat"
    native_all = native_session / "RatA_day1.ripplesALL.event.mat"
    generated_root = tmp_path / "detected"
    generated = generated_root / "RatB_day1" / "RatB_day1.ripplesNREM.generated.event.mat"
    _event_file(native)
    _event_file(native_all)
    _event_file(generated)
    pd.DataFrame(
        [
            {
                "detected_events": 40,
                "ripple_channel_source": "ca1_shank_unit_mode_fallback",
            }
        ]
    ).to_csv(
        generated.parent / "hc11_lfp_ripple_detection_qc.csv", index=False
    )
    validation = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {
                "native_validation_available": True,
                "overlap_precision": 0.8,
                "overlap_recall": 0.9,
                "ripple_channel_source": "ca1_shank_unit_mode_fallback",
            }
        ]
    ).to_csv(validation, index=False)
    monkeypatch.setattr(
        prepare,
        "event_phase_counts",
        lambda *_: {
            "events": 40,
            "finite_events": 40,
            "valid_duration_events": 40,
            "nrem_events": 40,
            "pre_events": 20,
            "post_events": 20,
        },
    )

    manifest, qc, gates, _ = prepare.build_event_manifest(
        root,
        generated_root,
        [validation],
        min_validation_precision=0.7,
        min_validation_recall=0.8,
        min_events_per_phase=20,
        expected_sessions=2,
        expected_animals=2,
    )
    assert len(manifest) == 2
    assert set(manifest["event_source"]) == {
        "published_all_intersect_current_nrem",
        "lfp_detected_method_validated",
    }
    native_row = manifest[manifest["session"].eq("RatA_day1")].iloc[0]
    assert Path(native_row["ripple_event_path"]) == native_all.resolve()
    assert qc["detector_qc_passed"].all()
    assert qc["session_detector_qc_passed"].all()
    assert qc["nrem_restriction_passed"].all()
    assert bool(gates.set_index("gate").loc["overall", "passed"])

    weak = pd.read_csv(validation)
    weak["overlap_recall"] = 0.5
    weak.to_csv(validation, index=False)
    manifest, qc, gates, _ = prepare.build_event_manifest(
        root,
        generated_root,
        [validation],
        min_validation_precision=0.7,
        min_validation_recall=0.8,
        min_events_per_phase=20,
        expected_sessions=2,
        expected_animals=2,
    )
    assert len(manifest) == 1
    assert not bool(qc.loc[qc["session"].eq("RatB_day1"), "detector_qc_passed"].iloc[0])
    assert not bool(gates.set_index("gate").loc["overall", "passed"])
