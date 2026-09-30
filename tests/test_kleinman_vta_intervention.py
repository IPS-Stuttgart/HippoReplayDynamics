import itertools
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from scripts import calibrate_kleinman_vta_intervention as calibration
from scripts import kleinman_vta_intervention as vta
from scripts import run_kleinman_vta_intervention as driver
from scripts.kleinman_conditional_spatial_score import anchor_score, spatial_score


def inventory():
    request, author, packets = [], [], []
    for animal, group in vta.ANIMALS.items():
        for i in range(4):
            session = f"202105{24 + i}_run1"
            drug = i % 2
            r = {"animal": animal, "session": session, "released_drug": "CNO" if drug else "saline", "released_novel": "familiar"}
            request.append(r)
            author.append(
                dict(
                    **r,
                    physical_track_id="track_A",
                    pretraining_track=True,
                    experience_number_on_track=i + 1,
                    cno_dose_mg_kg=drug * 2,
                    injection_to_recording_minutes=30,
                    same_track_used_under_both_drugs=True,
                    confirmed_group=group,
                    treatment_allocation="author-described alternating schedule",
                    author_confirmed=True,
                    source_reference="author-response-fixture",
                )
            )
            packets.append(
                {
                    "animal": animal,
                    "session": session,
                    "drug": drug,
                    "novel": 0,
                    "packet_id": "p0",
                    "selected": True,
                    "availability_descriptor": True,
                    "reference_traversals": "[0,1,2]",
                    "past_traversal": 3,
                    "baseline_traversal": 4,
                    "target_traversal": 5,
                    "span_start_s": 0,
                    "span_end_s": 51,
                    "past_start_s": 30,
                    "past_end_s": 31,
                    "baseline_start_s": 40,
                    "baseline_end_s": 41,
                    "target_start_s": 50,
                    "target_end_s": 51,
                }
            )
    previous = pd.DataFrame([{"animal": "Con_1", "session": "older", "traversal": 5, "role": "target", "source_artifact": "old.csv"}])
    return pd.DataFrame(request), pd.DataFrame(author), pd.DataFrame(packets), previous


def score_frame():
    rows = []
    for i, (animal, group) in enumerate(vta.ANIMALS.items()):
        for track in ("a", "b"):
            for drug in (0, 1):
                change = ([-0.05, 0.02, 0.06, -0.4, -0.5, -0.6][i]) * drug
                rows.append(
                    {
                        "animal": animal,
                        "group": group,
                        "session": track + str(drug),
                        "packet_id": "p",
                        "physical_track_id": track,
                        "drug": drug,
                        "preceding_score": 0.3,
                        "preceding_information": 2.0,
                        "prospective_score": 0.3 + 2 * change,
                        "prospective_information": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_missing_author_data_is_not_a_negative_biological_result():
    request, _, _, _ = inventory()
    m = vta.import_metadata(request)
    assert not m.metadata_verified.any()
    assert m.metadata_exclusion.str.contains("missing_physical_track_id").all()
    assert m.physical_track_id.eq("").all()


def test_verified_metadata_keeps_group_and_release_labels():
    request, author, _, _ = inventory()
    m = vta.import_metadata(request, author)
    assert m.metadata_verified.all()
    assert m.groupby("group").animal.nunique().to_dict() == {"control": 3, "experimental": 3}


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("released_drug", "CNO", "author_release_conflict"),
        ("confirmed_group", "experimental", "experimental_control_conflict"),
        ("cno_dose_mg_kg", 2, "drug_dose_conflict"),
        ("experience_number_on_track", 1.5, "invalid_experience_order"),
        ("pretraining_track", False, "track_metadata_conflict"),
    ],
)
def test_metadata_conflicts(field, value, error):
    request, author, _, _ = inventory()
    author[field] = author[field].astype(object)
    author.loc[0, field] = value
    with pytest.raises(ValueError, match=error):
        vta.import_metadata(request, author)


def test_duplicate_and_unknown_metadata_rejected():
    request, author, _, _ = inventory()
    with pytest.raises(ValueError, match="duplicate_identity"):
        vta.import_metadata(pd.concat([request, request.iloc[:1]]), author)
    author.loc[0, "session"] = "not_in_release"
    with pytest.raises(ValueError, match="unknown_author_session"):
        vta.import_metadata(request, author)


def test_request_deadline_not_started_by_draft():
    now = datetime(2026, 9, 30, tzinfo=UTC)
    assert vta.request_status({"sent_at_utc": None}, now) == "awaiting_author_request_approval"
    assert vta.request_status({"sent_at_utc": "2026-09-17T00:00:00Z", "clarification_rounds": 1}, now) == "awaiting_author_metadata"
    assert vta.request_status({"sent_at_utc": "2026-09-16T00:00:00Z", "clarification_rounds": 1}, now) == "blocked_metadata"
    assert vta.request_status({"feasibility_working_days_used": 6}, now) == "stopped_feasibility_budget"
    with pytest.raises(ValueError, match="one_clarification"):
        vta.request_status({"clarification_rounds": 2}, now)


def test_primary_cohort_all_six_but_prior_readouts_excluded():
    request, author, packets, previous = inventory()
    previous.loc[0, ["session", "traversal"]] = [packets.iloc[0].session, 3]
    ledger, audit, coverage = vta.cohort_tables(vta.import_metadata(request, author), packets, previous)
    assert coverage.paired_drugs.all()
    assert audit.supported.all()
    assert len(ledger.loc[ledger.primary_eligible]) == 23
    assert ledger.iloc[0].previous_readout_overlap
    assert "exploratory_only" in ledger.iloc[0].exclusion_reason


def test_prior_reference_only_reuse_allowed():
    request, author, packets, previous = inventory()
    previous.loc[0, ["session", "traversal"]] = [packets.iloc[0].session, 1]
    ledger, _, _ = vta.cohort_tables(vta.import_metadata(request, author), packets, previous)
    assert ledger.primary_eligible.all()


def test_missing_drug_and_disjoint_experience_stop():
    request, author, packets, previous = inventory()
    packets = packets.loc[~(packets.animal.eq("Exp_4") & packets.drug.eq(1))]
    _, audit, coverage = vta.cohort_tables(vta.import_metadata(request, author), packets, previous)
    assert not coverage.loc[coverage.animal.eq("Exp_4"), "paired_drugs"].item()
    assert not audit.loc[audit.animal.eq("Exp_4"), "both_drugs"].item()
    request, author, packets, previous = inventory()
    author.loc[author.released_drug.eq("CNO"), "experience_number_on_track"] += 10
    ledger, audit, coverage = vta.cohort_tables(vta.import_metadata(request, author), packets, previous)
    assert not audit.experience_overlap.any() and not ledger.primary_eligible.any()
    assert not coverage.paired_drugs.any()


@pytest.mark.parametrize("change", ["chronology", "reference_leak", "overlap", "duplicate"])
def test_temporal_and_overlap_errors(change):
    _, _, packets, _ = inventory()
    if change == "chronology":
        packets.loc[0, "target_start_s"] = 40
    elif change == "reference_leak":
        packets.loc[0, "reference_traversals"] = "[0,1,5]"
    else:
        copy = packets.iloc[[0]].copy()
        if change == "overlap":
            copy["packet_id"] = "other_direction"
        packets = pd.concat([packets, copy], ignore_index=True)
    with pytest.raises(ValueError):
        vta.validate_packets(packets)


def test_independent_animal_DID_and_two_sided_test():
    tracks, animals = vta.animal_contrasts(score_frame())
    r = vta.contrast_inference(animals)
    experimental = np.array([-0.4, -0.5, -0.6])
    control = np.array([-0.05, 0.02, 0.06])
    assert r["D"] == pytest.approx(experimental.mean() - control.mean())
    assert r["p_two_sided"] == pytest.approx(ttest_ind(experimental, control, equal_var=False).pvalue)
    assert len(tracks) == 12 and len(animals) == 6
    assert r["ci_high"] < 0


def test_packet_count_does_not_multiply_animal_or_track_weight():
    original = score_frame()
    extras = pd.concat([original.loc[original.animal.eq("Exp_1") & original.physical_track_id.eq("a")].assign(packet_id=f"copy{i}") for i in range(30)])
    _, before = vta.animal_contrasts(original)
    _, after = vta.animal_contrasts(pd.concat([original, extras]))
    np.testing.assert_allclose(before.drug_difference, after.drug_difference)
    assert vta.contrast_inference(before)["D"] == pytest.approx(vta.contrast_inference(after)["D"])


def test_zero_information_not_a_biological_zero():
    f = score_frame()
    f.loc[f.animal.eq("Exp_4") & f.drug.eq(1), "prospective_information"] = 0
    _, animals = vta.animal_contrasts(f)
    assert np.isnan(animals.loc[animals.animal.eq("Exp_4"), "drug_difference"].item())
    assert vta.contrast_inference(animals)["status"] == "missing_paired_information"


def test_independent_conditional_score_enumeration():
    b, y = np.array([2, 1, 2]), np.array([0, 2, 1])
    t0, t1, feature = np.array([1, 2, 1]), np.array([2, 1, 3]), np.array([-1.0, 0.2, 1.1])
    c, total = b + y, int(y.sum())
    states = [np.array(z) for z in itertools.product(*[range(int(n) + 1) for n in c]) if sum(z) == total]
    weights = np.array([math.prod(math.comb(int(n), int(k)) * float(t / s) ** int(k) for n, k, t, s in zip(c, z, t1, t0, strict=True)) for z in states])
    weights /= weights.sum()
    values = np.array([z @ feature for z in states])
    mean, variance = weights @ values, weights @ values**2 - (weights @ values) ** 2
    r = spatial_score(b, y, t0, t1, feature)
    assert r["score"] == pytest.approx(y @ feature - mean, abs=1e-10)
    assert r["information"] == pytest.approx(variance, abs=1e-10)
    x, u, v = np.array([-1.0, 0.0, 1.0]), np.array([0.2, -0.3, 0.4]), np.array([1.0, 2.0, 3.0])
    centered = x - sum(x * v) / sum(v)
    assert anchor_score(x, u, v) == pytest.approx((sum(centered * u), sum(centered**2 * v)))


def calibration_estimates(null_hits=0):
    rows = []
    for scenario, effect in calibration.SCENARIOS:
        for rep in range(1000):
            rows.append({"scenario": scenario, "effect": effect, "replicate": rep, "status": "scored", "D": effect or 0.01, "flag": rep < (null_hits if effect == 0 else 900)})
    return pd.DataFrame(rows)


def test_fixed_1000_validation_fails_null_mc_bound_and_missing_draws():
    good = calibration_estimates()
    summary = vta.calibration_summary(good)
    assert summary.complete.all() and summary.loc[summary.effect.eq(0), "null_pass"].all()
    bad = vta.calibration_summary(calibration_estimates(70))
    assert not bad.loc[bad.effect.eq(0), "null_pass"].any()
    assert bad.loc[bad.effect.eq(0), "mc95_high"].gt(0.075).all()
    assert not vta.calibration_summary(good.iloc[1:]).complete.all()
    with pytest.raises(ValueError, match="incomplete_calibration"):
        vta.calibration_summary(good.loc[good.scenario.ne("bursting")])


def test_locked_bank_cannot_be_retried_elsewhere(tmp_path):
    driver.lock_stage(tmp_path, "native", tmp_path / "run1", {"code_commit": "abc", "created_at_utc": "today"})
    with pytest.raises(FileExistsError):
        driver.lock_stage(tmp_path, "native", tmp_path / "run2", {"code_commit": "abc", "created_at_utc": "today"})


def test_frozen_cohort_tampering_rejected(tmp_path):
    driver.write_json(tmp_path / "freeze.json", {"status": "ready_for_calibration", "cohort_sha256": "not-the-csv"})
    with pytest.raises(ValueError, match="valid_frozen"):
        driver.checked_freeze(tmp_path, tmp_path / "raw")


def bank_fixture():
    rates = np.array([[2.0, 8.0, 15.0, 5.0], [14.0, 5.0, 2.0, 7.0], [3.0, 12.0, 5.0, 1.0], [5.0, 2.0, 9.0, 14.0]])
    return {
        "identity": "Exp_1/s/p",
        "animal": "Exp_1",
        "group": "experimental",
        "session": "s",
        "packet_id": "p",
        "physical_track_id": "a",
        "drug": 1,
        "generating_rates": rates,
        "reference_exposure": np.ones(4) * 2,
        "readout_exposure": np.ones((3, 4)) * 0.5,
        "ripple_counts": np.array([1, 9, 2, 20]),
        "background_counts": np.array([5, 3, 10, 1]),
        "ripple_exposure_s": 0.5,
        "background_exposure_s": 5.0,
        "readout_midpoints_s": np.array([30.0, 55.0, 90.0]),
    }


def test_simulator_uses_shared_R4_and_only_drug_nuisance_for_null(monkeypatch):
    b = bank_fixture()
    captured = []
    original = calibration.paired_scores

    def capture(counts, exposure, models, predictor):
        captured.append((counts.copy(), exposure.copy()))
        return original(counts, exposure, models, predictor)

    monkeypatch.setattr(calibration, "paired_scores", capture)
    a = calibration.simulate_packet(b, "development", "drug_recruitment_exposure", 0, 0)
    c = calibration.simulate_packet(b, "development", "drug_recruitment_exposure", 0, 0)
    assert a == c
    assert captured[0][0].shape[1] == 3
    assert not np.array_equal(captured[0][1], b["readout_exposure"])
    # Scalar expression changes count scale, not the fixed generating map.
    original_rates = b["generating_rates"].copy()
    for scenario in vta.NULLS:
        r = calibration.simulate_packet(b, "development", scenario, 0, 0)
        assert np.isfinite([r[k] for k in vta.SCORE_COLUMNS]).all()
    np.testing.assert_array_equal(original_rates, b["generating_rates"])


def test_development_and_validation_draws_are_distinct():
    a = calibration.rng_for("development", "same", 1).integers(0, 2**31, 20)
    b = calibration.rng_for("validation", "same", 1).integers(0, 2**31, 20)
    assert not np.array_equal(a, b)


def test_prepare_without_metadata_never_accesses_raw_outcomes(monkeypatch, tmp_path):
    request, _, _, _ = inventory()
    request.to_csv(tmp_path / "request.csv", index=False)
    driver.write_json(tmp_path / "state.json", {"sent_at_utc": None})
    monkeypatch.setattr(driver, "provenance", lambda *a, **k: {"input_file_paths": {}, "input_file_sha256": {}, "code_commit": "test"})
    args = SimpleNamespace(
        request_table=tmp_path / "request.csv", request_state=tmp_path / "state.json", verified_metadata=None, dataset_root=tmp_path / "DOES_NOT_EXIST", output_dir=tmp_path / "out"
    )
    driver.prepare(args)
    result = driver.read_json(args.output_dir / "decision.json")
    assert result["status"] == "awaiting_author_request_approval" and not result["drug_effect_scored"]
    assert not (args.output_dir / "freeze.json").exists()


def test_prepare_complete_design_freezes_inputs_and_prior_history(monkeypatch, tmp_path):
    request, author, packets, _ = inventory()
    request.to_csv(tmp_path / "request.csv", index=False)
    author.to_csv(tmp_path / "author.csv", index=False)
    packets.to_csv(tmp_path / "packets.csv", index=False)
    driver.write_json(tmp_path / "manifest.json", {"outputs": {"packets.csv": driver.file_sha256(tmp_path / "packets.csv")}})
    pd.DataFrame([{"animal": "Con_1", "session": "older", "baseline_traversal": 4, "target_traversal": 5}]).to_csv(tmp_path / "old.csv", index=False)
    driver.write_json(tmp_path / "previous.json", {"history_complete_confirmed_by": "test", "sources": [{"path": "old.csv", "sha256": driver.file_sha256(tmp_path / "old.csv")}]})
    driver.write_json(tmp_path / "state.json", {"sent_at_utc": None})
    review = {k: True for k in ("design_approved", "dose_timing_reviewed", "allocation_reviewed", "recording_days_reviewed", "experience_overlap_reviewed")}
    review.update(
        reviewer="test",
        assessment="synthetic design only",
        metadata_sha256=driver.file_sha256(tmp_path / "author.csv"),
        packet_inventory_sha256=driver.file_sha256(tmp_path / "packets.csv"),
    )
    driver.write_json(tmp_path / "review.json", review)
    captured = {}

    def fake_provenance(inputs, **kwargs):
        captured.update(inputs)
        return {"input_file_paths": {}, "input_file_sha256": {}, "code_commit": "test"}

    monkeypatch.setattr(driver, "provenance", fake_provenance)
    args = SimpleNamespace(
        request_table=tmp_path / "request.csv",
        request_state=tmp_path / "state.json",
        verified_metadata=tmp_path / "author.csv",
        dataset_root=tmp_path / "raw",
        output_dir=tmp_path / "out",
        packet_inventory=tmp_path / "packets.csv",
        previous_selections=tmp_path / "previous.json",
        design_review=tmp_path / "review.json",
    )
    driver.prepare(args)
    assert driver.read_json(args.output_dir / "decision.json")["status"] == "ready_for_calibration"
    assert len(pd.read_csv(args.output_dir / "frozen_cohort.csv")) == 24
    assert len([k for k in captured if k.startswith("raw:")]) == 72
    assert "previous_selection_0" in captured and "packet_manifest" in captured
    assert not (args.output_dir / "primary_contrast.csv").exists()


def test_readout_registry_hashes_and_all_roles(tmp_path):
    f = pd.DataFrame([{"animal": "Exp_1", "session": "s", "past_traversal": 3, "baseline_traversal": 4, "target_traversal": 5}])
    f.to_csv(tmp_path / "old.csv", index=False)
    driver.write_json(tmp_path / "previous.json", {"history_complete_confirmed_by": "test", "sources": [{"path": "old.csv", "sha256": driver.file_sha256(tmp_path / "old.csv")}]})
    registry, _ = driver.previous_registry(tmp_path / "previous.json")
    assert set(registry.traversal) == {3, 4, 5}
    assert set(registry.role) == {"past", "baseline", "target"}
    f.assign(target_traversal=6).to_csv(tmp_path / "old.csv", index=False)
    with pytest.raises(ValueError, match="changed_previous"):
        driver.previous_registry(tmp_path / "previous.json")


def test_native_rejects_old_pooled_calibration_before_scoring(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "checked_freeze", lambda *args: ({"freeze_sha256": "abc"}, pd.DataFrame()))
    monkeypatch.setattr(driver, "checked_artifact", lambda *args: {"engineering_screen_passed": True})
    with pytest.raises(ValueError, match="same_cohort_passing_intervention"):
        driver.native_score(SimpleNamespace(frozen_cohort=tmp_path, dataset_root=tmp_path, calibration_dir=tmp_path))


def test_launcher_refuses_other_host(monkeypatch, tmp_path):
    from scripts import launch_kleinman_vta_job as launch

    monkeypatch.setattr(launch.socket, "gethostname", lambda: "gpuserver6000")
    monkeypatch.setattr(launch.sys, "argv", ["launch", "--job-dir", str(tmp_path / "job"), "--", "run_kleinman_vta_intervention.py", "prepare"])
    with pytest.raises(SystemExit, match="restricted to gpuserver4090"):
        launch.main()
    assert not (tmp_path / "job").exists()
