import json

import pandas as pd
import pytest

from scripts import audit_kleinman_vta_archived_coverage as audit
from tests.test_kleinman_vta_intervention import inventory


def test_counts_are_upper_bounds_not_authorized_cohort():
    request, _, packets, previous = inventory()
    previous.loc[0, ["session", "traversal"]] = [packets.iloc[0].session, 3]
    ledger, coverage = audit.coverage_tables(request, packets, previous)
    assert len(coverage) == 12
    assert coverage.remaining_packets_upper_bound.sum() == 23
    assert coverage.same_track_support.eq("unknown").all()
    assert not coverage.complete_scoring_history_verified.any()
    assert not ledger.iloc[0].remaining_upper_bound_only


@pytest.mark.parametrize("role,traversal", [("past", 3), ("baseline", 4), ("target", 5)])
def test_each_readout_role_excludes_the_packet(role, traversal):
    request, _, packets, previous = inventory()
    previous.loc[0, ["session", "traversal", "role"]] = [packets.iloc[0].session, traversal, role]
    ledger, _ = audit.coverage_tables(request, packets, previous)
    assert ledger.known_previous_readout_overlap.sum() == 1


def test_reference_only_reuse_and_duplicate_history_do_not_remove_extra_packets():
    request, _, packets, previous = inventory()
    previous.loc[0, ["session", "traversal"]] = [packets.iloc[0].session, 1]
    ledger, _ = audit.coverage_tables(request, packets, pd.concat([previous, previous]))
    assert ledger.remaining_upper_bound_only.all()


def test_missing_drug_or_all_packets_keeps_explicit_zero_denominators():
    request, _, packets, previous = inventory()
    previous.loc[0, "session"] = packets.iloc[0].session
    packets = packets.loc[~(packets.animal.eq("Con_1") & packets.drug.eq(1))]
    _, coverage = audit.coverage_tables(request, packets, previous)
    assert coverage.loc[coverage.animal.eq("Con_1") & coverage.drug.eq("CNO"), "remaining_packets_upper_bound"].item() == 0
    _, empty = audit.coverage_tables(request, packets.iloc[:0], previous)
    assert len(empty) == 12
    assert empty.remaining_packets_upper_bound.eq(0).all()


def test_novel_and_unavailable_packets_do_not_count():
    request, _, packets, previous = inventory()
    previous.loc[0, "session"] = packets.iloc[0].session
    packets.loc[0, "availability_descriptor"] = False
    packets.loc[1, "novel"] = 1
    request.loc[1, "released_novel"] = "novel"
    ledger, coverage = audit.coverage_tables(request, packets, previous)
    assert coverage.remaining_packets_upper_bound.sum() == 22
    assert not ledger.iloc[:2].remaining_upper_bound_only.any()


def test_conflicting_labels_and_unknown_history_rejected():
    request, _, packets, previous = inventory()
    with pytest.raises(ValueError, match="session_not_in_request"):
        audit.coverage_tables(request, packets, previous)
    previous.loc[0, "session"] = packets.iloc[0].session
    packets.loc[0, "drug"] = 1
    with pytest.raises(ValueError, match="metadata_conflict"):
        audit.coverage_tables(request, packets, previous)


def test_selection_source_must_match_both_producer_and_native_consumer(tmp_path):
    path = tmp_path / "selected.csv"
    path.write_text("animal,session,baseline_traversal,target_traversal\nCon_1,s,2,3\n")
    digest = audit.file_sha256(path)
    producer, consumer = tmp_path / "producer.json", tmp_path / "native.json"
    producer.write_text(json.dumps({"outputs": {path.name: digest}}))
    consumer.write_text(json.dumps({"input_file_sha256": {"selection": digest}}))
    frame = audit.verified_table(path, producer, consumer, "selection")
    assert len(audit.readout_rows(frame, str(path))) == 2
    consumer.write_text(json.dumps({"input_file_sha256": {"selection": "wrong"}}))
    with pytest.raises(ValueError, match="native_selection_hash"):
        audit.verified_table(path, producer, consumer, "selection")
    path.write_text(path.read_text() + "Con_2,s,2,3\n")
    with pytest.raises(ValueError, match="archived_table_hash"):
        audit.verified_table(path, producer)


def test_archive_audit_never_unlocks_freeze_even_with_remaining_coverage(tmp_path, monkeypatch):
    request, _, packets, _ = inventory()
    base = tmp_path / "paper/research/updates"

    def save(frame, relative, manifest_name="manifest.json"):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        (path.parent / manifest_name).write_text(json.dumps({"outputs": {path.name: audit.file_sha256(path)}}))
        return path

    save(request, "kleinman_metadata_request_20260924/kleinman_track_metadata_request.csv", "request_manifest.json")
    save(packets, "kleinman_nonoverlapping_design_20260924/packets.csv")
    for relative, native, key in audit.HISTORY:
        selection = save(packets.iloc[:1], relative)
        path = base / native
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"input_file_sha256": {key: audit.file_sha256(selection)}}))
    monkeypatch.setattr(audit, "provenance", lambda *args, **kwargs: {"code_commit": "fixture", "git_dirty": False})
    output = tmp_path / "out"
    result = audit.audit(tmp_path / "paper", output)
    assert result["all_six_have_both_drugs_in_upper_bound"]
    assert result["remaining_familiar_packets_upper_bound"] == 23
    assert result["known_previous_unique_readouts"] == 3
    assert not result["cohort_freezable"]
    assert not result["native_outcomes_scored"]
    assert not result["drug_effect_scored"]
    assert not result["complete_scoring_history_verified"]
    spec = json.loads((output / "previous_scoring_registry_provisional.json").read_text())
    assert spec["history_complete_confirmed_by"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert audit.file_sha256(output / name) == digest
    assert not (output / "frozen_cohort.csv").exists()
    with pytest.raises(FileExistsError):
        audit.audit(tmp_path / "paper", output)
