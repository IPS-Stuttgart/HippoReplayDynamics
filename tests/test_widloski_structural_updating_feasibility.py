"""Synthetic inputs only. No replay content or biological outcome is needed."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts import audit_widloski_structural_updating_feasibility as audit
from scripts import launch_widloski_structural_updating as launcher
from scripts import widloski_structural_updating_core as core


@pytest.fixture
def protocol():
    return audit.read_json(audit.DEFAULT_PROTOCOL)


@pytest.fixture
def graph():
    coordinates = {"A": (0, 0), "B": (10, 0), "C": (0, 10), "Q": (20, 0), "D": (10, 10), "X": (40, 0), "Y": (40, 10), "Z": (50, 0)}
    triples = [("A", "B", 10), ("A", "C", 10), ("B", "D", 10), ("B", "Q", 10), ("Q", "D", 20), ("C", "D", 10), ("X", "Y", 10), ("X", "Z", 10), ("Y", "D", 30), ("Z", "D", 30)]
    edges = [{"id": a + b, "u": a, "v": b, "length_cm": n} for a, b, n in triples]
    return {
        "documented": True,
        "behaviorally_checked": True,
        "documentation": "Synthetic checked fixture",
        "length_unit": "cm",
        "nodes": [{"id": n, "x_cm": x, "y_cm": y, "radius_cm": 1} for n, (x, y) in coordinates.items()],
        "edges": edges,
        "configurations": [{"id": "old", "open_edges": [e["id"] for e in edges]}, {"id": "new", "open_edges": [e["id"] for e in edges if e["id"] != "BD"]}],
        "destinations": ["D"],
    }


def file_spec(path, clock="clock"):
    return {"path": str(path), "sha256": core.file_hash(path), "clock_id": clock, "documentation": "Synthetic column contract in this test"}


@pytest.fixture
def metadata(tmp_path, graph):
    witness = tmp_path / "witness.json"
    witness.write_text('{"synthetic": true}')
    sessions = []
    for i in (1, 2):
        sid = f"Billy3/2020010{i}/run1"
        clock = f"clock{i}"
        t = np.arange(51) / 10
        x = np.where(t < 2.5, 0, 40) if i == 1 else np.clip((t - 2) * 20, 0, 10)
        pos = pd.DataFrame({"t_s": t, "x_cm": x, "y_cm": 0, "speed_cm_s": np.where((t >= 2) & (t <= 2.5), 20, 0), "supported": True})
        position = tmp_path / f"position{i}.csv"
        pos.to_csv(position, index=False)
        ev = pd.DataFrame(
            [("early", 0.2, 0.4, True), ("straddles", 1.8, 2.1, True), ("late", 3, 3.2, True), ("invalid", 0.5, 0.7, False)],
            columns=["event_id", "start_time_s", "end_time_s", "validated"],
        )
        event = tmp_path / f"events{i}.csv"
        ev.to_csv(event, index=False)
        traversal = tmp_path / f"traversals{i}.csv"
        pd.DataFrame([("first", "AB", "A", "B", 2, 2.5)] if i == 2 else [], columns=["traversal_id", "edge_id", "from_node", "to_node", "start_time_s", "end_time_s"]).to_csv(
            traversal, index=False
        )
        sessions.append(
            {
                "animal": "Billy3",
                "session_id": sid,
                "date": f"2020-01-0{i}",
                "session_order": i,
                "clock_id": clock,
                "time_alignment_verified": True,
                "entry_time_s": 0,
                "end_time_s": 5,
                "configuration_id": "old" if i == 1 else "new",
                "home_goal_node": "D",
                "traversals_complete": True,
                "previous_session_id": "Billy3/20200101/run1" if i == 2 else None,
                "intervening_exposure_complete": True,
                "position": file_spec(position, clock),
                "events": file_spec(event, clock),
                "traversals": file_spec(traversal, clock),
                "capabilities": {c: {**file_spec(witness), "status": "verified"} for c in core.CAPABILITIES},
            }
        )
    return {"schema_version": 1, "reviewed_by": "synthetic fixture", "documentation": "Not real metadata", "graph": graph, "sessions": sessions}


def make_hdf(path, records):
    with h5py.File(path, "w") as h:
        data = h.create_group("Data")
        refs = h.create_group("#refs#")
        for field in ["ratName", "file", "rat", "day", "sessionNum"]:
            ds = data.create_dataset(field, (len(records), 1), dtype=h5py.ref_dtype)
            for i, record in enumerate(records):
                v = record[field]
                a = np.array([ord(c) for c in v], dtype=np.uint16)[:, None] if isinstance(v, str) else np.array([[v]])
                ds[i, 0] = refs.create_dataset(f"{field}{i}", data=a).ref
        # These intentionally unusable fields must never be interpreted as tracking.
        data.create_dataset("ratLocs", data=[np.nan])
        data.create_dataset("replays", data=[np.nan])


@pytest.fixture
def source_args(tmp_path, protocol):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    make_hdf(dataset / "sessions.mat", [{"ratName": "Billy3", "file": "20200101", "rat": 1, "day": 1, "sessionNum": 1}])
    stats = np.zeros((2, 46))
    stats[:, 42:] = [[1, 1, 1, 1], [2, 1, 1, 1]]
    savemat(dataset / "events.mat", {"replays_stats": stats})
    source = audit.read_json(audit.DEFAULT_SOURCES)
    source["files"] = [
        {"filename": "sessions.mat", "reader": "session_inventory_v1", "source_url": "synthetic", "documentation": "Synthetic identity contract"},
        {"filename": "events.mat", "reader": "event_identities_v1", "source_url": "synthetic", "documentation": "Synthetic identity contract"},
    ]
    manifest, p = tmp_path / "source.json", tmp_path / "protocol.json"
    manifest.write_text(json.dumps(source))
    p.write_text(json.dumps(protocol))
    return audit.parser().parse_args(
        ["--dataset-root", str(dataset), "--source-manifest", str(manifest), "--protocol", str(p), "--output-dir", str(tmp_path / "out"), "--stage", "inventory"]
    )


def test_documented_matlab_and_hdf5_import(source_args):
    summary = audit.inventory(source_args)
    assert summary["source_integrity_valid"]
    assert (summary["sessions"], summary["resolved_event_rows"]) == (1, 2)
    rows = audit.read_session_inventory(source_args.dataset_root / "sessions.mat", ["Billy3"])
    assert rows[0]["session_id"] == "Billy3/20200101/run1"
    assert audit.inspect_file(source_args.dataset_root / "events.mat")[0]["shape"] == [2, 46]


def test_missing_identity_never_assigned_by_row_order(source_args):
    p = source_args.dataset_root / "events.mat"
    a = np.zeros((2, 46))
    a[:, 42:] = [[1, 1, 1, 1], [np.nan] * 4]
    savemat(p, {"replays_stats": a})
    result = audit.inventory(source_args)
    assert result["unresolved_event_rows"] == 1
    assert not result["source_integrity_valid"]
    rows = pd.read_csv(source_args.output_dir / "inventory" / (audit.PREFIX + "native_event_identity_inventory.csv"))
    assert pd.isna(rows.iloc[1].session_id)


@pytest.mark.parametrize("bad", ["undocumented", "checksum", "duplicate", "animal"])
def test_source_conflicts_block_inventory(source_args, bad):
    if bad in {"undocumented", "checksum"}:
        s = audit.read_json(source_args.source_manifest)
        if bad == "undocumented":
            del s["files"][0]["documentation"]
            source_args.source_manifest.write_text(json.dumps(s))
            with pytest.raises(core.MetadataError, match="Undocumented"):
                audit.inventory(source_args)
            return
        s["files"][0]["expected_md5"] = "0" * 32
        source_args.source_manifest.write_text(json.dumps(s))
    else:
        r = {"ratName": "other" if bad == "animal" else "Billy3", "file": "20200101", "rat": 1, "day": 1, "sessionNum": 1}
        make_hdf(source_args.dataset_root / "sessions.mat", [r, r] if bad == "duplicate" else [r])
    assert not audit.inventory(source_args)["source_integrity_valid"]


def test_duplicate_native_event_id_rejected(tmp_path):
    p = tmp_path / "duplicate.mat"
    a = np.zeros((2, 46))
    a[:, 42:] = [1, 1, 1, 1]
    savemat(p, {"replays_stats": a})
    with pytest.raises(core.MetadataError, match="Duplicate"):
        audit.read_event_identities(p, ["Billy3"])


def test_graph_remote_change_local_exclusion_and_ties(graph):
    rows, exclusions = core.route_contrasts(graph, "old", "new")
    a = next(r for r in rows if r["junction"] == "A")
    assert a["old_best"] == ["B", "C"]
    assert a["new_best"] == ["C"]
    # Independently sum the simple routes A-B-Q-D and A-C-D.
    assert a["old_costs"] == [10 + 10, 10 + 10]
    assert a["new_costs"] == [10 + 10 + 20, 10 + 10]
    assert a["structural_predictors"] == [-20, 0]
    assert any(e["junction"] == "B" and e["reason"] == "local_connectivity_changed" for e in exclusions)
    assert next(r for r in rows if r["junction"] == "X")["kind"] == "unaffected"
    assert all(r["kind"] == "unaffected" for r in core.route_contrasts(graph, "old", "old")[0])


def test_goal_change_alone_is_not_barrier_change(metadata, tmp_path, protocol):
    metadata["sessions"][1].update(configuration_id="old", home_goal_node="C")
    result = core.build_opportunities(metadata, tmp_path, protocol)
    assert result["opportunities"] == []
    assert any(e["reason"] == "no_barrier_change" for e in result["exclusions"])


def test_undocumented_geometry_and_unit_contract(graph):
    for key, value in [("documented", False), ("behaviorally_checked", False), ("length_unit", "m")]:
        bad = {**graph, key: value}
        with pytest.raises(core.MetadataError):
            core.load_graph(bad)


@pytest.mark.parametrize("change", ["identity", "order", "date", "alignment", "clock"])
def test_metadata_identity_order_and_clock_conflicts(metadata, tmp_path, protocol, change):
    s = metadata["sessions"][1]
    if change == "identity":
        s["session_id"] = metadata["sessions"][0]["session_id"]
    elif change == "order":
        s["session_order"] = 1
    elif change == "date":
        s["date"] = "2019-12-01"
    elif change == "alignment":
        s["time_alignment_verified"] = False
    else:
        s["position"]["clock_id"] = "unrelated-clock"
        result = core.build_opportunities(metadata, tmp_path, protocol)
        assert not result["measurement_complete"]
        assert any("Clock mismatch" in e["reason"] for e in result["exclusions"])
        return
    with pytest.raises(core.MetadataError):
        core.build_opportunities(metadata, tmp_path, protocol)


def test_missing_intervening_exposure_not_bridged(metadata, tmp_path, protocol):
    metadata["sessions"][1]["intervening_exposure_complete"] = False
    result = core.build_opportunities(metadata, tmp_path, protocol)
    assert not result["opportunities"]
    assert any("unobserved_exposure" in e["reason"] for e in result["exclusions"])


def test_gap_censors_at_left_sample_never_resumes():
    s = {"entry_time_s": 0, "end_time_s": 1.0}
    p = pd.DataFrame({"t_s": [0, 0.1, 0.2, 0.6, 0.7, 1], "x_cm": 0, "y_cm": 0, "speed_cm_s": 0, "supported": True})
    p = core.tracking(p, s)
    assert core.history_horizon(p, s) == (0.2, "tracking_gap")
    assert core.immobile_intervals(p, 0, 0.2) == [[0.0, 0.2]]
    assert not core.contained(0.1, 0.3, [[0, 0.2], [0.6, 0.7]])
    p.loc[0, "supported"] = False
    assert core.history_horizon(p, s) == (0, "unsupported_entry")


def test_first_traversal_containment_and_duplicate_memberships(metadata, tmp_path, protocol):
    result = core.build_opportunities(metadata, tmp_path, protocol)
    assert result["measurement_complete"]
    a = next(o for o in result["opportunities"] if o["junction"] == "A")
    assert a["end_time_s"] == 2  # departure, not arrival at t=2.5
    assert a["censor_reason"] == "first_outgoing_traversal"
    assert a["matched_control_id"]
    members = [m["event_id"] for m in result["memberships"] if m["opportunity_id"] == a["opportunity_id"]]
    assert members == ["early"]
    coverage = core.coverage(result["opportunities"], result["memberships"], protocol)
    b = coverage[0]
    assert b["event_opportunity_memberships"] > b["unique_events"]
    assert b["eligible_episodes"] == 1
    assert b["status"] == "fail"  # no vacuous minimum-coverage pass


def test_event_clock_validation_and_overlap(metadata, tmp_path, protocol):
    s = metadata["sessions"][1]
    p = Path(s["events"]["path"])
    frame = pd.read_csv(p)
    frame.loc[1, ["start_time_s", "end_time_s"]] = [0.3, 0.5]
    frame.to_csv(p, index=False)
    s["events"] = file_spec(p, s["clock_id"])
    result = core.build_opportunities(metadata, tmp_path, protocol)
    assert any(e["reason"] == "overlapping_native_intervals_retained_as_distinct_ids" for e in result["exclusions"])
    frame.loc[0, "end_time_s"] = 6
    with pytest.raises(core.MetadataError, match="outside session clock"):
        core.validate_events(frame, s)
    frame.loc[0, "end_time_s"] = 0.4
    frame.loc[1, "event_id"] = frame.loc[0, "event_id"]
    with pytest.raises(core.MetadataError, match="Duplicate"):
        core.validate_events(frame, s)


def design_fixture(confounded=False):
    rows = []
    rng = np.random.default_rng(22)
    for episode in range(12):
        for junction in ["A", "X"]:
            for alternative in ["B", "C"]:
                d = dict.fromkeys(core.DESIGN_COLUMNS, "")
                d.update(
                    opportunity_id=f"{episode}{junction}",
                    animal="Billy3",
                    date="2020-01-01",
                    episode_id=str(episode),
                    old_configuration="old",
                    new_configuration=f"new{episode % 2}",
                    old_goal="D",
                    new_goal=f"G{episode % 3}",
                    junction=junction,
                    destination="D",
                    alternative=alternative,
                    exposure_s=2.0,
                    old_relative_cost_cm=0.0,
                    structural_predictor_cm=float(episode % 3) if confounded else float(rng.normal()),
                )
                rows.append(d)
    return rows


def test_perfect_confound_and_crossed_design():
    bad, _ = core.confounding_audit(design_fixture(True))
    good, support = core.confounding_audit(design_fixture())
    assert not bad["identifiable"] and bad["target_rank_gain"] == 0
    assert good["identifiable"] and good["target_rank_gain"] == 1
    assert all(s["supported"] for s in support)
    assert core.confounding_audit([])[0]["status"] == "unavailable"
    one_change = design_fixture()
    for row in one_change:
        row["new_configuration"] = "only-one"
    assert not core.confounding_audit(one_change)[0]["identifiable"]


def test_empty_missing_vs_zero_and_ready_gates(protocol):
    absent = core.coverage([], [], protocol, available=False)
    empty = core.coverage([], [], protocol, available=True)
    assert all(r["unique_events"] is None and r["status"] == "unavailable" for r in absent)
    assert all(r["unique_events"] == 0 and r["status"] == "fail" for r in empty)
    caps = [{"capability": c, "status": "verified"} for c in core.CAPABILITIES]
    good = [{"animal": a, "status": "pass"} for a in protocol["animals"]]
    confound = {"status": "pass", "identifiable": True, "reason": "synthetic"}

    def decision(capabilities=caps, animals=good, novelty="distinct_contrast_supported", c=confound, available=True):
        return core.gates_and_decision(capabilities, animals, novelty, c, available)[1]["decision"]

    assert decision() == "ready_for_readout_feasibility"
    assert decision(animals=empty) == "insufficient_opportunities"
    assert decision(novelty="unresolved") == "novelty_unresolved"
    assert decision(c={"status": "fail", "identifiable": False, "reason": "collinear"}) == "confounded_design"
    assert decision(capabilities=[]) == "blocked_data"
    cap = copy.deepcopy(caps)
    next(c for c in cap if c["capability"] == "barrier_geometry")["status"] = "unresolved"
    assert decision(capabilities=cap) == "blocked_metadata"
    cap = copy.deepcopy(caps)
    next(c for c in cap if c["capability"] == "sorted_spikes")["status"] = "absent"
    assert decision(capabilities=cap) == "blocked_data"


def test_full_cli_blocked_pipeline_and_nonreading_report(source_args, monkeypatch):
    args = [
        "--dataset-root",
        str(source_args.dataset_root),
        "--source-manifest",
        str(source_args.source_manifest),
        "--protocol",
        str(source_args.protocol),
        "--output-dir",
        str(source_args.output_dir),
    ]
    assert audit.main(args + ["--stage", "inventory"]) == 0
    assert audit.main(args + ["--stage", "opportunities"]) == 0
    # Reporting remains possible after the recordings have been disconnected.
    for p in source_args.dataset_root.iterdir():
        p.unlink()
    monkeypatch.setattr(audit, "loadmat", lambda *a, **k: pytest.fail("report reopened a recording"))
    monkeypatch.setattr(core, "build_opportunities", lambda *a, **k: pytest.fail("report recomputed opportunities"))
    assert audit.main(args + ["--stage", "report"]) == 0
    assert audit.main(args + ["--stage", "report"]) == 0
    d = audit.read_json(source_args.output_dir / "report" / "decision.json")
    assert d["decision"] == "blocked_data"
    assert d["eligible_unique_events"] is None
    assert not d["author_request_sent"] and not d["biological_analysis_performed"]
    assert "unavailable, not zero" in (source_args.output_dir / "report" / (audit.PREFIX + "report.md")).read_text()


def test_artifact_and_input_changes_rejected(source_args):
    audit.inventory(source_args)
    # Unchanged restart is safe; source changes are not.
    audit.inventory(source_args)
    p = source_args.dataset_root / "events.mat"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(core.MetadataError, match="changed|mismatch"):
        audit.inventory(source_args)
    with pytest.raises(core.MetadataError, match="changed|mismatch"):
        audit.opportunities(source_args)
    out = source_args.output_dir / "inventory"
    (out / "inventory_summary.json").write_text("{}")
    with pytest.raises(core.MetadataError, match="Artifact hash mismatch"):
        audit.verify_stage(out)


def test_unverified_capability_and_tampered_file(metadata, tmp_path):
    assert all(r["status"] == "verified" for r in core.capability_rows(metadata, tmp_path))
    Path(metadata["sessions"][0]["capabilities"]["sorted_spikes"]["path"]).write_text("changed")
    assert all(r["status"] == "unresolved" for r in core.capability_rows(metadata, tmp_path))


def test_known_empty_inventory_differs_from_missing(source_args):
    savemat(source_args.dataset_root / "events.mat", {"replays_stats": np.empty((0, 46))})
    known = audit.inventory(source_args)
    assert known["published_events"] == 0 and known["source_integrity_valid"]
    source_args.output_dir = source_args.output_dir.parent / "out-missing"
    (source_args.dataset_root / "events.mat").unlink()
    missing = audit.inventory(source_args)
    assert missing["published_events"] is None and not missing["source_integrity_valid"]
    audit.opportunities(source_args)
    audit.report(source_args.output_dir)


def test_verified_end_to_end_opportunities_and_behavior_panel(source_args, metadata, tmp_path):
    records = [{"ratName": "Billy3", "file": f"2020010{i}", "rat": 1, "day": i, "sessionNum": 1} for i in (1, 2)]
    make_hdf(source_args.dataset_root / "sessions.mat", records)
    a = np.zeros((2, 46))
    a[:, 42:] = [[1, 1, 1, 1], [1, 1, 2, 1]]
    savemat(source_args.dataset_root / "events.mat", {"replays_stats": a})
    source_args.verified_metadata = tmp_path / "verified.json"
    source_args.verified_metadata.write_text(json.dumps(metadata))
    assert audit.inventory(source_args)["metadata_valid"]
    d = audit.opportunities(source_args)
    assert d["opportunity_measurement_available"] and d["eligible_unique_events"] > 0
    assert d["decision"] == "novelty_unresolved"
    audit.report(source_args.output_dir)
    assert (source_args.output_dir / "opportunities/geometry_timeline_1.png").stat().st_size > 1000
    summary = pd.read_csv(source_args.output_dir / "opportunities" / (audit.PREFIX + "animal_coverage.csv"))
    assert not summary.status.eq("pass").any()


def test_source_metadata_conflict_is_explicit(source_args, metadata, tmp_path):
    source_args.verified_metadata = tmp_path / "metadata-conflict.json"
    # The source has one session; silently selecting this unrelated cohort is forbidden.
    source_args.verified_metadata.write_text(json.dumps(metadata))
    result = audit.inventory(source_args)
    assert not result["metadata_valid"]
    assert any("every source session" in issue for issue in result["issues"])


def test_changed_protocol_refuses_checkpoint_mix(source_args):
    audit.inventory(source_args)
    p = audit.read_json(source_args.protocol)
    p["annotation"] = "changed contract"
    source_args.protocol.write_text(json.dumps(p))
    with pytest.raises(core.MetadataError, match="changed inputs"):
        audit.inventory(source_args)


def test_launcher_detaches_and_requires_clean_server_checkout(tmp_path, monkeypatch):
    calls = []
    job = tmp_path / "job"
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(launcher.sys, "argv", ["launch", "--job-dir", str(job), "--", "--output-dir", str(tmp_path / "out")])
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda cmd, **kw: "" if "status" in cmd else "a" * 40)

    class Detached:
        pid = 123

        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(launcher.subprocess, "Popen", Detached)
    assert launcher.main() == 0
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["stdin"] == launcher.subprocess.DEVNULL
    assert audit.read_json(job / "launch.json")["supervisor_pid"] == 123
    assert audit.read_json(job / "job.json")["code_commit"] == "a" * 40
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **kw: " M concurrent.py")
    with pytest.raises(SystemExit):
        launcher.main()


def test_launcher_supervisor_records_failed_stage(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    (job / "job.json").write_text(json.dumps({"arguments": ["--output-dir", "unused"]}))
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(launcher.sys, "argv", ["launch", "--job-dir", str(job), "--supervise"])
    calls = []

    def failure(cmd, **kwargs):
        calls.append(cmd)
        return 2

    monkeypatch.setattr(launcher.subprocess, "call", failure)
    assert launcher.main() == 2
    assert len(calls) == 1
    assert audit.read_json(job / "status.json")["failed_stage"] == "inventory"
