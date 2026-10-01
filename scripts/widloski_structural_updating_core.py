"""Effect-blind geometry, history and coverage checks; no replay decoder."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CAPABILITIES = (
    "full_session_position",
    "sorted_spikes",
    "cross_session_unit_identities",
    "replay_intervals",
    "barrier_geometry",
    "session_chronology",
    "goal_trial_timing",
    "prechange_encoding_support",
)
DATA_CAPABILITIES = {"full_session_position", "sorted_spikes", "cross_session_unit_identities", "replay_intervals"}
OPPORTUNITY_CAPABILITIES = {"full_session_position", "replay_intervals", "barrier_geometry", "session_chronology", "goal_trial_timing"}
SESSION_COLUMNS = ["animal", "date", "session_id", "day_index", "session_number", "source_row", "published_events", "identity_status"]
OPPORTUNITY_COLUMNS = [
    "opportunity_id",
    "animal",
    "date",
    "session_id",
    "previous_session_id",
    "episode_id",
    "old_configuration",
    "new_configuration",
    "old_goal",
    "new_goal",
    "goal_changed",
    "junction",
    "destination",
    "kind",
    "start_time_s",
    "end_time_s",
    "censor_reason",
    "exposure_s",
    "outdegree",
    "old_min_cost_cm",
    "route_change_magnitude_cm",
    "eligible_event_count",
    "matched_control_id",
    "matched_affected_ids",
    "match_status",
]
MEMBERSHIP_COLUMNS = ["animal", "date", "session_id", "episode_id", "opportunity_id", "event_id", "start_time_s", "end_time_s", "kind"]
EXCLUSION_COLUMNS = ["animal", "session_id", "junction", "destination", "event_id", "reason"]
DESIGN_COLUMNS = [
    "opportunity_id",
    "animal",
    "date",
    "episode_id",
    "old_configuration",
    "new_configuration",
    "old_goal",
    "new_goal",
    "junction",
    "destination",
    "alternative",
    "old_cost_cm",
    "new_cost_cm",
    "old_relative_cost_cm",
    "new_relative_cost_cm",
    "structural_predictor_cm",
    "exposure_s",
]


class MetadataError(ValueError):
    """An input cannot be interpreted under the documented adapter contract."""


def require(condition, message):
    if not condition:
        raise MetadataError(message)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_file(spec, base, clock_id=None):
    require(isinstance(spec, dict), "Missing documented file specification")
    require(bool(spec.get("documentation")), "Missing file/column documentation")
    path = Path(spec.get("path", ""))
    path = path if path.is_absolute() else Path(base) / path
    require(path.is_file(), f"Missing declared file: {path}")
    require(file_hash(path) == spec.get("sha256"), f"Input hash mismatch: {path}")
    if clock_id is not None:
        require(spec.get("clock_id") == clock_id, f"Clock mismatch: {path}")
    return path.resolve()


def capability_rows(metadata, base):
    rows = []
    for session in metadata.get("sessions", []):
        for capability in CAPABILITIES:
            spec = session.get("capabilities", {}).get(capability, {})
            status = spec.get("status", "unresolved")
            require(status in {"verified", "absent", "unresolved"}, f"Invalid availability status for {capability}")
            reason, path, sha = spec.get("reason", "No evidence supplied"), "", ""
            if status == "verified":
                try:
                    path = str(checked_file(spec, base))
                    sha = spec["sha256"]
                    reason = spec["documentation"]
                except MetadataError as exc:
                    status, reason = "unresolved", str(exc)
            rows.append(
                {
                    "animal": session["animal"],
                    "session_id": session["session_id"],
                    "capability": capability,
                    "status": status,
                    "reason": reason,
                    "evidence_path": path,
                    "evidence_sha256": sha,
                }
            )
    return rows


def validate_metadata(metadata):
    require(metadata.get("schema_version") == 1, "Unsupported metadata schema")
    require(metadata.get("reviewed_by") and metadata.get("documentation"), "Verified metadata require a reviewer and documentation")
    sessions = metadata.get("sessions", [])
    keys, orders = set(), set()
    for s in sessions:
        key = (s.get("animal"), s.get("session_id"))
        require(all(isinstance(x, str) and x for x in key), "Missing animal/session identity")
        require(key not in keys, "Conflicting/duplicate session identities")
        keys.add(key)
        order = s.get("session_order")
        require(type(order) is int and order >= 1, "Session order must be a documented positive integer")
        require((s["animal"], order) not in orders, "Duplicate animal/session order")
        orders.add((s["animal"], order))
        require(s.get("date") and s.get("clock_id"), "Missing recording day or clock")
        try:
            date.fromisoformat(s["date"])
        except (TypeError, ValueError) as exc:
            raise MetadataError("Recording date must be an ISO calendar date") from exc
        require(s.get("time_alignment_verified") is True, "Unverified timestamp alignment")
        bounds = np.asarray([s.get("entry_time_s"), s.get("end_time_s")], dtype=float)
        require(np.isfinite(bounds).all() and bounds[1] > bounds[0], "Invalid session boundaries")
    return sorted(sessions, key=lambda s: (s["animal"], s["session_order"]))


def load_graph(graph):
    require(graph.get("documented") is True and graph.get("behaviorally_checked") is True and graph.get("documentation"), "Maze graph must be documented and behaviorally checked")
    require(graph.get("length_unit") == "cm", "Graph lengths must be verified centimetres")
    nodes, edges, configs = {}, {}, {}
    for n in graph.get("nodes", []):
        key = n["id"]
        require(key not in nodes, "Duplicate graph node")
        values = np.array([n["x_cm"], n["y_cm"], n["radius_cm"]], float)
        require(np.isfinite(values).all() and values[2] > 0, "Invalid documented junction region")
        nodes[key] = n
    require(bool(nodes), "Empty maze graph")
    pairs = set()
    for e in graph.get("edges", []):
        pair = tuple(sorted((e["u"], e["v"])))
        require(e["id"] not in edges and pair not in pairs, "Duplicate graph edge")
        require(e["u"] in nodes and e["v"] in nodes and e["u"] != e["v"], "Unknown edge endpoint or self-loop")
        require(np.isfinite(e["length_cm"]) and e["length_cm"] > 0, "Invalid graph length")
        edges[e["id"]] = e
        pairs.add(pair)
    for c in graph.get("configurations", []):
        require(c["id"] not in configs, "Duplicate configuration identity")
        opened = c["open_edges"]
        require(len(opened) == len(set(opened)) and set(opened) <= set(edges), "Unknown/duplicate configuration edge")
        configs[c["id"]] = set(opened)
    destinations = graph.get("destinations", [])
    require(destinations and len(destinations) == len(set(destinations)) and set(destinations) <= set(nodes), "Undocumented destinations")
    return nodes, edges, configs, sorted(destinations)


def adjacency(nodes, edges, opened):
    ids = sorted(nodes)
    index = {n: i for i, n in enumerate(ids)}
    a = np.zeros((len(ids), len(ids)))
    neighbors = {n: {} for n in ids}
    for eid in opened:
        e = edges[eid]
        a[index[e["u"]], index[e["v"]]] = e["length_cm"]
        a[index[e["v"]], index[e["u"]]] = e["length_cm"]
        neighbors[e["u"]][e["v"]] = e["length_cm"]
        neighbors[e["v"]][e["u"]] = e["length_cm"]
    return ids, index, neighbors, dijkstra(csr_matrix(a), directed=False)


def route_contrasts(graph, old_configuration, new_configuration, tolerance=1e-8):
    """Compare all fixed destinations, never the previous goal against a new goal."""
    nodes, edges, configs, destinations = load_graph(graph)
    require(old_configuration in configs and new_configuration in configs, "Missing configuration geometry")
    ids, index, old_neighbors, _ = adjacency(nodes, edges, configs[old_configuration])
    _, _, new_neighbors, _ = adjacency(nodes, edges, configs[new_configuration])
    rows, excluded = [], []
    for junction in ids:
        if old_neighbors[junction] != new_neighbors[junction]:
            excluded.append({"junction": junction, "destination": "", "reason": "local_connectivity_changed"})
            continue
        if len(old_neighbors[junction]) < 2:
            continue
        alternatives = sorted(old_neighbors[junction])
        # A committed outgoing alternative cannot immediately return through this junction.
        old_open = {e for e in configs[old_configuration] if junction not in (edges[e]["u"], edges[e]["v"])}
        new_open = {e for e in configs[new_configuration] if junction not in (edges[e]["u"], edges[e]["v"])}
        _, _, _, old_dist = adjacency(nodes, edges, old_open)
        _, _, _, new_dist = adjacency(nodes, edges, new_open)
        for destination in destinations:
            if destination == junction:
                continue
            a = np.array([old_neighbors[junction][v] + old_dist[index[v], index[destination]] for v in alternatives])
            b = np.array([new_neighbors[junction][v] + new_dist[index[v], index[destination]] for v in alternatives])
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                excluded.append({"junction": junction, "destination": destination, "reason": "destination_unreachable"})
                continue
            ra, rb = a - a.min(), b - b.min()
            delta = ra - rb
            rows.append(
                {
                    "junction": junction,
                    "destination": destination,
                    "alternatives": alternatives,
                    "old_costs": a.tolist(),
                    "new_costs": b.tolist(),
                    "old_relative": ra.tolist(),
                    "new_relative": rb.tolist(),
                    "structural_predictors": delta.tolist(),
                    "outdegree": len(alternatives),
                    "old_min_cost_cm": float(a.min()),
                    "old_best": [v for v, c in zip(alternatives, ra, strict=True) if c <= tolerance],
                    "new_best": [v for v, c in zip(alternatives, rb, strict=True) if c <= tolerance],
                    "kind": "affected" if np.max(np.abs(delta)) > tolerance else "unaffected",
                    "route_change_magnitude_cm": float(np.max(np.abs(delta))),
                }
            )
    return rows, excluded


def read_table(spec, base, clock, columns):
    frame = pd.read_csv(checked_file(spec, base, clock), keep_default_na=False)
    require(set(columns) <= set(frame), f"Missing columns: {sorted(set(columns) - set(frame))}")
    return frame


def bool_array(series, name):
    values = []
    for value in series:
        require(str(value).lower() in {"true", "false", "1", "0"}, f"Invalid boolean: {name}")
        values.append(str(value).lower() in {"true", "1"})
    return np.array(values, dtype=bool)


def tracking(frame, session):
    result = frame.copy()
    for col in ("t_s", "x_cm", "y_cm", "speed_cm_s"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    t = result.t_s.to_numpy()
    require(len(t) >= 2 and np.isfinite(t).all() and np.all(np.diff(t) > 0), "Tracking times are missing, duplicate or out of order")
    require(np.all((t >= session["entry_time_s"]) & (t <= session["end_time_s"])), "Tracking outside declared session clock")
    result["supported"] = bool_array(result.supported, "tracking.supported")
    valid = result.supported.to_numpy() & np.isfinite(result[["x_cm", "y_cm", "speed_cm_s"]]).all(axis=1).to_numpy()
    valid &= result.speed_cm_s.to_numpy() >= 0
    result["supported"] = valid
    return result


def history_horizon(frame, session, gap_s=0.25, tolerance=1e-8):
    """Censor at the last supported sample BEFORE a missing interval, not after it."""
    t, valid = frame.t_s.to_numpy(), frame.supported.to_numpy()
    entry = session["entry_time_s"]
    if t[0] > entry + tolerance or not valid[0]:
        return entry, "unsupported_entry"
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > gap_s + tolerance or not valid[i]:
            return float(t[i - 1]), "tracking_gap"
    if t[-1] < session["end_time_s"] - tolerance:
        return float(t[-1]), "tracking_ends_before_session"
    return float(session["end_time_s"]), "session_end"


def immobile_intervals(frame, start, end, speed=5.0, gap_s=0.25):
    t = frame.t_s.to_numpy()
    good = frame.supported.to_numpy() & (frame.speed_cm_s.to_numpy() < speed)
    intervals = []
    for i in range(len(t) - 1):
        if good[i] and good[i + 1] and t[i + 1] - t[i] <= gap_s + 1e-8:
            lo, hi = max(start, t[i]), min(end, t[i + 1])
            if hi > lo:
                if intervals and abs(intervals[-1][1] - lo) < 1e-8:
                    intervals[-1][1] = float(hi)
                else:
                    intervals.append([float(lo), float(hi)])
    return intervals


def contained(start, end, intervals):
    return any(start >= lo and end <= hi for lo, hi in intervals)


def previous_visit(frame, node):
    distance = np.hypot(frame.x_cm - node["x_cm"], frame.y_cm - node["y_cm"])
    return bool(np.any(frame.supported & (distance <= node["radius_cm"])))


def validate_traversals(frame, session, graph, positions):
    nodes, edges, configs, _ = load_graph(graph)
    opened = configs[session["configuration_id"]]
    require(not frame.traversal_id.astype(str).duplicated().any(), "Duplicate traversal identifiers")
    bounds = frame[["start_time_s", "end_time_s"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    require(np.isfinite(bounds).all(), "Non-finite traversal timestamps")
    previous_end = -np.inf
    supported = immobile_intervals(positions, session["entry_time_s"], session["end_time_s"], speed=np.inf)
    for row in frame.to_dict("records"):
        start, end = float(row["start_time_s"]), float(row["end_time_s"])
        require(session["entry_time_s"] <= start < end <= session["end_time_s"], "Traversal outside session clock")
        require(start >= previous_end, "Overlapping or out-of-order traversals")
        previous_end = end
        require(row["edge_id"] in opened, "Behavioral traversal crosses a blocked/unknown edge")
        edge = edges[row["edge_id"]]
        require({row["from_node"], row["to_node"]} == {edge["u"], edge["v"]}, "Traversal endpoint/edge mismatch")
        require(contained(start, end, supported), "Traversal crosses unsupported tracking")
        for time, node_id in ((start, row["from_node"]), (end, row["to_node"])):
            node = nodes[node_id]
            x = np.interp(time, positions.t_s, positions.x_cm)
            y = np.interp(time, positions.t_s, positions.y_cm)
            require(np.hypot(x - node["x_cm"], y - node["y_cm"]) <= node["radius_cm"] + 1e-8, "Traversal is not supported by the declared junction regions")
    return frame


def validate_events(frame, session):
    result = frame.copy()
    result["event_id"] = result.event_id.astype(str)
    require(not result.event_id.duplicated().any() and result.event_id.ne("").all(), "Duplicate/missing native event identifiers")
    for col in ("start_time_s", "end_time_s"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    require(np.isfinite(result[["start_time_s", "end_time_s"]]).all().all(), "Non-finite event timestamps")
    require(
        ((result.start_time_s >= session["entry_time_s"]) & (result.end_time_s <= session["end_time_s"]) & (result.end_time_s > result.start_time_s)).all(),
        "Native event outside session clock",
    )
    result["validated"] = bool_array(result.validated, "event.validated")
    return result.sort_values(["start_time_s", "end_time_s", "event_id"])


def build_opportunities(metadata, base, protocol):
    sessions = validate_metadata(metadata)
    graph = metadata["graph"]
    nodes, _, configs, _ = load_graph(graph)
    availability = pd.DataFrame(capability_rows(metadata, base))
    exclusions, opportunities, memberships, design, episodes, timelines = [], [], [], [], [], []
    previous, cached = {}, {}

    def exclude(s, reason, junction="", destination="", event_id=""):
        exclusions.append({"animal": s["animal"], "session_id": s["session_id"], "junction": junction, "destination": destination, "event_id": event_id, "reason": reason})

    def load_position(s):
        key = (s["animal"], s["session_id"])
        if key not in cached:
            frame = read_table(s["position"], base, s["clock_id"], ["t_s", "x_cm", "y_cm", "speed_cm_s", "supported"])
            cached[key] = tracking(frame, s)
        return cached[key]

    for s in sessions:
        old = previous.get(s["animal"])
        previous[s["animal"]] = s
        if old is None:
            exclude(s, "no_observed_previous_session")
            continue
        if s.get("previous_session_id") != old["session_id"] or s["session_order"] != old["session_order"] + 1 or s.get("intervening_exposure_complete") is not True:
            exclude(s, "missing_intervening_session_or_unobserved_exposure")
            continue
        if s["date"] < old["date"] or (s["clock_id"] == old["clock_id"] and s["entry_time_s"] < old["end_time_s"]):
            raise MetadataError("Session ordering conflicts with dates or clock bounds")
        if s["configuration_id"] not in configs or old["configuration_id"] not in configs:
            exclude(s, "missing_configuration_geometry")
            continue
        if configs[s["configuration_id"]] == configs[old["configuration_id"]]:
            exclude(s, "no_barrier_change")
            continue
        if s["home_goal_node"] not in nodes or old["home_goal_node"] not in nodes:
            exclude(s, "missing_verified_goal")
            continue
        pair_caps = availability[(availability.animal == s["animal"]) & availability.session_id.isin([s["session_id"], old["session_id"]])]
        current_caps = pair_caps[pair_caps.session_id == s["session_id"]]
        old_caps = pair_caps[pair_caps.session_id == old["session_id"]]
        if (
            not current_caps[current_caps.capability.isin(OPPORTUNITY_CAPABILITIES)].status.eq("verified").all()
            or not old_caps[old_caps.capability.isin({"full_session_position", "barrier_geometry", "session_chronology", "goal_trial_timing"})].status.eq("verified").all()
        ):
            exclude(s, "unverified_opportunity_inputs")
            continue
        try:
            pos, old_pos = load_position(s), load_position(old)
            require(s.get("traversals_complete") is True, "Missing complete independently verified traversal history")
            traversals = read_table(s["traversals"], base, s["clock_id"], ["traversal_id", "edge_id", "from_node", "to_node", "start_time_s", "end_time_s"])
            traversals = validate_traversals(traversals, s, graph, pos)
            events = validate_events(read_table(s["events"], base, s["clock_id"], ["event_id", "start_time_s", "end_time_s", "validated"]), s)
        except (MetadataError, KeyError, ValueError) as exc:
            exclude(s, f"invalid_inputs: {exc}")
            continue
        contrasts, geometry_exclusions = route_contrasts(graph, old["configuration_id"], s["configuration_id"], protocol["route_tolerance_cm"])
        for e in geometry_exclusions:
            exclude(s, **e)
        episode = f"{s['animal']}:{old['session_id']}->{s['session_id']}"
        horizon, horizon_reason = history_horizon(pos, s, protocol["tracking_gap_s"])
        episodes.append(
            {
                "animal": s["animal"],
                "date": s["date"],
                "episode_id": episode,
                "session_id": s["session_id"],
                "old_configuration": old["configuration_id"],
                "new_configuration": s["configuration_id"],
                "old_goal": old["home_goal_node"],
                "new_goal": s["home_goal_node"],
                "goal_changed": s["home_goal_node"] != old["home_goal_node"],
                "history_end_s": horizon,
                "history_end_reason": horizon_reason,
                "native_events": len(events),
                "validated_native_events": int(events.validated.sum()),
            }
        )
        event_rows = events.to_dict("records")
        # Distinct native events can overlap; expose them, without row-wise deduplication.
        overlapping = set()
        for i, e in enumerate(event_rows):
            for f in event_rows[i + 1 :]:
                if f["start_time_s"] >= e["end_time_s"]:
                    break
                overlapping.update([e["event_id"], f["event_id"]])
        for event_id in sorted(overlapping):
            exclude(s, "overlapping_native_intervals_retained_as_distinct_ids", event_id=event_id)
        for c in contrasts:
            node, destination = c["junction"], c["destination"]
            if not previous_visit(old_pos, nodes[node]):
                exclude(s, "no_documented_previous_junction_experience", node, destination)
                continue
            end, reason = horizon, horizon_reason
            outgoing = traversals[traversals.from_node == node]
            if len(outgoing) and float(outgoing.start_time_s.min()) <= end:
                end, reason = float(outgoing.start_time_s.min()), "first_outgoing_traversal"
            intervals = immobile_intervals(pos, s["entry_time_s"], end, protocol["immobility_speed_cm_s"], protocol["tracking_gap_s"])
            oid = f"{episode}:{node}:{destination}"
            members = [e for e in event_rows if e["validated"] and contained(e["start_time_s"], e["end_time_s"], intervals)]
            exposure = float(sum(hi - lo for lo, hi in intervals))
            row = {
                "opportunity_id": oid,
                "animal": s["animal"],
                "date": s["date"],
                "session_id": s["session_id"],
                "previous_session_id": old["session_id"],
                "episode_id": episode,
                "old_configuration": old["configuration_id"],
                "new_configuration": s["configuration_id"],
                "old_goal": old["home_goal_node"],
                "new_goal": s["home_goal_node"],
                "goal_changed": s["home_goal_node"] != old["home_goal_node"],
                "junction": node,
                "destination": destination,
                "kind": c["kind"],
                "start_time_s": s["entry_time_s"],
                "end_time_s": end,
                "censor_reason": reason,
                "exposure_s": exposure,
                "outdegree": c["outdegree"],
                "old_min_cost_cm": c["old_min_cost_cm"],
                "route_change_magnitude_cm": c["route_change_magnitude_cm"],
                "eligible_event_count": len(members),
                "matched_control_id": "",
                "matched_affected_ids": "",
                "match_status": "unmatched",
            }
            opportunities.append(row)
            for e in members:
                memberships.append(
                    {
                        **{k: row[k] for k in ("animal", "date", "session_id", "episode_id", "opportunity_id", "kind")},
                        **{k: e[k] for k in ("event_id", "start_time_s", "end_time_s")},
                    }
                )
            for i, alternative in enumerate(c["alternatives"]):
                design.append(
                    {
                        **{k: row[k] for k in DESIGN_COLUMNS if k in row},
                        "alternative": alternative,
                        "old_cost_cm": c["old_costs"][i],
                        "new_cost_cm": c["new_costs"][i],
                        "old_relative_cost_cm": c["old_relative"][i],
                        "new_relative_cost_cm": c["new_relative"][i],
                        "structural_predictor_cm": c["structural_predictors"][i],
                    }
                )
            timelines.append({"opportunity_id": oid, "immobile_intervals": intervals, "event_intervals": [[e["start_time_s"], e["end_time_s"]] for e in members]})
    # Match from geometry/history only; never choose a control for its event count.
    for row in opportunities:
        if row["kind"] != "affected":
            continue
        controls = [
            c
            for c in opportunities
            if c["kind"] == "unaffected" and c["episode_id"] == row["episode_id"] and c["destination"] == row["destination"] and c["outdegree"] == row["outdegree"]
        ]
        if controls:
            control = min(controls, key=lambda c: (abs(c["old_min_cost_cm"] - row["old_min_cost_cm"]), c["opportunity_id"]))
            row["matched_control_id"], row["match_status"] = control["opportunity_id"], "matched_geometry_only"
            ids = [v for v in control["matched_affected_ids"].split(";") if v]
            control["matched_affected_ids"] = ";".join(sorted(ids + [row["opportunity_id"]]))
            control["match_status"] = "matched_geometry_only"
    complete = not any(e["reason"].startswith(("invalid_inputs:", "unverified_opportunity_inputs", "missing_configuration_geometry", "missing_verified_goal")) for e in exclusions)
    return {
        "opportunities": opportunities,
        "memberships": memberships,
        "design": design,
        "exclusions": exclusions,
        "episodes": episodes,
        "timelines": timelines,
        "measurement_complete": complete,
    }


def confounding_audit(design):
    frame = pd.DataFrame(design, columns=DESIGN_COLUMNS)
    unavailable = {
        "status": "unavailable",
        "reason": "No matched, exposed opportunity design",
        "n_rows": 0,
        "nuisance_rank": None,
        "augmented_rank": None,
        "target_rank_gain": None,
        "structural_residual_fraction": None,
        "supported_goal_strata": 0,
        "within_goal_support": False,
        "identifiable": False,
    }
    if frame.empty:
        return unavailable, []
    frame = frame[frame.exposure_s > 0].copy()
    if frame.empty:
        return unavailable, []
    frame["goal_pair"] = frame.old_goal.astype(str) + "->" + frame.new_goal.astype(str)
    frame["animal_day"] = frame.animal.astype(str) + ":" + frame.date.astype(str)
    frame["configuration_pair"] = frame.old_configuration.astype(str) + "->" + frame.new_configuration.astype(str)
    support = []
    for key, g in frame.groupby("goal_pair", sort=True):
        support.append(
            {
                "goal_pair": key,
                "episodes": g.episode_id.nunique(),
                "animals": g.animal.nunique(),
                "configuration_pairs": g.configuration_pair.nunique(),
                "structural_range_cm": float(np.ptp(g.structural_predictor_cm)),
                "supported": g.episode_id.nunique() >= 2 and g.configuration_pair.nunique() >= 2 and np.ptp(g.structural_predictor_cm) > 1e-8,
            }
        )
    categories = pd.get_dummies(frame[["animal_day", "goal_pair", "junction", "destination", "alternative"]].astype(str), drop_first=True, dtype=float)
    num = frame[["old_relative_cost_cm", "exposure_s"]].to_numpy(float)
    num = (num - num.mean(axis=0)) / np.where(num.std(axis=0) > 1e-8, num.std(axis=0), 1)
    x = np.column_stack([np.ones(len(frame)), categories.to_numpy(), num])
    z = frame.structural_predictor_cm.to_numpy(float)
    z = z - z.mean()
    scale = np.std(z)
    if scale > 1e-8:
        z /= scale
    residual = z - x @ np.linalg.lstsq(x, z, rcond=1e-10)[0]
    rank = int(np.linalg.matrix_rank(x, tol=1e-8))
    augmented = int(np.linalg.matrix_rank(np.column_stack([x, z]), tol=1e-8))
    fraction = float(np.linalg.norm(residual) / np.linalg.norm(z)) if np.linalg.norm(z) > 1e-8 else 0.0
    supported = sum(bool(s["supported"]) for s in support)
    identified = augmented == rank + 1 and fraction > 1e-8 and supported > 0
    return {
        "status": "pass" if identified else "fail",
        "reason": "Algebraic support only, not causal identification" if identified else "Structural contrast lacks rank or within-goal overlap",
        "n_rows": len(frame),
        "nuisance_rank": rank,
        "augmented_rank": augmented,
        "target_rank_gain": augmented - rank,
        "structural_residual_fraction": fraction,
        "supported_goal_strata": supported,
        "within_goal_support": supported > 0,
        "identifiable": identified,
    }, support


def coverage(opportunities, memberships, protocol, available=True):
    by_id = {o["opportunity_id"]: o for o in opportunities}
    good = [
        o
        for o in opportunities
        if o["kind"] == "affected"
        and o["eligible_event_count"] > 0
        and o["exposure_s"] > 0
        and o["matched_control_id"] in by_id
        and by_id[o["matched_control_id"]]["eligible_event_count"] > 0
        and by_id[o["matched_control_id"]]["exposure_s"] > 0
    ]
    rows = []
    for animal in protocol["animals"]:
        os = [o for o in good if o["animal"] == animal]
        unique = {(m["session_id"], m["event_id"]) for m in memberships if m["animal"] == animal}
        episodes, days = len({o["episode_id"] for o in os}), len({o["date"] for o in os})
        rows.append(
            {
                "animal": animal,
                "eligible_episodes": episodes if available else None,
                "eligible_days": days if available else None,
                "unique_events": len(unique) if available else None,
                "event_opportunity_memberships": sum(m["animal"] == animal for m in memberships) if available else None,
                "status": ("pass" if episodes >= protocol["minimum_episodes_per_animal"] and days >= protocol["minimum_days_per_animal"] else "fail")
                if available
                else "unavailable",
            }
        )
    return rows


def gates_and_decision(availability, animals, novelty, confounding, structural_inputs_available, inventory_valid=True):
    gates = []

    def add(gate, passed, reason, unavailable=False):
        gates.append({"gate": gate, "status": "unavailable" if unavailable else ("pass" if passed else "fail"), "reason": reason})

    add("source_integrity_and_identity", inventory_valid, "Hashes, documented adapters, and unique identities must agree")
    for capability in CAPABILITIES:
        relevant = [a for a in availability if a["capability"] == capability]
        add(capability, bool(relevant) and all(r["status"] == "verified" for r in relevant), "Verified evidence required for every supplied session")
    add("novelty_review_resolved", novelty == "distinct_contrast_supported", "Exact experience-conditioned contrast still requires a documented distinctness review")
    add("opportunity_inputs_verified", structural_inputs_available, "Geometry, complete exposure history, aligned tracking, and native intervals required")
    add("goal_topology_separable", confounding.get("identifiable") is True, confounding["reason"], confounding["status"] == "unavailable")
    add(
        "all_animals_repeated_coverage",
        bool(animals) and all(a["status"] == "pass" for a in animals),
        "Each expected animal needs two eligible changes across two days, with matched unaffected events",
        not structural_inputs_available,
    )
    statuses = {g["gate"]: g["status"] for g in gates}
    if not inventory_valid or any(statuses[c] != "pass" for c in DATA_CAPABILITIES):
        decision = "blocked_data"
    elif any(statuses[c] != "pass" for c in set(CAPABILITIES) - DATA_CAPABILITIES) or not structural_inputs_available:
        decision = "blocked_metadata"
    elif statuses["novelty_review_resolved"] != "pass":
        decision = "novelty_unresolved"
    elif statuses["goal_topology_separable"] != "pass":
        decision = "confounded_design"
    elif statuses["all_animals_repeated_coverage"] != "pass":
        decision = "insufficient_opportunities"
    else:
        decision = "ready_for_readout_feasibility"
    add("overall", decision == "ready_for_readout_feasibility", decision)
    return gates, {
        "decision": decision,
        "failed_gates": [g["gate"] for g in gates if g["status"] != "pass"],
        "biological_analysis_performed": False,
        "power_established": False,
        "author_request_sent": False,
        "interpretation": "Not retraversed does not mean unknown: barriers were visible.",
    }
