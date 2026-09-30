"""Region-blind paths and separately evaluated terminal-content labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class Geometry:
    grid: np.ndarray
    component: np.ndarray
    predecessors: np.ndarray
    euclidean: np.ndarray

    @classmethod
    def from_grid(cls, grid, graph):
        return cls(np.asarray(grid), graph[0], graph[2], cdist(grid, grid))


class BlindPath:
    """Backward-in-time geometry law; no region or tuning input is accepted."""

    def __init__(self, geometry, kind, speed, jump_lengths, intervals, scale, rng):
        if kind not in ("stationary", "moving", "jumping") or scale <= 0 or speed <= 0:
            raise ValueError("invalid dynamics")
        self.geometry, self.kind, self.rng = geometry, kind, rng
        self.speed, self.scale = float(speed)*scale, scale
        self.jump_lengths, self.intervals = np.asarray(jump_lengths), np.asarray(intervals)
        if not len(self.intervals) or not np.all(self.intervals > 0) or not len(self.jump_lengths) or not np.all(self.jump_lengths > 0):
            raise ValueError("positive empirical pools required")
        self.ages = [0.]
        self.nodes = [int(rng.choice(geometry.component))]
        self.requested_lengths, self.realized_lengths = [], []
        self.destination = self.nodes[0]
        interval = float(rng.choice(self.intervals, p=self.intervals/self.intervals.sum()))
        self.next_interval = max(float(rng.uniform(0, interval)), 1e-12)

    def extend(self, horizon):
        if horizon <= 0:
            raise ValueError("positive horizon required")
        if self.kind == "stationary":
            self.ages = [0., max(horizon, self.ages[-1])]
            self.nodes = [self.nodes[0], self.nodes[0]]
            return self
        geo, rng = self.geometry, self.rng
        while self.ages[-1] < horizon:
            current = self.nodes[-1]
            if self.kind == "jumping":
                requested = float(rng.choice(self.jump_lengths))*self.scale
                support = geo.component[geo.component != current]
                difference = np.abs(geo.euclidean[current, support]-requested)
                possible = support[difference <= max(4., float(difference.min())+1e-10)]
                target = int(rng.choice(possible))
                duration = self.next_interval
                self.next_interval = float(rng.choice(self.intervals))
                self.requested_lengths.append(requested)
                self.realized_lengths.append(float(geo.euclidean[current, target]))
            else:
                while current == self.destination:
                    self.destination = int(rng.choice(geo.component))
                target = int(geo.predecessors[self.destination, current])
                if target < 0:
                    raise ValueError("broken occupied-grid route")
                duration = float(np.linalg.norm(geo.grid[target]-geo.grid[current]))/self.speed
            self.ages.append(self.ages[-1]+duration)
            self.nodes.append(target)
            if len(self.nodes) > 100000:
                raise RuntimeError("path exceeded safety cap")
        return self

    def intensities(self, ages, rates):
        age = np.clip(np.asarray(ages), 0., self.ages[-1])
        knots, nodes = np.asarray(self.ages), np.asarray(self.nodes)
        i = np.clip(np.searchsorted(knots, age, side="left")-1, 0, len(knots)-2)
        if self.kind != "moving":
            return rates[:, nodes[i]].T
        weight = (age-knots[i])/(knots[i+1]-knots[i])
        return rates[:, nodes[i]].T*(1-weight[:, None])+rates[:, nodes[i+1]].T*weight[:, None]


def circle_interval(p, q, center, radius):
    """Fractional interval of a line segment inside a disc, or None."""
    p, q, center = np.asarray(p), np.asarray(q), np.asarray(center)
    v, offset = q-p, p-center
    a = float(v@v)
    if a < 1e-20:
        return (0., 1.) if offset@offset <= radius*radius else None
    b, c = 2*float(v@offset), float(offset@offset)-radius*radius
    disc = b*b-4*a*c
    if disc < 0:
        return None
    left = max(0., (-b-np.sqrt(disc))/(2*a))
    right = min(1., (-b+np.sqrt(disc))/(2*a))
    return (left, right) if right > left else None


def path_content(path, center, radius, duration):
    if duration > path.ages[-1]+1e-10 or duration <= 0:
        raise ValueError("path must cover requested duration")
    grid, ages, nodes = path.geometry.grid, np.asarray(path.ages), np.asarray(path.nodes)
    visits = []
    crossings = np.zeros((2, 2), int)
    late = False
    for k in range(len(ages)-1):
        start, end = float(ages[k]), min(float(ages[k+1]), duration)
        if end <= start:
            break
        p = grid[nodes[k]]
        q = grid[nodes[k+1]] if path.kind == "moving" else p
        if path.kind == "moving":
            q = p+(q-p)*(end-start)/(ages[k+1]-start)
        inside = circle_interval(p, q, center, radius)
        if inside is not None:
            visits.append((start+inside[0]*(end-start), start+inside[1]*(end-start)))
        if path.kind == "jumping" and ages[k+1] < duration:
            before = int(np.linalg.norm(grid[nodes[k+1]]-center) <= radius)
            after = int(np.linalg.norm(p-center) <= radius)
            crossings[before, after] += 1
            late |= bool(before != after and ages[k+1] <= .005)
    merged = []
    for a, b in visits:
        if merged and a <= merged[-1][1]+1e-12:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    dwell = float(sum(b-a for a, b in merged))
    dwell = min(duration, max(0., dwell))
    k = int(np.clip(np.searchsorted(ages, duration, side="left")-1, 0, len(ages)-2))
    origin = grid[nodes[k]]
    if path.kind == "moving":
        origin = origin+(grid[nodes[k+1]]-origin)*(duration-ages[k])/(ages[k+1]-ages[k])
    return {"origin_home": bool(np.linalg.norm(origin-center) <= radius), "dwell_s": dwell, "occupancy_fraction": dwell/duration,
            "label": dwell >= .020-1e-9,
            "longest_visit_s": max((b-a for a, b in merged), default=0.),
            "endpoint_home": bool(np.linalg.norm(grid[nodes[0]]-center) <= radius),
            "jump_crossings": crossings, "late_crossing": late}


def allocate_identities(path, ages, rates, rng):
    if len(rates) >= 65536 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("positive finite rates and fewer than 65536 cells required")
    intensity = path.intensities(ages, rates)
    probability = intensity/intensity.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.
    return (rng.random(len(ages))[:, None] > cumulative).sum(axis=1).astype(np.uint16)


def sample_conditional_event(geometry, kind, speed, lengths, intervals, scale, center,
                             delta, wanted, template, rates, rng, max_attempts=100000):
    """Rejection is outside the region-blind generator and includes native support."""
    geometry_rejections, spike_rejections = 0, 0
    pre_support_occupancy = []
    for attempt in range(1, max_attempts+1):
        path = BlindPath(geometry, kind, speed, lengths, intervals, scale, rng).extend(.1)
        content = path_content(path, center, 20., delta)
        if content["label"] != bool(wanted):
            geometry_rejections += 1
            continue
        pre_support_occupancy.append(content["occupancy_fraction"])
        path.extend(max(.1, template["end"]-template["start"]))
        identity = allocate_identities(path, template["end"]-template["times"], rates, rng)
        active = len(np.unique(identity))
        if active < int(np.ceil(.1*len(rates))):
            spike_rejections += 1
            continue
        contents = [path_content(path, center, 20., d) for d in (.02, .04, .06, .1)]
        return {"path": path, "identities": identity, "active": active, "contents": contents,
                "attempts": attempt, "geometry_rejections": geometry_rejections, "spike_rejections": spike_rejections,
                "pre_support_occupancy": np.asarray(pre_support_occupancy)}
    raise RuntimeError(f"conditional sampling failed: {kind}, delta={delta}, label={wanted}")
