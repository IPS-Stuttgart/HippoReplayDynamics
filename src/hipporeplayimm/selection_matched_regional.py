"""Count-time-preserving regional simulation and calibration diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from hipporeplayimm.regional_content_frontier import calls_from_bf, mixture_fit, regional_log_bf

GENERATORS = ("stationary", "moving", "late_jump")


def endpoint_interval(start, end):
    n = int(np.floor((end-start+1e-9)/.005))
    if n < 4:
        raise ValueError("event too short")
    return start+(n-4)*.005, start+n*.005


def latent_intensities(times, start, end, target, kind, grid, rates, graph, region, rng):
    """Fixed-map interpolation; terminal state is the requested target."""
    component, distances, predecessors = graph
    if target not in component:
        raise ValueError("target outside connected simulation component")
    endpoint = endpoint_interval(start, end)[1]
    if kind == "stationary":
        return np.broadcast_to(rates[:, target], (len(times), len(rates))).copy()
    opposite = component[region[component] != region[target]]
    if not len(opposite):
        raise ValueError("both classes need graph support")
    if kind == "late_jump":
        origin = int(opposite[np.argmax(distances[target, opposite])])
        return np.where((times >= endpoint-.005)[:, None], rates[:, target], rates[:, origin])
    if kind != "moving":
        raise ValueError("unknown generator")
    origin = int(rng.choice(opposite))
    route = [origin]
    while route[-1] != target:
        route.append(int(predecessors[target, route[-1]]))
        if route[-1] < 0 or len(route) > len(grid):
            raise ValueError("broken geodesic")
    route = np.asarray(route)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(grid[route], axis=0), axis=1))]
    location = np.clip(lengths[-1]-500*(endpoint-times), 0., lengths[-1])
    right = np.clip(np.searchsorted(lengths, location, side="right"), 1, len(route)-1)
    fraction = (location-lengths[right-1])/(lengths[right]-lengths[right-1])
    return rates[:, route[right-1]].T*(1-fraction[:, None])+rates[:, route[right]].T*fraction[:, None]


def draw_candidate(times, start, end, target, kind, grid, rates, graph, region, rng, perturbation="none"):
    intensity = latent_intensities(times, start, end, target, kind, grid, rates, graph, region, rng)
    home_cells = region[np.argmax(rates, axis=1)]
    if perturbation == "home_participation_half":
        intensity[:, home_cells] *= .5
    elif perturbation == "home_nospatial_2hz":
        intensity[:, home_cells] += 2.
    elif perturbation not in ("none", "within_region_shift", "global_gain_x2"):
        raise ValueError("unknown perturbation")
    # A common gain cancels exactly after conditioning on the population count.
    probability = intensity/intensity.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.
    identity = (rng.random(len(times))[:, None] > cumulative).sum(axis=1)
    a, b = endpoint_interval(start, end)
    mask = (times >= a) & (times < b)
    counts = np.bincount(identity[mask], minlength=len(rates))
    return counts, len(np.unique(identity)), identity


def draw_panel(templates, prevalence, generator, grid, rates, graph, region, rng, perturbation="none"):
    n = len(templates)
    labels = np.zeros(n, int)
    labels[rng.permutation(n)[:round(n*prevalence)]] = 1
    counts, active, targets, identities, kinds = [], [], [], [], []
    for template, z in zip(templates, labels, strict=True):
        support = graph[0][region[graph[0]] == bool(z)]
        if perturbation == "within_region_shift":
            x = grid[support, 0]
            weights = np.exp(2*(x-x.mean())/max(x.std(), 1.))
            target = int(rng.choice(support, p=weights/weights.sum()))
        else:
            target = int(rng.choice(support))
        kind = str(rng.choice(GENERATORS)) if generator == "mix" else generator
        c, a, identity = draw_candidate(template["times"], template["start"], template["end"], target, kind,
            grid, rates, graph, region, rng, perturbation)
        counts.append(c)
        active.append(a)
        targets.append(target)
        identities.append(identity)
        kinds.append(kind)
    return {"counts": np.asarray(counts, np.uint16), "active": np.asarray(active), "targets": np.asarray(targets),
                "labels": labels, "identities": np.concatenate(identities), "kinds": np.asarray(kinds)}


def read_population(counts, all_ids, population):
    lookup = {int(cell): i for i, cell in enumerate(all_ids)}
    ix = [lookup[int(cell)] for cell in population["ids"]]
    n = counts[:, ix]
    bf = regional_log_bf(n, population["rates"], population["region"])
    area = population["region"].mean()
    mass = expit(bf+np.log(area/(1-area)))
    return bf, calls_from_bf(bf, n.sum(axis=1)), mass


def calibration_hist(calls, truth, accepted):
    result = np.zeros((2, 3), int)
    np.add.at(result, (truth[accepted], calls[accepted]), 1)
    return result


def evaluate(calls, truth, accepted, calibration):
    observed = np.bincount(calls[accepted], minlength=3)
    fit = mixture_fit(calibration, observed)
    prevalence = float(truth[accepted].mean()) if accepted.any() else np.nan
    emission = (calibration+.5)/(calibration.sum(axis=1)[:, None]+1.5)
    expected = (1-fit["prevalence"])*emission[0]+fit["prevalence"]*emission[1]
    residual = observed/max(observed.sum(), 1)-expected
    return dict(**fit, events=int(accepted.sum()), true_prevalence=prevalence,
                signed_error=fit["prevalence"]-prevalence, absolute_error=abs(fit["prevalence"]-prevalence),
                residual_against=float(residual[0]), residual_neutral=float(residual[1]), residual_for=float(residual[2]))


def threshold(values, alpha=.05):
    v = np.asarray(values, float)
    if len(v) < 20 or not np.isfinite(v).all():
        raise ValueError("at least 20 finite independent null values needed")
    # Finite-bank upper order statistic, conservative at the requested tail.
    rank = min(len(v)-1, int(np.ceil((len(v)+1)*(1-alpha)))-1)
    return float(np.sort(v)[rank])
