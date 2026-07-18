"""Load balancer: replicate hot experts based on measured routing stats.

Input: global per-(layer, expert) token counts from a profiling window.
Output: a placement plan = naive contiguous placement + replicas of hot experts
on under-loaded ranks. Dispatch splits a replicated expert's traffic evenly
across its replicas (round-robin), so a replica on rank r moves 1/n_replicas of
that expert's load onto r. The planner greedily adds the replica that most
reduces the layer's max rank load, within a per-rank replica-slot budget.
"""
from __future__ import annotations

import torch

from .ep import Placement


def _rank_loads(loads, owners, world):
    rl = [0.0] * world
    for e, own in enumerate(owners):
        share = loads[e] / len(own)
        for r in own:
            rl[r] += share
    return rl


def _best_replica_addition(loads, owners, world, slots):
    """The (expert, cold rank) replica that most lowers the layer's busiest rank.

    Returns (new_max, expert, rank), or None if no in-budget addition reduces it.
    """
    cur_max = max(_rank_loads(loads, owners, world))
    best = None
    for e in range(len(owners)):
        if loads[e] / len(owners[e]) == 0:
            continue
        for r in range(world):
            if r in owners[e] or slots[r] <= 0:
                continue
            trial = [o[:] for o in owners]
            trial[e] = sorted(trial[e] + [r])
            new_max = max(_rank_loads(loads, trial, world))
            if new_max < cur_max and (best is None or new_max < best[0]):
                best = (new_max, e, r)
    return best


def _plan_layer(loads, world, per, slots_per_rank, target_ratio):
    """Greedily replicate hot experts for one layer. Returns (owners, before, after)."""
    owners = [[e // per] for e in range(len(loads))]
    slots = [slots_per_rank] * world
    before = _rank_loads(loads, owners, world)
    mean = sum(before) / world
    while mean and max(_rank_loads(loads, owners, world)) / mean > target_ratio:
        best = _best_replica_addition(loads, owners, world, slots)
        if best is None:
            break
        _, e, r = best
        owners[e] = sorted(owners[e] + [r])
        slots[r] -= 1
    return owners, before, _rank_loads(loads, owners, world)


def plan_balanced(counts: torch.Tensor, world: int, slots_per_rank: int = 2,
                  target_ratio: float = 1.05) -> tuple[Placement, dict]:
    """counts: (num_layers, num_experts) global routing counts.

    slots_per_rank: extra expert copies each rank may host per layer
    (memory overhead = slots_per_rank / (num_experts / world)).
    """
    num_layers, num_experts = counts.shape
    per = num_experts // world
    all_owners = []
    report = {"layers": [], "replicas_total": 0}

    for l in range(num_layers):
        loads = counts[l].double().tolist()
        owners, before, after = _plan_layer(loads, world, per, slots_per_rank, target_ratio)
        mean = sum(before) / world
        replicas = sum(len(o) for o in owners) - num_experts
        report["replicas_total"] += replicas
        report["layers"].append({
            "layer": l,
            "imbalance_before": max(before) / mean if mean else 1.0,
            "imbalance_after": max(after) / mean if mean else 1.0,
            "replicas": replicas,
        })
        all_owners.append(owners)

    return Placement(all_owners), report
