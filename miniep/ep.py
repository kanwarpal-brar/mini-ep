"""Expert parallelism: placement, all-to-all dispatch/combine, per-rank expert shards.

Every rank runs the full non-expert model on its own shard of the batch
(data-parallel attention) and owns a subset of experts per layer. Each MoE layer
does: local top-k routing -> exchange per-rank entry counts -> all-to-all of
(hidden state, expert id) entries to owner ranks -> owners run their experts ->
all-to-all back -> combine in canonical token order (so results are invariant
to placement and world size, which keeps the correctness gates sharp).
"""
from __future__ import annotations

import time

import torch
import torch.distributed as dist

from .modeling import flatten_topk, run_experts, scatter_combine


class Placement:
    """owners[layer][expert] = list of ranks holding that expert (>=1).

    Dispatch policy for replicated experts: round-robin per (source rank, layer,
    expert), which is deterministic and splits a hot expert's traffic evenly across replicas.
    """

    def __init__(self, owners: list[list[list[int]]]):
        self.owners = owners
        self.num_layers = len(owners)
        self.num_experts = len(owners[0])
        # per-(layer, expert) round-robin counters, zeroed before every forward
        self._rr = torch.zeros(self.num_layers, self.num_experts, dtype=torch.long)
        self._tables = [self._build_table(owners[l]) for l in range(self.num_layers)]

    @staticmethod
    def _build_table(owners_l: list[list[int]]):
        """Dispatch tables for one layer: table[e, j] = rank of e's j-th replica."""
        n_exp = len(owners_l)
        rmax = max(len(o) for o in owners_l)
        table = torch.zeros(n_exp, rmax, dtype=torch.long)
        n_rep = torch.zeros(n_exp, dtype=torch.long)
        for e, o in enumerate(owners_l):
            table[e, :len(o)] = torch.tensor(o)
            n_rep[e] = len(o)
        return table, n_rep

    @staticmethod
    def naive(num_layers: int, num_experts: int, world: int) -> "Placement":
        assert num_experts % world == 0
        per = num_experts // world
        return Placement(
            [[[e // per] for e in range(num_experts)] for _ in range(num_layers)])

    @staticmethod
    def from_plan(plan: dict) -> "Placement":
        owners = [
            [sorted(layer_owners[str(e)]) for e in range(len(layer_owners))]
            for layer_owners in (plan["owners"][str(l)] for l in range(len(plan["owners"])))
        ]
        return Placement(owners)

    def to_plan(self) -> dict:
        return {"owners": {
            str(l): {str(e): self.owners[l][e] for e in range(self.num_experts)}
            for l in range(self.num_layers)}}

    def experts_of_rank(self, rank: int) -> dict[int, list[int]]:
        return {
            l: sorted(e for e in range(self.num_experts) if rank in self.owners[l][e])
            for l in range(self.num_layers)
        }

    def reset_rr(self):
        self._rr.zero_()

    def dest_ranks(self, layer: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """expert_ids: 1-D tensor of expert ids -> 1-D tensor of destination ranks.

        Vectorized round-robin: the i-th occurrence (in flat order) of expert e
        goes to replica (rr[e] + i) % n_rep[e], then rr[e] += count[e] (exactly
        the per-entry loop's assignment, without a Python loop on the hot path).
        """
        table, n_rep = self._tables[layer]
        if (n_rep == 1).all():
            return table[expert_ids, 0]  # no replicas: placement fixed, rr unused
        m = expert_ids.numel()
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        # within-expert occurrence index of each entry, in original flat order
        srt, order = torch.sort(expert_ids, stable=True)
        starts = torch.cumsum(counts, 0) - counts  # sorted position of e's 1st entry
        within_sorted = torch.arange(m, dtype=torch.long) - starts[srt]
        within = torch.empty_like(within_sorted)
        within[order] = within_sorted
        slot = (self._rr[layer][expert_ids] + within) % n_rep[expert_ids]
        self._rr[layer] += counts
        return table[expert_ids, slot]


def detect_all_to_all(group) -> bool:
    try:
        world = dist.get_world_size(group)
        x = torch.zeros(world, dtype=torch.float32)
        out = torch.zeros(world, dtype=torch.float32)
        dist.all_to_all_single(out, x, group=group)
        return True
    except Exception:
        return False


class Comm:
    """all_to_all_single with uneven splits; falls back to isend/irecv on
    backends without native alltoall (older gloo)."""

    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.native = detect_all_to_all(group)

    def exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        # send_counts: (world,) int64 -> recv_counts (world,)
        gathered = [torch.zeros_like(send_counts) for _ in range(self.world)]
        dist.all_gather(gathered, send_counts, group=self.group)
        return torch.stack(gathered)[:, self.rank].contiguous()

    def all_to_all(self, x: torch.Tensor, send_counts, recv_counts) -> torch.Tensor:
        """x: (sum(send_counts), ...) grouped by destination rank."""
        out_shape = (int(recv_counts.sum().item()),) + tuple(x.shape[1:])
        out = torch.empty(out_shape, dtype=x.dtype)
        if self.native:
            dist.all_to_all_single(
                out, x.contiguous(),
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts.tolist(), group=self.group)
            return out
        return self._all_to_all_fallback(x, send_counts, recv_counts, out)

    def _all_to_all_fallback(self, x, send_counts, recv_counts, out) -> torch.Tensor:
        """Pairwise isend/irecv for backends without native all_to_all_single."""
        s_off = [0] + torch.cumsum(send_counts, 0).tolist()
        r_off = [0] + torch.cumsum(recv_counts, 0).tolist()
        reqs = []
        for peer in range(self.world):
            if peer == self.rank:
                out[r_off[peer]:r_off[peer + 1]] = x[s_off[peer]:s_off[peer + 1]]
                continue
            if send_counts[peer] > 0:
                reqs.append(dist.isend(x[s_off[peer]:s_off[peer + 1]].contiguous(),
                                       peer, group=self.group))
            if recv_counts[peer] > 0:
                buf = torch.empty((int(recv_counts[peer]),) + tuple(x.shape[1:]), dtype=x.dtype)
                reqs.append((dist.irecv(buf, peer, group=self.group), buf,
                             r_off[peer], r_off[peer + 1]))
        for r in reqs:
            if isinstance(r, tuple):
                req, buf, a, b = r
                req.wait()
                out[a:b] = buf
            else:
                r.wait()
        return out


class EPStats:
    """Per-rank counters for one run. Times in seconds."""

    def __init__(self, num_layers, num_experts):
        self.route_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)  # source-side
        self.recv_tokens = 0          # expert entries computed on this rank
        self.recv_tokens_per_layer = torch.zeros(num_layers, dtype=torch.long)
        self.compute_time = 0.0       # expert FFN time
        self.a2a_time = 0.0           # collective time incl. straggler wait
        self.steps = 0

    def snapshot(self) -> dict:
        return {
            "recv_tokens": int(self.recv_tokens),
            "recv_tokens_per_layer": self.recv_tokens_per_layer.tolist(),
            "compute_time": self.compute_time,
            "a2a_time": self.a2a_time,
            "route_counts": self.route_counts.tolist(),
            "steps": self.steps,
        }


class EPMoEBackend:
    """Drop-in for LocalMoEBackend, executing experts across ranks.

    local_experts: sorted global expert ids resident on this rank for this layer.
    w_in/w_out are the corresponding stacked slices (local row order).
    """

    def __init__(self, layer, comm: Comm, placement: Placement,
                 local_experts: list[int], w_in, w_out, stats: EPStats):
        self.layer = layer
        self.comm = comm
        self.placement = placement
        self.w_in = w_in
        self.w_out = w_out
        self.stats = stats
        self._update_map(local_experts)

    def _update_map(self, local_experts):
        self.local_experts = local_experts
        g2l = torch.full((self.placement.num_experts,), -1, dtype=torch.long)
        for i, e in enumerate(local_experts):
            g2l[e] = i
        self.g2l = g2l

    def __call__(self, x_flat, topk_idx, topk_gate, valid_mask=None):
        n, k = topk_idx.shape
        token_ids, slot_ids, keep_e = flatten_topk(n, k, valid_mask)
        flat_expert = topk_idx.reshape(-1)[keep_e]
        flat_gate = topk_gate.reshape(-1)[keep_e]

        self.stats.route_counts[self.layer] += torch.bincount(
            flat_expert, minlength=self.placement.num_experts)

        dest = self.placement.dest_ranks(self.layer, flat_expert)
        order = torch.argsort(dest, stable=True)
        send_counts = torch.bincount(dest, minlength=self.comm.world)

        t0 = time.perf_counter()
        recv_counts = self.comm.exchange_counts(send_counts)
        recv_experts = self.comm.all_to_all(flat_expert[order], send_counts, recv_counts)
        recv_x = self.comm.all_to_all(x_flat[token_ids[order]], send_counts, recv_counts)
        t1 = time.perf_counter()

        local_ids = self.g2l[recv_experts]
        assert (local_ids >= 0).all(), f"layer {self.layer}: got expert not resident here"
        out = run_experts(recv_x, local_ids, self.w_in, self.w_out)
        t2 = time.perf_counter()

        back = self.comm.all_to_all(out, recv_counts, send_counts)
        t3 = time.perf_counter()

        # undo the destination sort so accumulation is in canonical token order,
        # independent of placement/world size (see scatter_combine for the rest)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel())
        gated = back[inv] * flat_gate.unsqueeze(1)
        result = scatter_combine(gated, token_ids, slot_ids, n, k, x_flat.shape[1])

        recv_total = int(recv_counts.sum().item())
        self.stats.recv_tokens += recv_total
        self.stats.recv_tokens_per_layer[self.layer] += recv_total
        self.stats.compute_time += t2 - t1
        self.stats.a2a_time += (t1 - t0) + (t3 - t2)
        return result
