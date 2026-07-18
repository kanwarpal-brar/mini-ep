"""torchrun entrypoint: expert-parallel worker.

Every rank holds the full non-expert weights plus its shard of experts, runs the
same number of forward passes in lockstep (one batched prefill, then one forward
per decode step) so the per-layer all-to-alls always line up.

Modes:
  gateb  correctness gate: PARITY prompts sharded across ranks must reproduce
         the HF reference continuations exactly (and logit fingerprints in tol)
  bench  load benchmark on the SKEW workload; writes per-rank stats JSON

Usage:
  uv run torchrun --nproc-per-node 4 -m miniep.worker --mode gateb
  uv run torchrun --nproc-per-node 4 -m miniep.worker --mode bench \
      --plan results/plan_balanced.json --out results/bench_balanced.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from . import GATE_LOGIT_TOL, LOGIT_FP_KEY, LOGIT_FP_SIZE
from .ep import Comm, EPMoEBackend, EPStats, Placement
from .modeling import GraniteMoeModel, KVCache
from .weights import expert_weight_keys, load_config, load_weights

ROOT = Path(__file__).resolve().parents[1]
_PREVIEW_TOKENS = 12  # tokens shown per side when a gate mismatch is printed


class Shard:
    """One rank's slice of the global batch, right-padded, with padded KV."""

    def __init__(self, prompts_ids: list[list[int]], num_layers: int, pad_id: int = 0):
        assert prompts_ids
        self.lengths = torch.tensor([len(p) for p in prompts_ids])
        self.T = int(self.lengths.max().item())
        self.B = len(prompts_ids)
        self.input = torch.full((self.B, self.T), pad_id, dtype=torch.long)
        for i, p in enumerate(prompts_ids):
            self.input[i, :len(p)] = torch.tensor(p)
        self.kv = KVCache(num_layers)
        self.generated: list[list[int]] = [[] for _ in range(self.B)]
        self.last_tokens: torch.Tensor | None = None
        self.decoded_steps = 0

    def prefill_masks(self):
        # attn (B,1,T,T) bool keep-mask: causal AND key within true length
        causal = torch.tril(torch.ones(self.T, self.T, dtype=torch.bool))
        key_valid = (torch.arange(self.T)[None, :] < self.lengths[:, None])  # (B,T)
        mask = causal[None, :, :] & key_valid[:, None, :]
        moe_valid = key_valid  # same predicate for query positions
        return mask[:, None, :, :], moe_valid

    def decode_masks(self):
        # keys: prompt part [0,T) valid where < len; decode part all valid
        t = self.decoded_steps
        key_valid = torch.cat([
            (torch.arange(self.T)[None, :] < self.lengths[:, None]).expand(self.B, self.T),
            torch.ones(self.B, t + 1, dtype=torch.bool),
        ], dim=1)[:, : self.T + t + 1]
        return key_valid[:, None, None, :], None

    def positions_decode(self):
        return (self.lengths + self.decoded_steps).unsqueeze(1)  # (B,1)


class Engine:
    def __init__(self, placement: Placement, mem_report=True):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.cfg = load_config()
        self.placement = placement
        self.comm = Comm()
        self.stats = EPStats(self.cfg.num_hidden_layers, self.cfg.num_local_experts)
        my_experts = placement.experts_of_rank(self.rank)
        w = load_weights(experts_per_layer=my_experts)
        self.backends = []
        for l in range(self.cfg.num_hidden_layers):
            in_key, out_key = expert_weight_keys(l)
            self.backends.append(EPMoEBackend(
                l, self.comm, placement, my_experts[l],
                w[in_key], w[out_key], self.stats))
        self.model = GraniteMoeModel(self.cfg, w, self.backends)
        if mem_report and self.rank == 0:
            n_exp = sum(len(v) for v in my_experts.values())
            print(f"[rank0] world={self.world} native_a2a={self.comm.native} "
                  f"experts/rank/layer≈{n_exp / self.cfg.num_hidden_layers:.1f}")

    @torch.no_grad()
    def prefill(self, shard: Shard) -> torch.Tensor:
        """Returns (B,) next tokens (greedy)."""
        self.placement.reset_rr()
        mask, moe_valid = shard.prefill_masks()
        t0 = time.perf_counter()
        logits = self.model(shard.input, kv_cache=shard.kv, attn_mask=mask,
                            moe_valid=moe_valid, logits_at=shard.lengths - 1)
        self.stats.steps += 1
        last = logits[:, 0]  # (B, V)
        shard.last_tokens = last.argmax(dim=-1)
        shard.prefill_time = time.perf_counter() - t0
        shard.last_prefill_logits = last
        return shard.last_tokens

    @torch.no_grad()
    def prefill_dummy(self, shard: Shard):
        """Prefill that adds zero expert load (idle rank keeping collectives aligned)."""
        self.placement.reset_rr()
        mask, _ = shard.prefill_masks()
        logits = self.model(shard.input, kv_cache=shard.kv, attn_mask=mask,
                            moe_valid=torch.zeros(shard.B, shard.T, dtype=torch.bool),
                            logits_at=shard.lengths - 1)
        self.stats.steps += 1
        shard.last_tokens = logits[:, 0].argmax(dim=-1)
        return shard.last_tokens

    @torch.no_grad()
    def decode_step(self, shard: Shard, active: torch.Tensor | None = None):
        """One lockstep greedy decode step. active: (B,) bool (rows still generating)."""
        self.placement.reset_rr()
        for i in range(shard.B):
            if active is None or active[i]:
                shard.generated[i].append(int(shard.last_tokens[i]))
        mask, _ = shard.decode_masks()
        step_input = shard.last_tokens.unsqueeze(1)
        moe_valid = active.unsqueeze(1) if active is not None else None
        logits = self.model(step_input, positions=shard.positions_decode(),
                            kv_cache=shard.kv, attn_mask=mask, moe_valid=moe_valid)
        self.stats.steps += 1
        shard.last_tokens = logits[:, -1].argmax(dim=-1)
        shard.decoded_steps += 1


def load_placement(args, cfg) -> Placement:
    if args.plan:
        return Placement.from_plan(json.loads(Path(args.plan).read_text()))
    return Placement.naive(cfg.num_hidden_layers, cfg.num_local_experts,
                           dist.get_world_size())


def shard_indices(n_items: int, rank: int, world: int) -> list[int]:
    return [i for i in range(n_items) if i % world == rank]


def mode_gateb(args):
    rank, world = dist.get_rank(), dist.get_world_size()
    ref = json.loads((ROOT / "results" / "reference.json").read_text())
    prompts = ref["prompts"]
    engine = Engine(load_placement(args, load_config()))
    mine = shard_indices(len(prompts), rank, world)
    shard = Shard([prompts[i]["input_ids"] for i in mine], engine.cfg.num_hidden_layers,
                  pad_id=engine.cfg.pad_token_id)

    engine.prefill(shard)
    for _ in range(ref["max_new_tokens"]):
        engine.decode_step(shard)

    my_results = []
    for row, i in enumerate(mine):
        p = prompts[i]
        got = shard.generated[row]
        tok_ok = got == p["generated_ids"]
        head = shard.last_prefill_logits[row, :LOGIT_FP_SIZE]
        fp_diff = (head - torch.tensor(p[LOGIT_FP_KEY])).abs().max().item()
        my_results.append((i, tok_ok, fp_diff,
                           got[:_PREVIEW_TOKENS], p["generated_ids"][:_PREVIEW_TOKENS]))

    gathered = [None] * world
    dist.gather_object(my_results, gathered if rank == 0 else None, dst=0)
    fails = _report_gateb(gathered, world) if rank == 0 else 0
    # all ranks exit with the gate verdict so torchrun/CI reports red on failure
    fails_t = torch.tensor([fails])
    dist.broadcast(fails_t, src=0)
    dist.barrier()
    sys.exit(1 if fails_t.item() else 0)


def _report_gateb(gathered, world) -> int:
    """Print per-prompt gate results, persist the verdict, return the failure count."""
    fails = 0
    for res in gathered:
        for i, tok_ok, fp_diff, got, want in sorted(res):
            ok = tok_ok and fp_diff < GATE_LOGIT_TOL
            fails += 0 if ok else 1
            print(f"[{'OK ' if ok else 'FAIL'}] prompt {i}: tokens_match={tok_ok} "
                  f"logit_fp_diff={fp_diff:.2e}")
            if not tok_ok:
                print("   want:", want, "\n   got :", got)
    print(f"GATE B (world={world}):", "PASS" if fails == 0 else f"FAIL ({fails})")
    (ROOT / "results" / f"gateb_world{world}.json").write_text(
        json.dumps({"world": world, "pass": fails == 0}))
    return fails


def mode_bench(args):
    from .prompts import SKEW
    from transformers import AutoTokenizer
    from . import MODEL_ID

    rank, world = dist.get_rank(), dist.get_world_size()
    cfg = load_config()
    placement = load_placement(args, cfg)
    engine = Engine(placement)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    all_ids = [tok(p).input_ids for p in SKEW]
    mine = [all_ids[i] for i in shard_indices(len(all_ids), rank, world)]

    # count the work: valid prompt positions + decode positions, summed globally
    my_positions = sum(len(p) for p in mine) * args.iters
    my_positions += len(mine) * args.decode_steps * args.iters

    dist.barrier()
    t_start = time.perf_counter()
    iter_times = []
    for _ in range(args.iters):
        it0 = time.perf_counter()
        shard = Shard(mine, cfg.num_hidden_layers, pad_id=cfg.pad_token_id)
        engine.prefill(shard)
        for _ in range(args.decode_steps):
            engine.decode_step(shard)
        dist.barrier()
        iter_times.append(time.perf_counter() - it0)
    wall = time.perf_counter() - t_start

    payload = {
        "rank": rank,
        "positions": my_positions,
        "stats": engine.stats.snapshot(),
    }
    gathered = [None] * world
    dist.gather_object(payload, gathered if rank == 0 else None, dst=0)
    if rank == 0:
        _report_bench(gathered, world, args, wall, iter_times)
    dist.barrier()


def _report_bench(gathered, world, args, wall, iter_times):
    """Write the bench JSON and print the per-rank load / imbalance summary."""
    total_positions = sum(g["positions"] for g in gathered)
    out = {
        "world": world,
        "plan": args.plan or "naive",
        "iters": args.iters,
        "decode_steps": args.decode_steps,
        "wall_time": wall,
        "iter_times": iter_times,
        "positions_per_s": total_positions / wall,
        "total_positions": total_positions,
        "ranks": gathered,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    loads = [g["stats"]["recv_tokens"] for g in gathered]
    comp = [g["stats"]["compute_time"] for g in gathered]
    a2a = [g["stats"]["a2a_time"] for g in gathered]
    mean_load = sum(loads) / len(loads)
    print(f"wall {wall:.1f}s  positions/s {total_positions / wall:.1f}")
    print(f"per-rank expert entries: {loads}  imbalance max/mean: "
          f"{max(loads) / mean_load:.2f}")
    print(f"per-rank compute_time: {[round(c, 1) for c in comp]}")
    print(f"per-rank a2a(+wait)  : {[round(a, 1) for a in a2a]}")
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gateb", "bench", "serve"], required=True)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--plan", default=None, help="placement plan JSON (default: naive)")
    ap.add_argument("--out", default=str(ROOT / "results" / "bench.json"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    dist.init_process_group("gloo")
    try:
        if args.mode == "gateb":
            mode_gateb(args)
        elif args.mode == "bench":
            mode_bench(args)
        else:
            from .server import serve
            serve(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
