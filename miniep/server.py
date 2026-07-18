"""Serving mode: FastAPI frontend on rank 0, lockstep EP workers behind it.

All ranks sit in a command loop; rank 0 broadcasts one command per iteration so
collectives stay aligned:
  idle       nothing queued; everyone sleeps briefly
  round      admit queued requests (round-robin across ranks), prefill, then
             decode until every sequence hits EOS or its token budget
  stats      gather per-rank stats snapshots to rank 0
  rebalance  build a replication plan from routing stats accumulated so far,
             hot-load newly assigned expert weights from the checkpoint, swap
             the placement in place, no restart
  shutdown   exit

Batching is per-round (static): requests that arrive mid-round join the next
round. Continuous batching is future work.
"""
from __future__ import annotations

import queue
import threading
import time

import torch
import torch.distributed as dist

from . import MODEL_ID
from .balancer import plan_balanced
from .ep import Placement
from .weights import expert_weight_keys, load_weights


class Op:
    """Lockstep command ops broadcast from rank 0."""
    IDLE = "idle"
    ROUND = "round"
    STATS = "stats"
    REBALANCE_STATS = "rebalance_stats"
    REBALANCE = "rebalance"
    SHUTDOWN = "shutdown"


IDLE_MS = 30  # idle-tick sleep, matched by rank 0 and the workers


def _broadcast_cmd(cmd, rank):
    box = [cmd if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def apply_placement(engine, plan_dict):
    """Swap placement live: load newly owned expert weights, drop the rest."""
    new_placement = Placement.from_plan(plan_dict)
    my = new_placement.experts_of_rank(engine.rank)
    w = load_weights(experts_per_layer=my, only_experts=True)
    for l, backend in enumerate(engine.backends):
        in_key, out_key = expert_weight_keys(l)
        backend.placement = new_placement
        backend.w_in = w[in_key]
        backend.w_out = w[out_key]
        backend._update_map(my[l])
    engine.placement = new_placement


def run_round(engine, seqs_for_me, max_steps):
    """seqs_for_me: list of {id, ids, max_new}. Returns {seq_id: token_list}."""
    from .worker import Shard

    dummy = not seqs_for_me
    pad_id = engine.cfg.pad_token_id
    eos_id = engine.cfg.eos_token_id
    prompt_ids = [s["ids"] for s in seqs_for_me] or [[pad_id]]
    shard = Shard(prompt_ids, engine.cfg.num_hidden_layers, pad_id=pad_id)
    if dummy:
        # dummy shard keeps collectives aligned but must add zero expert load
        engine.prefill_dummy(shard)
    else:
        engine.prefill(shard)

    budgets = [s["max_new"] for s in seqs_for_me] or [0]
    active = torch.tensor([not dummy] * shard.B)
    for step in range(max_steps):
        if not dummy:
            for i in range(shard.B):
                if active[i] and (len(shard.generated[i]) >= budgets[i]
                                  or (shard.generated[i] and shard.generated[i][-1] == eos_id)):
                    active[i] = False
        engine.decode_step(shard, active=active)
    out = {}
    for s, gen, budget in zip(seqs_for_me, shard.generated, budgets):
        toks = gen[:budget]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        out[s["id"]] = toks
    return out


def _worker_loop(engine, t_start):
    """Non-coordinator ranks: obey broadcast commands so collectives stay aligned."""
    while True:
        cmd = _broadcast_cmd(None, engine.rank)
        op = cmd["op"]
        if op == Op.SHUTDOWN:
            return
        elif op == Op.IDLE:
            time.sleep(cmd["ms"] / 1000)
        elif op == Op.ROUND:
            results = run_round(engine, cmd["assign"].get(engine.rank, []), cmd["max_steps"])
            dist.gather_object(results, None, dst=0)
        elif op == Op.STATS:
            dist.gather_object(_stats_payload(engine, t_start), None, dst=0)
        elif op == Op.REBALANCE_STATS:
            dist.all_reduce(engine.stats.route_counts.clone())
        elif op == Op.REBALANCE:
            apply_placement(engine, cmd["plan"])


def _submit_control(control_q, cmd, timeout):
    """Enqueue a control command for the coordinator loop and block for its result."""
    done = threading.Event()
    slot = {}
    control_q.put({**cmd, "done": done, "slot": slot})
    done.wait(timeout=timeout)
    return slot.get("payload", {})


def _register_routes(app, tok, requests_q, control_q):
    @app.post("/generate")
    def generate(body: dict):
        # sync handler on purpose: FastAPI runs it in the threadpool, so the
        # blocking done.wait() below cannot stall the event loop
        prompt = body["prompt"]
        max_new = int(body.get("max_new_tokens", 64))
        if body.get("chat", True):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True)["input_ids"]
        else:
            ids = tok(prompt).input_ids
        done = threading.Event()
        slot = {}
        requests_q.put({"ids": ids, "max_new": max_new, "done": done, "slot": slot})
        done.wait(timeout=600)
        return {"text": tok.decode(slot.get("tokens", [])),
                "tokens": len(slot.get("tokens", [])),
                "rank": slot.get("rank"), "round_s": slot.get("round_s")}

    @app.get("/stats")
    def stats():
        return _submit_control(control_q, {"op": Op.STATS}, timeout=60)

    @app.post("/rebalance")
    def rebalance(slots_per_rank: int = 2):
        return _submit_control(control_q, {"op": Op.REBALANCE, "slots": slots_per_rank},
                               timeout=300)


def _handle_stats(engine, ctl, t_start, world):
    _broadcast_cmd({"op": Op.STATS}, 0)
    gathered = [None] * world
    dist.gather_object(_stats_payload(engine, t_start), gathered, dst=0)
    ctl["slot"]["payload"] = _stats_summary(gathered)
    ctl["done"].set()


def _handle_rebalance(engine, ctl, world):
    _broadcast_cmd({"op": Op.REBALANCE_STATS}, 0)
    counts = engine.stats.route_counts.clone()
    dist.all_reduce(counts)  # source-side counts summed over ranks = global
    placement, report = plan_balanced(counts, world, slots_per_rank=ctl["slots"])
    plan = placement.to_plan()
    t0 = time.perf_counter()
    _broadcast_cmd({"op": Op.REBALANCE, "plan": plan}, 0)
    apply_placement(engine, plan)
    ib = [l["imbalance_before"] for l in report["layers"]]
    ia = [l["imbalance_after"] for l in report["layers"]]
    ctl["slot"]["payload"] = {
        "replicas_added": report["replicas_total"],
        "predicted_imbalance_before_avg": round(sum(ib) / len(ib), 3),
        "predicted_imbalance_after_avg": round(sum(ia) / len(ia), 3),
        "apply_seconds": round(time.perf_counter() - t0, 2),
    }
    ctl["done"].set()


def _drain_requests(requests_q, max_batch):
    """Block briefly for one request, then greedily pull up to max_batch more."""
    pending = []
    try:
        pending.append(requests_q.get(timeout=0.05))
        while len(pending) < max_batch:
            pending.append(requests_q.get_nowait())
    except queue.Empty:
        pass
    return pending


def _run_generation_round(engine, pending, world, seq_counter):
    """Assign requests round-robin across ranks, run one batched round, fulfil each
    request's future. Returns the advanced seq_counter."""
    assign = {r: [] for r in range(world)}
    for i, req in enumerate(pending):
        seq_counter += 1
        req["seq_id"] = seq_counter
        assign[i % world].append(
            {"id": seq_counter, "ids": req["ids"], "max_new": req["max_new"]})
    max_steps = max(r["max_new"] for r in pending)
    t0 = time.perf_counter()
    _broadcast_cmd({"op": Op.ROUND, "assign": assign, "max_steps": max_steps}, 0)
    my_results = run_round(engine, assign[0], max_steps)
    gathered = [None] * world
    dist.gather_object(my_results, gathered, dst=0)
    round_s = time.perf_counter() - t0
    merged = {}
    for r, res in enumerate(gathered):
        for sid, toks in res.items():
            merged[sid] = (toks, r)
    for req in pending:
        toks, r = merged[req["seq_id"]]
        req["slot"].update(tokens=toks, rank=r, round_s=round(round_s, 2))
        req["done"].set()
    return seq_counter


def _coordinator_loop(engine, requests_q, control_q, t_start, world, args):
    """Rank 0: each tick, handle one queued control command, else run a generation round."""
    seq_counter = 0
    while True:
        try:
            ctl = control_q.get_nowait()
        except queue.Empty:
            ctl = None
        if ctl and ctl["op"] == Op.STATS:
            _handle_stats(engine, ctl, t_start, world)
            continue
        if ctl and ctl["op"] == Op.REBALANCE:
            _handle_rebalance(engine, ctl, world)
            continue

        pending = _drain_requests(requests_q, args.max_batch)
        if not pending:
            _broadcast_cmd({"op": Op.IDLE, "ms": IDLE_MS}, 0)
            time.sleep(IDLE_MS / 1000)
            continue
        seq_counter = _run_generation_round(engine, pending, world, seq_counter)


def serve(args):
    from .worker import Engine, load_placement, load_config

    rank, world = dist.get_rank(), dist.get_world_size()
    engine = Engine(load_placement(args, load_config()))
    t_start = time.perf_counter()

    if rank != 0:
        _worker_loop(engine, t_start)
        return

    # ---- rank 0: HTTP frontend + coordinator loop ----
    from fastapi import FastAPI
    import uvicorn
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    requests_q: "queue.Queue[dict]" = queue.Queue()
    control_q: "queue.Queue[dict]" = queue.Queue()
    app = FastAPI(title="mini-EP")
    _register_routes(app, tok, requests_q, control_q)

    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=args.port,
                                   log_level="warning"),
        daemon=True).start()
    print(f"[rank0] serving on http://127.0.0.1:{args.port}  (world={world})")

    _coordinator_loop(engine, requests_q, control_q, t_start, world, args)


def _stats_payload(engine, t_start):
    s = engine.stats.snapshot()
    s["wall"] = time.perf_counter() - t_start
    s["rank"] = engine.rank
    s["experts_per_layer"] = [
        len(engine.backends[l].local_experts)
        for l in range(engine.cfg.num_hidden_layers)]
    del s["route_counts"]
    return s


def _stats_summary(gathered):
    loads = [g["recv_tokens"] for g in gathered]
    mean = sum(loads) / len(loads) if any(loads) else 1
    # per-layer imbalance across ranks: the straggler metric that matters
    # (aggregate per-rank load can look balanced while per-layer is not)
    per_layer = torch.tensor([g["recv_tokens_per_layer"] for g in gathered],
                             dtype=torch.float64)  # (world, L)
    layer_mean = per_layer.mean(dim=0)
    layer_imb = torch.where(layer_mean > 0,
                            per_layer.max(dim=0).values / layer_mean.clamp(min=1),
                            torch.zeros_like(layer_mean))
    return {
        "world": len(gathered),
        "per_rank_expert_tokens": loads,
        "imbalance_max_over_mean": max(loads) / mean if any(loads) else None,
        "per_layer_imbalance_avg": round(float(layer_imb.mean()), 3),
        "per_layer_imbalance_worst": round(float(layer_imb.max()), 3),
        "per_rank_compute_s": [round(g["compute_time"], 2) for g in gathered],
        "per_rank_a2a_wait_s": [round(g["a2a_time"], 2) for g in gathered],
        "per_rank_busy_frac": [
            round(g["compute_time"] / g["wall"], 3) for g in gathered],
        "experts_per_layer_per_rank": [
            round(sum(g["experts_per_layer"]) / len(g["experts_per_layer"]), 1)
            for g in gathered],
    }
