# mini-EP: Explained from the Ground Up

This document explains everything you need to understand mini-EP, from the
underlying ideas up to the code and the experiment.

**Who this is for.** You're comfortable with linear algebra, basic programming,
and the general notion of a neural network. You do **not** need to know how
transformers, Mixture-of-Experts, or distributed computing work; Part 1 builds
those up. Specialized terms are defined the first time they appear, and the key
ones are collected in the [Glossary](#glossary).

**How to read it.**
- **Part 1**: the background concepts mini-EP is built on.
- **Part 2**: the problem the project demonstrates.
- **Part 3**: the fix.
- **Part 4**: how the code is built.
- **Part 5**: how correctness and the results are established.
- **Part 6**: limitations and the hardware caveats.

Already know transformers and MoE? Skip to [Part 2](#part-2--the-problem-mini-ep-demonstrates).

---

## Part 1: Prerequisites

### 1.1 How a language model generates text

A **large language model (LLM)** turns text into a sequence of integers
(**tokens**) with a **tokenizer**, then predicts the next token over and over.

- **Vocabulary**: the fixed set of possible tokens (here, 49,155 of them).
- **Embedding**: a lookup table mapping each token id to a vector of size `H`
  (the **hidden size**; here `H = 1024`). This vector, carried and transformed
  through the model, is the token's **hidden state**.
- **Logits**: at the end, the model produces one score per vocabulary entry.
  The highest-scoring token is the prediction. Taking the argmax every step is
  **greedy decoding** (deterministic; no randomness).
- **Decoder-only transformer**: the model is a stack of `L` identical
  **blocks** (here `L = 24`), each transforming the hidden states. "Decoder-only"
  means it generates left-to-right.
- **Autoregressive**: generation is one token at a time: predict, append the
  result to the input, predict again.

Generation has two phases:

- **Prefill**: process the entire prompt in one forward pass (all prompt tokens
  at once) to produce the first new token.
- **Decode**: generate the rest one token per forward pass.

- **KV cache**: attention (below) needs the key/value vectors of *all* previous
  tokens. Rather than recompute them every step, the model caches them, so producing
  each new token reuses past work instead of reprocessing the whole sequence.

### 1.2 Inside a block: attention + feed-forward

Each transformer block has two sub-layers, each wrapped in a normalization step
and a **residual connection** (add the sub-layer's output back to its input; this
stabilizes deep networks):

1. **Attention**: lets each token pull in information from other tokens. Each
   token forms a **query (Q)**, **key (K)**, and **value (V)** vector; a token's
   output is a weighted average of the values of the tokens it can see, weighted by
   query·key similarity. A **causal mask** restricts each token to attend only to
   earlier tokens (you can't see the future when generating). Two refinements
   this model uses:
   - **GQA (grouped-query attention)**: several query heads share one key/value
     head (here 16 query heads, 8 KV heads), which shrinks the KV cache.
   - **RoPE (rotary position embedding)**: encodes token position by rotating Q
     and K by an angle proportional to position, so attention is
     position-aware.
   - **RMSNorm**: the normalization used before each sub-layer; rescale a vector
     by its root-mean-square. Cheaper than LayerNorm, no mean-subtraction.

2. **Feed-forward network (FFN)**: a small multi-layer perceptron applied to
   **each token independently** (no mixing between tokens). In a standard
   transformer this is where most of the parameters and compute live. This model
   uses a **gated** FFN: `down(SiLU(gate(x)) * up(x))`, where `SiLU` is a smooth
   activation function and `gate`/`up`/`down` are linear layers.

**The one fact to carry forward:** attention mixes information *across* tokens;
the FFN processes *each token on its own*. That independence is exactly what lets
us cut the FFN into pieces and run them on different machines.

### 1.3 Mixture of Experts (MoE)

A **Mixture-of-Experts** layer replaces the single FFN with many smaller FFNs
called **experts**, plus a small **router** that decides, per token, which
experts to use.

- **Top-k routing**: the router (a linear layer `hidden → num_experts`) scores
  all `E` experts for a token; only the top `k` are run. Here `E = 32`,
  `k = 8`.
- **Gating**: the top-`k` scores are passed through a **softmax** (turning
  scores into weights that sum to 1). The layer's output is the gate-weighted sum
  of those `k` experts' outputs.
- **Sparse activation**: although the model *contains* 32 experts per layer,
  any given token only runs 8. So **total parameters** (capacity, which helps
  quality) can be large while **active parameters** (compute per token) stay
  small. Granite here is "1.3B total / ~400M active."

- **Why load is uneven (the seed of the whole project).** During training, a
  **load-balancing loss** nudges the router to spread tokens across experts. But
  that's an *average* over the training mix. At inference time, on a *narrow*
  workload (say, all code), the router concentrates traffic on a handful of
  experts: expert usage is **heavy-tailed**. Some experts are hammered; most sit
  nearly idle. Hold that thought.

### 1.4 Running a big model across many devices

One device often isn't enough: either the weights don't fit in memory, or one
device is too slow. So the model is split across several workers. Terms:

- **Rank / world size**: a **rank** is one worker process; **world size** is how
  many there are. mini-EP runs on **`torch.distributed`**, PyTorch's multi-process
  toolkit. On a real cluster each rank drives one GPU; on this CPU box each rank
  is a process (see [Part 6](#part-6--limitations-and-hardware-notes)).
- **Backend**: the library that moves tensors between ranks: **NCCL** on NVIDIA
  GPUs, **gloo** on CPU. mini-EP's logic is identical either way.

Two ways to split work:

- **Data parallelism**: copy the whole model to every rank, split the *batch*.
  Each rank does the same thing on different inputs.
- **Expert parallelism (EP)**: split the *experts* across ranks. Each rank keeps
  the full non-expert weights (attention, norms, router, embeddings) and owns
  only `E / world` experts per layer (here, 8 of 32 across 4 ranks). Attention
  runs data-parallel (each rank on its own slice of the batch); only the expert
  FFN is sharded.

**The catch that defines EP:** a token sitting on rank *r* may be routed to an
expert owned by rank *s*. So every MoE layer has to ship each token to the rank
that owns its expert, run it there, and ship the result back. That shipping is
done with **collective operations**, coordinated communication where every rank
participates at once:

- **all-reduce**: sum a tensor across all ranks; everyone gets the total.
- **all-gather**: every rank gets a copy of every rank's piece.
- **all-to-all**: every rank sends a (possibly different-sized) chunk to every
  other rank simultaneously. This is the natural primitive for "regroup the
  tokens by which rank owns their expert." mini-EP uses it in two phases per MoE layer:
  **dispatch** (send each token to its expert's owner), then **combine** (send the
  results back).

Because every rank must enter the same collective together, **each MoE layer is a
synchronization barrier**: the layer cannot finish until *every* rank has done
its share and reached the exchange. All ranks advance in **lockstep**. If one
rank has more work, everyone else waits for it; that rank is a **straggler**.

### 1.5 Determinism and floating point

Floating-point addition is **not associative**: `(a + b) + c` can differ from
`a + (b + c)` in the last bits. So if you sum the same numbers in a different
order (which happens naturally when the work is split differently across ranks),
you get slightly different results.

mini-EP cares about this because it *verifies* its distributed engine against a
simple single-process reference (the "gates" in Part 5). It can't make the two
match to the last bit (distributing the work changes the shapes of the underlying
matrix multiplies, and that alone perturbs the low-order bits), but it removes
every source of nondeterminism it *controls*. The main one: each token's `k`
expert results are always summed back together in the same fixed **canonical
order**, whatever ranks computed them. So the engine is repeatable run-to-run and
independent of *where* experts live, and the only residual gap from the reference
is rounding on the order of `1e-5`, small enough that greedy decoding still picks
identical tokens (which is what the gates check).

One more determinism trap, relevant on GPUs: accumulating many values into shared
memory slots (e.g. PyTorch's `index_add_`) uses **atomic adds**, whose completion
order is nondeterministic, so results wobble run to run. mini-EP avoids this by
writing each contribution to its **own unique slot** (a **scatter**) and only
then summing. Same math, but deterministic. This is a recurring design choice you'll
see in Part 4.

---

## Part 2: The problem mini-EP demonstrates

Put Parts 1.3 and 1.4 together. Experts are sharded across ranks with the obvious
**naive placement**: rank 0 owns experts 0–7, rank 1 owns 8–15, and so on, the
same fixed split for all 24 layers. Now run a skewed (code) workload, where each
layer's traffic piles onto a few **hot experts**.

Those hot experts live on whichever rank happens to own them. That rank does far
more expert compute in that layer than the others (it's the straggler) and
because the layer ends in an all-to-all barrier, **every other rank waits for
it.** The layer runs only as fast as its busiest worker.

**The subtle, central insight.** If you measure each rank's *total* expert load
summed over all 24 layers, it looks almost balanced, about **1.06×** max-vs-mean.
That's because a rank that's hot in one layer tends to be cold in another, and it
averages out. But that aggregate number is a mirage: **per layer**, the imbalance
is large (**1.27×** on average, **1.98×** in the worst layer), and you pay the
per-layer maximum at **every one of the 24 barriers**. The straggler
*rotates* from rank to rank as you go down the layers. Averaging hides a cost you
pay 24 separate times.

**Why this is worth fixing.** The accelerators are the expensive resource. A rank
that finishes its experts early then blocks at the layer's all-to-all barrier is
burning paid-for compute doing nothing. Because every layer waits for its busiest
rank, the imbalance stretches the expert-compute critical path (the sum of the
per-layer maxima) to **26.6%** above a perfectly balanced layer (the *straggler
tax*), a cost paid at each of the 24 barriers with no improvement in model quality
to show for it. It surfaces as lower throughput, higher latency, and worse
cost-per-token, and the tail only gets heavier at larger scale (more experts, more
ranks, longer context).

---

## Part 3: The fix (replicate hot experts)

An expert is a pair of weight matrices, data you can copy. So put a **copy
(replica)** of a hot expert on a second, under-loaded rank, and **split that
expert's tokens across the copies** (round-robin: first token to copy A, second
to copy B, and so on). If a hot expert has two copies, half its load moves off the
overloaded rank.

**This does not change the output.** The replicas are weight-identical and the
split is a deterministic round-robin, so every token is still processed by the
same expert with the same weights; only the *location* of the computation moves.
The model computes the same function and produces the same tokens as the naive
placement. (That equivalence is exactly what **Gate C** in Part 5 verifies.)

**The planner** (`balancer.py`) decides where to add replicas, greedily, one
layer at a time:

1. Start from naive placement and compute each rank's load for this layer
   (a replicated expert contributes `load / number_of_copies` to each of its
   ranks).
2. While the layer's `max/mean` rank load exceeds a target (**1.05**) and ranks
   still have spare **replica slots**: try every `(hot expert → currently-cold
   rank)` pairing, and add the one replica that most reduces the layer's busiest
   rank.
3. Stop when balanced enough or out of slots.

The budget is `slots_per_rank` (default **2**): each rank may host at most 2 extra
expert copies per layer.

**The cost.** Replicas take memory. Here the planner added **71 replicas** across
the 24×32 = 768 expert slots, i.e. **+9.2%** expert-weight memory, to cut the
straggler tax from 26.6% to 4.1%. Replication only helps *hot* experts; it can't
fix a layer whose load is spread thin but uneven.

---

## Part 4: How mini-EP is built

Everything below the Hugging Face tokenizer is written from scratch (~1,300 lines):
the model, the expert-parallel engine, the balancer, and the server. This part
walks through the machinery.

### 4.1 One MoE layer, end to end (the core loop)

This is the heart of the project: `EPMoEBackend.__call__` in `miniep/ep.py`. On
each rank the layer receives the tokens' hidden states `x` (shape `N × H`, where
`N` is this rank's token count) plus, for each token, its `k` chosen expert ids
and gate weights. Then:

1. **Flatten to entries.** Each valid `(token, expert-slot)` pair becomes one
   **entry**: a hidden vector, an expert id, and a gate weight. A token with
   `k = 8` experts produces up to 8 entries. Padded or inactive positions are
   dropped here via a validity mask, so they never reach an expert or the stats.
2. **Record routing counts.** Tally how many entries each expert got (this is
   what the balancer and `/stats` later consume).
3. **Pick a destination rank per entry.** `Placement.dest_ranks` maps each
   entry's expert id to the rank that will run it, round-robining across that
   expert's replicas. Then **sort** the entries by destination rank so all
   entries going to the same rank are contiguous.
4. **Exchange counts** (`all_gather`) so every rank learns how many entries it
   will receive from each other rank.
5. **Dispatch** (`all_to_all`) the expert ids and the hidden vectors. After this,
   each rank holds exactly the entries for the experts it owns.
6. **Run the experts locally.** `run_experts` groups the received entries by
   expert and does the two matmuls (`gate`/`up` → SiLU → `down`) per expert.
7. **Combine** (`all_to_all`) the results back to the ranks the entries came
   from.
8. **Reassemble deterministically.** Undo the sort from step 3, multiply each
   result by its gate weight, **scatter** it into a unique `(token, slot)` cell,
   and sum over the `k` slots to get each token's output. Because every
   `(token, slot)` cell is unique, this needs no atomic adds and the order is
   fixed regardless of world size or placement, the source of the engine's
   determinism (Part 1.5).
9. **Record timings**: expert-compute time and all-to-all(+wait) time, per rank.

Steps 1–9 are only the FFN. Attention already ran independently on each rank's own
tokens (data-parallel), so it needs no communication.

### 4.2 Module map

- **`miniep/modeling.py`**: the from-scratch GraniteMoE model: `RMSNorm`, RoPE,
  GQA attention, the gated-SiLU experts, `KVCache`, and the block/model classes.
  The MoE computation is delegated to a pluggable **backend** so the *same* model
  runs single-process (`LocalMoEBackend`, the reference) or across ranks
  (`EPMoEBackend`). It also faithfully reproduces Granite's architecture-specific
  scalar multipliers (embedding ×12, attention ×1/64, residual ×0.22, logits ÷6).
- **`miniep/ep.py`**: the expert-parallel core:
  - **`Placement`**: `owners[layer][expert] = [ranks]`. Provides `naive()`
    (contiguous split), load/save to JSON, and `dest_ranks` (the vectorized
    round-robin dispatch from step 3, computed without a Python loop but
    bit-equivalent to one).
  - **`Comm`**: wraps `all_to_all` (with a manual point-to-point `isend`/`irecv`
    fallback for older backends) and `exchange_counts`.
  - **`EPStats`**: per-rank counters (entries received per layer, expert-compute
    time, all-to-all time) plus global per-`(layer, expert)` routing counts.
  - **`EPMoEBackend`**: the loop in 4.1.
- **`miniep/balancer.py`**: `plan_balanced(...)`, the greedy planner from Part 3.
- **`miniep/weights.py`**: loads Granite's weights straight from the `safetensors`
  checkpoint (the on-disk weight file), slicing out **only the experts a rank
  owns** (so no rank ever holds all 32). `only_experts=True` reloads the expert
  tensors for the live hot-swap.
- **`miniep/worker.py`**: the `torchrun` entrypoint and the lockstep engine
  (next section). Has three modes: `gateb` (correctness gate), `bench`
  (benchmark), `serve`.
- **`miniep/server.py`**: the serving frontend and coordinator (next section).
- **`miniep/prompts.py`**: `PARITY` (8 diverse prompts for the gates) and `SKEW`
  (repeated code snippets, the domain-skewed workload that concentrates router
  traffic).
- **`scripts/`**: `reference.py` (Hugging Face ground truth), `check_parity.py`
  (Gate A), `make_plan.py` (bench → balanced plan), `make_charts.py`,
  `demo_serve.py`.

### 4.3 Keeping ranks in lockstep

Because every MoE layer is a collective barrier (Part 1.4), all ranks must issue
the **same** sequence of collectives. mini-EP guarantees this:

- **The engine** runs one batched prefill, then one forward per decode step, on
  every rank together. A rank with no real work (an idle worker, or a padded
  batch slot) runs a **masked dummy** forward: it participates in every
  collective but contributes zero expert load, so the barriers still line up and
  the stats stay clean.
- **The server** (`serve` mode) makes rank 0 an HTTP frontend *and* the
  coordinator. Every tick, rank 0 **broadcasts one command**: `idle`, `round`
  (run a batch of requests), `stats`, or `rebalance`; all ranks call matching
  collectives. It exposes:
  - `POST /generate`: enqueue a prompt; requests are batched per round.
  - `GET /stats`: per-rank load plus the **per-layer imbalance** metric (the
    aggregate-vs-per-layer gap this whole project is about).
  - `POST /rebalance`: sum the routing counts seen so far (`all_reduce`), build a
    fresh plan, and **hot-swap** it live. Each rank re-slices the experts for its
    new assignment from the checkpoint (only the expert tensors; the rest of the
    model stays resident) and swaps them in, with no restart and no weight transfer
    between ranks. Measured at roughly one second.

---

## Part 5: Proving it works

### Correctness gates

The distributed engine is only trustworthy if it reproduces a simple reference.
`reference.py` runs the stock Hugging Face model once to produce ground truth;
every later stage must match it. Two terms:

- **Teacher forcing**: feed the model the *known* full sequence and compare the
  logits at every position in a single pass (tests the math directly, without
  compounding generation errors).
- **Logit fingerprint**: the first 64 logit values at the final position; a cheap
  numeric signature to compare against the reference.

| Gate | What it checks | Result |
|------|----------------|--------|
| **A** | from-scratch model vs Hugging Face (single process): teacher-forced logits + 32-token greedy continuations on 8 prompts | **bitwise-identical** logits (max abs diff `0.0`), all continuations match |
| **B** | expert-parallel engine (world size 2 and 4) vs reference | continuations match, logit fingerprint agrees to `< 3e-5` |
| **C** | EP **with replicated placement** vs reference | outputs unchanged (replicas don't alter the math) |

Gate A uses a tolerance of `2e-3` (headroom for fp32 summation-order noise across
implementations) but lands at `0.0`. Every gate script exits non-zero on
failure, so they double as CI checks.

### The benchmark and its metrics

`bench` mode runs the `SKEW` workload on 4 ranks, first under naive placement,
then under the balanced plan `make_plan.py` derived from the naive run's routing
counts. The metrics:

- **Per-layer load imbalance**: for one layer, `max_rank(load) / mean_rank(load)`;
  reported averaged over the 24 layers, and for the single worst layer.
- **Straggler overhead**: `Σ_layers max_rank(load) / Σ_layers mean_rank(load) − 1`.
  The extra expert-compute time paid because each layer waits for its busiest
  rank. This is the number that matters.
- **Worker busy fraction**: `expert-compute / (expert-compute + all-to-all-and-wait)`.
  How much of a worker's MoE time is real expert work vs. moving data and waiting.
- **Throughput**: token-positions (prefill + decode) processed per second across
  the whole cluster.

Results (4 workers, skewed code workload, 3 iterations):

| | naive | balanced |
|---|---|---|
| per-layer imbalance (avg / worst) | 1.27× / 1.98× | **1.04× / 1.16×** |
| straggler overhead | 26.6% | **4.1%** |
| worker busy fraction | 62–73% | 69–79% |
| wall time | 158.1 s | **148.4 s** |
| throughput | 81.9 tok-pos/s | **87.3 (+6.6%)** |

**Why is +6.6% so much smaller than the 26.6% → 4.1% straggler-tax drop?** Look at
where the time goes. Each rank's *total* expert compute is already nearly balanced
(the ~1.06× aggregate from Part 2); the imbalance is per-layer, not aggregate. It
manifests as the *other* ranks waiting at each layer's barrier (no single rank
computes more). That wait is lumped together with the data
transfer in the all-to-all time, which ate 27–38% of each worker's MoE time in the
naive run (the 62–73% busy fraction). Balancing shrinks the wait, lifting the busy
fraction to 69–79% and shortening the critical path (the ~6% wall-time win). It
can't do more: the expert compute itself (about half of each step) and the
non-expert work (attention, the output projection, the all-to-all transfers,
framework overhead) are the same wherever experts live. The straggler tax is the
clean measure of the imbalance; throughput is that gain diluted across the whole
pipeline.

**Determinism.** Runs are repeatable: per-iteration times were 52.3–53.2 s (naive)
vs 49.1–49.9 s (balanced), non-overlapping, and routing counts are bit-identical
across reruns, so the improvement is real signal, not noise.

---

## Part 6: Limitations and hardware notes

**CPU ranks stand in for GPUs.** This was built on a 4-core, GPU-less ARM box, so
4 `torch.distributed` (gloo) ranks play the role of 4 GPUs. Everything the demo
shows (expert sharding, all-to-all dispatch/combine, per-layer stragglers,
replication) is backend-independent. Porting to a real multi-GPU node means
initializing NCCL instead of gloo and placing each rank's tensors on its GPU;
the engine and collective logic are unchanged. Three choices keep that port clean:
the deterministic scatter combine (no atomic adds), vectorized dispatch (no
per-entry Python loop), and experts-only reload on hot-swap. One caveat on the
numbers: the balance between compute and communication shifts with hardware, batch
size, and model, so treat this run as a faithful demonstration of the *mechanism*
and a direct measurement of the *imbalance*, not a throughput prediction for a
particular GPU. The quantity the balancer targets, per-layer expert-load
imbalance, is hardware-independent; how much wall time it buys back depends on how
large a share of each step is expert compute.

**The model is a substitution too.** The original target (OLMoE-1B-7B) is 14 GB and
won't run on 4 CPU cores. `granite-3.0-1b-a400m-instruct` is the same architecture
class (many small experts, top-k softmax routing) at a size that fits, and nothing
in the engine is Granite-specific.

**Real limitations of the approach itself:**

- **Load drift.** The balanced plan is fit to one measured traffic window. When the
  workload shifts, imbalance creeps back. `/rebalance` re-fits on demand, but there
  is no automatic trigger, no smoothing over time, and no hysteresis; a production
  system would rebalance continuously and predictively.
- **Replication costs memory** and only helps hot experts; it can't help a layer
  whose load is spread thin but uneven.
- **Serving is deliberately minimal**: per-round (static) batching, greedy
  decoding, no KV paging or chunked prefill. The demo is about placement and
  balancing, not serving throughput.

---

## Glossary

- **Active parameters**: the weights used per token (`k` experts + the
  non-expert layers), as opposed to total parameters.
- **all-gather / all-reduce / all-to-all**: collective operations; every rank gets
  every rank's data / a summed tensor / a personalized chunk from each rank,
  respectively.
- **Atomic add**: a hardware-serialized accumulation into a shared memory slot;
  order (and thus floating-point result) is nondeterministic. Avoided here.
- **Autoregressive**: generating one token at a time, feeding each output back in.
- **Backend (gloo / NCCL)**: the library moving tensors between ranks; gloo on CPU,
  NCCL on NVIDIA GPUs.
- **Causal mask**: restriction that a token may attend only to earlier tokens.
- **Collective operation**: communication in which all ranks participate together.
- **Combine**: the second all-to-all of a MoE layer, returning expert outputs to
  the ranks their tokens came from.
- **Data parallelism**: replicate the model, split the batch across ranks.
- **Decode**: the generation phase producing one token per forward pass.
- **Dispatch**: the first all-to-all of a MoE layer, sending tokens to expert
  owners.
- **Embedding**: the lookup mapping a token id to its hidden vector.
- **Expert**: one of the small FFNs inside a MoE layer.
- **Expert parallelism (EP)**: sharding experts across ranks; attention stays
  data-parallel.
- **FFN (feed-forward network)**: the per-token MLP sub-layer of a transformer
  block.
- **Gating**: the softmax-weighted combination of a token's chosen experts.
- **Greedy decoding**: always pick the highest-probability next token.
- **GQA (grouped-query attention)**: multiple query heads sharing one key/value
  head to shrink the KV cache.
- **Hidden state / hidden size (H)**: the per-token vector carried through the
  model, and its dimension (1024 here).
- **Hot expert**: an expert receiving disproportionately many tokens.
- **KV cache**: stored key/value vectors of past tokens, so decode stays cheap.
- **Lockstep**: all ranks executing the same sequence of forwards/collectives in
  sync.
- **Logits**: the model's raw per-vocabulary scores for the next token.
- **MoE (Mixture of Experts)**: a layer of many experts + a router selecting a few
  per token.
- **Naive placement**: the baseline contiguous expert-to-rank assignment.
- **Prefill**: the phase that processes the whole prompt in one pass.
- **Rank / world size**: one worker process / the number of them.
- **Replica**: an extra copy of an expert placed on another rank to share its load.
- **Residual connection**: adding a sub-layer's input to its output.
- **RMSNorm**: root-mean-square normalization.
- **RoPE (rotary position embedding)**: positional encoding via rotation of Q/K.
- **Router**: the linear layer that scores experts per token.
- **Scatter**: writing values to distinct indices (no accumulation), hence
  deterministic.
- **SiLU**: the smooth activation `x · sigmoid(x)` used in the experts.
- **Sparse activation**: running only a few of the available experts per token.
- **Straggler**: the slowest rank at a barrier; everyone waits for it.
- **Teacher forcing**: evaluating on a known sequence in one pass, comparing
  per-position logits.
- **Token / tokenizer / vocabulary**: the integer units of text / the encoder /
  the full set of tokens.
- **Top-k routing**: running only the `k` highest-scored experts per token.
- **`torch.distributed`**: PyTorch's multi-process/-device communication toolkit.
