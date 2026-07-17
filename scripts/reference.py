"""Single-process HF reference run: the ground truth every later stage is gated on.

Produces results/reference.json:
  - greedy continuations (token ids + text) for the PARITY prompt set
  - final-position logits fingerprint per prompt (for tolerance checks)
  - per-layer, per-expert routing counts for PARITY and SKEW prompt sets
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniep import LOGIT_FP_KEY, LOGIT_FP_SIZE, MODEL_ID
from miniep.prompts import PARITY, SKEW

MAX_NEW_TOKENS = 32
OUT = Path(__file__).resolve().parents[1] / "results"


def find_router_modules(model):
    """Locate each layer's router Linear (hidden -> num_experts) by name."""
    routers = {}
    for name, mod in model.named_modules():
        # transformers 4.x: ...block_sparse_moe.router.layer ; 5.x: ...block_sparse_moe.router
        if name.endswith("block_sparse_moe.router.layer") or name.endswith("block_sparse_moe.router"):
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            routers[layer_idx] = mod
    assert routers, "no router modules found; transformers layout changed?"
    return routers


class RoutingCounter:
    def __init__(self, model, num_experts, top_k):
        self.num_experts = num_experts
        self.top_k = top_k
        routers = find_router_modules(model)
        self.counts = torch.zeros(len(routers), num_experts, dtype=torch.long)
        self.handles = [
            mod.register_forward_hook(self._make_hook(i)) for i, mod in sorted(routers.items())
        ]

    def _make_hook(self, layer_idx):
        def hook(_mod, _inp, out):
            # transformers 5 router returns (top_k_index, top_k_weights, logits);
            # a bare Linear returns the (tokens, num_experts) logits
            if isinstance(out, tuple):
                top = out[0].reshape(-1)
            else:
                top = out.topk(self.top_k, dim=-1).indices.reshape(-1)
            self.counts[layer_idx] += torch.bincount(top, minlength=self.num_experts)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def main():
    OUT.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    cfg = model.config
    print(f"loaded {MODEL_ID}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params fp32")

    results = {"model": MODEL_ID, "max_new_tokens": MAX_NEW_TOKENS, "prompts": []}

    # --- greedy generation on PARITY set (correctness ground truth) ---
    for prompt in PARITY:
        ids = tok(prompt, return_tensors="pt").input_ids
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=cfg.pad_token_id,
            )
            logits_last = model(ids).logits[0, -1]
        gen = out[0, ids.shape[1]:]
        results["prompts"].append({
            "prompt": prompt,
            "input_ids": ids[0].tolist(),
            "generated_ids": gen.tolist(),
            "text": tok.decode(gen),
            LOGIT_FP_KEY: logits_last[:LOGIT_FP_SIZE].tolist(),
            "last_logits_absmean": logits_last.abs().mean().item(),
        })
        print(f"[{time.time()-t0:5.1f}s] {prompt[:40]!r} -> {tok.decode(gen)[:60]!r}")

    # --- routing histograms (teacher-forced over prompt tokens) ---
    for set_name, prompts in [("parity", PARITY), ("skew", SKEW)]:
        counter = RoutingCounter(model, cfg.num_local_experts, cfg.num_experts_per_tok)
        n_tok = 0
        t0 = time.time()
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt").input_ids
                n_tok += ids.numel()
                model(ids)
        counter.close()
        counts = counter.counts
        per_layer = counts.float()
        mm = (per_layer.max(dim=1).values / per_layer.mean(dim=1)).tolist()
        print(f"{set_name}: {n_tok} tokens, {time.time()-t0:.1f}s; "
              f"per-layer max/mean expert load: min {min(mm):.2f} max {max(mm):.2f}")
        with open(OUT / f"expert_load_{set_name}.json", "w") as f:
            json.dump({"set": set_name, "tokens": n_tok, "counts": counts.tolist()}, f)

    with open(OUT / "reference.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", OUT / "reference.json")


if __name__ == "__main__":
    main()
