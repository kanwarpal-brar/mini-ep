"""Gate A: from-scratch model must match the HF reference.

Checks, on the PARITY prompt set:
  1. teacher-forced full-sequence logits: max abs diff within tolerance
  2. greedy continuations identical to results/reference.json
"""
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniep import GATE_LOGIT_TOL, MODEL_ID
from miniep.modeling import greedy_generate
from miniep.weights import build_local_model

REF = Path(__file__).resolve().parents[1] / "results" / "reference.json"


def main():
    ref = json.loads(REF.read_text())
    prompt_ids = [torch.tensor([p["input_ids"]]) for p in ref["prompts"]]

    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    hf_logits = []
    with torch.no_grad():
        for ids in prompt_ids:
            hf_logits.append(hf(ids).logits[0])
    del hf
    gc.collect()

    model, _cfg = build_local_model()
    failures = 0
    per_prompt = []
    with torch.no_grad():
        for p, ids, hf_l in zip(ref["prompts"], prompt_ids, hf_logits):
            mine = model(ids)[0]
            diff = (mine - hf_l).abs().max().item()
            gen = greedy_generate(model, ids, ref["max_new_tokens"])
            tok_ok = gen == p["generated_ids"]
            status = "OK " if (diff < GATE_LOGIT_TOL and tok_ok) else "FAIL"
            if status == "FAIL":
                failures += 1
            per_prompt.append({"prompt": p["prompt"], "max_abs_diff": diff,
                               "tokens_match": tok_ok})
            print(f"[{status}] logit_maxdiff={diff:.2e} tokens_match={tok_ok}  {p['prompt'][:40]!r}")
            if not tok_ok:
                print("   ref :", p["generated_ids"][:16])
                print("   mine:", gen[:16])

    out = REF.parent / "gatea.json"
    out.write_text(json.dumps({"tol": GATE_LOGIT_TOL, "pass": failures == 0,
                               "prompts": per_prompt}, indent=1))
    print("wrote", out)
    print("GATE A:", "PASS" if failures == 0 else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
