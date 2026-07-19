"""Build a balanced placement plan from a naive-placement bench run.

Usage: uv run python scripts/make_plan.py results/bench_naive.json \
           --out results/plan_balanced.json --slots 2
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniep.balancer import plan_balanced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_json")
    ap.add_argument("--out", default="results/plan_balanced.json")
    ap.add_argument("--slots", type=int, default=2)
    args = ap.parse_args()

    bench = json.loads(Path(args.bench_json).read_text())
    world = bench["world"]
    counts = sum(torch.tensor(r["stats"]["route_counts"]) for r in bench["ranks"])

    placement, report = plan_balanced(counts, world, slots_per_rank=args.slots)
    Path(args.out).write_text(json.dumps(placement.to_plan()))

    ib = [l["imbalance_before"] for l in report["layers"]]
    ia = [l["imbalance_after"] for l in report["layers"]]
    print(f"world={world} slots/rank/layer={args.slots} "
          f"replicas added: {report['replicas_total']} "
          f"(mem overhead {report['replicas_total'] / (counts.shape[0] * counts.shape[1]):.1%})")
    print(f"predicted per-layer rank imbalance (max/mean): "
          f"before avg {sum(ib)/len(ib):.2f} worst {max(ib):.2f} -> "
          f"after avg {sum(ia)/len(ia):.2f} worst {max(ia):.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
