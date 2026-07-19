"""Exercise the serving endpoint: concurrent requests, stats, live rebalance.

  terminal 1: uv run torchrun --nproc-per-node 4 -m miniep.worker --mode serve
  terminal 2: uv run python scripts/demo_serve.py
"""
import argparse
import concurrent.futures as cf
import json

import httpx

QUESTIONS = [
    "Write a Python function that checks if a string is a palindrome.",
    "What is a mixture-of-experts model?",
    "Write a SQL query that counts orders per customer.",
    "Explain what a load balancer does in one paragraph.",
    "Write a JavaScript function that debounces another function.",
    "What causes the straggler problem in distributed computing?",
    "Write a bash one-liner to find the 10 largest files in a directory.",
    "Explain the difference between a process and a thread.",
]


def fire(base, prompt, max_new):
    r = httpx.post(f"{base}/generate",
                   json={"prompt": prompt, "max_new_tokens": max_new},
                   timeout=600)
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8008")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--rebalance", action="store_true")
    args = ap.parse_args()

    print(f"firing {len(QUESTIONS)} concurrent requests...")
    with cf.ThreadPoolExecutor(len(QUESTIONS)) as ex:
        futs = [ex.submit(fire, args.base, q, args.max_new) for q in QUESTIONS]
        for q, f in zip(QUESTIONS, futs):
            res = f.result()
            if "rank" not in res:
                print(f"\n--- SERVER ERROR for {q!r}: {res}")
                continue
            print(f"\n--- [rank {res['rank']}] {q}\n{res['text'].strip()[:200]}")

    stats = httpx.get(f"{args.base}/stats", timeout=60).json()
    print("\n/stats:", json.dumps(stats, indent=1))

    if args.rebalance:
        rb = httpx.post(f"{args.base}/rebalance", timeout=300).json()
        print("\n/rebalance:", json.dumps(rb, indent=1))


if __name__ == "__main__":
    main()
