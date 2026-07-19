"""Render README charts from results/*.json (light-mode PNGs).

Palette: validated reference palette (dataviz skill); categorical slots in
documented order (blue #2a78d6, green #008300), sequential blue ramp for
heatmaps, muted gray for dead time, chrome inks for text/grid.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"   # slot 1: naive / primary series
GREEN = "#008300"  # slot 2: balanced
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "font.size": 9,
})


def load(name):
    return json.loads((RES / name).read_text())


def per_layer_loads(bench):
    return np.array([r["stats"]["recv_tokens_per_layer"] for r in bench["ranks"]],
                    dtype=float)  # (world, L)


def chart_expert_distribution():
    data = load("expert_load_skew.json")
    counts = np.array(data["counts"], dtype=float)  # (L, E)
    imb = counts.max(axis=1) / counts.mean(axis=1)
    worst = int(imb.argmax())
    c = counts[worst]
    order = np.argsort(-c)
    mean = c.mean()

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    colors = [BLUE if v > 2 * mean else SEQ[1] for v in c[order]]
    ax.bar(range(len(c)), c[order], color=colors, width=0.82)
    ax.axhline(mean, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(len(c) - 0.5, mean, " mean", va="bottom", ha="right", color=INK2, fontsize=8)
    hot = (c[order] > 2 * mean).sum()
    ax.set_title(f"Expert load is heavy-tailed: layer {worst}, code workload")
    ax.set_xlabel("expert (sorted by tokens routed)")
    ax.set_ylabel("tokens routed")
    ax.set_xticks([0, 7, 15, 23, 31])
    ax.text(0.99, 0.86, f"top expert ≈ {imb[worst]:.1f}× mean;  {hot} experts above 2× mean",
            transform=ax.transAxes, ha="right", color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "expert_distribution.png")
    plt.close(fig)


def chart_load_heatmap(naive, balanced):
    ln = per_layer_loads(naive)
    lb = per_layer_loads(balanced)
    # normalize each layer column to share-of-layer so color = share of that
    # layer's traffic handled by the rank (0.25 = perfectly balanced at world=4)
    def shares(x):
        return x / x.sum(axis=0, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    vmax = max(shares(ln).max(), shares(lb).max())

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.4), dpi=200, sharex=True)
    for ax, mat, label in [(axes[0], shares(ln), "naive contiguous placement"),
                           (axes[1], shares(lb), "balanced (hot experts replicated)")]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks(range(mat.shape[0]), [f"rank {r}" for r in range(mat.shape[0])])
        ax.set_title(label, loc="left", fontsize=9.5)
        ax.grid(False)
        worst = mat.max(axis=0)
        ax.set_ylabel("")
    axes[1].set_xlabel("MoE layer")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("share of layer's expert tokens", fontsize=8)
    cb.outline.set_visible(False)
    imb_n = (ln.max(0) / ln.mean(0)).mean()
    imb_b = (lb.max(0) / lb.mean(0)).mean()
    fig.suptitle(f"Per-layer load per rank: avg per-layer imbalance "
                 f"{imb_n:.2f}× → {imb_b:.2f}×  (1.00× = perfect)",
                 fontsize=11, fontweight="bold", x=0.075, ha="left")
    fig.savefig(OUT / "load_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def chart_straggler(naive, balanced):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200, sharey=True)
    tallest = 0
    for ax, bench, label in [(axes[0], naive, "naive"),
                             (axes[1], balanced, "balanced")]:
        world = bench["world"]
        comp = [r["stats"]["compute_time"] for r in bench["ranks"]]
        wait = [r["stats"]["a2a_time"] for r in bench["ranks"]]
        tallest = max(tallest, max(c + w for c, w in zip(comp, wait)))
        xs = np.arange(world)
        ax.bar(xs, comp, 0.62, color=BLUE, label="expert compute")
        ax.bar(xs, wait, 0.62, bottom=comp, color=MUTED, label="all-to-all + waiting",
               edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(xs, [f"rank {r}" for r in xs])
        thr = bench["positions_per_s"]
        ax.set_title(f"{label}: {bench['wall_time']:.0f}s wall, {thr:.0f} tok-pos/s",
                     loc="left", fontsize=9.5)
        for x, (c, w) in enumerate(zip(comp, wait)):
            ax.text(x, c + w + 1.2, f"{c / (c + w):.0%} busy", ha="center",
                    fontsize=7.5, color=INK2)
    for ax in axes:
        ax.set_ylim(0, tallest * 1.16)
    axes[0].set_ylabel("seconds (whole run)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8, ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 0.89))
    fig.suptitle("Same workload, same ranks; replication turns waiting into work",
                 fontsize=11, fontweight="bold", x=0.06, y=0.985, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(OUT / "straggler.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    naive = load("bench_naive.json")
    balanced = load("bench_balanced.json")
    chart_expert_distribution()
    chart_load_heatmap(naive, balanced)
    chart_straggler(naive, balanced)
    imb_n = (per_layer_loads(naive).max(0) / per_layer_loads(naive).mean(0)).mean()
    imb_b = (per_layer_loads(balanced).max(0) / per_layer_loads(balanced).mean(0)).mean()
    print(f"imbalance {imb_n:.3f} -> {imb_b:.3f}; "
          f"throughput {naive['positions_per_s']:.1f} -> {balanced['positions_per_s']:.1f} "
          f"({balanced['positions_per_s'] / naive['positions_per_s'] - 1:+.1%})")
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
