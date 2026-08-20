#!/usr/bin/env python
"""
Build Figure 3 (runtime scalability) from the ``run_test_runtime.py`` outputs.

Reads every JSON in ``results/test_results/`` and plots mean explanation time
against graph size on log-log axes, one line per method. Methods that could not
be run at a given size (CF and CF^2 exhaust memory beyond ~6,400 nodes) simply
have no point there, which is how the missing segments in Figure 3 arise.

Usage (from the repository root):
    python scripts/plot_scalability.py
    python scripts/plot_scalability.py --synthetic_type neighbor_feature \
        --nodes 100 400 1600 6400 25600 --out figures/scalability.png
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

METHOD_STYLE = {
    "cf-greedy": {"label": "CF-Greedy (Ours)", "color": "#2ca02c", "marker": "s"},
    "cf": {"label": "CF", "color": "#1f77b4", "marker": "o"},
    "cff": {"label": r"CF$^2$", "color": "#ff7f0e", "marker": "^"},
}
METHOD_ORDER = ["cf-greedy", "cf", "cff"]


def collect(results_dir, synthetic_type):
    """method -> {n_nodes: [times]} over all matching result files."""
    times = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        return times, files

    for path in files:
        with open(path, "r") as f:
            payload = json.load(f)
        config = payload.get("config", {})
        if synthetic_type and config.get("synthetic_type") != synthetic_type:
            continue
        n_nodes = config.get("n_nodes")
        if n_nodes is None:
            print(f"  (skipping {os.path.basename(path)}: no n_nodes in config)")
            continue
        for method, md in payload.get("methods", {}).items():
            if "running_time" in md:
                times[method][int(n_nodes)].append(float(md["running_time"]))
    return times, files


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results/test_results")
    parser.add_argument("--synthetic_type", default="neighbor_feature",
                        choices=["neighbor_feature", "neighbor_only", "all"],
                        help="Which synthetic family to plot ('all' pools both)")
    parser.add_argument("--nodes", type=int, nargs="*", default=None,
                        help="Restrict to these node counts (default: all found)")
    parser.add_argument("--out", default="figures/scalability.png")
    parser.add_argument("--show", action="store_true", help="Also open a window")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    synthetic_type = None if args.synthetic_type == "all" else args.synthetic_type
    times, files = collect(args.results_dir, synthetic_type)

    if not files:
        print(f"No JSON files found in {args.results_dir}.")
        print("Generate them first:")
        print("  bash scripts/gen_scalability_commands.sh")
        print("  # then execute the commands in all_commands_runtime.txt")
        return 1
    if not times:
        print(f"No runs matched synthetic_type={args.synthetic_type}.")
        return 1

    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 17,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    all_node_counts = set()

    for method in METHOD_ORDER:
        if method not in times:
            continue
        node_counts = sorted(times[method])
        if args.nodes:
            node_counts = [n for n in node_counts if n in set(args.nodes)]
        if not node_counts:
            continue
        all_node_counts.update(node_counts)
        means = [float(np.mean(times[method][n])) for n in node_counts]
        style = METHOD_STYLE[method]
        ax.plot(node_counts, means, style["marker"] + "-",
                color=style["color"], linewidth=2.5, markersize=9,
                label=style["label"], zorder=3)

    if not all_node_counts:
        print("Nothing left to plot after filtering by --nodes.")
        return 1

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(sorted(all_node_counts))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.get_xaxis().set_tick_params(which="minor", size=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    ax.set_xlabel(r"Number of Nodes ($\log_2$ scale)")
    ax.set_ylabel(r"Runtime in seconds ($\log_{10}$ scale)")
    ax.set_title("Runtime Scalability Comparison", pad=12)
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.2, pad=6)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Figure written to {args.out}")

    # Also print the underlying numbers so they can be checked against Figure 3.
    print("\nMean runtime (seconds):")
    print(f'{"nodes":>8}  ' + "  ".join(f"{METHOD_STYLE[m]['label']:>18}" for m in METHOD_ORDER))
    for n in sorted(all_node_counts):
        cells = []
        for m in METHOD_ORDER:
            vals = times.get(m, {}).get(n)
            cells.append(f"{np.mean(vals):>18.3f}" if vals else f"{'-':>18}")
        print(f"{n:>8}  " + "  ".join(cells))

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
