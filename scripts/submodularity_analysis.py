#!/usr/bin/env python
"""
Empirical validation of relaxed additivity and approximate submodularity
(Section 3.3 / Figure 2 of the paper).

The objective is the cumulative risk reduction achieved by a set of feature
interventions applied to a target node v:

    f(S) = p(v | G) - p(v | G with S applied),      f(empty set) = 0

where p is the trained GCN's predicted risk. Over many random trials we sample
two disjoint intervention sets A and B (each of cardinality < k) at a randomly
chosen correctly-predicted at-risk node and record two diagnostics:

    Additivity ratio   r_add = f(A u B) / (f(A) + f(B))
    Modularity ratio   r_mod = sum_{x in B} [f(A u {x}) - f(A)] / [f(A u B) - f(A)]

r_add ~ 1 means interventions act roughly independently (relaxed additivity with
a small slack alpha). r_mod <= 1 means diminishing returns (submodularity);
r_mod ~ 1 means near-modularity. Trials whose denominator is exactly zero are
recorded as a ratio of 1.0.

The figure in the paper was produced on the military network, which cannot be
released (IRB). This script runs the identical analysis on the synthetic
graphs, which are regenerable from source.

Usage (from the repository root):
    python scripts/submodularity_analysis.py
    python scripts/submodularity_analysis.py --synthetic_type neighbor_only \
        --m_attrs 6 --n_trials 5000 --out figures/submodularity.png
"""

import argparse
import os
import random
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gen_graph_data import generate_graph_withAvgDegree, stratified_split_data  # noqa: E402
from src.model import GCN_Mulin, train_with_early_stopping  # noqa: E402
from src.intervention_design_model import apply_interventions  # noqa: E402


def risk(model, data, node_idx, interventions, device):
    """Predicted risk p(v) after applying ``interventions`` to node ``node_idx``."""
    modified = apply_interventions(data, node_idx, interventions).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(modified.x, modified.edge_index)
    return float(pred[node_idx].item())


def one_trial(model, data, target_nodes, k, device, rng):
    """Return (r_add, r_mod) for one random (node, A, B) draw, or None if degenerate."""
    node_idx = rng.choice(target_nodes)
    num_features = data.x.shape[1]

    max_a = min(k - 1, num_features // 2)
    if max_a < 1:
        return None
    a_size = rng.randint(1, max_a)
    A = set(rng.sample(range(num_features), a_size))
    rest = sorted(set(range(num_features)) - A)
    max_b = min(k - 1, len(rest))
    if max_b < 1:
        return None
    b_size = rng.randint(1, max_b)
    B = set(rng.sample(rest, b_size))

    # f(S) = baseline risk - risk after applying S (all sampled features -> 0)
    p0 = risk(model, data, node_idx, set(), device)

    def f(feature_set):
        interventions = {('feature', int(i), 0) for i in feature_set}
        return p0 - risk(model, data, node_idx, interventions, device)

    f_A = f(A)
    f_B = f(B)
    f_AB = f(A | B)

    denom_add = f_A + f_B
    r_add = 1.0 if denom_add == 0 else f_AB / denom_add

    sum_marginals = sum(f(A | {x}) - f_A for x in B)
    denom_mod = f_AB - f_A
    r_mod = 1.0 if denom_mod == 0 else sum_marginals / denom_mod

    return r_add, r_mod


def train_backbone(data, args, device):
    data = stratified_split_data(data, num_val=args.val_ratio, num_test=args.test_ratio,
                                 is_undirected=True, seed=args.seed)
    data = data.to(device)
    model = GCN_Mulin(in_features=data.x.shape[1],
                      hidden_dim=args.hidden_dim,
                      dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.BCELoss()

    if hasattr(data, 'val_mask') and data.val_mask.sum() > 0:
        positive_indices_val = torch.where(data.val_mask & (data.y == 1))[0]
    else:
        positive_indices_val = None

    model, *_ = train_with_early_stopping(
        model, data, optimizer, criterion,
        max_epochs=args.max_epochs, patience=args.patience,
        check_every=args.check_every,
        positive_indices_val=positive_indices_val, device=device,
    )
    return model, data


def plot(r_add, r_mod, out_path, bins, show):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, title, xlabel in (
        (axes[0], r_add, "Distribution of Additivity Ratios", "Additivity Ratio"),
        (axes[1], r_mod, "Distribution of Modularity Ratios", "Modularity Ratio"),
    ):
        clipped = np.clip(values, 0.0, 2.0)
        weights = np.ones_like(clipped) / len(clipped)
        ax.hist(clipped, bins=bins, range=(0.0, 2.0), weights=weights, color="#3b7fb5")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction")
        ax.set_xlim(0.0, 2.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Figure written to {out_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Graph
    parser.add_argument("--synthetic_type", default="neighbor_feature",
                        choices=["neighbor_feature", "neighbor_only"])
    parser.add_argument("--n_nodes", type=int, default=250)
    parser.add_argument("--m_attrs", type=int, default=10)
    parser.add_argument("--total_edges", type=int, default=300)
    parser.add_argument("--label_percent", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--check_every", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    # Analysis
    parser.add_argument("--k", type=int, default=5,
                        help="Maximum cardinality of the sampled sets A and B")
    parser.add_argument("--n_trials", type=int, default=5000)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--out", default="figures/submodularity.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    data, G, _, _, _ = generate_graph_withAvgDegree(
        n_nodes=args.n_nodes,
        m_attrs=args.m_attrs,
        total_edges=args.total_edges,
        label_percent=args.label_percent,
        seed=args.seed,
        keep_features=(args.synthetic_type == "neighbor_feature"),
    )
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{data.x.shape[1]} features, {int(data.y.sum())} positives")

    model, data = train_backbone(data, args, device)

    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
    y_pred = (preds > 0.5).long()
    target_nodes = [i for i in range(data.num_nodes)
                    if data.y[i] == 1 and y_pred[i] == 1]
    if not target_nodes:
        print("No correctly predicted at-risk nodes; nothing to analyze.")
        return 1
    print(f"Correctly predicted at-risk nodes: {len(target_nodes)}")

    r_add, r_mod = [], []
    for t in range(args.n_trials):
        result = one_trial(model, data, target_nodes, args.k, device, rng)
        if result is None:
            continue
        r_add.append(result[0])
        r_mod.append(result[1])
        if (t + 1) % 500 == 0:
            print(f"  {t + 1}/{args.n_trials} trials")

    r_add = np.asarray(r_add)
    r_mod = np.asarray(r_mod)
    print(f"\nCompleted {len(r_add)} trials.")
    for name, arr in (("Additivity", r_add), ("Modularity", r_mod)):
        print(f"  {name:<11} mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"in [0.9, 1.1]={np.mean((arr >= 0.9) & (arr <= 1.1)) * 100:.2f}%  "
              f"<= 1: {np.mean(arr <= 1.0) * 100:.2f}%")

    plot(r_add, r_mod, args.out, args.bins, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
