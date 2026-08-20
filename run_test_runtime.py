#!/usr/bin/env python
"""
Runtime Test Script — untrained GNN, no training step.

Measures pure algorithm runtime for one or more explanation methods
(cf-greedy, cf, cff) using a randomly-initialized GCN_Mulin model.

This is what produces Figure 3 (runtime scalability). Training is skipped on
purpose so the measurement isolates explanation-generation cost.

Usage:
    python run_test_runtime.py --synthetic_type neighbor_feature \
        --n_nodes 1600 --total_edges 6400 --m_attrs 10 --methods cf-greedy cf cff

    # Parallel mode (CF and CF^2 only - CF-Greedy is already fast):
    python run_test_runtime.py --n_nodes 400 --total_edges 1600 \
        --methods cf cff --parallel --num_workers 30
"""

import argparse
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected

from src.model import GCN_Mulin, generate_cff_explanations
from src.intervention_design_model import (
    generate_counterfactuals_batch,
    greedy_intervention_selection,
    frequency_intervention_selection,
    random_intervention_selection,
)
from src.intervention_design_model_cff import generate_counterfactuals_batch_cff
from src.intervention_design_model_cf import convert_cf_to_counterfactuals, generate_cf_explanations
from data.gen_graph_data import stratified_split_data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime test with untrained GNN — no training step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--methods", type=str, nargs="+",
                        default=["cf-greedy", "cf", "cff"],
                        choices=["cf-greedy", "cf", "cff"],
                        help="Explanation method(s) to benchmark")
    parser.add_argument("--seed", type=int, default=42)

    # ---- Synthetic data ----
    parser.add_argument("--n_nodes", type=int, default=250)
    parser.add_argument("--m_attrs", type=int, default=5)
    parser.add_argument("--total_edges", type=int, default=300)
    parser.add_argument("--synthetic_type", type=str, default="neighbor_feature",
                        choices=["neighbor_feature", "neighbor_only"])

    # ---- Model ----
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)

    # ---- Experiment ----
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # ---- CF-Greedy ----
    parser.add_argument("--cf_greedy_max_interventions", type=int, default=5)
    parser.add_argument("--cf_greedy_edge_mode", type=str, default="edge_features",
                        choices=["edge_features", "edges", "none"])
    parser.add_argument("--cf_greedy_only_delete_edges", action="store_true")

    # ---- CF ----
    parser.add_argument("--cf_lr", type=float, default=0.1)
    parser.add_argument("--cf_beta", type=float, default=1.0)
    parser.add_argument("--cf_epochs", type=int, default=500)
    parser.add_argument("--cf_n_hops", type=int, default=2)
    parser.add_argument("--cf_diff_threshold", type=float, default=0.1)

    # ---- CFF ----
    parser.add_argument("--cff_mode", type=str, default="edge+feature",
                        choices=["edge", "feature", "edge+feature"])
    parser.add_argument("--cff_lr", type=float, default=0.01)
    parser.add_argument("--cff_epochs", type=int, default=500)
    parser.add_argument("--cff_lam", type=float, default=100.0)
    parser.add_argument("--cff_alp", type=float, default=0.05)
    parser.add_argument("--cff_gam", type=float, default=0.1)
    parser.add_argument("--cff_feat_TH", type=float, default=0.7)
    parser.add_argument("--cff_edge_TH", type=float, default=0.7)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)

    # ---- Parallel ----
    parser.add_argument("--parallel", action="store_true",
                        help="Enable multiprocessing for CF and CFF (CF-Greedy unaffected)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (reuse same loaders as run.py)
# ---------------------------------------------------------------------------

def load_dataset(args):
    from data.gen_graph_data import generate_graph_withAvgDegree
    data, G, attributes, w, cutoff_score = generate_graph_withAvgDegree(
        n_nodes=args.n_nodes,
        m_attrs=args.m_attrs,
        total_edges=args.total_edges,
        seed=args.seed,
        keep_features=(args.synthetic_type == "neighbor_feature"),
    )
    feature_names = [f"F{i}" for i in range(data.x.shape[1])]
    print(f"Generated Synthetic ({args.synthetic_type}): nodes={G.number_of_nodes()}, "
          f"edges={G.number_of_edges()}, features={data.x.shape[1]}, "
          f"positives={data.y.sum().item()}")
    return data, None, None, feature_names, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    methods_to_run = args.methods

    if not methods_to_run:
        print("No valid methods to run. Exiting.")
        sys.exit(1)

    # Output file
    if args.output_file is None:
        import math
        budget_limit = args.budget * math.log(1.0 / args.epsilon)
        output_dir = "results/test_results"
        method_tag = "+".join(methods_to_run)
        filename = (f"runtime_{args.synthetic_type}_node{args.n_nodes}_attr{args.m_attrs}_"
                    f"edge{args.total_edges}_{method_tag}.json")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, filename)

    if args.parallel:
        import os as _os
        _nw = args.num_workers if args.num_workers else max(1, _os.cpu_count() - 1)
        print(f"Parallel mode enabled: {_nw} workers (CF and CFF only)")

    print(f"Methods: {methods_to_run}")
    print(f"Output:  {args.output_file}")

    # Load data
    data, pca_components, feature_partitions, feature_names, unchangeable_indices, forbidden_indices = \
        load_dataset(args)

    # Prepare device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        current_device = torch.device("cpu")
    else:
        current_device = torch.device(args.device)

    # Prepare data (same as experiment_runner)
    if data.edge_index.shape[1] > 0:
        data.edge_index = to_undirected(data.edge_index)

    data = stratified_split_data(
        data,
        num_val=args.val_ratio,
        num_test=args.test_ratio,
        is_undirected=True,
        seed=args.seed if (args.val_ratio > 0 or args.test_ratio > 0) else None,
    )
    data = data.to(current_device)

    features = data.x.float()
    labels = data.y.long()
    edge_index = data.edge_index
    idx_train = torch.where(data.train_mask)[0]

    # Build untrained model (random weights — no training)
    model = GCN_Mulin(
        in_features=features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pca_components=pca_components,
    ).to(current_device)
    model.eval()
    print(f"Using randomly-initialized (untrained) GCN_Mulin on {current_device}")

    # Use all positive-labeled training nodes as target nodes
    # (skip y_pred filter — untrained model predictions are meaningless)
    idx_train_positive = idx_train[(labels[idx_train] == 1)]
    target_nodes = [i.item() for i in idx_train_positive]
    print(f"Target nodes (positive-labeled, train set): {len(target_nodes)}")

    if len(target_nodes) == 0:
        print("No target nodes found. Exiting.")
        sys.exit(1)

    # Placeholder y_pred (all 1s for positive nodes) so CF functions don't crash
    with torch.no_grad():
        out = model(features, edge_index)
        y_pred = (out > 0.5).long()

    # Run each method
    method_results = {}

    for current_method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Method: {current_method.upper()}")
        print(f"{'='*60}")

        if current_method == "cf-greedy":
            start_time = time.time()
            counterfactuals = generate_counterfactuals_batch(
                data=data,
                positive_node_indices=target_nodes,
                model=model,
                max_interventions=args.cf_greedy_max_interventions,
                edge_mode=args.cf_greedy_edge_mode,
                feature_partitions=feature_partitions,
                unchangeable_features=unchangeable_indices,
                forbidden_target_features=forbidden_indices,
                flip_direction="positive_to_negative",
                device=current_device,
                verbose=args.verbose,
                only_delete_edges=args.cf_greedy_only_delete_edges,
            )
            running_time = time.time() - start_time
            positive_node_indices_method = target_nodes
            explanation_stats = {
                "num_counterfactuals": len([cf for cf in counterfactuals if len(cf) > 0])
            }

        elif current_method == "cf":
            start_time = time.time()
            if args.parallel:
                from src.parallel_utils import generate_cf_explanations_parallel
                cf_results, cf_serial_time = generate_cf_explanations_parallel(
                    model=model,
                    features=features,
                    edge_index=edge_index,
                    labels=labels,
                    target_indices=target_nodes,
                    y_pred=y_pred,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    pca_components=pca_components,
                    cf_lr=args.cf_lr,
                    cf_beta=args.cf_beta,
                    cf_epochs=args.cf_epochs,
                    cf_n_hops=args.cf_n_hops,
                    num_workers=args.num_workers,
                    verbose=args.verbose,
                )
            else:
                cf_results = generate_cf_explanations(
                    model=model,
                    features=features,
                    edge_index=edge_index,
                    labels=labels,
                    target_indices=target_nodes,
                    y_pred=y_pred,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    pca_components=pca_components,
                    cf_lr=args.cf_lr,
                    cf_beta=args.cf_beta,
                    cf_epochs=args.cf_epochs,
                    cf_n_hops=args.cf_n_hops,
                    verbose=args.verbose,
                )
            convert_start = time.time()
            counterfactuals = convert_cf_to_counterfactuals(cf_results, data, args.cf_diff_threshold)
            convert_time = time.time() - convert_start
            if args.parallel:
                running_time = cf_serial_time + convert_time
            else:
                running_time = time.time() - start_time
            positive_node_indices_method = [r["node_idx"] for r in cf_results]
            cf_success_rate = (sum(1 for r in cf_results if r["cf_found"]) / len(cf_results)
                               if cf_results else 0)
            explanation_stats = {
                "cf_success_rate": cf_success_rate,
                "num_explanations": len(cf_results),
            }

        else:  # cff
            start_time = time.time()
            if args.parallel:
                from src.parallel_utils import generate_cff_explanations_parallel
                cff_results, cff_serial_time = generate_cff_explanations_parallel(
                    base_model=model,
                    features=features,
                    edge_index=edge_index,
                    labels=labels,
                    target_indices=target_nodes,
                    mode=args.cff_mode,
                    lr=args.cff_lr,
                    epochs=args.cff_epochs,
                    lam=args.cff_lam,
                    alp=args.cff_alp,
                    gam=args.cff_gam,
                    num_workers=args.num_workers,
                    verbose=args.verbose,
                    unchangeable_indices=unchangeable_indices,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    pca_components=pca_components,
                )
            else:
                cff_results = generate_cff_explanations(
                    base_model=model,
                    features=features,
                    edge_index=edge_index,
                    labels=labels,
                    target_indices=target_nodes,
                    mode=args.cff_mode,
                    lr=args.cff_lr,
                    epochs=args.cff_epochs,
                    lam=args.cff_lam,
                    alp=args.cff_alp,
                    gam=args.cff_gam,
                    verbose=args.verbose,
                    unchangeable_indices=unchangeable_indices,
                )
            convert_start = time.time()
            counterfactuals = generate_counterfactuals_batch_cff(
                data=data,
                cff_results=cff_results,
                feat_TH=args.cff_feat_TH,
                edge_TH=args.cff_edge_TH,
                verbose=False,
                unchangeable_indices=unchangeable_indices,
            )
            convert_time = time.time() - convert_start
            if args.parallel:
                running_time = cff_serial_time + convert_time
            else:
                running_time = time.time() - start_time
            positive_node_indices_method = [r["node_idx"] for r in cff_results]
            factual_rate = sum(1 for r in cff_results if r["factual_success"]) / len(cff_results)
            cf_rate = sum(1 for r in cff_results if r["counterfactual_success"]) / len(cff_results)
            explanation_stats = {
                "factual_success_rate": factual_rate,
                "counterfactual_success_rate": cf_rate,
                "num_explanations": len(cff_results),
            }

        print(f"  Explanation time: {running_time:.2f}s")

        # Intervention selection
        print("  Running intervention selection...")
        sel_g, cost_g, cov_g, cost_hist_g = greedy_intervention_selection(
            counterfactuals=counterfactuals,
            data=data,
            positive_node_indices=positive_node_indices_method,
            model=model,
            budget=args.budget,
            epsilon=args.epsilon,
            feature_names=feature_names,
            feature_partitions=feature_partitions,
            verbose=False,
            device=current_device,
        )
        sel_f, cost_f, cov_f, cost_hist_f = frequency_intervention_selection(
            counterfactuals=counterfactuals,
            data=data,
            positive_node_indices=positive_node_indices_method,
            model=model,
            budget=args.budget,
            epsilon=args.epsilon,
            feature_names=feature_names,
            verbose=False,
            device=current_device,
        )
        sel_r, cost_r, cov_r, cost_hist_r = random_intervention_selection(
            counterfactuals=counterfactuals,
            data=data,
            positive_node_indices=positive_node_indices_method,
            model=model,
            budget=args.budget,
            epsilon=args.epsilon,
            feature_names=feature_names,
            verbose=False,
            device=current_device,
        )

        method_results[current_method] = {
            "running_time": running_time,
            "explanation": explanation_stats,
            "greedy": {
                "selected_clauses": [list(c) for c in sel_g],
                "total_cost": cost_g,
                "coverage_history": cov_g,
                "cost_history": cost_hist_g,
                "final_coverage": cov_g[-1] if cov_g else 0,
            },
            "frequency": {
                "selected_clauses": [list(c) for c in sel_f],
                "total_cost": cost_f,
                "coverage_history": cov_f,
                "cost_history": cost_hist_f,
                "final_coverage": cov_f[-1] if cov_f else 0,
            },
            "random": {
                "selected_clauses": [list(c) for c in sel_r],
                "total_cost": cost_r,
                "coverage_history": cov_r,
                "cost_history": cost_hist_r,
                "final_coverage": cov_r[-1] if cov_r else 0,
            },
        }
        print(f"  Greedy coverage={cov_g[-1] if cov_g else 0:.3f} | "
              f"Freq={cov_f[-1] if cov_f else 0:.3f} | "
              f"Rand={cov_r[-1] if cov_r else 0:.3f}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Runtime Summary (untrained model):")
    for m, r in method_results.items():
        print(f"  {m.upper():12s}: {r['running_time']:.2f}s")
    print(f"{'='*60}")

    # ---- Save results ----
    output = {
        "note": "Untrained (random) GNN — runtime test only, results are not meaningful.",
        "config": {
            "synthetic_type": args.synthetic_type,
            "n_nodes": args.n_nodes,
            "m_attrs": args.m_attrs,
            "total_edges": args.total_edges,
            "methods": methods_to_run,
            "num_target_nodes": len(target_nodes),
            "parallel": args.parallel,
            "num_workers": args.num_workers,
            "model": {"hidden_dim": args.hidden_dim, "dropout": args.dropout},
            "budget": args.budget,
            "epsilon": args.epsilon,
            "cf_greedy_params": {
                "max_interventions": args.cf_greedy_max_interventions,
                "edge_mode": args.cf_greedy_edge_mode,
            },
            "cf_params": {
                "lr": args.cf_lr, "beta": args.cf_beta,
                "epochs": args.cf_epochs, "n_hops": args.cf_n_hops,
            },
            "cff_params": {
                "mode": args.cff_mode, "lr": args.cff_lr,
                "epochs": args.cff_epochs, "lam": args.cff_lam,
            },
        },
        "methods": method_results,
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
