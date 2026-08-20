#!/usr/bin/env python
"""
Main Runner Script for G2I Intervention-Design Experiments

Runs the full pipeline (train GCN -> generate local counterfactuals with
CF-Greedy / CF / CF^2 -> select a DNF intervention policy) on synthetic graphs
and writes a JSON result file that `scripts/make_table3.py` aggregates into
Table 3 of the paper.

Two synthetic families are supported (see Section 4.1.2 of the paper):
    neighbor_feature  diathesis-stress labels (intrinsic features x neighborhood)
    neighbor_only     labels determined by neighborhood composition alone

Usage:
    python run.py --synthetic_type neighbor_feature --n_nodes 250 \
        --total_edges 300 --m_attrs 10 --N 10
    python run.py --synthetic_type neighbor_only --n_nodes 500 \
        --total_edges 3000 --m_attrs 6 --N 10
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from src.experiment_runner import run_ALL


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run G2I intervention-design experiments on synthetic graphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========================
    # Synthetic data parameters
    # ========================
    synthetic_group = parser.add_argument_group('Synthetic Data Parameters')
    synthetic_group.add_argument('--n_nodes', type=int, default=250,
                                 help='Number of nodes in synthetic graph')
    synthetic_group.add_argument('--m_attrs', type=int, default=5,
                                 help='Number of attributes/features')
    synthetic_group.add_argument('--alpha', type=float, default=5.0,
                                 help='Alpha parameter for score calculation')
    synthetic_group.add_argument('--beta', type=float, default=0.0,
                                 help='Beta parameter for neighbor influence')
    synthetic_group.add_argument('--total_edges', type=int, default=300,
                                 help='Total number of edges in graph')
    synthetic_group.add_argument('--label_percent', type=float, default=0.17,
                                 help='Percentage of positive labels')
    synthetic_group.add_argument('--sbm_fraction', type=float, default=0.6,
                                 help='Fraction of edges from SBM structure')
    synthetic_group.add_argument('--n_comms', type=int, default=6,
                                 help='Number of communities in SBM')
    synthetic_group.add_argument('--p_intra', type=float, default=0.08,
                                 help='Intra-community edge probability')
    synthetic_group.add_argument('--p_inter', type=float, default=0.006,
                                 help='Inter-community edge probability')
    synthetic_group.add_argument('--synthetic_seed', type=int, default=42,
                                 help='Random seed for synthetic data generation')
    synthetic_group.add_argument('--synthetic_type', type=str, default='neighbor_feature',
                                 choices=['neighbor_feature', 'neighbor_only'],
                                 help="Synthetic family. 'neighbor_feature' uses the "
                                      "diathesis-stress label rule (node attributes AND "
                                      "neighborhood); 'neighbor_only' derives labels from "
                                      "the neighborhood alone.")

    # ========================
    # Experiment parameters
    # ========================
    exp_group = parser.add_argument_group('Experiment Parameters')
    exp_group.add_argument('--N', type=int, default=1,
                           help='Number of experiment runs')
    exp_group.add_argument('--budget', type=float, default=10.0,
                           help='Budget for intervention selection')
    exp_group.add_argument('--epsilon', type=float, default=0.1,
                           help='Epsilon for budget calculation')
    exp_group.add_argument('--output_file', type=str, default=None,
                           help='Output JSON file path (auto-generated if not specified)')
    exp_group.add_argument('--val_ratio', type=float, default=0.0,
                           help='Validation set ratio')
    exp_group.add_argument('--test_ratio', type=float, default=0.1,
                           help='Test set ratio')

    # ========================
    # Model parameters
    # ========================
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--hidden_dim', type=int, default=32,
                             help='Hidden dimension of GCN')
    model_group.add_argument('--dropout', type=float, default=0.5,
                             help='Dropout rate')
    model_group.add_argument('--lr', type=float, default=0.01,
                             help='Learning rate')
    model_group.add_argument('--weight_decay', type=float, default=5e-4,
                             help='Weight decay')
    model_group.add_argument('--max_epochs', type=int, default=2000,
                             help='Maximum training epochs')
    model_group.add_argument('--patience', type=int, default=5,
                             help='Early stopping patience')
    model_group.add_argument('--check_every', type=int, default=50,
                             help='Check validation every N epochs')

    # ========================
    # Explanation method selection
    # ========================
    method_group = parser.add_argument_group('Explanation Methods')
    method_group.add_argument('--explanation_method', type=str, nargs='+',
                              default=['all'],
                              help='Explanation method(s): cf-greedy, cf, cff, or all')

    # ========================
    # CF-Greedy parameters
    # ========================
    cf_greedy_group = parser.add_argument_group('CF-Greedy Parameters')
    cf_greedy_group.add_argument('--cf_greedy_max_interventions', type=int, default=5,
                                 help='Maximum interventions per counterfactual')
    cf_greedy_group.add_argument('--cf_greedy_edge_mode', type=str, default='edge_features',
                                 choices=['edge_features', 'edges', 'none'],
                                 help="Intervention space for CF-Greedy: 'none' = target-node "
                                      "features only, 'edges' = also add/remove edges, "
                                      "'edge_features' = also perturb neighbor features "
                                      "(the setting used for the paper's Table 3)")
    cf_greedy_group.add_argument('--cf_greedy_only_delete_edges', action='store_true',
                                 help='Only delete edges (no additions)')

    # ========================
    # CF-GNNExplainer parameters
    # ========================
    cf_group = parser.add_argument_group('CF-GNNExplainer Parameters')
    cf_group.add_argument('--cf_lr', type=float, default=0.1,
                          help='Learning rate for CF optimization')
    cf_group.add_argument('--cf_beta', type=float, default=1.0,
                          help='Beta weight for graph distance loss')
    cf_group.add_argument('--cf_epochs', type=int, default=500,
                          help='Number of CF optimization epochs')
    cf_group.add_argument('--cf_n_hops', type=int, default=2,
                          help='Number of hops for neighborhood subgraph')
    cf_group.add_argument('--cf_diff_threshold', type=float, default=0.1,
                          help='Difference threshold for neighbor conditions')

    # ========================
    # CFF parameters
    # ========================
    cff_group = parser.add_argument_group('CFF Parameters')
    cff_group.add_argument('--cff_mode', type=str, default='edge+feature',
                           choices=['edge', 'feature', 'edge+feature'],
                           help='CFF explanation mode')
    cff_group.add_argument('--cff_lr', type=float, default=0.01,
                           help='Learning rate for CFF')
    cff_group.add_argument('--cff_epochs', type=int, default=500,
                           help='Number of CFF optimization epochs')
    cff_group.add_argument('--cff_lam', type=float, default=100.0,
                           help='Lambda for CFF loss')
    cff_group.add_argument('--cff_alp', type=float, default=0.05,
                           help='Alpha for CFF loss')
    cff_group.add_argument('--cff_gam', type=float, default=0.1,
                           help='Gamma for CFF loss')
    cff_group.add_argument('--cff_feat_TH', type=float, default=0.7,
                           help='Feature threshold for CFF')
    cff_group.add_argument('--cff_edge_TH', type=float, default=0.7,
                           help='Edge threshold for CFF')

    # ========================
    # Parallel options
    # ========================
    parser.add_argument('--parallel', action='store_true',
                        help='Enable multiprocessing for CF and CFF (CF-Greedy unaffected)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')

    # ========================
    # Other options
    # ========================
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu, cuda, cuda:0, cuda:1, etc.)')

    return parser.parse_args()


def load_dataset(args):
    """Generate the synthetic graph described by ``args``."""
    from data.gen_graph_data import generate_graph_withAvgDegree

    # The two synthetic families differ only in the label rule: the
    # Neighbor-Feature variant gates risk on the node's own attributes as well
    # as on its neighborhood, the Neighbor-Only variant uses the neighborhood
    # alone. `keep_features` is the switch inside the generator.
    keep_features = (args.synthetic_type == 'neighbor_feature')

    data, G, attributes, w, cutoff_score = generate_graph_withAvgDegree(
        n_nodes=args.n_nodes,
        m_attrs=args.m_attrs,
        alpha=args.alpha,
        beta=args.beta,
        total_edges=args.total_edges,
        label_percent=args.label_percent,
        sbm_fraction=args.sbm_fraction,
        n_comms=args.n_comms,
        p_intra=args.p_intra,
        p_inter=args.p_inter,
        seed=args.synthetic_seed,
        keep_features=keep_features
    )

    feature_names = [f"F{i}" for i in range(data.x.shape[1])]

    print(f"\nGenerated Synthetic Dataset ({args.synthetic_type}):")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Features: {data.x.shape[1]}")
    print(f"  Positive labels: {data.y.sum().item()}")

    # Synthetic data doesn't use PCA, feature_partitions, or unchangeable/forbidden indices
    return data, None, None, feature_names, None, None


def main():
    args = parse_args()

    # Set global random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    data, pca_components, feature_partitions, feature_names, unchangeable_indices, forbidden_indices = load_dataset(args)

    # Process explanation methods
    if args.explanation_method == ['all']:
        explanation_method = 'all'
    elif len(args.explanation_method) == 1:
        explanation_method = args.explanation_method[0]
    else:
        explanation_method = args.explanation_method

    # Generate output file path if not specified
    if args.output_file is None:
        import math
        budget_limit = args.budget * math.log(1.0 / args.epsilon)

        output_dir = f'results/synthetic_{args.synthetic_type}'
        filename = (f"node{args.n_nodes}_attr{args.m_attrs}_edge{args.total_edges}_"
                   f"N{args.N}_budgetlim{budget_limit:.1f}_"
                   f"cfgreedy{args.cf_greedy_edge_mode}_cff{args.cff_mode}_"
                   f"cfbeta{args.cf_beta}_cfflam{args.cff_lam}_"
                   f"cfffeatT{args.cff_feat_TH}_cffedgeT{args.cff_edge_TH}.json")

        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, filename)

    print(f"\nOutput file: {args.output_file}")
    print(f"Explanation methods: {explanation_method}")
    print(f"Runs: {args.N}, Budget: {args.budget}, Epsilon: {args.epsilon}")

    # Run experiments
    results = run_ALL(
        data=data,
        feature_names=feature_names,
        pca_components=pca_components,
        feature_partitions=feature_partitions,
        unchangeable_indices=unchangeable_indices,
        forbidden_indices=forbidden_indices,
        N=args.N,
        budget=args.budget,
        epsilon=args.epsilon,
        output_file=args.output_file,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        check_every=args.check_every,
        explanation_method=explanation_method,
        cf_greedy_max_interventions=args.cf_greedy_max_interventions,
        cf_greedy_edge_mode=args.cf_greedy_edge_mode,
        cf_greedy_only_delete_edges=args.cf_greedy_only_delete_edges,
        cf_lr=args.cf_lr,
        cf_beta=args.cf_beta,
        cf_epochs=args.cf_epochs,
        cf_n_hops=args.cf_n_hops,
        cf_diff_threshold=args.cf_diff_threshold,
        cff_mode=args.cff_mode,
        cff_lr=args.cff_lr,
        cff_epochs=args.cff_epochs,
        cff_lam=args.cff_lam,
        cff_alp=args.cff_alp,
        cff_gam=args.cff_gam,
        cff_feat_TH=args.cff_feat_TH,
        cff_edge_TH=args.cff_edge_TH,
        parallel=args.parallel,
        num_workers=args.num_workers,
        verbose=args.verbose,
        device=args.device
    )

    print(f"\nExperiment completed! Results saved to {args.output_file}")

    return results


if __name__ == "__main__":
    main()
