import json
import time
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from torch_geometric.utils import to_undirected
from data.gen_graph_data import split_data, stratified_split_data, generate_graph
from src.model import (
    GCN_Mulin, GCN_Mulin_Perturb, CFExplainer,
    train_with_early_stopping, evaluate,
    generate_cff_explanations
)
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from src.intervention_design_model import (
    generate_counterfactuals_batch,
    greedy_intervention_selection,
    frequency_intervention_selection,
    random_intervention_selection
)
from src.intervention_design_model_cff import generate_counterfactuals_batch_cff
from src.intervention_design_model_cf import convert_cf_to_counterfactuals, generate_cf_explanations


def run_ALL(
    data,
    # Data options
    feature_names=None,
    pca_components=None,           # Optional PCA for real data
    feature_partitions=None,       # For cf-greedy method
    unchangeable_indices=None,     # For cf-greedy method (real data)
    forbidden_indices=None,        # For cf-greedy method (real data)
    # Experiment options
    N=10,
    budget=5.0,
    epsilon=0.1,
    output_file='experiment_results.json',
    # Data split options
    val_ratio=0.0,                 # Validation ratio (0 for synthetic)
    test_ratio=0.0,                # Test ratio (0 for synthetic)
    # Model options
    hidden_dim=32,
    dropout=0.5,
    lr=0.01,
    weight_decay=1e-5,
    max_epochs=2000,
    # Early stopping parameters
    patience=5,
    check_every=50,
    # Explanation method: "cf-greedy", "cf", "cff", or list like ["cf", "cff"], or "all"
    explanation_method="all",
    # CF-Greedy specific options (L2G local counterfactual)
    cf_greedy_max_interventions=5,
    cf_greedy_edge_mode='edge_features', #'none' (ignore), 'edges' (add/remove edges), 'edge_features' (flip neighbor features)
    cf_greedy_only_delete_edges=False,
    # CF specific options (CF-GNNExplainer)
    cf_lr=0.1,
    cf_beta=0.1,
    cf_epochs=500,
    cf_n_hops=1,
    cf_diff_threshold=0.1,
    # CFF specific options
    cff_mode="edge+feature",       # "edge", "feature", or "edge+feature"
    cff_lr=0.01,
    cff_epochs=500,
    cff_lam=100,
    cff_alp=0.6,
    cff_gam=0.5,
    cff_feat_TH=0.5,               # Feature importance threshold (higher = fewer interventions)
    cff_edge_TH=0.5,               # Edge importance threshold (higher = fewer interventions)
    # Parallel options
    parallel=False,                 # Enable multiprocessing for CF and CFF
    num_workers=None,               # Number of parallel workers (default: cpu_count - 1)
    # Other options
    verbose=False,
    device='cpu'  # 'cpu', 'cuda', or 'cuda:0', etc.
):
    """
    Run multiple explanation methods in a single experiment (trains model once, runs all methods).

    This function supports running one, two, or all three explanation methods on the same
    trained model for fair comparison. Results are stored per-method.

    Args:
        data: PyTorch Geometric Data object (required).

        Data options:
            feature_names: List of feature names. If None, auto-generated as F0, F1, ...
            pca_components: Optional PCA components for dimensionality reduction (for real data).
                           Shape: (n_components, n_features). Model will use built-in PCA layer.
            feature_partitions: Feature partition indices for cf-greedy method.
            unchangeable_indices: Indices of unchangeable features (for real data constraints).
            forbidden_indices: Indices of forbidden target features (for real data constraints).

        Experiment options:
            N: Number of experiment runs.
            budget: Budget for intervention selection.
            epsilon: Epsilon for greedy selection.
            output_file: Path to save JSON results.

        Data split options:
            val_ratio: Validation set ratio (default 0 for synthetic, use 0.2 for real).
            test_ratio: Test set ratio (default 0 for synthetic, use 0.3 for real).

        Model options:
            hidden_dim: Hidden dimension for GCN layers.
            dropout: Dropout rate.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            max_epochs: Maximum training epochs.
            patience: Early stopping patience (number of checks without improvement).
            check_every: Number of epochs between early stopping checks.

        Explanation method:
            explanation_method: Can be:
                - Single method: "cf-greedy", "cf", or "cff"
                - Multiple methods: ["cf", "cff"] or ["cf-greedy", "cf", "cff"]
                - All methods: "all" (runs all three methods, default)

                Methods:
                - "cf-greedy": Local counterfactual explanation with greedy search (L2G)
                - "cf": CF-GNNExplainer (edge-based counterfactual)
                - "cff": CFF explainer (edge and/or feature importance)

        CF-Greedy specific options:
            cf_greedy_max_interventions: Max interventions per node.
            cf_greedy_edge_mode: Edge mode ('edge_features', etc.).
            cf_greedy_only_delete_edges: Only consider edge deletions.

        CF specific options (CF-GNNExplainer):
            cf_lr: Learning rate for CF optimization.
            cf_beta: Beta weight for graph distance loss.
            cf_epochs: Number of CF optimization epochs.
            cf_n_hops: Number of hops for neighborhood subgraph.
            cf_diff_threshold: Threshold for feature difference in counterfactual conversion.

        CFF specific options:
            cff_mode: "edge", "feature", or "edge+feature".
            cff_lr, cff_epochs, cff_lam, cff_alp, cff_gam: CFF hyperparameters.

        Parallel options:
            parallel: If True, use multiprocessing for CF and CFF methods (default False).
                      CF-Greedy is not parallelized (already fast).
                      Uses torch.multiprocessing with 'spawn' context (works on macOS + Linux).
            num_workers: Number of parallel workers (default: cpu_count - 1).

        Other options:
            verbose: Print detailed progress.

    Returns:
        results: Dictionary with all run results.

    Examples:
        # Synthetic data with CF-Greedy
        results = run_Mulin(
            data=synthetic_data,
            explanation_method="cf-greedy",
            N=10,
            budget=5.0
        )

        # Real data with CF-GNNExplainer and PCA
        results = run_Mulin(
            data=real_data,
            pca_components=pca.components_,
            explanation_method="cf",
            val_ratio=0.2,
            test_ratio=0.3,
            N=10,
            budget=8.0
        )

        # Real data with CFF and PCA
        results = run_Mulin(
            data=real_data,
            pca_components=pca.components_,
            explanation_method="cff",
            cff_mode="edge+feature",
            val_ratio=0.2,
            test_ratio=0.3,
            N=10,
            budget=8.0
        )
    """
    # Validate and normalize explanation method(s)
    valid_methods = ["cf-greedy", "cf", "cff"]

    if explanation_method == "all":
        methods_to_run = valid_methods.copy()
    elif isinstance(explanation_method, str):
        if explanation_method not in valid_methods:
            raise ValueError(f"explanation_method must be one of {valid_methods}, got '{explanation_method}'")
        methods_to_run = [explanation_method]
    elif isinstance(explanation_method, list):
        for m in explanation_method:
            if m not in valid_methods:
                raise ValueError(f"Invalid method '{m}'. Must be one of {valid_methods}")
        methods_to_run = explanation_method
    else:
        raise ValueError(f"explanation_method must be a string, list, or 'all', got {type(explanation_method)}")

    # Generate feature names if not provided
    if feature_names is None:
        if pca_components is not None:
            feature_names = [f"PC{i}" for i in range(pca_components.shape[0])]
        else:
            feature_names = [f"F{i}" for i in range(data.x.shape[1])]

    # Initialize results
    results = {
        "runs": [],
        "config": {
            "N": N,
            "budget": budget,
            "epsilon": epsilon,
            "explanation_methods": methods_to_run,
            "use_pca": pca_components is not None,
            "parallel": parallel,
            "num_workers": num_workers,
            "model_params": {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs
            }
        }
    }

    # Add method-specific config for all methods that will run
    if "cf-greedy" in methods_to_run:
        results["config"]["cf_greedy_params"] = {
            "max_interventions": cf_greedy_max_interventions,
            "edge_mode": cf_greedy_edge_mode,
            "only_delete_edges": cf_greedy_only_delete_edges
        }
    if "cf" in methods_to_run:
        results["config"]["cf_params"] = {
            "lr": cf_lr,
            "beta": cf_beta,
            "epochs": cf_epochs,
            "n_hops": cf_n_hops,
            "diff_threshold": cf_diff_threshold
        }
    if "cff" in methods_to_run:
        results["config"]["cff_params"] = {
            "mode": cff_mode,
            "lr": cff_lr,
            "epochs": cff_epochs,
            "lam": cff_lam,
            "alp": cff_alp,
            "gam": cff_gam,
            "feat_TH": cff_feat_TH,
            "edge_TH": cff_edge_TH
        }

    # Print parallel info
    if parallel:
        import os as _os
        _nw = num_workers if num_workers else max(1, _os.cpu_count() - 1)
        print(f"Parallel mode enabled: {_nw} workers (CF and CFF only)")

    # Run experiments
    for run_id in range(N):
        print(f"\n{'='*80}")
        print(f"Run {run_id + 1}/{N} | Methods: {', '.join(m.upper() for m in methods_to_run)}")
        print(f"{'='*80}")

        # Set seed for reproducibility
        seed = 42 + run_id
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Setup device
        if device == 'cuda' and not torch.cuda.is_available():
            print("  Warning: CUDA not available, using CPU")
            current_device = torch.device('cpu')
        else:
            current_device = torch.device(device)

        if run_id == 0:
            print(f"  Using device: {current_device}")

        # Clone data for this run
        run_data = data.clone()

        # IMPORTANT: Convert to undirected graph for consistent GNN training and CF explanation
        # NetworkX graphs and edge list files often have single-direction edges
        if run_data.edge_index.shape[1] > 0:
            run_data.edge_index = to_undirected(run_data.edge_index)

        # Split data
        run_data = stratified_split_data(
            run_data,
            num_val=val_ratio,
            num_test=test_ratio,
            is_undirected=True,
            seed=seed if (val_ratio > 0 or test_ratio > 0) else None
        )

        # Prepare data and move to device
        run_data = run_data.to(current_device)
        features = run_data.x.float()
        labels = run_data.y.long()
        edge_index = run_data.edge_index
        idx_train = torch.where(run_data.train_mask)[0]

        # Initialize model and move to device
        model = GCN_Mulin(
            in_features=features.shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            pca_components=pca_components
        ).to(current_device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        # Training with early stopping
        if verbose:
            print(f"\nTraining GCN_Mulin with early stopping...")

        # Identify positive indices in validation set for early stopping
        if hasattr(run_data, 'val_mask') and run_data.val_mask.sum() > 0:
            positive_indices_val = torch.where((run_data.val_mask) & (run_data.y == 1))[0]
        else:
            positive_indices_val = None

        # Train model with early stopping
        model, losses, train_accs, val_accs, final_pos_val_acc, test_acc = train_with_early_stopping(
            model, run_data, optimizer, criterion,
            max_epochs=max_epochs,
            patience=patience,
            check_every=check_every,
            positive_indices_val=positive_indices_val,
            device=current_device
        )

        # Evaluate model
        model.eval()
        with torch.no_grad():
            out = model(features, edge_index)
            y_pred = (out > 0.5).long()

        final_acc = (y_pred == labels).float().mean().item()
        pos_acc = (y_pred[labels == 1] == 1).float().mean().item() if (labels == 1).sum() > 0 else 0.0
        print(f"  Final Acc: {final_acc:.4f} | Positive Acc: {pos_acc:.4f}")

        # Get target nodes (true positives)
        idx_train_positive = idx_train[(labels[idx_train] == 1)]
        positive_label_indices = [i.item() for i in idx_train_positive]  # All label=1 nodes
        target_nodes = [i for i in positive_label_indices if y_pred[i] == 1]  # True positives only

        if len(target_nodes) == 0:
            print(f"  No target nodes, skipping run {run_id + 1}")
            continue

        print(f"  Target nodes: {len(target_nodes)}")

        # Store results for each method
        method_results = {}

        # Run each explanation method
        for current_method in methods_to_run:
            print(f"\n--- Method: {current_method.upper()} ---")

            # Generate explanations based on method
            if current_method == "cf-greedy":
                # CF-Greedy: Local counterfactual explanation with greedy search
                print(f"Generating CF-Greedy counterfactuals...")
                start_time = time.time()
                counterfactuals = generate_counterfactuals_batch(
                    data=run_data,
                    positive_node_indices=target_nodes,
                    model=model,
                    max_interventions=cf_greedy_max_interventions,
                    edge_mode=cf_greedy_edge_mode,
                    feature_partitions=feature_partitions,
                    unchangeable_features=unchangeable_indices,
                    forbidden_target_features=forbidden_indices,
                    flip_direction='positive_to_negative',
                    device=current_device,
                    verbose=False,
                    only_delete_edges=cf_greedy_only_delete_edges
                )
                running_time = time.time() - start_time
                positive_node_indices_method = target_nodes
                explanation_stats = {
                    "num_counterfactuals": len([cf for cf in counterfactuals if len(cf) > 0])
                }

            elif current_method == "cf":
                # CF: CF-GNNExplainer (edge-based counterfactual)
                print(f"Generating CF-GNNExplainer explanations{'  (parallel)' if parallel else ''}...")
                start_time = time.time()

                if parallel:
                    from src.parallel_utils import generate_cf_explanations_parallel
                    cf_results, cf_serial_time = generate_cf_explanations_parallel(
                        model=model,
                        features=features,
                        edge_index=edge_index,
                        labels=labels,
                        target_indices=target_nodes,
                        y_pred=y_pred,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        pca_components=pca_components,
                        cf_lr=cf_lr,
                        cf_beta=cf_beta,
                        cf_epochs=cf_epochs,
                        cf_n_hops=cf_n_hops,
                        num_workers=num_workers,
                        verbose=verbose
                    )
                else:
                    # Use encapsulated CF explanation function
                    cf_results = generate_cf_explanations(
                        model=model,
                        features=features,
                        edge_index=edge_index,
                        labels=labels,
                        target_indices=target_nodes,
                        y_pred=y_pred,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        pca_components=pca_components,
                        cf_lr=cf_lr,
                        cf_beta=cf_beta,
                        cf_epochs=cf_epochs,
                        cf_n_hops=cf_n_hops,
                        verbose=verbose
                    )

                # Convert CF results to counterfactuals
                print(f"Converting CF to counterfactuals...")
                convert_start = time.time()
                counterfactuals = convert_cf_to_counterfactuals(cf_results, run_data, cf_diff_threshold)
                convert_time = time.time() - convert_start

                if parallel:
                    # Use sum of per-node times + conversion = equivalent serial time
                    running_time = cf_serial_time + convert_time
                else:
                    running_time = time.time() - start_time
                positive_node_indices_method = [r['node_idx'] for r in cf_results]

                cf_success_rate = sum(1 for r in cf_results if r['cf_found']) / len(cf_results) if cf_results else 0
                explanation_stats = {
                    "cf_success_rate": cf_success_rate,
                    "num_explanations": len(cf_results)
                }

            else:  # cff
                # CFF: Counterfactual and Factual reasoning
                print(f"Generating CFF explanations (mode: {cff_mode}){'  (parallel)' if parallel else ''}...")
                start_time = time.time()

                if parallel:
                    from src.parallel_utils import generate_cff_explanations_parallel
                    cff_results, cff_serial_time = generate_cff_explanations_parallel(
                        base_model=model,
                        features=features,
                        edge_index=edge_index,
                        labels=labels,
                        target_indices=target_nodes,
                        mode=cff_mode,
                        lr=cff_lr,
                        epochs=cff_epochs,
                        lam=cff_lam,
                        alp=cff_alp,
                        gam=cff_gam,
                        num_workers=num_workers,
                        verbose=verbose,
                        unchangeable_indices=unchangeable_indices,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        pca_components=pca_components
                    )
                else:
                    cff_results = generate_cff_explanations(
                        base_model=model,
                        features=features,
                        edge_index=edge_index,
                        labels=labels,
                        target_indices=target_nodes,
                        mode=cff_mode,
                        lr=cff_lr,
                        epochs=cff_epochs,
                        lam=cff_lam,
                        alp=cff_alp,
                        gam=cff_gam,
                        verbose=verbose,
                        unchangeable_indices=unchangeable_indices
                    )

                # Convert CFF results to counterfactuals
                print(f"Converting CFF to counterfactuals...")
                convert_start = time.time()
                counterfactuals = generate_counterfactuals_batch_cff(
                    data=run_data,
                    cff_results=cff_results,
                    feat_TH=cff_feat_TH,
                    edge_TH=cff_edge_TH,
                    verbose=False,
                    unchangeable_indices=unchangeable_indices
                )
                convert_time = time.time() - convert_start

                if parallel:
                    # Use sum of per-node times + conversion = equivalent serial time
                    running_time = cff_serial_time + convert_time
                else:
                    running_time = time.time() - start_time
                positive_node_indices_method = [r['node_idx'] for r in cff_results]

                factual_rate = sum(1 for r in cff_results if r['factual_success']) / len(cff_results)
                cf_rate = sum(1 for r in cff_results if r['counterfactual_success']) / len(cff_results)
                explanation_stats = {
                    "factual_success_rate": factual_rate,
                    "counterfactual_success_rate": cf_rate,
                    "num_explanations": len(cff_results)
                }

            # Run intervention selection methods
            print(f"Running intervention selection...")

            # Greedy
            sel_g, cost_g, cov_g, cost_hist_g = greedy_intervention_selection(
                counterfactuals=counterfactuals,
                data=run_data,
                positive_node_indices=positive_node_indices_method,
                model=model,
                budget=budget,
                epsilon=epsilon,
                feature_names=feature_names,
                feature_partitions=feature_partitions,
                verbose=False,
                device=current_device
            )

            # Frequency
            sel_f, cost_f, cov_f, cost_hist_f = frequency_intervention_selection(
                counterfactuals=counterfactuals,
                data=run_data,
                positive_node_indices=positive_node_indices_method,
                model=model,
                budget=budget,
                epsilon=epsilon,
                feature_names=feature_names,
                verbose=False,
                device=current_device
            )

            # Random
            sel_r, cost_r, cov_r, cost_hist_r = random_intervention_selection(
                counterfactuals=counterfactuals,
                data=run_data,
                positive_node_indices=positive_node_indices_method,
                model=model,
                budget=budget,
                epsilon=epsilon,
                feature_names=feature_names,
                verbose=False,
                device=current_device
            )

            # Store results for this method
            method_results[current_method] = {
                "running_time": running_time,
                "explanation": explanation_stats,
                "greedy": {
                    "selected_clauses": [list(clause) for clause in sel_g],
                    "total_cost": cost_g,
                    "coverage_history": cov_g,
                    "cost_history": cost_hist_g,
                    "final_coverage": cov_g[-1] if cov_g else 0
                },
                "frequency": {
                    "selected_clauses": [list(clause) for clause in sel_f],
                    "total_cost": cost_f,
                    "coverage_history": cov_f,
                    "cost_history": cost_hist_f,
                    "final_coverage": cov_f[-1] if cov_f else 0
                },
                "random": {
                    "selected_clauses": [list(clause) for clause in sel_r],
                    "total_cost": cost_r,
                    "coverage_history": cov_r,
                    "cost_history": cost_hist_r,
                    "final_coverage": cov_r[-1] if cov_r else 0
                }
            }

            print(f"  {current_method}: Greedy={cov_g[-1] if cov_g else 0}, Freq={cov_f[-1] if cov_f else 0}, Rand={cov_r[-1] if cov_r else 0}")

        # Store run results
        run_result = {
            "run_id": run_id,
            "seed": seed,
            "num_target_nodes": len(target_nodes),
            "num_positive_labels": len(positive_label_indices),
            "positive_label_indices": positive_label_indices,
            "training": {
                "final_acc": final_acc,
                "pos_acc": pos_acc,
                "train_accs": [acc.item() if torch.is_tensor(acc) else acc for acc in train_accs],
                "val_accs": [acc.item() if torch.is_tensor(acc) else acc for acc in val_accs],
                "final_pos_val_acc": final_pos_val_acc.item() if torch.is_tensor(final_pos_val_acc) else final_pos_val_acc,
                "test_acc": test_acc
            },
            "methods": method_results
        }

        results["runs"].append(run_result)

        print(f"\nRun {run_id + 1} completed!")

    # Save results
    import os
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*80}")
    print(f"All {N} runs completed. Results saved to {output_file}")
    print(f"{'='*80}")

    return results




if __name__ == "__main__":
    # This can be run standalone if needed, but mainly called from notebook
    pass