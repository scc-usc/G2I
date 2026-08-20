"""
Parallel execution utilities for CF and CFF explanation methods.
Uses torch.multiprocessing with 'spawn' context for cross-platform compatibility (macOS + Linux).

Usage:
    These functions are called from experiment_runner.py when parallel=True.
    They mirror the serial versions but distribute per-node work across multiple processes.
"""

import os
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import List, Dict, Optional, Tuple


# ============================================================
# CF-GNNExplainer Parallel
# ============================================================

_cf_state = {}


def _init_cf_worker(config):
    """Initialize CF worker process with model and shared data."""
    global _cf_state
    from src.model import GCN_Mulin

    model = GCN_Mulin(
        in_features=config['in_features'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        pca_components=config['pca_components']
    )
    model.load_state_dict(config['model_state_dict'])
    model.eval()

    _cf_state = {
        'model': model,
        'features': config['features'],
        'labels': config['labels'],
        'adj': config['adj'],
        'edge_index_tuple': config['edge_index_tuple'],
        'hidden_dim': config['hidden_dim'],
        'dropout': config['dropout'],
        'pca_components': config['pca_components'],
        'cf_lr': config['cf_lr'],
        'cf_beta': config['cf_beta'],
        'cf_epochs': config['cf_epochs'],
        'cf_n_hops': config['cf_n_hops'],
    }


def _cf_process_node(task):
    """Process a single node for CF-GNNExplainer (worker function)."""
    node_idx, y_pred_val = task
    t0 = time.time()

    from src.model import CFExplainer
    from src.utils import get_neighbourhood

    s = _cf_state
    model = s['model']
    features = s['features']
    labels = s['labels']

    orig_pred = int(y_pred_val)
    true_label = int(labels[node_idx].item())

    # Get neighborhood subgraph
    try:
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
            node_idx, s['edge_index_tuple'], s['cf_n_hops'], features, labels
        )
    except IndexError:
        return {
            'node_idx': node_idx, 'true_label': true_label,
            'orig_pred': orig_pred, 'cf_found': False,
            'removed_edges': [], 'edge_scores': {},
            'node_dict': {node_idx: 0},
            '_node_time': time.time() - t0
        }

    new_idx = node_dict[node_idx]

    if sub_adj.dim() < 2 or sub_adj.shape[0] < 2:
        return {
            'node_idx': node_idx, 'true_label': true_label,
            'orig_pred': orig_pred, 'cf_found': False,
            'removed_edges': [], 'edge_scores': {},
            'node_dict': node_dict,
            '_node_time': time.time() - t0
        }

    # Create explainer and run optimization
    explainer = CFExplainer(
        model=model, sub_adj=sub_adj, sub_feat=sub_feat,
        n_hid=s['hidden_dim'], dropout=s['dropout'],
        sub_labels=sub_labels, y_pred_orig=orig_pred,
        new_idx=new_idx, beta=s['cf_beta'],
        pca_components=s['pca_components']
    )

    cf_optimizer = optim.Adam(explainer.cf_model.parameters(), lr=s['cf_lr'])
    cf_examples = explainer.explain(
        cf_optimizer=cf_optimizer, cf_epochs=s['cf_epochs'], verbose=False
    )

    # Get edge importance scores
    edge_scores = {}
    reverse_node_dict = {v: k for k, v in node_dict.items()}
    try:
        P = torch.sigmoid(explainer.cf_model.P_vec).detach().view(sub_adj.shape[0], -1)
        P_sym = (P + P.T) / 2
        for i in range(sub_adj.shape[0]):
            for j in range(i + 1, sub_adj.shape[1]):
                if sub_adj[i, j] > 0:
                    u = reverse_node_dict.get(i)
                    v = reverse_node_dict.get(j)
                    if u is not None and v is not None:
                        score = P_sym[i, j].item()
                        edge_scores[(u, v)] = score
                        edge_scores[(v, u)] = score
    except Exception:
        pass

    # Convert to removed edges
    removed_edges = []
    if len(cf_examples) > 0:
        for (ii, jj) in cf_examples:
            u = reverse_node_dict.get(ii)
            v = reverse_node_dict.get(jj)
            if u is not None and v is not None and u != v:
                removed_edges.append((u, v))

    return {
        'node_idx': node_idx, 'true_label': true_label,
        'orig_pred': orig_pred, 'cf_found': len(removed_edges) > 0,
        'removed_edges': removed_edges, 'edge_scores': edge_scores,
        'node_dict': node_dict,
        '_node_time': time.time() - t0
    }


def generate_cf_explanations_parallel(
    model, features, edge_index, labels, target_indices, y_pred,
    hidden_dim=32, dropout=0.5, pca_components=None,
    cf_lr=0.1, cf_beta=0.1, cf_epochs=500, cf_n_hops=2,
    num_workers=None, verbose=False
) -> Tuple[List[Dict], float]:
    """
    Parallel version of generate_cf_explanations.

    Distributes per-node CF-GNNExplainer optimization across multiple processes.
    Falls back to serial if num_workers <= 1 or only 1 target node.

    Args:
        Same as generate_cf_explanations, plus:
        num_workers: Number of parallel workers (default: cpu_count - 1)

    Returns:
        (cf_results, serial_time): results list and sum of per-node times (equivalent serial time)
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, len(target_indices))

    # Fall back to serial for trivial cases
    if num_workers <= 1:
        from src.intervention_design_model_cf import generate_cf_explanations
        t0 = time.time()
        results = generate_cf_explanations(
            model, features, edge_index, labels, target_indices, y_pred,
            hidden_dim, dropout, pca_components,
            cf_lr, cf_beta, cf_epochs, cf_n_hops, verbose
        )
        return results, time.time() - t0

    # Prepare adjacency (same as serial version)
    adj = to_dense_adj(edge_index, max_num_nodes=features.shape[0]).squeeze(0)
    eit = dense_to_sparse(adj)

    # Config dict for worker initialization (all moved to CPU for serialization)
    config = {
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'in_features': features.shape[1],
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'pca_components': pca_components,
        'features': features.detach().cpu(),
        'labels': labels.detach().cpu(),
        'adj': adj.detach().cpu(),
        'edge_index_tuple': (eit[0].detach().cpu(), eit[1].detach().cpu()),
        'cf_lr': cf_lr,
        'cf_beta': cf_beta,
        'cf_epochs': cf_epochs,
        'cf_n_hops': cf_n_hops,
    }

    # Build per-node tasks: (node_idx, y_pred_value)
    tasks = [(int(idx), int(y_pred[idx].item())) for idx in target_indices]

    if verbose:
        print(f"  Running CF parallel with {num_workers} workers on {len(tasks)} nodes...")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers, initializer=_init_cf_worker, initargs=(config,)) as pool:
        results = []
        for result in pool.imap_unordered(_cf_process_node, tasks):
            results.append(result)
            if verbose and len(results) % 10 == 0:
                print(f"    Completed {len(results)}/{len(tasks)} nodes...")

    if verbose:
        print(f"    All {len(results)} nodes completed.")

    # Sum per-node times = equivalent serial time
    serial_time = sum(r.pop('_node_time', 0) for r in results)

    # Reorder results to match input order
    idx_map = {r['node_idx']: r for r in results}
    return [idx_map[int(idx)] for idx in target_indices], serial_time


# ============================================================
# CFF Explainer Parallel
# ============================================================

_cff_state = {}


def _init_cff_worker(config):
    """Initialize CFF worker process with model and shared data."""
    global _cff_state
    from src.model import GCN_Mulin

    model = GCN_Mulin(
        in_features=config['in_features'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        pca_components=config['pca_components']
    )
    model.load_state_dict(config['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    _cff_state = {
        'model': model,
        'features': config['features'],
        'edge_index': config['edge_index'],
        'labels': config['labels'],
        'mode': config['mode'],
        'lr': config['lr'],
        'epochs': config['epochs'],
        'lam': config['lam'],
        'alp': config['alp'],
        'gam': config['gam'],
        'unchangeable_indices': config['unchangeable_indices'],
    }


def _cff_process_node(task):
    """Process a single node for CFF explainer (worker function)."""
    node_idx, orig_pred, true_label = task
    t0 = time.time()

    from src.model import train_cff_explainer

    s = _cff_state
    _, result = train_cff_explainer(
        s['model'], s['features'], s['edge_index'],
        node_idx, orig_pred,
        mode=s['mode'], lr=s['lr'], epochs=s['epochs'],
        lam=s['lam'], alp=s['alp'], gam=s['gam'],
        verbose=False, unchangeable_indices=s['unchangeable_indices']
    )
    result['true_label'] = true_label
    result['_node_time'] = time.time() - t0
    return result


def generate_cff_explanations_parallel(
    base_model, features, edge_index, labels, target_indices,
    mode="edge+feature", lr=0.01, epochs=500, lam=500, alp=0.6, gam=0.5,
    num_workers=None, verbose=True, unchangeable_indices=None,
    hidden_dim=32, dropout=0.5, pca_components=None
) -> Tuple[List[Dict], float]:
    """
    Parallel version of generate_cff_explanations.

    Distributes per-node CFF training across multiple processes.
    Falls back to serial if num_workers <= 1 or only 1 target node.

    Args:
        Same as generate_cff_explanations, plus:
        num_workers: Number of parallel workers (default: cpu_count - 1)
        hidden_dim, dropout, pca_components: Model constructor args for worker reconstruction

    Returns:
        (cff_results, serial_time): results list and sum of per-node times (equivalent serial time)
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    num_workers = min(num_workers, len(target_indices))

    # Fall back to serial for trivial cases
    if num_workers <= 1:
        from src.model import generate_cff_explanations
        t0 = time.time()
        results = generate_cff_explanations(
            base_model, features, edge_index, labels, target_indices,
            mode, lr, epochs, lam, alp, gam, verbose=verbose,
            unchangeable_indices=unchangeable_indices
        )
        return results, time.time() - t0

    # Freeze base model and get predictions
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        predictions = base_model(features, edge_index)

    # Config dict for worker initialization
    config = {
        'model_state_dict': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
        'in_features': features.shape[1],
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'pca_components': pca_components,
        'features': features.detach().cpu(),
        'edge_index': edge_index.detach().cpu(),
        'labels': labels.detach().cpu(),
        'mode': mode,
        'lr': lr,
        'epochs': epochs,
        'lam': lam,
        'alp': alp,
        'gam': gam,
        'unchangeable_indices': unchangeable_indices,
    }

    # Build per-node tasks: (node_idx, orig_pred, true_label)
    tasks = []
    for idx in target_indices:
        node_idx = int(idx) if isinstance(idx, torch.Tensor) else idx
        true_label = labels[node_idx].item()
        orig_pred = int(predictions[node_idx].item() > 0.5)
        tasks.append((node_idx, orig_pred, true_label))

    if verbose:
        print(f"  Running CFF parallel with {num_workers} workers on {len(tasks)} nodes...")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers, initializer=_init_cff_worker, initargs=(config,)) as pool:
        results = []
        for result in pool.imap_unordered(_cff_process_node, tasks):
            results.append(result)
            if verbose and len(results) % 10 == 0:
                print(f"    Completed {len(results)}/{len(tasks)} nodes...")

    if verbose:
        print(f"    All {len(results)} nodes completed.")

    # Sum per-node times = equivalent serial time
    serial_time = sum(r.pop('_node_time', 0) for r in results)

    # Reorder results to match input order
    idx_map = {r['node_idx']: r for r in results}
    target_list = [int(idx) if isinstance(idx, torch.Tensor) else idx for idx in target_indices]
    return [idx_map[idx] for idx in target_list], serial_time
