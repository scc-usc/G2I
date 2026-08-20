import torch
import torch.optim as optim
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import List, Dict, Set, Tuple, Optional


def generate_cf_explanations(
    model,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    target_indices: List[int],
    y_pred: torch.Tensor,
    hidden_dim: int = 32,
    dropout: float = 0.5,
    pca_components = None,
    cf_lr: float = 0.1,
    cf_beta: float = 0.1,
    cf_epochs: int = 500,
    cf_n_hops: int = 2,
    verbose: bool = False
) -> List[Dict]:
    """
    Generate CF-GNNExplainer explanations for target nodes.
    
    This function encapsulates the CF-GNNExplainer logic, finding edges to remove
    that flip predictions (edge-based counterfactual explanations).
    
    Args:
        model: Trained GNN model (GCN_Mulin)
        features: Node feature tensor [num_nodes, num_features]
        edge_index: Edge index tensor [2, num_edges]
        labels: Node label tensor [num_nodes]
        target_indices: List of node indices to explain
        y_pred: Predictions for all nodes [num_nodes]
        hidden_dim: Hidden dimension of the model
        dropout: Dropout rate of the model
        pca_components: PCA components if used in model
        cf_lr: Learning rate for CF optimization
        cf_beta: Beta weight for graph distance loss
        cf_epochs: Number of CF optimization epochs
        cf_n_hops: Number of hops for neighborhood subgraph
        verbose: Print progress information
    
    Returns:
        List of dictionaries containing CF explanation results:
        - node_idx: Node index
        - true_label: Ground truth label
        - orig_pred: Original prediction
        - cf_found: Whether a counterfactual was found
        - removed_edges: List of (u, v) edges to remove
        - node_dict: Mapping from original to subgraph indices
    """
    from src.model import CFExplainer
    from src.utils import get_neighbourhood
    
    # Prepare adjacency for CF-GNNExplainer
    # Note: edge_index should already be undirected (converted at data loading time)
    adj = to_dense_adj(edge_index, max_num_nodes=features.shape[0]).squeeze(0)
    edge_index_tuple = dense_to_sparse(adj)
    
    cf_results = []
    
    for i, node_idx in enumerate(target_indices):
        node_idx = int(node_idx)
        orig_pred = int(y_pred[node_idx].item())
        true_label = int(labels[node_idx].item())
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing node {i+1}/{len(target_indices)}...")
        
        # Get neighborhood subgraph
        try:
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
                node_idx, edge_index_tuple, cf_n_hops, features, labels
            )
        except IndexError:
            cf_results.append({
                'node_idx': node_idx,
                'true_label': true_label,
                'orig_pred': orig_pred,
                'cf_found': False,
                'removed_edges': [],
                'node_dict': {node_idx: 0}
            })
            continue
        
        new_idx = node_dict[node_idx]
        
        if sub_adj.dim() < 2 or sub_adj.shape[0] < 2:
            cf_results.append({
                'node_idx': node_idx,
                'true_label': true_label,
                'orig_pred': orig_pred,
                'cf_found': False,
                'removed_edges': [],
                'node_dict': node_dict
            })
            continue
        
        # Create explainer (with PCA components)
        explainer = CFExplainer(
            model=model,
            sub_adj=sub_adj,
            sub_feat=sub_feat,
            n_hid=hidden_dim,
            dropout=dropout,
            sub_labels=sub_labels,
            y_pred_orig=orig_pred,
            new_idx=new_idx,
            beta=cf_beta,
            pca_components=pca_components
        )
        
        # Run CF optimization with Adam optimizer
        cf_optimizer = optim.Adam(explainer.cf_model.parameters(), lr=cf_lr)
        cf_examples = explainer.explain(
            cf_optimizer=cf_optimizer,
            cf_epochs=cf_epochs,
            verbose=False
        )

        # Get edge importance scores from perturbation mask
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
        except:
            pass

        # Convert to removed edges
        removed_edges = []
        if len(cf_examples) > 0:
            for (ii, jj) in cf_examples:
                u = reverse_node_dict.get(ii)
                v = reverse_node_dict.get(jj)
                if u is not None and v is not None and u != v:
                    removed_edges.append((u, v))

        cf_results.append({
            'node_idx': node_idx,
            'true_label': true_label,
            'orig_pred': orig_pred,
            'cf_found': len(removed_edges) > 0,
            'removed_edges': removed_edges,
            'edge_scores': edge_scores,
            'node_dict': node_dict
        })
    
    return cf_results


# ============================================================
# Original function (commented out) - uses neighbor_mean conditions
# ============================================================
# def convert_cf_to_counterfactuals(cf_results, data, diff_threshold=0.1):
#     """Convert CF-GNNExplainer results to counterfactual interventions"""
#     counterfactuals = []
#
#     for result in cf_results:
#         node_idx = result['node_idx']
#         removed_edges = result['removed_edges']
#
#         if not result['cf_found'] or len(removed_edges) == 0:
#             counterfactuals.append(set())
#             continue
#
#         edge_interventions = set()
#         for (u, v) in removed_edges:
#             neighbor = v if u == node_idx else u
#             edge_interventions.add(('drop_edge', neighbor))
#
#         if len(edge_interventions) == 0:
#             counterfactuals.append(set())
#             continue
#
#         edge_index = data.edge_index
#         incoming_mask = edge_index[1] == node_idx
#         current_neighbors = set(edge_index[0][incoming_mask].tolist())
#         outgoing_mask = edge_index[0] == node_idx
#         current_neighbors.update(edge_index[1][outgoing_mask].tolist())
#
#         new_neighbors = set(current_neighbors)
#         for intervention in edge_interventions:
#             if intervention[0] == 'drop_edge':
#                 new_neighbors.discard(intervention[1])
#
#         if len(current_neighbors) == 0 or len(new_neighbors) == 0:
#             counterfactuals.append(set())
#             continue
#
#         current_neighbor_features = data.x[list(current_neighbors)].mean(dim=0)
#         new_neighbor_features = data.x[list(new_neighbors)].mean(dim=0)
#
#         neighbor_conditions = set()
#         for feat_idx in range(data.x.shape[1]):
#             diff = new_neighbor_features[feat_idx].item() - current_neighbor_features[feat_idx].item()
#
#             if abs(diff) > diff_threshold:
#                 if diff > 0:
#                     operator = '>'
#                     threshold = new_neighbor_features[feat_idx].item()
#                 else:
#                     operator = '<'
#                     threshold = new_neighbor_features[feat_idx].item()
#
#                 neighbor_conditions.add(('neighbor_mean', feat_idx, operator, round(threshold, 3)))
#
#         counterfactuals.append(neighbor_conditions)
#
#     return counterfactuals


# ============================================================
# New function - uses reduce_feature_ratio with discretization
# ============================================================
def discretize_ratio(ratio: float, levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) -> float:
    """Discretize ratio to specified levels"""
    for level in levels:
        if ratio <= level:
            return level
    return levels[-1]


def convert_cf_to_counterfactuals(cf_results, data, diff_threshold=0.1, ratio_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Convert CF-GNNExplainer results to counterfactual interventions.

    Uses reduce_feature_ratio format:
    - ('reduce_feature_ratio', feature_idx, discretized_ratio)
    - ratio indicates the proportion of neighbors with the feature to remove

    Args:
        cf_results: CF-GNNExplainer results
        data: PyG Data object
        diff_threshold: Feature difference threshold (unused, kept for compatibility)
        ratio_levels: Discretization levels, default [0.1, ..., 1.0] for 10% to 100%

    Returns:
        List of Sets, each set contains ('reduce_feature_ratio', feat_idx, ratio) tuples
    """
    counterfactuals = []

    for result in cf_results:
        node_idx = result['node_idx']
        removed_edges = result['removed_edges']

        if not result['cf_found'] or len(removed_edges) == 0:
            counterfactuals.append(set())
            continue

        # Find neighbors involved in removed edges
        removed_neighbors = set()
        for (u, v) in removed_edges:
            neighbor = v if u == node_idx else u
            removed_neighbors.add(neighbor)

        # Get all current neighbors
        edge_index = data.edge_index
        current_neighbors = set()
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() == node_idx:
                current_neighbors.add(edge_index[1, i].item())
            if edge_index[1, i].item() == node_idx:
                current_neighbors.add(edge_index[0, i].item())

        if len(current_neighbors) == 0:
            counterfactuals.append(set())
            continue

        # For each feature, calculate reduction ratio
        interventions = set()
        n_features = data.x.shape[1]

        for feat_idx in range(n_features):
            # Count neighbors with this feature (feature value = 1)
            neighbors_with_feature = sum(1 for n in current_neighbors if data.x[n, feat_idx].item() == 1)

            if neighbors_with_feature == 0:
                continue

            # Count removed neighbors with this feature
            removed_with_feature = sum(1 for n in removed_neighbors if data.x[n, feat_idx].item() == 1)

            if removed_with_feature == 0:
                continue

            # Calculate and discretize reduction ratio
            ratio = removed_with_feature / neighbors_with_feature
            discretized = discretize_ratio(ratio, ratio_levels)

            interventions.add(('reduce_feature_ratio', feat_idx, discretized))

        counterfactuals.append(interventions)

    return counterfactuals


