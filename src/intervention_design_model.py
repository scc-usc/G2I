
import random
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import List, Set, Dict, Tuple, Optional, Union
from torch_geometric.data import Data

def get_coverage_count(val):
    """
    Robustly get the count of covered nodes, regardless of type (set, list, int, etc).
    Args:
        val: Coverage variable (set, list, int, etc)
    Returns:
        int: Number of covered nodes
    """
    if isinstance(val, (set, list, tuple)):
        return len(val)
    elif hasattr(val, '__len__'):
        return len(val)
    else:
        return int(val) if val is not None else 0
"""
Intervention Design via Counterfactual Explanations (Graph-Based)
Implements greedy algorithm for selecting interventions to flip positive instances to negative
using a trained GNN model. Supports node feature changes, edge additions, and edge removals.
"""




# ============================================================
# Original function (commented out) - uses neighbor_mean conditions
# ============================================================
# def convert_edge_changes_to_neighbor_conditions(
#     node_idx: int,
#     data: Data,
#     edge_interventions: Set[Tuple],
#     diff_threshold: float = 0.1
# ) -> Set[Tuple]:
#     """
#     Convert edge additions/deletions into interpretable neighbor feature conditions.
#     """
#     if len(edge_interventions) == 0:
#         return set()
#
#     edge_index = data.edge_index
#     current_neighbors = edge_index[0][edge_index[1] == node_idx].tolist()
#
#     new_neighbors = set(current_neighbors)
#     for intervention in edge_interventions:
#         if intervention[0] == 'add_edge':
#             new_neighbors.add(intervention[1])
#         elif intervention[0] == 'drop_edge':
#             new_neighbors.discard(intervention[1])
#
#     if len(current_neighbors) == 0 or len(new_neighbors) == 0:
#         return set()
#
#     current_neighbor_features = data.x[current_neighbors].mean(dim=0)
#     new_neighbor_features = data.x[list(new_neighbors)].mean(dim=0)
#
#     neighbor_conditions = set()
#     for feat_idx in range(data.x.shape[1]):
#         diff = new_neighbor_features[feat_idx].item() - current_neighbor_features[feat_idx].item()
#
#         if abs(diff) > diff_threshold:
#             if diff > 0:
#                 operator = '>'
#                 threshold = new_neighbor_features[feat_idx].item()
#             else:
#                 operator = '<'
#                 threshold = new_neighbor_features[feat_idx].item()
#
#             neighbor_conditions.add(('neighbor_mean', feat_idx, operator, round(threshold, 3)))
#
#     return neighbor_conditions


# ============================================================
# New function - uses reduce_feature_ratio with discretization
# ============================================================
def _discretize_ratio(ratio: float, levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) -> float:
    """Discretize ratio to specified levels"""
    for level in levels:
        if ratio <= level:
            return level
    return levels[-1]


def convert_edge_changes_to_neighbor_conditions(
    node_idx: int,
    data: Data,
    edge_interventions: Set[Tuple],
    diff_threshold: float = 0.1,
    ratio_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
) -> Set[Tuple]:
    """
    Convert edge additions/deletions into reduce_feature_ratio conditions.

    For drop_edge: uses reduce_feature_ratio format
    For add_edge: uses increase_feature_ratio format

    Args:
        node_idx: Index of the node
        data: PyTorch Geometric Data object
        edge_interventions: Set of edge intervention tuples (('add_edge', neighbor) or ('drop_edge', neighbor))
        diff_threshold: Unused, kept for compatibility
        ratio_levels: Discretization levels, default [0.1, ..., 1.0] for 10% to 100%

    Returns:
        Set of tuples: ('reduce_feature_ratio', feat_idx, ratio) or ('increase_feature_ratio', feat_idx, ratio)
    """
    if len(edge_interventions) == 0:
        return set()

    # Separate add_edge and drop_edge
    added_neighbors = set()
    removed_neighbors = set()
    for intervention in edge_interventions:
        if intervention[0] == 'add_edge':
            added_neighbors.add(intervention[1])
        elif intervention[0] == 'drop_edge':
            removed_neighbors.add(intervention[1])

    # Get all current neighbors
    edge_index = data.edge_index
    current_neighbors = set()
    for i in range(edge_index.shape[1]):
        if edge_index[0, i].item() == node_idx:
            current_neighbors.add(edge_index[1, i].item())
        if edge_index[1, i].item() == node_idx:
            current_neighbors.add(edge_index[0, i].item())

    if len(current_neighbors) == 0 and len(added_neighbors) == 0:
        return set()

    interventions = set()
    n_features = data.x.shape[1]

    # Process drop_edge -> reduce_feature_ratio
    if len(removed_neighbors) > 0 and len(current_neighbors) > 0:
        for feat_idx in range(n_features):
            neighbors_with_feature = sum(1 for n in current_neighbors if data.x[n, feat_idx].item() == 1)
            if neighbors_with_feature == 0:
                continue

            removed_with_feature = sum(1 for n in removed_neighbors if data.x[n, feat_idx].item() == 1)
            if removed_with_feature == 0:
                continue

            ratio = removed_with_feature / neighbors_with_feature
            discretized = _discretize_ratio(ratio, ratio_levels)
            interventions.add(('reduce_feature_ratio', feat_idx, discretized))

    # Process add_edge -> increase_feature_ratio
    if len(added_neighbors) > 0:
        # Calculate new neighbor set after adding
        new_neighbors = current_neighbors | added_neighbors
        for feat_idx in range(n_features):
            # Count added neighbors with this feature
            added_with_feature = sum(1 for n in added_neighbors if data.x[n, feat_idx].item() == 1)
            if added_with_feature == 0:
                continue

            # Calculate increase ratio (relative to original neighbor count)
            if len(current_neighbors) > 0:
                neighbors_with_feature = sum(1 for n in current_neighbors if data.x[n, feat_idx].item() == 1)
                if neighbors_with_feature > 0:
                    ratio = added_with_feature / neighbors_with_feature
                else:
                    ratio = 1.0  # No feature before, has now, set to 100%
            else:
                ratio = 1.0  # No neighbors before

            discretized = _discretize_ratio(ratio, ratio_levels)
            interventions.add(('increase_feature_ratio', feat_idx, discretized))

    return interventions


def convert_neighbor_feature_changes_to_neighbor_conditions(
    node_idx: int,
    data: Data,
    neighbor_feature_interventions: Set[Tuple],
    diff_threshold: float = 0.1
) -> Set[Tuple]:
    """
    Convert neighbor feature changes into interpretable neighbor feature conditions.
    
    Only considers features that were actually altered in the neighbor interventions.
    Measures the distribution of those features among neighbors before and after changes.
    For altered features whose distribution changed significantly, creates threshold conditions
    like ('neighbor_mean', feature_idx, '>', value).
    
    Args:
        node_idx: Index of the node
        data: PyTorch Geometric Data object
        neighbor_feature_interventions: Set of neighbor feature intervention tuples 
                                       (('neighbor_feature', neighbor_idx, feat_idx, target_value))
        diff_threshold: Minimum difference in feature distribution to create a condition
    
    Returns:
        Set of neighbor condition tuples: ('neighbor_mean', feature_idx, operator, threshold)
        where operator is '>' or '<'
    """
    if len(neighbor_feature_interventions) == 0:
        return set()
    
    # Get current neighbors (incoming neighbors: nodes j with edge j -> node_idx)
    edge_index = data.edge_index
    current_neighbors = edge_index[0][edge_index[1] == node_idx].tolist()
    
    # If no neighbors, return empty set
    if len(current_neighbors) == 0:
        return set()
    
    # Identify which features were actually changed in the interventions
    changed_features = set()
    for interv in neighbor_feature_interventions:
        _, _, feat_idx, _ = interv
        changed_features.add(feat_idx)
    
    # Compute initial feature distributions
    initial_neighbor_features = data.x[current_neighbors].mean(dim=0)
    
    # Apply neighbor feature changes
    data_modified = data.clone()
    for interv in neighbor_feature_interventions:
        _, neighbor_idx, feat_idx, target_value = interv
        data_modified.x[neighbor_idx, feat_idx] = target_value
    
    # Compute new feature distributions (only for current neighbors)
    new_neighbor_features = data_modified.x[current_neighbors].mean(dim=0)
    
    # Find features with significant distribution changes (only for changed features)
    neighbor_conditions = set()
    for feat_idx in changed_features:
        diff = new_neighbor_features[feat_idx].item() - initial_neighbor_features[feat_idx].item()
        
        if abs(diff) > diff_threshold:
            # Create threshold condition
            if diff > 0:
                # New distribution is higher, so we need neighbor_mean > new_value
                operator = '>'
                threshold = new_neighbor_features[feat_idx].item()
            else:
                # New distribution is lower, so we need neighbor_mean < new_value
                operator = '<'
                threshold = new_neighbor_features[feat_idx].item()
            
            neighbor_conditions.add(('neighbor_mean', feat_idx, operator, round(threshold, 3)))
    
    return neighbor_conditions


def get_local_counterfactual(
    node_idx: int,
    data: Data,
    model: nn.Module,
    max_interventions: int = 5,
    edge_mode: str = 'none',  # 'none', 'edges', 'edge_features'
    only_delete_edges: bool = False,
    flip_direction: str = 'positive_to_negative',
    device: str = 'cpu',
    early_stop: bool = False,  # parameter to control early termination on flip
    ignore_node_features: bool = False,  # flag to ignore node feature interventions
    feature_partitions: Optional[List[List[int]]] = None,  # Groupings for one-hot encodings
    unchangeable_features: Optional[List[int]] = None,  # List of feature indices that cannot be perturbed
    forbidden_target_features: Optional[List[int]] = None, # List of feature indices that cannot be suggested as target states
    verbose: bool = False
) -> Set[Tuple]:
    """
    Generate a local counterfactual explanation for a node using the GNN model.
    
    Uses a greedy approach to find minimal changes (feature changes, edge additions/removals, or neighbor feature changes)
    that flip the prediction for the target node.
    
    Args:
        node_idx: Index of the target node
        data: PyTorch Geometric Data object (x, edge_index, y)
        model: Trained GNN model
        max_interventions: Maximum number of interventions to consider
        edge_mode: How to handle edges/neighbors - 'none' (ignore), 'edges' (add/remove edges), 'edge_features' (flip neighbor features)
        only_delete_edges: If True and edge_mode=='edges', only consider edge deletions (not additions)
        flip_direction: 'positive_to_negative' or 'negative_to_positive'
        device: Device to run the model on
        early_stop: If True, stop early when a flip is found
        ignore_node_features: If True, skip node feature interventions and only consider edge/neighbor interventions
        feature_partitions: Optional list of lists, where each inner list contains feature indices
                            belonging to the same one-hot encoding partition.
        unchangeable_features: Optional list of feature indices that are immutable.
        forbidden_target_features: Optional list of feature indices that cannot be set to 1/True (e.g. "missing" categories).
        verbose: Whether to print progress
    
    Returns:
        Set of intervention tuples:
        - ('feature', feature_idx, target_value) for feature changes
        - ('neighbor_mean', feature_idx, operator, threshold) for neighbor conditions
        Note: Edge/neighbor interventions are converted to neighbor conditions
    """
    model.eval()
    data = data.to(device)
    n_features = int(data.x.shape[1])
    n_nodes = int(data.x.shape[0])

    # Default partitions to singletons if not provided
    if feature_partitions is None:
        feature_partitions = [[i] for i in range(n_features)]
    
    unchangeable_set = set(unchangeable_features) if unchangeable_features is not None else set()
    forbidden_set = set(forbidden_target_features) if forbidden_target_features is not None else set()

    # helper: get current prediction probability for node
    def node_prob(d):
        with torch.no_grad():
            out = model(d.x, d.edge_index)
            return float(out[node_idx].item())

    # starting probability and target direction
    base_prob = node_prob(data)
    if flip_direction == 'positive_to_negative':
        target_class = False
    else:
        target_class = True

    # if already in target state, return empty
    if (base_prob > 0.5) == target_class:
        return set()

    # Tracking
    cf: Set[Tuple] = set()
    edge_interventions: Set[Tuple] = set()
    neighbor_feature_interventions: Set[Tuple] = set()

    # data_modified will accumulate applied interventions
    data_modified = data.clone()

    for step in range(max_interventions):
        # Recompute current base probability from modified data
        current_prob = node_prob(data_modified)

        # Build candidate set fresh each iteration
        candidates: List[Tuple] = []  # list of (candidate_tuple, kind)

        # 1) Self feature flips
        if not ignore_node_features:
            for partition in feature_partitions:
                # Skip if any feature in partition is unchangeable
                if any(idx in unchangeable_set for idx in partition):
                    continue
                
                # If any feature in partition is already in cf, skip entire partition
                if any(any(ci[0] == 'feature' and ci[1] == f for ci in cf) for f in partition):
                    continue
                
                if len(partition) == 1:
                    feat_idx = partition[0]
                    orig = float(data_modified.x[node_idx, feat_idx].item())
                    if orig > 0:
                        if feat_idx not in forbidden_set and feat_idx not in unchangeable_set:
                            candidates.append((('feature', feat_idx, 0), 'feature'))
                    if orig < 1:
                        if feat_idx not in forbidden_set and feat_idx not in unchangeable_set:
                            candidates.append((('feature', feat_idx, 1), 'feature'))
                else:
                    # One-hot logic: only the "changed to" part matters.
                    # Setting one feature in partition to 1 implies others are 0.
                    for f_target in partition:
                        # Only consider flipping the feature 'ON' (to 1)
                        # The 'OFF' part is handled by the partition-aware application logic.
                        if float(data_modified.x[node_idx, f_target].item()) < 1:
                            if f_target not in forbidden_set and f_target not in unchangeable_set:
                                candidates.append((('feature', f_target, 1), 'feature'))

        # 2) Neighbor feature flips (incoming neighbors)
        if edge_mode == 'edge_features':
            edge_index = data_modified.edge_index.cpu().numpy()
            incoming = sorted(int(x) for x in edge_index[0, edge_index[1] == node_idx])
            for neigh in incoming:
                for partition in feature_partitions:
                    # If any feature in partition for this neighbor is already in cf
                    if any(any(ci[0] == 'neighbor_feature' and ci[1] == neigh and ci[2] == f for ci in cf) for f in partition):
                        continue
                    
                    if len(partition) == 1:
                        feat_idx = partition[0]
                        orig = float(data_modified.x[neigh, feat_idx].item())
                        if orig > 0:
                            if feat_idx not in forbidden_set and feat_idx not in unchangeable_set:
                                candidates.append((('neighbor_feature', neigh, feat_idx, 0), 'neighbor_feature'))
                        if orig < 1:
                            if feat_idx not in forbidden_set and feat_idx not in unchangeable_set:
                                candidates.append((('neighbor_feature', neigh, feat_idx, 1), 'neighbor_feature'))
                    else:
                        # One-hot logic for neighbor: only store the 'target' feature set to 1
                        for f_target in partition:
                            if float(data_modified.x[neigh, f_target].item()) < 1:
                                if f_target not in forbidden_set and f_target not in unchangeable_set:
                                    candidates.append((('neighbor_feature', neigh, f_target, 1), 'neighbor_feature'))

        # 3) Edge add/drop candidates
        if edge_mode == 'edges':
            # current incoming neighbors
            edge_index = data_modified.edge_index.cpu().numpy()
            incoming = set(int(x) for x in edge_index[0, edge_index[1] == node_idx])
            # drop candidates
            for neigh in sorted(incoming):
                if ('drop_edge', neigh) in cf:
                    continue
                candidates.append((( 'drop_edge', neigh), 'drop_edge'))

            # add candidates (limit search for speed)
            if not only_delete_edges:
                potential = [n for n in range(n_nodes) if n not in incoming and n != node_idx]
                for neigh in sorted(potential)[:20]:
                    if ('add_edge', neigh) in cf:
                        continue
                    candidates.append((( 'add_edge', neigh), 'add_edge'))

        # Evaluate candidates for marginal improvement toward target
        best_candidate = None
        best_improvement = 0.0

        for cand, kind in candidates:
            # Evaluate applying this candidate along with existing counterfactuals
            to_apply = cf | (set(cand) if isinstance(cand[0], tuple) else {cand})
            test = apply_interventions(data, node_idx, to_apply, feature_partitions)
            
            new_prob = node_prob(test)

            if flip_direction == 'positive_to_negative':
                improvement = current_prob - new_prob
            else:
                improvement = new_prob - current_prob

            if verbose:
                print(f"try {cand}: current_prob={current_prob:.4f}, new_prob={new_prob:.4f}, improvement={improvement:.4f}")

            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = cand

        # If no candidate improves probability, stop
        if best_candidate is None or best_improvement <= 0:
            break

        # Apply best candidate to data_modified and record
        cand = best_candidate
        to_add = set(cand) if isinstance(cand[0], tuple) else {cand}
        
        for item in to_add:
            cf.add(item)
            if item[0] == 'neighbor_feature':
                neighbor_feature_interventions.add(item)
            elif item[0] in ('drop_edge', 'add_edge'):
                edge_interventions.add(item)

        # Update data_modified using the unified application logic
        data_modified = apply_interventions(data, node_idx, cf, feature_partitions)

        if verbose:
            print(f"Selected {cand} (improvement={best_improvement:.4f}); applied at step {step+1}")

        # Early stop if flip achieved
        with torch.no_grad():
            out_now = model(data_modified.x, data_modified.edge_index)
            pred_now = (out_now[node_idx].item() > 0.5)
        if pred_now == target_class:
            if verbose:
                print("Target class achieved by interventions")
            break

    # Final verification
    with torch.no_grad():
        out_final = model(data_modified.x, data_modified.edge_index)
        final_pred = (out_final[node_idx].item() > 0.5)

    if final_pred != target_class:
        return set()

    # Convert any edge/neigh-feature interventions to neighbor conditions
    cf_final = set(interv for interv in cf if interv[0] == 'feature')
    if edge_mode == 'edges' and len(edge_interventions) > 0:
        neighbor_conds = convert_edge_changes_to_neighbor_conditions(node_idx, data, edge_interventions, diff_threshold=0.1)
        cf_final.update(neighbor_conds)
    if edge_mode == 'edge_features' and len(neighbor_feature_interventions) > 0:
        neighbor_conds = convert_neighbor_feature_changes_to_neighbor_conditions(node_idx, data, neighbor_feature_interventions, diff_threshold=0.1)
        cf_final.update(neighbor_conds)

    return cf_final


def apply_interventions(
    data: Data, 
    node_idx: int, 
    interventions: Set[Tuple],
    feature_partitions: Optional[List[List[int]]] = None
) -> Data:
    """
    Apply a set of interventions to a graph for a specific node.
    If partitions are provided, setting a one-hot feature to 1 will zero others in partition.
    
    Args:
        data: PyTorch Geometric Data object
        node_idx: Target node index
        interventions: Set of intervention tuples
        feature_partitions: Optional groupings for one-hot features
    
    Returns:
        Modified data object
    """
    data_modified = data.clone()
    
    # Pre-process partitions for quick lookup
    idx_to_partition = {}
    if feature_partitions:
        for p in feature_partitions:
            if len(p) > 1:
                for idx in p:
                    idx_to_partition[idx] = p

    for interv in interventions:
        if interv[0] == 'feature':
            _, feat_idx, target_value = interv
            # Partition-aware application: if setting a one-hot feature to 1, zero others
            if target_value == 1 and feat_idx in idx_to_partition:
                partition = idx_to_partition[feat_idx]
                data_modified.x[node_idx, partition] = 0
            data_modified.x[node_idx, feat_idx] = target_value
            
        elif interv[0] == 'drop_edge':
            _, neighbor_idx = interv
            edge_list = data_modified.edge_index.t().tolist()
            edge_list = [e for e in edge_list if not (e[0] == node_idx and e[1] == neighbor_idx)
                         and not (e[1] == node_idx and e[0] == neighbor_idx)]
            if edge_list:
                data_modified.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                data_modified.edge_index = torch.empty(2, 0, dtype=torch.long)
            
        elif interv[0] == 'add_edge':
            _, neighbor_idx = interv
            new_edges = torch.tensor([[node_idx, neighbor_idx], [neighbor_idx, node_idx]], 
                                     dtype=torch.long).t()
            data_modified.edge_index = torch.cat([data_modified.edge_index, new_edges], dim=1)
            
        elif interv[0] == 'reduce_feature_ratio':
            _, feat_idx, ratio = interv
            # Find neighbors with feature
            edge_list = data_modified.edge_index.t().tolist()
            current_neighbors = [e[1] for e in edge_list if e[0] == node_idx]
            target_neighbors = []
            for neigh in current_neighbors:
                if data_modified.x[neigh, feat_idx].item() > 0.5:
                    target_neighbors.append(neigh)
            
            if target_neighbors:
                num_to_remove = max(1, int(len(target_neighbors) * ratio))
                # Deterministic removal
                neighbors_to_remove = set(target_neighbors[:num_to_remove])
                if neighbors_to_remove:
                    edge_list = [e for e in edge_list if not (
                        (e[0] == node_idx and e[1] in neighbors_to_remove) or
                        (e[1] == node_idx and e[0] in neighbors_to_remove)
                    )]
                    if edge_list:
                        data_modified.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    else:
                        data_modified.edge_index = torch.empty(2, 0, dtype=torch.long)

        elif interv[0] == 'increase_feature_ratio':
            _, feat_idx, ratio = interv
            # Find neighbors with feature for base count
            edge_list = data_modified.edge_index.t().tolist()
            current_neighbors = set(e[1] for e in edge_list if e[0] == node_idx)
            neighbors_with_feat = sum(1 for n in current_neighbors 
                                    if data_modified.x[n, feat_idx].item() > 0.5)
            
            num_to_add = max(1, int(neighbors_with_feat * ratio)) if neighbors_with_feat > 0 else 1
            
            # Find candidates (non-neighbors with feature)
            num_nodes = data_modified.x.shape[0]
            candidates = []
            for n in range(num_nodes):
                if n != node_idx and n not in current_neighbors:
                    if data_modified.x[n, feat_idx].item() > 0.5:
                        candidates.append(n)
                        if len(candidates) >= num_to_add:
                            break
            
            if candidates:
                new_edges_list = []
                for cand in candidates[:num_to_add]:
                    new_edges_list.append([node_idx, cand])
                    new_edges_list.append([cand, node_idx])
                new_edges = torch.tensor(new_edges_list, dtype=torch.long).t()
                data_modified.edge_index = torch.cat([data_modified.edge_index, new_edges], dim=1)
            
        elif interv[0] == 'neighbor_feature':
            _, neighbor_idx, feat_idx, target_value = interv
            if target_value == 1 and feat_idx in idx_to_partition:
                partition = idx_to_partition[feat_idx]
                data_modified.x[neighbor_idx, partition] = 0
            data_modified.x[neighbor_idx, feat_idx] = target_value
    
    return data_modified


# ============================================================
# Original compute_coverage function (commented out) - no actual edge operations
# ============================================================
# def compute_coverage(
#     S: Set[frozenset],
#     data: Data,
#     positive_node_indices: List[int],
#     counterfactuals: List[Set[Tuple]],
#     model: nn.Module,
#     device: str = 'cpu',
#     feature_partitions: Optional[List[List[int]]] = None
# ) -> int:
#     """
#     Compute coverage: number of positive nodes that would flip when any selected clause is applied.
#     (Original version: uses condition matching, no actual edge operations)
#     """
#     if not S:
#         return 0
#
#     model.eval()
#     data = data.to(device)
#     covered_nodes = set()
#     node_cf_map = {node_idx: cf for node_idx, cf in zip(positive_node_indices, counterfactuals)}
#
#     with torch.no_grad():
#         for node_idx in positive_node_indices:
#             if node_idx in covered_nodes:
#                 continue
#             node_cf = node_cf_map.get(node_idx, set())
#             for clause in S:
#                 if frozenset(node_cf) == clause:
#                     covered_nodes.add(node_idx)
#                     break
#                 feature_interventions = {interv for interv in clause if interv[0] == 'feature'}
#                 if feature_interventions:
#                     data_modified = apply_interventions(data, node_idx, feature_interventions, feature_partitions)
#                     out = model(data_modified.x, data_modified.edge_index)
#                     pred = (out[node_idx] > 0.5).item()
#                     if not pred:
#                         covered_nodes.add(node_idx)
#                         break
#                 node_feature_conds = {interv for interv in node_cf if interv[0] == 'feature'}
#                 if feature_interventions == node_feature_conds:
#                     ratio_types = ('neighbor_mean', 'reduce_feature_ratio', 'increase_feature_ratio')
#                     clause_neighbor_conds = {interv for interv in clause if interv[0] in ratio_types}
#                     node_neighbor_conds = {interv for interv in node_cf if interv[0] in ratio_types}
#                     if _check_neighbor_compatibility(node_neighbor_conds, clause_neighbor_conds):
#                         covered_nodes.add(node_idx)
#                         break
#     return len(covered_nodes)


# ============================================================
# New: Helper function to apply edge interventions
# ============================================================
def apply_edge_interventions(
    data: Data,
    node_idx: int,
    edge_interventions: Set[Tuple],
    seed: int = 42
) -> Data:
    """
    Apply edge interventions to graph (reduce_feature_ratio / increase_feature_ratio).

    Args:
        data: PyTorch Geometric Data object
        node_idx: Target node index
        edge_interventions: Set of edge interventions containing:
            - ('reduce_feature_ratio', feat_idx, ratio): Remove ratio proportion of edges to neighbors with feature
            - ('increase_feature_ratio', feat_idx, ratio): Add ratio proportion of edges to non-neighbors with feature
        seed: Random seed for reproducibility

    Returns:
        Modified Data object (edge_index updated)
    """
    import random
    random.seed(seed)

    data_modified = data.clone()
    edge_index = data_modified.edge_index.clone()

    # Get current neighbors (undirected graph, bidirectional)
    current_neighbors = set()
    edge_list = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_list.append((src, dst))
        if src == node_idx:
            current_neighbors.add(dst)
        if dst == node_idx:
            current_neighbors.add(src)

    edges_to_remove = set()
    edges_to_add = set()

    for interv in edge_interventions:
        if interv[0] == 'reduce_feature_ratio':
            feat_idx, ratio = interv[1], interv[2]

            # Find neighbors with this feature
            neighbors_with_feature = [n for n in current_neighbors if data.x[n, feat_idx].item() == 1]

            if len(neighbors_with_feature) > 0:
                # Calculate number to remove based on ratio
                num_to_remove = max(1, int(len(neighbors_with_feature) * ratio))
                num_to_remove = min(num_to_remove, len(neighbors_with_feature))

                # Randomly select neighbors to remove
                to_remove = random.sample(neighbors_with_feature, num_to_remove)

                for neighbor in to_remove:
                    edges_to_remove.add((node_idx, neighbor))
                    edges_to_remove.add((neighbor, node_idx))

        elif interv[0] == 'increase_feature_ratio':
            feat_idx, ratio = interv[1], interv[2]

            # Find non-neighbors with this feature
            all_nodes = set(range(data.x.shape[0]))
            non_neighbors = all_nodes - current_neighbors - {node_idx}
            non_neighbors_with_feature = [n for n in non_neighbors if data.x[n, feat_idx].item() == 1]

            if len(non_neighbors_with_feature) > 0:
                # Calculate current neighbors with feature as baseline
                current_with_feature = sum(1 for n in current_neighbors if data.x[n, feat_idx].item() == 1)
                base = max(current_with_feature, 1)

                # Calculate number to add based on ratio
                num_to_add = max(1, int(base * ratio))
                num_to_add = min(num_to_add, len(non_neighbors_with_feature))

                # Randomly select nodes to add
                to_add = random.sample(non_neighbors_with_feature, num_to_add)

                for neighbor in to_add:
                    edges_to_add.add((node_idx, neighbor))
                    edges_to_add.add((neighbor, node_idx))

    # Build new edge list
    new_edges = []
    for edge in edge_list:
        if edge not in edges_to_remove:
            new_edges.append(edge)

    for edge in edges_to_add:
        if edge not in new_edges:
            new_edges.append(edge)

    # Convert back to tensor
    if len(new_edges) > 0:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    else:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long)

    data_modified.edge_index = new_edge_index
    return data_modified


# ============================================================
# New compute_coverage function - performs actual edge operations
# ============================================================
def compute_coverage(
    S: Set[frozenset],
    data: Data,
    positive_node_indices: List[int],
    counterfactuals: List[Set[Tuple]],
    model: nn.Module,
    device: str = 'cpu',
    feature_partitions: Optional[List[List[int]]] = None
) -> int:
    """
    Compute coverage: number of positive nodes that would flip when any selected clause is applied.

    New version: Actually performs edge deletion and addition operations, then runs model to verify flip.

    Coverage algorithm:
    1. (Self Cover) If clause is exactly the node's counterfactual, directly cover
    2. (Direct Flip) Apply all interventions in clause (feature changes + edge operations), check for flip

    Args:
        S: Set of selected clauses (each is a frozenset of intervention tuples).
        data: PyTorch Geometric Data object.
        positive_node_indices: List of positive node indices to check coverage for.
        counterfactuals: List of original counterfactuals (C_j) for each positive node.
        model: Trained GNN model.
        device: Device to run the model on.
        feature_partitions: Optional groupings for one-hot features.

    Returns:
        Number of nodes that would be flipped by at least one selected clause.
    """
    if not S:
        return 0

    model.eval()
    data = data.to(device)
    covered_nodes = set()

    # Create a mapping from node_idx to its counterfactual for quick lookup
    node_cf_map = {node_idx: cf for node_idx, cf in zip(positive_node_indices, counterfactuals)}

    with torch.no_grad():
        for node_idx in positive_node_indices:
            if node_idx in covered_nodes:
                continue

            node_cf = node_cf_map.get(node_idx, set())

            for clause in S:
                # --- 1. Self Cover: clause is exactly the node's counterfactual ---
                if frozenset(node_cf) == clause:
                    covered_nodes.add(node_idx)
                    break

                # --- 2. Direct Flip Check: actually apply interventions and verify ---
                # Separate different types of interventions
                feature_interventions = {interv for interv in clause if interv[0] == 'feature'}
                edge_interventions = {interv for interv in clause
                                     if interv[0] in ('reduce_feature_ratio', 'increase_feature_ratio')}

                # Skip if no interventions
                if not feature_interventions and not edge_interventions:
                    continue

                # Apply interventions
                data_modified = data.clone()

                # Apply feature interventions
                if feature_interventions:
                    data_modified = apply_interventions(data_modified, node_idx, feature_interventions, feature_partitions)

                # Apply edge interventions (actual edge deletion/addition)
                if edge_interventions:
                    data_modified = apply_edge_interventions(data_modified, node_idx, edge_interventions)

                # Run model to check if prediction flips
                out = model(data_modified.x, data_modified.edge_index)
                pred = (out[node_idx] > 0.5).item()

                if not pred:  # Prediction flipped to negative
                    covered_nodes.add(node_idx)
                    break

    return len(covered_nodes)


# ============================================================
# Original function (commented out) - for neighbor_mean conditions
# ============================================================
# def _check_neighbor_compatibility(node_conds: Set[Tuple], clause_conds: Set[Tuple]) -> bool:
#     """
#     Check if clause's neighbor conditions are stricter than or equal to node's neighbor conditions.
#     """
#     for clause_cond in clause_conds:
#         _, feat_idx, operator, clause_threshold = clause_cond
#
#         node_matching = [nc for nc in node_conds if nc[1] == feat_idx]
#
#         if not node_matching:
#             continue
#
#         node_cond = node_matching[0]
#         _, _, node_op, node_threshold = node_cond
#
#         if operator != node_op:
#             return False
#
#         if operator == '>':
#             if clause_threshold < node_threshold:
#                 return False
#         elif operator == '<':
#             if clause_threshold > node_threshold:
#                 return False
#
#     return True


# ============================================================
# New function - supports reduce_feature_ratio and increase_feature_ratio
# ============================================================
def _check_neighbor_compatibility(node_conds: Set[Tuple], clause_conds: Set[Tuple]) -> bool:
    """
    Check if clause's conditions are stricter than or equal to node's conditions.

    Supported condition types:
    - ('neighbor_mean', feat_idx, operator, threshold) - original format
    - ('reduce_feature_ratio', feat_idx, ratio) - new format (reduction ratio)
    - ('increase_feature_ratio', feat_idx, ratio) - new format (increase ratio)

    For reduce_feature_ratio: higher ratio means stricter (remove more)
    For increase_feature_ratio: higher ratio means stricter (add more)

    Args:
        node_conds: Node's conditions (from its counterfactual C_j)
        clause_conds: Clause's conditions (C_i)

    Returns:
        True if clause conditions are stricter or equal to node conditions
    """
    # Separate different types of conditions
    def get_conds_by_type(conds):
        neighbor_mean = {}
        reduce_ratio = {}
        increase_ratio = {}
        for cond in conds:
            if cond[0] == 'neighbor_mean':
                neighbor_mean[cond[1]] = (cond[2], cond[3])  # feat_idx -> (operator, threshold)
            elif cond[0] == 'reduce_feature_ratio':
                reduce_ratio[cond[1]] = cond[2]  # feat_idx -> ratio
            elif cond[0] == 'increase_feature_ratio':
                increase_ratio[cond[1]] = cond[2]  # feat_idx -> ratio
        return neighbor_mean, reduce_ratio, increase_ratio

    node_nm, node_reduce, node_increase = get_conds_by_type(node_conds)
    clause_nm, clause_reduce, clause_increase = get_conds_by_type(clause_conds)

    # Check reduce_feature_ratio conditions
    for feat_idx, clause_ratio in clause_reduce.items():
        if feat_idx in node_reduce:
            node_ratio = node_reduce[feat_idx]
            # clause ratio must be >= node ratio (stricter = remove more)
            if clause_ratio < node_ratio:
                return False
        # If node doesn't have this feature condition, clause having stricter condition is OK

    # Check increase_feature_ratio conditions
    for feat_idx, clause_ratio in clause_increase.items():
        if feat_idx in node_increase:
            node_ratio = node_increase[feat_idx]
            # clause ratio must be >= node ratio (stricter = add more)
            if clause_ratio < node_ratio:
                return False

    # Check neighbor_mean conditions (backward compatible with old format)
    for feat_idx, (clause_op, clause_th) in clause_nm.items():
        if feat_idx in node_nm:
            node_op, node_th = node_nm[feat_idx]
            if clause_op != node_op:
                return False
            if clause_op == '>':
                if clause_th < node_th:
                    return False
            elif clause_op == '<':
                if clause_th > node_th:
                    return False

    return True

# def _check_neighbor_compatibility(node_conds: Set[Tuple], clause_conds: Set[Tuple], tolerance: float = 0.5) -> bool:
#     """
#     Check if clause's neighbor conditions are partially compatible with node's neighbor conditions.
    
#     For coverage: clause C_i covers node x_j if C_i's neighbor requirements are stricter/equal to C_j's,
#     OR if they are within a small tolerance.
    
#     Args:
#         node_conds: Node's neighbor conditions (from its counterfactual C_j)
#         clause_conds: Clause's neighbor conditions (C_i)
#         tolerance: Allowable numerical difference for threshold matching
    
#     Returns:
#         True if clause conditions are compatible with node conditions
#     """
#     # For each condition in the clause, check if clause is stricter than node
#     for clause_cond in clause_conds:
#         _, feat_idx, operator, clause_threshold = clause_cond
        
#         # Find if node has a condition on the same feature
#         node_matching = [nc for nc in node_conds if nc[1] == feat_idx]
        
#         if not node_matching:
#             # Node has no constraint on this feature, so clause's constraint is "stricter" (adds a constraint)
#             # Compatibility assumption: adding a constraint that the node didn't need is generally OK 
#             # if we assume the node's original explanation was just "minimal sufficient".
#             continue
        
#         # Check if clause's condition is stricter than node's condition
#         node_cond = node_matching[0]  # Should only be one per feature
#         _, _, node_op, node_threshold = node_cond
        
#         # Must have same direction
#         if operator != node_op:
#             return False
        
#         # Check if clause's condition is strictly stricter OR within tolerance
#         if operator == '>':
#             # Stricter means higher threshold. 
#             # Relaxed: Clause can be slightly lower than node's threshold by `tolerance`
#             # e.g., Clause > 0.45 covers Node > 0.5 (diff 0.05 < 0.1)
#             if clause_threshold < (node_threshold - tolerance):
#                 return False
#         elif operator == '<':
#             # Stricter means lower threshold.
#             # Relaxed: Clause can be slightly higher than node's threshold by `tolerance`
#             # e.g., Clause < 0.55 covers Node < 0.5 (diff 0.05 < 0.1)
#             if clause_threshold > (node_threshold + tolerance):
#                 return False
    
#     return True

def marginal_coverage(
    S: Union[Set[frozenset], List[frozenset]], 
    new_clause: frozenset, 
    data: Data,
    positive_node_indices: List[int],
    counterfactuals: List[Set[Tuple]],
    model: nn.Module,
    device: str = 'cpu',
    feature_partitions: Optional[List[List[int]]] = None
) -> int:
    """
    Compute marginal coverage gain of adding new_clause to S.
    
    Args:
        S: Current set/list of selected clauses
        new_clause: Candidate clause to add
        data: PyTorch Geometric Data object
        positive_node_indices: List of positive node indices
        counterfactuals: List of original counterfactuals for each positive node
        model: Trained GNN model
        device: Device to run the model on
        feature_partitions: Optional groupings for one-hot features
    
    Returns:
        Marginal coverage gain (number of newly covered nodes)
    """
    S_set = set(S)
    coverage_before = compute_coverage(S_set, data, positive_node_indices, counterfactuals, model, device, feature_partitions)
    coverage_after = compute_coverage(S_set | {new_clause}, data, positive_node_indices, counterfactuals, model, device, feature_partitions)
    return coverage_after - coverage_before


def greedy_intervention_selection(
    counterfactuals: List[Set[Tuple]], 
    data: Data,
    positive_node_indices: List[int],
    model: nn.Module,
    budget: float,
    epsilon: float = 0.1,
    use_edges: bool = True,
    device: str = 'cpu',
    verbose: bool = True,
    feature_names: Optional[List[str]] = None,
    feature_partitions: Optional[List[List[int]]] = None
) -> Tuple[List[frozenset], float, List[int], List[float]]:
    """
    Greedy algorithm for intervention design with budget constraint.
    Selects clauses (counterfactuals) to maximize coverage on a graph.
    Achieves (1-ε, 1+ln(1/ε)) bicriteria approximation.
    
    Args:
        counterfactuals: List of counterfactual sets (one per positive node)
                        Each counterfactual is a set of intervention tuples
        data: PyTorch Geometric Data object
        positive_node_indices: List of positive node indices
        model: Trained GNN model
        budget: Budget constraint B
        epsilon: Approximation parameter
        use_edges: Whether edge interventions were used (affects interpretation only)
        device: Device to run the model on
        verbose: Whether to print progress
        feature_names: Optional list of names for each feature index
        feature_partitions: Optional list of index groups for one-hot encoded features
    
    Returns:
        S: Selected set of clauses (each clause is a frozenset of intervention tuples)
        total_cost: Total cost g(S) = sum of clause sizes
        coverage_history: List of coverage at each iteration
        cost_history: List of cumulative cost at each iteration
    """
    S = []  # List of selected clauses (in order)
    total_cost = 0.0
    budget_limit = budget * np.log(1 / epsilon)
    coverage_history = [0]
    cost_history = [0.0]
    n_nodes = len(positive_node_indices)
    
    # Convert counterfactuals to frozensets and compute costs
    candidate_clauses = {frozenset(cf) for cf in counterfactuals if len(cf) > 0}
    
    # Augment candidate clauses if feature partitions are provided
    if feature_partitions:
        # Create a mapping from index to its partition
        idx_to_partition = {}
        for p in feature_partitions:
            if len(p) > 1:
                for idx in p:
                    idx_to_partition[idx] = p
        
        augmented_clauses = set()
        for clause in candidate_clauses:
            clause_list = list(clause)
            for i, interv in enumerate(clause_list):
                if interv[0] == 'node':
                    idx, val = interv[1], interv[2]
                    # If turning a dummy feature ON, try turning on others in same partition
                    if val == 1.0 and idx in idx_to_partition:
                        partition = idx_to_partition[idx]
                        for alt_idx in partition:
                            if alt_idx != idx:
                                # Create new clause with swapped feature
                                new_clause_list = clause_list.copy()
                                new_clause_list[i] = ('feature', alt_idx, 1.0)
                                augmented_clauses.add(frozenset(new_clause_list))
        
        if augmented_clauses:
            if verbose:
                print(f"Augmenting candidates: Added {len(augmented_clauses)} partition-aware variations")
            candidate_clauses.update(augmented_clauses)

    clause_costs = {clause: len(clause) for clause in candidate_clauses}
    
    iteration = 0
    max_iterations = len(candidate_clauses)
    
    if verbose:
        print(f"Starting greedy selection (budget={budget}, budget_limit={budget_limit:.2f})")
        print(f"Total positive nodes: {n_nodes}")
        print(f"Total candidate clauses: {len(candidate_clauses)}")
        print("-" * 60)
    
    while iteration < max_iterations:
        # Check if all nodes are covered
        coverage = compute_coverage(set(S), data, positive_node_indices, counterfactuals, model, device, feature_partitions)
        if coverage == n_nodes:
            if verbose:
                print(f"All nodes covered!")
            break
        
        # Check budget constraint
        if total_cost >= budget_limit:
            if verbose:
                print(f"Budget limit reached: {total_cost:.2f} >= {budget_limit:.2f}")
            break
        
        # Find clause with maximum marginal coverage per unit cost
        best_clause = None
        best_ratio = -1
        
        remaining_clauses = candidate_clauses - set(S)
        if not remaining_clauses:
            break
        
        for clause in remaining_clauses:
            cost = clause_costs[clause]
            if total_cost + cost > budget_limit:
                continue
            
            marginal = marginal_coverage(S, clause, data, positive_node_indices, counterfactuals, model, device, feature_partitions)
            if cost > 0 and marginal > 0:
                ratio = marginal / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_clause = clause
        
        if best_clause is None:
            if verbose:
                print("No more beneficial clauses within budget")
                # Debug: check remaining clauses
                remaining_clauses = candidate_clauses - set(S)
                print(f"  Remaining clauses: {len(remaining_clauses)}")
                for clause in remaining_clauses:
                    cost = clause_costs[clause]
                    if total_cost + cost <= budget_limit:
                        marginal = marginal_coverage(S, clause, data, positive_node_indices, counterfactuals, model, device, feature_partitions)
                        ratio = marginal / cost if cost > 0 else 0
                        print(f"    {format_interventions(clause, feature_names, feature_partitions)}: marginal={marginal}, cost={cost}, ratio={ratio:.3f}")
                    else:
                        print(f"    {format_interventions(clause, feature_names, feature_partitions)}: exceeds budget (cost={cost})")
            break
        
        # Add best clause to S
        S.append(best_clause)
        total_cost += clause_costs[best_clause]
        coverage = compute_coverage(set(S), data, positive_node_indices, counterfactuals, model, device, feature_partitions)
        coverage_history.append(coverage)
        cost_history.append(total_cost)
        
        if verbose:
            clause_str = format_interventions(best_clause, feature_names, feature_partitions)
            print(f"Iter {iteration+1}: Added clause ({clause_str}) "
                  f"(cost={clause_costs[best_clause]:.2f}, ratio={best_ratio:.3f}) | "
                  f"Coverage: {coverage}/{n_nodes} | "
                  f"Total cost: {total_cost:.2f}")
        
        iteration += 1
    
    if verbose:
        final_coverage = compute_coverage(S, data, positive_node_indices, counterfactuals, model, device, feature_partitions)
        print("-" * 60)
        print(f"Final results:")
        print(f"  Selected clauses:")
        for i, clause in enumerate(S, 1):
            clause_str = format_interventions(clause, feature_names, feature_partitions)
            print(f"    Clause {i}: ({clause_str})")
        print(f"  Number of clauses: {len(S)}")
        print(f"  Coverage: {final_coverage}/{n_nodes} "
              f"({100*final_coverage/n_nodes:.1f}%)")
        print(f"  Total cost: {total_cost:.2f} / {budget_limit:.2f}")
    
    return S, total_cost, coverage_history, cost_history


def frequency_intervention_selection(
    counterfactuals: List[Set[Tuple]],
    data: Data,
    positive_node_indices: List[int],
    model: nn.Module,
    budget: float,
    epsilon: float = 0.1,
    feature_names: List[str] = None,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[Set[frozenset], float, List[int]]:
    """
    Baseline: Pick top-k most frequent individual interventions.
    Creates a DNF where each clause has a single intervention.
    Stops early if all nodes are covered.
    """
    budget_limit = int(budget * np.log(1 / epsilon))
    n_nodes = len(positive_node_indices)

    # Flatten all interventions and count frequencies
    all_interv = []
    for cf in counterfactuals:
        for interv in cf:
            all_interv.append(interv)

    counts = Counter(all_interv)
    # Sort by frequency descending
    sorted_interv = [interv for interv, count in counts.most_common()]

    # Select top-k within budget, but stop early if all nodes covered
    coverage_history = [0]
    cost_history = [0.0]
    current_S = set()
    current_cost = 0.0

    for interv in sorted_interv[:budget_limit]:
        clause = frozenset({interv})
        current_S.add(clause)
        current_cost += len(clause)
        cov = compute_coverage(current_S, data, positive_node_indices, counterfactuals, model, device)
        coverage_history.append(cov)
        cost_history.append(current_cost)

        # Early stop if all nodes covered
        if cov == n_nodes:
            if verbose:
                print(f"All nodes covered!")
            break

    S = current_S
    total_cost = current_cost

    if verbose:
        print(f"Frequency Baseline: Selected {len(S)} single-feature clauses. Coverage: {coverage_history[-1]}/{n_nodes}")
        if len(S) > 0:
            print("  Selected clauses:")
            for i, clause in enumerate(S, 1):
                # We don't have partitions in the baseline function signature yet,
                # but we can add it or just use the default.
                print(f"    Clause {i}: ({format_interventions(clause, feature_names)})")

    return S, total_cost, coverage_history, cost_history


def random_intervention_selection(
    counterfactuals: List[Set[Tuple]],
    data: Data,
    positive_node_indices: List[int],
    model: nn.Module,
    budget: float,
    epsilon: float = 0.1,
    feature_names: List[str] = None,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[Set[frozenset], float, List[int]]:
    """
    Baseline: Pick k individual interventions at random from those present in CFs.
    Creates a DNF where each clause has a single intervention.
    Stops early if all nodes are covered.
    """
    budget_limit = int(budget * np.log(1 / epsilon))
    n_nodes = len(positive_node_indices)

    # Get set of all unique interventions
    unique_interv = set()
    for cf in counterfactuals:
        for interv in cf:
            unique_interv.add(interv)

    unique_interv_list = list(unique_interv)
    # Randomly shuffle and select up to k
    random.shuffle(unique_interv_list)
    size = min(len(unique_interv_list), budget_limit)

    # Select interventions one by one, stop early if all nodes covered
    coverage_history = [0]
    cost_history = [0.0]
    current_S = set()
    current_cost = 0.0

    for interv in unique_interv_list[:size]:
        clause = frozenset({interv})
        current_S.add(clause)
        current_cost += len(clause)
        cov = compute_coverage(current_S, data, positive_node_indices, counterfactuals, model, device)
        coverage_history.append(cov)
        cost_history.append(current_cost)

        # Early stop if all nodes covered
        if cov == n_nodes:
            if verbose:
                print(f"All nodes covered!")
            break

    S = current_S
    total_cost = current_cost

    if verbose:
        print(f"Random Baseline: Selected {len(S)} single-feature clauses. Coverage: {coverage_history[-1]}/{n_nodes}")
        if len(S) > 0:
            print("  Selected clauses:")
            for i, clause in enumerate(S, 1):
                print(f"    Clause {i}: ({format_interventions(clause, feature_names)})")

    return S, total_cost, coverage_history, cost_history


def format_interventions(
    interventions: Set[Tuple],
    feature_names: Optional[List[str]] = None,
    feature_partitions: Optional[List[List[int]]] = None
) -> str:
    """
    Format a set of interventions as a readable string.
    
    Args:
        interventions: Set of intervention tuples
        feature_names: Optional list of names for each feature index
        feature_partitions: Optional groupings for one-hot features to summarize swaps
    
    Returns:
        Formatted string like "X1=0, N(X2)>0.5, drop_edge(5)"
    """
    parts = []
    processed_interventions = set(interventions)
    
    def get_name(idx):
        if feature_names and idx < len(feature_names):
            return feature_names[idx]
        return f"X{idx}"

    # Handle one-hot partitions if provided
    if feature_partitions:
        for partition in feature_partitions:
            if len(partition) <= 1:
                continue
            
            # Find feature interventions in this partition
            p_interv = [i for i in processed_interventions if i[0] == 'feature' and i[1] in partition]
            
            if len(p_interv) == 1 and p_interv[0][2] == 1:
                # Single "ON" change in a partition: "Set ed_3.0"
                parts.append(f"Set {get_name(p_interv[0][1])}")
                processed_interventions.remove(p_interv[0])
            elif len(p_interv) >= 2:
                # Group multiple changes (e.g. from 1 to 0 and 0 to 1)
                turned_off = [i for i in p_interv if i[2] == 0]
                turned_on = [i for i in p_interv if i[2] == 1]
                
                if turned_off and turned_on:
                    parts.append(f"change {get_name(turned_off[0][1])} to {get_name(turned_on[0][1])}")
                elif turned_on:
                    parts.append(f"Set {get_name(turned_on[0][1])}")
                elif turned_off:
                    parts.append(f"Clear {get_name(turned_off[0][1])}")
                
                for i in p_interv:
                    processed_interventions.remove(i)
            
            # Do same for neighbor features
            for neigh_idx in set(i[1] for i in processed_interventions if i[0] == 'neighbor_feature'):
                n_intervs = [i for i in processed_interventions 
                            if i[0] == 'neighbor_feature' and i[1] == neigh_idx and i[2] in partition]
                if not n_intervs: continue
                
                if len(n_intervs) == 1 and n_intervs[0][3] == 1:
                    parts.append(f"N{neigh_idx}: Set {get_name(n_intervs[0][2])}")
                    processed_interventions.remove(n_intervs[0])
                elif len(n_intervs) >= 2:
                    turned_off = [i for i in n_intervs if i[3] == 0]
                    turned_on = [i for i in n_intervs if i[3] == 1]
                    if turned_off and turned_on:
                        parts.append(f"N{neigh_idx}: change {get_name(turned_off[0][2])} to {get_name(turned_on[0][2])}")
                    elif turned_on:
                        parts.append(f"N{neigh_idx}: Set {get_name(turned_on[0][2])}")
                    
                    for i in n_intervs:
                        processed_interventions.remove(i)

    # Process remaining interventions
    for interv in sorted(processed_interventions):
        if interv[0] == 'feature':
            parts.append(f"{get_name(interv[1])}={interv[2]}")
        elif interv[0] == 'neighbor_mean':
            feat_idx, operator, threshold = interv[1], interv[2], interv[3]
            parts.append(f"N({get_name(feat_idx)}){operator}{threshold}")
        elif interv[0] == 'reduce_feature_ratio':
            feat_idx, ratio = interv[1], interv[2]
            parts.append(f"reduce({get_name(feat_idx)}):{ratio:.0%}")
        elif interv[0] == 'increase_feature_ratio':
            feat_idx, ratio = interv[1], interv[2]
            parts.append(f"increase({get_name(feat_idx)}):{ratio:.0%}")
        elif interv[0] == 'drop_edge':
            parts.append(f"drop_edge({interv[1]})")
        elif interv[0] == 'add_edge':
            parts.append(f"add_edge({interv[1]})")
        elif interv[0] == 'neighbor_feature':
            neighbor_idx, feat_idx, target_value = interv[1], interv[2], interv[3]
            parts.append(f"N{neighbor_idx}({get_name(feat_idx)})={target_value}")

    return ", ".join(parts)


def generate_counterfactuals_batch(
    data: Data,
    positive_node_indices: List[int],
    model: nn.Module,
    max_interventions: int = 5,
    edge_mode: str = 'none',  # 'none', 'edges', 'edge_features'
    only_delete_edges: bool = False,
    flip_direction: str = 'positive_to_negative',
    device: str = 'cpu',
    verbose: bool = False,
    ignore_node_features: bool = False,  # Whether to ignore node feature interventions
    feature_partitions: Optional[List[List[int]]] = None,  # Groupings for one-hot encodings
    unchangeable_features: Optional[List[int]] = None,  # List of feature indices that cannot be perturbed
    forbidden_target_features: Optional[List[int]] = None # List of feature indices that cannot be suggested as targets
) -> List[Set[Tuple]]:
    """
    Generate counterfactuals for a batch of nodes in a graph.
    
    Args:
        data: PyTorch Geometric Data object
        positive_node_indices: List of node indices to generate CFs for
        model: Trained GNN model
        max_interventions: Maximum number of interventions to consider per node
        edge_mode: How to handle edges/neighbors - 'none' (ignore), 'edges' (add/remove edges), 'edge_features' (flip neighbor features)
        only_delete_edges: If True and edge_mode=='edges', only consider edge deletions (not additions)
        flip_direction: 'positive_to_negative' or 'negative_to_positive'
        device: Device to run the model on
        verbose: Whether to print progress
        ignore_node_features: If True, skip node feature interventions and only consider edge/neighbor interventions
        feature_partitions: Optional groupings for one-hot encodings
        unchangeable_features: Optional list of feature indices that are immutable.
        forbidden_target_features: Optional list of feature indices that cannot be suggested as targets.
    
    Returns:
        List of counterfactual sets, one for each node
    """
    model.eval()
    data = data.to(device)
    counterfactuals = []
    
    if verbose:
        direction_str = "positive→negative" if flip_direction == 'positive_to_negative' else "negative→positive"
        print(f"Generating counterfactuals for {len(positive_node_indices)} nodes ({direction_str})...")
    
    for idx, node_idx in enumerate(positive_node_indices):
        cf = get_local_counterfactual(
            node_idx, data, model, max_interventions,
            edge_mode=edge_mode,
            only_delete_edges=only_delete_edges,
            flip_direction=flip_direction,
            device=device,
            ignore_node_features=ignore_node_features,
            feature_partitions=feature_partitions,
            unchangeable_features=unchangeable_features,
            forbidden_target_features=forbidden_target_features
        )
        counterfactuals.append(cf)
        if verbose and (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(positive_node_indices)} nodes")
    
    if verbose:
        num_cfs = sum(1 for cf in counterfactuals if len(cf) > 0)
        print(f"Generated {num_cfs} counterfactuals")
    
    return counterfactuals
