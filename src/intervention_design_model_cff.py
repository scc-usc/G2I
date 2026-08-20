from typing import Set, Tuple, List, Optional
import torch
from torch_geometric.data import Data


# ============================================================
# Original function (commented out) - uses neighbor_mean conditions
# ============================================================
# def convert_edge_changes_to_neighbor_conditions_cff(
#     node_idx: int,
#     data: Data,
#     edge_interventions: Set[Tuple],
#     diff_threshold: float = 0.1
# ) -> Set[Tuple]:
#     """
#     Convert edge interventions (drop_edge) into interpretable neighbor feature conditions.
#     """
#     if len(edge_interventions) == 0:
#         return set()
#
#     edge_index = data.edge_index
#     incoming_mask = edge_index[1] == node_idx
#     current_neighbors = set(edge_index[0][incoming_mask].tolist())
#     outgoing_mask = edge_index[0] == node_idx
#     current_neighbors.update(edge_index[1][outgoing_mask].tolist())
#
#     new_neighbors = set(current_neighbors)
#     for intervention in edge_interventions:
#         if intervention[0] == 'drop_edge':
#             new_neighbors.discard(intervention[1])
#
#     if len(current_neighbors) == 0 or len(new_neighbors) == 0:
#         return set()
#
#     current_neighbors_list = list(current_neighbors)
#     new_neighbors_list = list(new_neighbors)
#
#     current_neighbor_features = data.x[current_neighbors_list].mean(dim=0)
#     new_neighbor_features = data.x[new_neighbors_list].mean(dim=0)
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
def _discretize_ratio_cff(ratio: float, levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) -> float:
    """Discretize ratio to specified levels"""
    for level in levels:
        if ratio <= level:
            return level
    return levels[-1]


def convert_edge_changes_to_neighbor_conditions_cff(
    node_idx: int,
    data: Data,
    edge_interventions: Set[Tuple],
    diff_threshold: float = 0.1,
    ratio_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
) -> Set[Tuple]:
    """
    Convert edge interventions (drop_edge) into reduce_feature_ratio conditions.

    Uses reduce_feature_ratio format:
    - ('reduce_feature_ratio', feature_idx, discretized_ratio)
    - ratio indicates the proportion of neighbors with the feature to remove

    Args:
        node_idx: Index of the node
        data: PyTorch Geometric Data object
        edge_interventions: Set of edge intervention tuples (('drop_edge', neighbor))
        diff_threshold: Unused, kept for compatibility
        ratio_levels: Discretization levels, default [0.1, ..., 1.0] for 10% to 100%

    Returns:
        Set of tuples: ('reduce_feature_ratio', feature_idx, discretized_ratio)
    """
    if len(edge_interventions) == 0:
        return set()

    # Find removed neighbors
    removed_neighbors = set()
    for intervention in edge_interventions:
        if intervention[0] == 'drop_edge':
            removed_neighbors.add(intervention[1])

    if len(removed_neighbors) == 0:
        return set()

    # Get all current neighbors
    edge_index = data.edge_index
    current_neighbors = set()
    for i in range(edge_index.shape[1]):
        if edge_index[0, i].item() == node_idx:
            current_neighbors.add(edge_index[1, i].item())
        if edge_index[1, i].item() == node_idx:
            current_neighbors.add(edge_index[0, i].item())

    if len(current_neighbors) == 0:
        return set()

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
        discretized = _discretize_ratio_cff(ratio, ratio_levels)

        interventions.add(('reduce_feature_ratio', feat_idx, discretized))

    return interventions


def convert_neighbor_feature_changes_to_neighbor_conditions_cff(
    node_idx: int,
    data: Data,
    neighbor_feature_interventions: Set[Tuple],
    diff_threshold: float = 0.1
) -> Set[Tuple]:
    """
    Convert neighbor feature changes into interpretable neighbor feature conditions.
    
    Args:
        node_idx: Index of the node
        data: PyTorch Geometric Data object
        neighbor_feature_interventions: Set of neighbor feature intervention tuples 
                                       (('neighbor_feature', neighbor_idx, feat_idx, target_value))
        diff_threshold: Minimum difference in feature distribution to create a condition
    
    Returns:
        Set of neighbor condition tuples: ('neighbor_mean', feature_idx, operator, threshold)
    """
    if len(neighbor_feature_interventions) == 0:
        return set()
    
    # Get current neighbors
    edge_index = data.edge_index
    incoming_mask = edge_index[1] == node_idx
    current_neighbors = set(edge_index[0][incoming_mask].tolist())
    outgoing_mask = edge_index[0] == node_idx
    current_neighbors.update(edge_index[1][outgoing_mask].tolist())
    
    if len(current_neighbors) == 0:
        return set()
    
    # Identify which features were actually changed
    changed_features = set()
    for interv in neighbor_feature_interventions:
        if interv[0] == 'neighbor_feature':
            changed_features.add(interv[2])  # feat_idx
    
    # Create modified features
    modified_x = data.x.clone()
    for interv in neighbor_feature_interventions:
        if interv[0] == 'neighbor_feature':
            neighbor_idx, feat_idx, target_value = interv[1], interv[2], interv[3]
            modified_x[neighbor_idx, feat_idx] = target_value
    
    # Compute distributions before and after
    current_neighbors_list = list(current_neighbors)
    current_neighbor_features = data.x[current_neighbors_list].mean(dim=0)
    new_neighbor_features = modified_x[current_neighbors_list].mean(dim=0)
    
    # Find features with significant distribution changes (only among changed features)
    neighbor_conditions = set()
    for feat_idx in changed_features:
        diff = new_neighbor_features[feat_idx].item() - current_neighbor_features[feat_idx].item()
        
        if abs(diff) > diff_threshold:
            if diff > 0:
                operator = '>'
                threshold = new_neighbor_features[feat_idx].item()
            else:
                operator = '<'
                threshold = new_neighbor_features[feat_idx].item()
            
            neighbor_conditions.add(('neighbor_mean', feat_idx, operator, round(threshold, 3)))
    
    return neighbor_conditions


def get_local_counterfactual_cff(
    node_idx: int,
    data: Data,
    feat_mask: Optional[torch.Tensor] = None,  # Shape: [num_nodes, feat_dim]
    edge_mask: Optional[torch.Tensor] = None,  # Shape: [num_edges]
    feat_TH: float = 0.5,
    edge_TH: float = 0.5,
    feature_partitions: Optional[List[List[int]]] = None,
    flip_direction: str = 'positive_to_negative',
    include_neighbor_features: bool = True,
    convert_to_neighbor_conditions: bool = True,  # Whether to convert edge/neighbor interventions to neighbor_mean
    diff_threshold: float = 0.1,  # Threshold for neighbor condition conversion
    unchangeable_indices: Optional[List[int]] = None  # Feature indices that cannot be changed
) -> Set[Tuple]:
    """
    Generate counterfactual interventions based on CFF explanation masks.

    Supports three modes based on which masks are provided:
    1. feature mode: only feat_mask provided
    2. edge mode: only edge_mask provided
    3. feature+edge mode: both masks provided

    Args:
        node_idx: Index of the target node
        data: PyTorch Geometric Data object (x, edge_index, y)
        feat_mask: Feature importance mask from CFF, shape [num_nodes, feat_dim].
        edge_mask: Edge importance mask from CFF, shape [num_edges].
        feat_TH: Threshold for feature importance (default 0.5)
        edge_TH: Threshold for edge importance (default 0.5)
        feature_partitions: Optional list of lists for one-hot encoded features.
        flip_direction: 'positive_to_negative' or 'negative_to_positive'
        include_neighbor_features: If True and edge_mask is provided, also generate
                                   neighbor feature interventions for neighbors on important edges.
        convert_to_neighbor_conditions: If True, convert edge/neighbor_feature interventions
                                        to neighbor_mean conditions (like intervention_design_model.py)
        diff_threshold: Minimum difference for neighbor condition conversion
        unchangeable_indices: List of feature indices that cannot be changed (will be skipped)

    Returns:
        Set of intervention tuples:
        - ('feature', feat_idx, target_value) for self feature changes
        - If convert_to_neighbor_conditions=False:
          - ('drop_edge', neighbor_idx) for edge removals
          - ('neighbor_feature', neighbor_idx, feat_idx, target_value) for neighbor feature changes
        - If convert_to_neighbor_conditions=True:
          - ('neighbor_mean', feat_idx, operator, threshold) for neighbor conditions
    """
    cf: Set[Tuple] = set()
    edge_interventions: Set[Tuple] = set()
    neighbor_feature_interventions: Set[Tuple] = set()

    n_features = int(data.x.shape[1])

    # Convert unchangeable_indices to set for fast lookup
    unchangeable_set = set(unchangeable_indices) if unchangeable_indices else set()

    # Default partitions to singletons if not provided
    if feature_partitions is None:
        feature_partitions = [[i] for i in range(n_features)]
    
    # Build partition lookup: feat_idx -> partition
    idx_to_partition = {}
    for partition in feature_partitions:
        for idx in partition:
            idx_to_partition[idx] = partition
    
    # =========================================================================
    # 1. Process Feature Mask -> Self Feature Interventions
    # =========================================================================
    if feat_mask is not None:
        node_feat_mask = feat_mask[node_idx]  # Shape: [feat_dim]
        node_features = data.x[node_idx]      # Shape: [feat_dim]

        processed_partitions = set()

        for feat_idx in range(n_features):
            # Skip unchangeable features
            if feat_idx in unchangeable_set:
                continue

            importance = float(node_feat_mask[feat_idx].item())

            if importance <= feat_TH:
                continue

            partition = idx_to_partition.get(feat_idx, [feat_idx])
            partition_key = tuple(sorted(partition))

            if partition_key in processed_partitions:
                continue

            # Check if any feature in this partition is unchangeable
            partition_has_unchangeable = any(p_idx in unchangeable_set for p_idx in partition)
            if partition_has_unchangeable:
                continue

            processed_partitions.add(partition_key)

            # Case 1: Continuous/Binary feature (partition length == 1)
            if len(partition) == 1:
                orig = float(node_features[feat_idx].item())
                if orig > 0.5:
                    cf.add(('feature', feat_idx, 0))
                else:
                    cf.add(('feature', feat_idx, 1))

            # Case 2: One-Hot encoded feature (partition length > 1)
            else:
                current_on_idx = None
                for p_idx in partition:
                    if float(node_features[p_idx].item()) > 0.5:
                        current_on_idx = p_idx
                        break

                best_target = None
                best_importance = -1

                for p_idx in partition:
                    if p_idx == current_on_idx:
                        continue
                    # Skip unchangeable features in partition
                    if p_idx in unchangeable_set:
                        continue
                    p_importance = float(node_feat_mask[p_idx].item())
                    if p_importance > best_importance:
                        best_importance = p_importance
                        best_target = p_idx

                if best_target is not None:
                    cf.add(('feature', best_target, 1))
    
    # =========================================================================
    # 2. Process Edge Mask -> Edge Interventions (drop_edge)
    # =========================================================================
    important_neighbors = set()
    
    if edge_mask is not None:
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        
        for edge_idx in range(num_edges):
            src = int(edge_index[0, edge_idx].item())
            dst = int(edge_index[1, edge_idx].item())
            
            if src != node_idx and dst != node_idx:
                continue
            
            importance = float(edge_mask[edge_idx].item())
            
            if importance > edge_TH:
                neighbor = dst if src == node_idx else src
                edge_interventions.add(('drop_edge', neighbor))
                important_neighbors.add(neighbor)
    
    # =========================================================================
    # 3. Process Neighbor Features (for neighbors on important edges)
    # =========================================================================
    if include_neighbor_features and feat_mask is not None and len(important_neighbors) > 0:
        for neighbor_idx in important_neighbors:
            neighbor_feat_mask = feat_mask[neighbor_idx]
            neighbor_features = data.x[neighbor_idx]

            processed_partitions = set()

            for feat_idx in range(n_features):
                # Skip unchangeable features
                if feat_idx in unchangeable_set:
                    continue

                importance = float(neighbor_feat_mask[feat_idx].item())

                if importance <= feat_TH:
                    continue

                partition = idx_to_partition.get(feat_idx, [feat_idx])
                partition_key = tuple(sorted(partition))

                if partition_key in processed_partitions:
                    continue

                # Check if any feature in this partition is unchangeable
                partition_has_unchangeable = any(p_idx in unchangeable_set for p_idx in partition)
                if partition_has_unchangeable:
                    continue

                processed_partitions.add(partition_key)

                if len(partition) == 1:
                    orig = float(neighbor_features[feat_idx].item())
                    if orig > 0.5:
                        neighbor_feature_interventions.add(('neighbor_feature', neighbor_idx, feat_idx, 0))
                    else:
                        neighbor_feature_interventions.add(('neighbor_feature', neighbor_idx, feat_idx, 1))
                else:
                    current_on_idx = None
                    for p_idx in partition:
                        if float(neighbor_features[p_idx].item()) > 0.5:
                            current_on_idx = p_idx
                            break

                    best_target = None
                    best_importance = -1

                    for p_idx in partition:
                        if p_idx == current_on_idx:
                            continue
                        # Skip unchangeable features in partition
                        if p_idx in unchangeable_set:
                            continue
                        p_importance = float(neighbor_feat_mask[p_idx].item())
                        if p_importance > best_importance:
                            best_importance = p_importance
                            best_target = p_idx

                    if best_target is not None:
                        neighbor_feature_interventions.add(('neighbor_feature', neighbor_idx, best_target, 1))
    
    # =========================================================================
    # 4. Final Output: Convert or Keep Raw Interventions
    # =========================================================================
    if convert_to_neighbor_conditions:
        # Keep only feature interventions, convert edge/neighbor to neighbor_mean
        cf_final = set(interv for interv in cf if interv[0] == 'feature')
        
        # Convert edge interventions to neighbor conditions
        if len(edge_interventions) > 0:
            neighbor_conds = convert_edge_changes_to_neighbor_conditions_cff(
                node_idx, data, edge_interventions, diff_threshold
            )
            cf_final.update(neighbor_conds)
        
        # Convert neighbor feature interventions to neighbor conditions
        if len(neighbor_feature_interventions) > 0:
            neighbor_conds = convert_neighbor_feature_changes_to_neighbor_conditions_cff(
                node_idx, data, neighbor_feature_interventions, diff_threshold
            )
            cf_final.update(neighbor_conds)
        
        return cf_final
    else:
        # Return raw interventions without conversion
        cf.update(edge_interventions)
        cf.update(neighbor_feature_interventions)
        return cf


def format_cff_interventions(
    interventions: Set[Tuple],
    feature_names: Optional[List[str]] = None,
    feature_partitions: Optional[List[List[int]]] = None
) -> str:
    """
    Format a set of CFF interventions as a readable string.
    
    Args:
        interventions: Set of intervention tuples
        feature_names: Optional list of names for each feature index
        feature_partitions: Optional groupings for one-hot features
    
    Returns:
        Formatted string like "X1=0, N(X2)>0.5, drop_edge(5)"
    """
    if len(interventions) == 0:
        return "No interventions"
    
    parts = []
    
    def get_name(idx):
        if feature_names and idx < len(feature_names):
            return feature_names[idx]
        return f"X{idx}"
    
    for interv in sorted(interventions):
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


def generate_counterfactuals_batch_cff(
    data: Data,
    cff_results: List[dict],
    node_list: Optional[List[int]] = None,
    feat_TH: float = 0.5,
    edge_TH: float = 0.5,
    feature_partitions: Optional[List[List[int]]] = None,
    include_neighbor_features: bool = True,
    convert_to_neighbor_conditions: bool = True,
    diff_threshold: float = 0.1,
    verbose: bool = False,
    unchangeable_indices: Optional[List[int]] = None
) -> List[Set[Tuple]]:
    """
    Generate counterfactual interventions for a batch of CFF results.

    Output format matches generate_counterfactuals_batch in intervention_design_model.py:
    Returns List[Set[Tuple]], one set per node.

    Supports three modes based on which masks are in cff_results:
    1. feature mode: only feat_mask provided
    2. edge mode: only edge_mask provided
    3. feature+edge mode: both masks provided

    Args:
        data: PyTorch Geometric Data object
        cff_results: List of CFF result dictionaries containing masks
        node_list: Optional list of node indices to process
        feat_TH: Threshold for feature importance
        edge_TH: Threshold for edge importance
        feature_partitions: Optional groupings for one-hot features
        include_neighbor_features: Whether to include neighbor feature interventions
        convert_to_neighbor_conditions: If True, convert to neighbor_mean conditions
        diff_threshold: Threshold for neighbor condition conversion
        verbose: Whether to print progress
        unchangeable_indices: List of feature indices that cannot be changed

    Returns:
        List of counterfactual sets, one for each node (same format as generate_counterfactuals_batch)
    """
    result_lookup = {r['node_idx']: r for r in cff_results}
    
    if node_list is None:
        nodes_to_process = [r['node_idx'] for r in cff_results]
    else:
        nodes_to_process = [n for n in node_list if n in result_lookup]
        if verbose and len(nodes_to_process) < len(node_list):
            missing = set(node_list) - set(nodes_to_process)
            print(f"Warning: {len(missing)} nodes not found in cff_results")
    
    counterfactuals: List[Set[Tuple]] = []
    
    if verbose:
        print(f"Generating counterfactuals for {len(nodes_to_process)} nodes...")
    
    for i, node_idx in enumerate(nodes_to_process):
        result = result_lookup[node_idx]
        orig_pred = result.get('orig_pred', 1)
        
        flip_direction = 'positive_to_negative' if orig_pred == 1 else 'negative_to_positive'
        
        feat_mask = result.get('feat_mask', None)
        edge_mask = result.get('edge_mask', None)
        
        cf = get_local_counterfactual_cff(
            node_idx=node_idx,
            data=data,
            feat_mask=feat_mask,
            edge_mask=edge_mask,
            feat_TH=feat_TH,
            edge_TH=edge_TH,
            feature_partitions=feature_partitions,
            flip_direction=flip_direction,
            include_neighbor_features=include_neighbor_features,
            convert_to_neighbor_conditions=convert_to_neighbor_conditions,
            diff_threshold=diff_threshold,
            unchangeable_indices=unchangeable_indices
        )
        
        counterfactuals.append(cf)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(nodes_to_process)} nodes")
    
    if verbose:
        num_with_cf = sum(1 for cf in counterfactuals if len(cf) > 0)
        print(f"Generated {num_with_cf} counterfactuals")
    
    return counterfactuals


def generate_counterfactuals_batch_cff_detailed(
    data: Data,
    cff_results: List[dict],
    node_list: Optional[List[int]] = None,
    feat_TH: float = 0.5,
    edge_TH: float = 0.5,
    feature_partitions: Optional[List[List[int]]] = None,
    feature_names: Optional[List[str]] = None,
    include_neighbor_features: bool = True,
    convert_to_neighbor_conditions: bool = True,
    diff_threshold: float = 0.1,
    verbose: bool = False,
    unchangeable_indices: Optional[List[int]] = None
) -> List[dict]:
    """
    Generate counterfactual interventions with detailed information for each node.

    Similar to generate_counterfactuals_batch_cff but returns detailed dictionaries
    instead of just the intervention sets.

    Args:
        data: PyTorch Geometric Data object
        cff_results: List of CFF result dictionaries containing masks
        node_list: Optional list of node indices to process
        feat_TH: Threshold for feature importance
        edge_TH: Threshold for edge importance
        feature_partitions: Optional groupings for one-hot features
        feature_names: Optional list of feature names for formatting
        include_neighbor_features: Whether to include neighbor feature interventions
        convert_to_neighbor_conditions: If True, convert to neighbor_mean conditions
        diff_threshold: Threshold for neighbor condition conversion
        verbose: Whether to print progress
        unchangeable_indices: List of feature indices that cannot be changed

    Returns:
        List of dictionaries with detailed counterfactual information:
        - 'node_idx': node index
        - 'orig_pred': original prediction
        - 'true_label': ground truth label (if available)
        - 'mode': CFF explanation mode
        - 'flip_direction': direction of flip
        - 'interventions': set of intervention tuples
        - 'formatted': formatted string representation
    """
    result_lookup = {r['node_idx']: r for r in cff_results}
    
    if node_list is None:
        nodes_to_process = [r['node_idx'] for r in cff_results]
    else:
        nodes_to_process = [n for n in node_list if n in result_lookup]
        if verbose and len(nodes_to_process) < len(node_list):
            missing = set(node_list) - set(nodes_to_process)
            print(f"Warning: {len(missing)} nodes not found in cff_results")
    
    counterfactuals = []
    
    if verbose:
        print(f"Generating counterfactuals for {len(nodes_to_process)} nodes...")
    
    for i, node_idx in enumerate(nodes_to_process):
        result = result_lookup[node_idx]
        orig_pred = result.get('orig_pred', 1)
        true_label = result.get('true_label', None)
        mode = result.get('mode', 'unknown')
        
        flip_direction = 'positive_to_negative' if orig_pred == 1 else 'negative_to_positive'
        
        feat_mask = result.get('feat_mask', None)
        edge_mask = result.get('edge_mask', None)
        
        cf = get_local_counterfactual_cff(
            node_idx=node_idx,
            data=data,
            feat_mask=feat_mask,
            edge_mask=edge_mask,
            feat_TH=feat_TH,
            edge_TH=edge_TH,
            feature_partitions=feature_partitions,
            flip_direction=flip_direction,
            include_neighbor_features=include_neighbor_features,
            convert_to_neighbor_conditions=convert_to_neighbor_conditions,
            diff_threshold=diff_threshold,
            unchangeable_indices=unchangeable_indices
        )
        
        cf_result = {
            'node_idx': node_idx,
            'orig_pred': orig_pred,
            'true_label': true_label,
            'mode': mode,
            'flip_direction': flip_direction,
            'interventions': cf,
            'formatted': format_cff_interventions(cf, feature_names=feature_names, feature_partitions=feature_partitions)
        }
        
        counterfactuals.append(cf_result)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(nodes_to_process)} nodes")
    
    if verbose:
        num_with_cf = sum(1 for c in counterfactuals if len(c['interventions']) > 0)
        print(f"Generated counterfactuals for {num_with_cf}/{len(counterfactuals)} nodes")
    
    return counterfactuals


def print_counterfactuals_cff(
    counterfactuals: List[dict],
    feature_names: Optional[List[str]] = None
) -> None:
    """
    Print counterfactual results in a formatted way.
    
    Args:
        counterfactuals: List of counterfactual result dictionaries
        feature_names: Optional list of feature names for display
    """
    print("=" * 80)
    print("COUNTERFACTUAL INTERVENTIONS FROM CFF")
    print("=" * 80)
    
    for cf_result in counterfactuals:
        node_idx = cf_result['node_idx']
        orig_pred = cf_result['orig_pred']
        true_label = cf_result.get('true_label', '?')
        mode = cf_result.get('mode', 'unknown')
        flip_dir = cf_result['flip_direction']
        interventions = cf_result['interventions']
        formatted = cf_result['formatted']
        
        print(f"\nNode {node_idx} (pred={orig_pred}, true={true_label}, mode={mode}):")
        print(f"  Interventions: {interventions}")
        print(f"  Formatted: {formatted}")
    
    print("\n" + "=" * 80)
    num_with_cf = sum(1 for c in counterfactuals if len(c['interventions']) > 0)
    print(f"Total: {len(counterfactuals)} nodes, {num_with_cf} with interventions")
    print("=" * 80)
