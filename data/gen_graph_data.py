import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

# ----------------------------
# Helper Functions
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def weighted_distance(A_i, A_j, w):
    diff = A_i - A_j
    return np.linalg.norm(w * diff)

def edge_probability(A_i, A_j, w, alpha=5.0, beta=0.0):
    d = weighted_distance(A_i, A_j, w)
    return sigmoid(-alpha * d + beta)


def generate_graph(
    n_nodes=250,
    m_attrs=6,
    alpha=5.0,
    beta=0.0,
    total_edges=280,
    label_percent=0.2,
    sbm_fraction=0.6,
    n_comms=6,
    p_intra=0.08,
    p_inter=0.006,
    seed=42,
    keep_features=True
):
    """
    Generate a graph with SBM structure and homophily-based edges.
    
    Args:
        n_nodes: Number of nodes in the graph
        m_attrs: Number of node attributes
        alpha: Alpha parameter for edge probability (controls distance sensitivity)
        beta: Beta parameter for edge probability (bias term)
        total_edges: Total number of edges in the graph
        label_percent: Percentage of nodes to label as positive
        sbm_fraction: Fraction of edges from SBM (rest from homophily)
        n_comms: Number of communities in SBM
        p_intra: Probability of intra-community edges in SBM
        p_inter: Probability of inter-community edges in SBM
        seed: Random seed for reproducibility
    
    Returns:
        data: PyTorch Geometric Data object with node features, edges, and labels
        G: NetworkX graph object
        attributes: Node attribute matrix
        w: Weight vector for attribute distances
    """
    np.random.seed(seed)
    p = label_percent
    
    # ----------------------------
    # Step 1: Generate Attributes and Weights
    # ----------------------------
    w = np.random.rand(m_attrs)
    w /= np.linalg.norm(w)
    attributes = np.random.randint(0, 2, size=(n_nodes, m_attrs))

    # ----------------------------
    # Step 2: Build SBM Graph
    # ----------------------------
    base, rem = divmod(n_nodes, n_comms)
    sizes = [base + (1 if i < rem else 0) for i in range(n_comms)]
    probs = [[p_intra if i == j else p_inter for j in range(n_comms)] for i in range(n_comms)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    # Attach attributes
    for i in G.nodes():
        G.nodes[i]['attr'] = attributes[i]

    # ----------------------------
    # Step 3: Trim SBM edges to desired count
    # ----------------------------
    sbm_target_edges = int(total_edges * sbm_fraction)
    edges_list = list(G.edges())
    if len(edges_list) > sbm_target_edges:
        np.random.shuffle(edges_list)
        edges_to_keep = edges_list[:sbm_target_edges]
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges_to_keep)
        for i in range(n_nodes):
            G.nodes[i]['attr'] = attributes[i]

    # ----------------------------
    # Step 4: Add Homophily-Based Edges
    # ----------------------------
    remaining_edges = total_edges - G.number_of_edges()
    existing_edges = set(map(frozenset, G.edges()))
    edge_scores = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if frozenset((i, j)) in existing_edges:
                continue
            A_i, A_j = attributes[i], attributes[j]
            score = edge_probability(A_i[:m_attrs-1], A_j[:m_attrs-1], w[:m_attrs-1], alpha, beta)
            edge_scores.append((i, j, score))

    edge_scores.sort(key=lambda x: x[2], reverse=True)
    for i, j, score in edge_scores[:remaining_edges]:
        G.add_edge(i, j, weight=score)

    # ----------------------------
    # Step 5: Assign Labels (Node Score Logic)
    # ----------------------------
    node_scores = []
    for i in range(n_nodes):
        attr_i = G.nodes[i]['attr']
        a = 1
        if keep_features and not (attr_i[0] == 1 and attr_i[m_attrs-1] == 0):
            a = 0
        
        neighbors = list(G.neighbors(i))
        if neighbors:
            count = sum(1 for j in neighbors if G.nodes[j]['attr'][m_attrs-1] == 0)
            b = count / len(neighbors)
        else:
            b = p

        score = a * b
        node_scores.append((i, score))

    node_scores.sort(key=lambda x: x[1], reverse=True)
    num_positive = int(label_percent * n_nodes)
    positive_nodes = set(i for i, _ in node_scores[:num_positive])

    # Compute cutoff score (threshold for positive class)
    cutoff_score = node_scores[num_positive - 1][1] if num_positive > 0 else None

    # Ensure all nodes with score >= cutoff_score are positive (handle ties)
    if cutoff_score is not None:
        positive_nodes = set(i for i, score in node_scores if score >= cutoff_score - 1e-6)

    label_array = np.array([1 if i in positive_nodes else 0 for i in range(n_nodes)], dtype=int)
    for i in range(n_nodes):
        G.nodes[i]['label'] = label_array[i]

    # ----------------------------
    # Step 6: Convert to PyG Data
    # ----------------------------
    x = torch.tensor(attributes.astype(np.float32))
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    y_tensor = torch.tensor(label_array, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    
    return data, G, attributes, w, cutoff_score


def generate_graph_withAvgDegree(
    n_nodes=250,
    m_attrs=6,
    alpha=5.0,
    beta=0.0,
    total_edges=280,
    label_percent=0.2,
    sbm_fraction=0.6,
    n_comms=6,
    p_intra=0.08,
    p_inter=0.006,
    seed=42,
    keep_features=True
):
    """
    Generate a graph with improved degree distribution and label assignment.

    Improvements over generate_graph():
    1. Edge generation: Shuffle before sorting to break tie-breaking bias toward low-index nodes,
       resulting in more uniform degree distribution across all nodes.
    2. Label assignment: Strictly take top num_positive nodes without expanding ties,
       avoiding the issue where cutoff_score=0 causes all nodes to be labeled positive.

    Args:
        n_nodes: Number of nodes in the graph
        m_attrs: Number of node attributes
        alpha: Alpha parameter for edge probability (controls distance sensitivity)
        beta: Beta parameter for edge probability (bias term)
        total_edges: Total number of edges in the graph
        label_percent: Percentage of nodes to label as positive
        sbm_fraction: Fraction of edges from SBM (rest from homophily)
        n_comms: Number of communities in SBM
        p_intra: Probability of intra-community edges in SBM
        p_inter: Probability of inter-community edges in SBM
        seed: Random seed for reproducibility
        keep_features: Whether to use node features in label assignment

    Returns:
        data: PyTorch Geometric Data object with node features, edges, and labels
        G: NetworkX graph object
        attributes: Node attribute matrix
        w: Weight vector for attribute distances
        cutoff_score: The score threshold for positive labels
    """
    np.random.seed(seed)
    p = label_percent

    # ----------------------------
    # Step 1: Generate Attributes and Weights
    # ----------------------------
    w = np.random.rand(m_attrs)
    w /= np.linalg.norm(w)
    attributes = np.random.randint(0, 2, size=(n_nodes, m_attrs))

    # ----------------------------
    # Step 2: Build SBM Graph
    # ----------------------------
    base, rem = divmod(n_nodes, n_comms)
    sizes = [base + (1 if i < rem else 0) for i in range(n_comms)]
    probs = [[p_intra if i == j else p_inter for j in range(n_comms)] for i in range(n_comms)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    for i in G.nodes():
        G.nodes[i]['attr'] = attributes[i]

    # ----------------------------
    # Step 3: Trim SBM edges to desired count
    # ----------------------------
    sbm_target_edges = int(total_edges * sbm_fraction)
    edges_list = list(G.edges())
    if len(edges_list) > sbm_target_edges:
        np.random.shuffle(edges_list)
        edges_to_keep = edges_list[:sbm_target_edges]
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges_to_keep)
        for i in range(n_nodes):
            G.nodes[i]['attr'] = attributes[i]

    # ----------------------------
    # Step 4: Add Homophily-Based Edges (with shuffle to break ties)
    # ----------------------------
    remaining_edges = total_edges - G.number_of_edges()
    existing_edges = set(map(frozenset, G.edges()))
    edge_scores = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if frozenset((i, j)) in existing_edges:
                continue
            A_i, A_j = attributes[i], attributes[j]
            score = edge_probability(A_i[:m_attrs-1], A_j[:m_attrs-1], w[:m_attrs-1], alpha, beta)
            edge_scores.append((i, j, score))

    # Shuffle first to break ties randomly, then sort by score
    # This prevents bias toward low-index nodes when edges have equal scores
    np.random.shuffle(edge_scores)
    edge_scores.sort(key=lambda x: x[2], reverse=True)
    for i, j, score in edge_scores[:remaining_edges]:
        G.add_edge(i, j, weight=score)

    # ----------------------------
    # Step 5: Assign Labels (strictly top num_positive, no tie expansion)
    # ----------------------------
    node_scores = []
    for i in range(n_nodes):
        attr_i = G.nodes[i]['attr']
        a = 1
        if keep_features and not (attr_i[0] == 1 and attr_i[m_attrs-1] == 0):
            a = 0

        neighbors = list(G.neighbors(i))
        if neighbors:
            count = sum(1 for j in neighbors if G.nodes[j]['attr'][m_attrs-1] == 0)
            b = count / len(neighbors)
        else:
            b = p

        score = a * b
        node_scores.append((i, score))

    node_scores.sort(key=lambda x: x[1], reverse=True)
    num_positive = int(label_percent * n_nodes)

    # Strictly take top num_positive nodes (no tie expansion)
    positive_nodes = set(i for i, _ in node_scores[:num_positive])
    cutoff_score = node_scores[num_positive - 1][1] if num_positive > 0 else None

    label_array = np.array([1 if i in positive_nodes else 0 for i in range(n_nodes)], dtype=int)
    for i in range(n_nodes):
        G.nodes[i]['label'] = label_array[i]

    # ----------------------------
    # Step 6: Convert to PyG Data
    # ----------------------------
    x = torch.tensor(attributes.astype(np.float32))
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    y_tensor = torch.tensor(label_array, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y_tensor)

    return data, G, attributes, w, cutoff_score


def split_data(data, num_val=0.2, num_test=0.2, is_undirected=True):
    """
    Split graph data into train/validation/test sets by nodes.
    
    Args:
        data: PyTorch Geometric Data object
        num_val: Fraction of nodes for validation
        num_test: Fraction of nodes for testing
        is_undirected: Whether the graph is undirected (not used in node splitting)
    
    Returns:
        data: Data object with train_mask, val_mask, test_mask added
    """
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    
    num_test_nodes = int(num_test * num_nodes)
    num_val_nodes = int(num_val * num_nodes)
    num_train_nodes = num_nodes - num_test_nodes - num_val_nodes
    
    train_indices = indices[:num_train_nodes]
    val_indices = indices[num_train_nodes:num_train_nodes + num_val_nodes]
    test_indices = indices[num_train_nodes + num_val_nodes:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def stratified_split_data(data, num_val=0.2, num_test=0.2, is_undirected=True, seed=42):
    """
    Create stratified train/val/test split based on labels in data.y.
    The ratio is applied per class, so each split keeps similar class distribution.
    
    Args:
        data: PyTorch Geometric Data object
        num_val: Fraction of nodes for validation
        num_test: Fraction of nodes for testing
        is_undirected: Whether the graph is undirected (not used in node splitting)
        seed: Random seed for reproducibility
    
    Returns:
        data: Data object with train_mask, val_mask, test_mask added
    """
    # Calculate train ratio
    train_ratio = 1.0 - (num_val + num_test)
    assert train_ratio > 0, "Train ratio must be positive"

    y_np = data.y.cpu().numpy()
    num_nodes = y_np.shape[0]
    num_classes = int(y_np.max()) + 1

    rng = np.random.RandomState(seed)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for c in range(num_classes):
        # Find indices of class c
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)

        n_c = len(idx_c)
        n_train = int(n_c * train_ratio)
        n_val = int(n_c * num_val)
        # The rest go to test to ensure sum is consistent
        n_test = n_c - n_train - n_val

        train_idx_c = idx_c[:n_train]
        val_idx_c = idx_c[n_train:n_train + n_val]
        test_idx_c = idx_c[n_train + n_val:]

        train_mask[train_idx_c] = True
        val_mask[val_idx_c] = True
        test_mask[test_idx_c] = True

    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)

    return data
