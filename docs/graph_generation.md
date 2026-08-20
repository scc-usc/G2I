# Graph Generation Process

This document describes the graph generation process implemented in `data/gen_graph_data.py`. The function `generate_graph` creates a synthetic graph combining Stochastic Block Model (SBM) structure with homophily-based edges, suitable for testing graph neural networks and counterfactual explanations.

## Overview

The graph generation follows these steps:
1. Generate node attributes and distance weights
2. Build an SBM graph with community structure
3. Trim SBM edges to a target count
4. Add homophily-based edges using attribute similarity
5. Assign node labels based on attributes and neighbor statistics
6. Convert to PyTorch Geometric format

## Parameters

- `n_nodes`: Total number of nodes (default: 250)
- `m_attrs`: Number of binary attributes per node (default: 6)
- `alpha`: Controls sensitivity to attribute distance in edge probability (default: 5.0)
- `beta`: Bias term in edge probability (default: 0.0)
- `total_edges`: Target total number of edges (default: 280)
- `label_percent`: Fraction of nodes labeled as positive (default: 0.2)
- `sbm_fraction`: Fraction of edges from SBM (default: 0.6)
- `n_comms`: Number of communities in SBM (default: 6)
- `p_intra`: Intra-community edge probability in SBM (default: 0.08)
- `p_inter`: Inter-community edge probability in SBM (default: 0.006)
- `seed`: Random seed for reproducibility (default: 42)
- `keep_features`: If True, use attribute-based logic for labels; if False, ignore attributes (default: True)

## Step-by-Step Generation

### 1. Generate Attributes and Weights

- **Weight Vector (`w`)**: A random vector of length `m_attrs`, normalized to unit length. Used for weighted distance calculations.
- **Node Attributes**: Binary matrix of shape `(n_nodes, m_attrs)`, where each entry is 0 or 1.

### 2. Build SBM Graph

- Uses NetworkX's `stochastic_block_model` to create a graph with `n_comms` communities.
- Community sizes are roughly equal, with remainders distributed.
- Edge probabilities: `p_intra` within communities, `p_inter` between communities.
- Attaches generated attributes to nodes.

### 3. Trim SBM Edges

- Calculates target SBM edges: `int(total_edges * sbm_fraction)`.
- If the SBM graph has more edges, randomly removes excess edges to reach the target.

### 4. Add Homophily-Based Edges

- Remaining edges to add: `total_edges - current_edges`.
- For each pair of nodes not already connected:
  - Compute weighted distance: `d = ||w * (A_i - A_j)||` (using first `m_attrs-1` attributes).
  - Edge probability: `sigmoid(-alpha * d + beta)`.
- Sort pairs by probability (descending) and add top `remaining_edges` edges.

### 5. Assign Labels

- For each node `i`:
  - `a = 1` if `attr[0] == 1` and `attr[m_attrs-1] == 0`, else `0` (if `keep_features=True`; else `a = 1`).
  - `b = fraction of neighbors where attr[m_attrs-1] == 0` (or `label_percent` if no neighbors).
  - Score = `a * b`.
- Sort nodes by score (descending), label top `int(label_percent * n_nodes)` as positive (1), others as negative (0).

### 6. Convert to PyTorch Geometric Data

- `x`: Tensor of node attributes (float32).
- `edge_index`: Tensor of edges (long, shape [2, num_edges]).
- `y`: Tensor of labels (long).

## Data Splitting

The `split_data` function uses PyTorch Geometric's `RandomLinkSplit` to split edges into train/validation/test sets, preserving the graph structure.

- `num_val`: Fraction for validation (default: 0.2).
- `num_test`: Fraction for test (default: 0.2).
- Assumes undirected graph.

## Usage Example

```python
from gen_graph_data import generate_graph, split_data

data, G, attributes, w = generate_graph(
    n_nodes=250, m_attrs=6, alpha=5.0, beta=0.0,
    total_edges=280, label_percent=0.2, sbm_fraction=0.6,
    n_comms=6, p_intra=0.08, p_inter=0.006, seed=42
)

train_data, val_data, test_data = split_data(data)
```

This generates a graph with mixed SBM and homophily edges, where labels depend on node attributes and neighbor statistics, making it suitable for testing GNN explainability.