"""
GNN Models for Graph Classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, MessagePassing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class GAT1(nn.Module):
    """
    1-layer Graph Attention Network for node classification.
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.5, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, concat=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()

class GCN1(nn.Module):
    """
    1-layer Graph Convolutional Network for node classification.
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


class GCN1_NoSelfLoops(nn.Module):
    """
    1-layer Graph Convolutional Network for node classification without self-loops.
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim, add_self_loops=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


class NeighborMeanLinear(MessagePassing):
    """
    Model that takes the average features of neighbors, then applies a linear layer followed by sigmoid.
    Supports optional edge_weight for CFF explainer masking.
    """
    def __init__(self, in_features):
        super().__init__(aggr='add')  # Use 'add' to support edge weights, then normalize manually
        self.linear = nn.Linear(in_features, 1)
        self.in_features = in_features

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with optional edge weights.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights for weighted aggregation (default: None)
        """
        # Default to uniform weights if not provided
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)

        # Weighted sum of neighbor features
        x_agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # Compute weighted degree (sum of edge weights per node) for normalization
        deg = self.propagate(
            edge_index,
            x=torch.ones(x.shape[0], 1, device=x.device),
            edge_weight=edge_weight
        )
        deg = deg.clamp(min=1e-6)  # Avoid division by zero

        # Weighted mean
        x_agg = x_agg / deg

        return torch.sigmoid(self.linear(x_agg)).squeeze()

    def message(self, x_j, edge_weight):
        """Weighted message passing."""
        return x_j * edge_weight.view(-1, 1)


class NeighborhoodMeanClassifier(nn.Module):
    """
    Linear classifier on the mean of neighborhood features.
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv = GCNConv(in_features, in_features, add_self_loops=False)
        # default to using bias in linear layer
        self.linear = nn.Linear(in_features, 1, bias=True)

    def forward(self, x, edge_index):
        # Compute mean of neighbors' features (GCNConv with add_self_loops=False approximates mean)
        x_agg = self.conv(x, edge_index)
        return torch.sigmoid(self.linear(x_agg)).squeeze()


class GCN_Mulin_Ori(nn.Module):
    """
    Original 2-layer Graph Convolutional Network for node classification.
    Supports edge_weight for CFF explainer masking.
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with optional edge_weight for CFF masking.
        """
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()

    def loss(self, pred, target):
        """Binary cross entropy loss"""
        return F.binary_cross_entropy(pred, target.float())


class GCN_Mulin(nn.Module):
    """
    3-layer Graph Convolutional Network with Jumping Knowledge for node classification.
    Concatenates outputs from all layers (JK architecture).
    Supports edge_weight for CFF explainer masking.

    Args:
        in_features: Number of input features (ignored if pca_components is provided)
        hidden_dim: Hidden dimension for GCN layers (default: 32)
        dropout: Dropout rate (default: 0.5)
        pca_components: Optional PCA components matrix for dimensionality reduction.
                       Shape: (n_components, n_features). If provided, input features
                       are first projected through a fixed (non-trainable) PCA layer.

    Examples:
        # Without PCA (synthetic data with 3 features)
        model = GCN_Mulin(in_features=3)

        # With PCA (real data with 48 features -> 18 components)
        pca = PCA(n_components=18).fit(X)
        model = GCN_Mulin(in_features=48, pca_components=pca.components_)
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.5, pca_components=None):
        super(GCN_Mulin, self).__init__()
        self.dropout = dropout
        self.use_pca = pca_components is not None

        # Optional PCA layer (fixed, not trainable)
        if self.use_pca:
            n_components, n_features = pca_components.shape
            self.pca_layer = nn.Linear(n_features, n_components, bias=False)
            self.pca_layer.weight.data = torch.tensor(pca_components, dtype=torch.float32)
            self.pca_layer.weight.requires_grad = False
            conv1_in = n_components
        else:
            self.pca_layer = None
            conv1_in = in_features

        # 3-layer GCN with Jumping Knowledge
        self.conv1 = GCNConv(conv1_in, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, 1)  # JK: concatenate all 3 layers

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with optional edge_weight for CFF masking.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights for weighted message passing
        """
        # Optional PCA projection
        if self.use_pca:
            x = self.pca_layer(x)

        # 3-layer GCN
        x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight=edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = F.relu(self.conv3(x2, edge_index, edge_weight=edge_weight))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # Jumping Knowledge: concatenate all layer outputs
        x_jk = torch.cat([x1, x2, x3], dim=1)
        out = self.fc(x_jk)
        return torch.sigmoid(out).squeeze()

    def loss(self, pred, target):
        """Binary cross entropy loss"""
        return F.binary_cross_entropy(pred, target.float())


class GCN_Mulin_Perturb(nn.Module):
    """
    Perturb version of GCN_Mulin for CF-GNNExplainer.
    Learns edge perturbation mask P to find counterfactual explanations.
    Supports optional PCA layer (must match the main model).
    """
    def __init__(self, in_features, hidden_dim, dropout, sub_adj, beta,
                 edge_additions=False, pca_components=None):
        super(GCN_Mulin_Perturb, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.beta = beta
        self.edge_additions = edge_additions
        self.num_nodes = sub_adj.shape[0]
        self.use_pca = pca_components is not None

        # Optional PCA layer (fixed, not trainable)
        if self.use_pca:
            n_components, n_features = pca_components.shape
            self.pca_layer = nn.Linear(n_features, n_components, bias=False)
            self.pca_layer.weight.data = torch.tensor(pca_components, dtype=torch.float32)
            self.pca_layer.weight.requires_grad = False
            conv1_in = n_components
        else:
            self.pca_layer = None
            conv1_in = in_features

        # GCN layers (same as GCN_Mulin)
        self.conv1 = GCNConv(conv1_in, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, 1)

        # Learnable perturbation parameter
        self.P_vec_size = sub_adj.shape[0] * sub_adj.shape[0]
        self.P_vec = nn.Parameter(torch.zeros(self.P_vec_size))

        # Store original adjacency
        self.sub_adj = sub_adj

    def forward(self, x, sub_adj):
        # Optional PCA projection
        if self.use_pca:
            x = self.pca_layer(x)

        # Get perturbed adjacency
        P = torch.sigmoid(self.P_vec).view(self.num_nodes, self.num_nodes)
        P_sym = (P + P.T) / 2  # Symmetrize

        if self.edge_additions:
            A_tilde = sub_adj + P_sym * (1 - sub_adj)  # Add edges
        else:
            A_tilde = sub_adj * (1 - P_sym)  # Remove edges

        # Convert dense adj to edge_index for GCNConv
        edge_index = A_tilde.nonzero().t().contiguous()
        edge_weight = A_tilde[edge_index[0], edge_index[1]]

        # Forward pass through GCN layers (Jumping Knowledge)
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # Jumping Knowledge concatenation
        x_jk = torch.cat([x1, x2, x3], dim=1)

        out = self.fc(x_jk)
        return torch.sigmoid(out).squeeze()

    def loss(self, output, y_pred_orig, new_idx):
        """BCE loss for counterfactual explanation."""
        pred_new = output[new_idx]

        # Target: flip the prediction
        if y_pred_orig == 1:
            target = torch.tensor(0.0)
        else:
            target = torch.tensor(1.0)

        # BCE loss
        eps = 1e-7
        pred_loss = -target * torch.log(pred_new + eps) - (1 - target) * torch.log(1 - pred_new + eps)

        # Graph distance loss (L1 norm of perturbation)
        P = torch.sigmoid(self.P_vec).view(self.num_nodes, self.num_nodes)
        P_sym = (P + P.T) / 2
        graph_dist = P_sym.sum() / 2  # Divide by 2 for symmetric

        return pred_loss + self.beta * graph_dist


class CFExplainer:
    """
    Counterfactual Explainer for GCN_Mulin model (binary classification).
    Uses CF-GNNExplainer method to find edge perturbations that flip predictions.
    """
    def __init__(self, model, sub_adj, sub_feat, n_hid, dropout,
                 sub_labels, y_pred_orig, new_idx, beta, pca_components=None):
        self.model = model
        self.model.eval()
        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.n_hid = n_hid
        self.dropout = dropout
        self.sub_labels = sub_labels
        self.y_pred_orig_fullgraph = y_pred_orig
        self.new_idx = new_idx
        self.beta = beta
        self.pca_components = pca_components

        # Create perturb model (with same PCA as main model)
        self.cf_model = GCN_Mulin_Perturb(
            in_features=sub_feat.shape[1],
            hidden_dim=n_hid,
            dropout=0.0,  # Use dropout=0 for deterministic predictions
            sub_adj=sub_adj,
            beta=beta,
            pca_components=pca_components
        )

        # Transfer weights from trained model
        self.cf_model.conv1.lin.weight.data = model.conv1.lin.weight.data.clone()
        self.cf_model.conv1.bias.data = model.conv1.bias.data.clone()
        self.cf_model.conv2.lin.weight.data = model.conv2.lin.weight.data.clone()
        self.cf_model.conv2.bias.data = model.conv2.bias.data.clone()
        self.cf_model.conv3.lin.weight.data = model.conv3.lin.weight.data.clone()
        self.cf_model.conv3.bias.data = model.conv3.bias.data.clone()
        self.cf_model.fc.weight.data = model.fc.weight.data.clone()
        self.cf_model.fc.bias.data = model.fc.bias.data.clone()

        # Freeze GCN weights, only train P_vec
        for name, param in self.cf_model.named_parameters():
            if 'P_vec' not in name:
                param.requires_grad = False

        # Compute original prediction on the subgraph
        self.cf_model.eval()
        with torch.no_grad():
            edge_idx = sub_adj.nonzero().t().contiguous()
            output_subgraph = model(sub_feat, edge_idx)
            self.y_pred_orig = (output_subgraph[new_idx] > 0.5).long().item()

    def explain(self, cf_optimizer, cf_epochs, verbose=False):
        """Run CF explanation optimization."""
        best_loss = float('inf')
        best_cf_example = []

        for epoch in range(cf_epochs):
            self.cf_model.train()
            cf_optimizer.zero_grad()

            output = self.cf_model(self.sub_feat, self.sub_adj)
            loss = self.cf_model.loss(output, self.y_pred_orig, self.new_idx)

            loss.backward()
            cf_optimizer.step()

            # Check if prediction flipped
            with torch.no_grad():
                pred_new = (output[self.new_idx] > 0.5).long().item()

            if pred_new != self.y_pred_orig and loss.item() < best_loss:
                best_loss = loss.item()
                # Get perturbed edges
                P = torch.sigmoid(self.cf_model.P_vec).view(self.sub_adj.shape[0], -1)
                P_sym = (P + P.T) / 2
                perturbed = (P_sym > 0.5) & (self.sub_adj > 0)
                best_cf_example = perturbed.nonzero().tolist()

            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch}: loss={loss.item():.4f}, pred={pred_new}")

        return best_cf_example


class SAGE1(nn.Module):
    """
    1-layer GraphSAGE for node classification.
    """
    def __init__(self, in_features, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


def train(model, data, optimizer, criterion, mask=None):
    """
    Train the model for one epoch.
    
    Args:
        model: GNN model
        data: PyTorch Geometric Data object
        optimizer: Optimizer
        criterion: Loss function
        mask: Optional mask for training nodes
    
    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    if mask is not None:
        loss = criterion(out[mask], data.y[mask].float())
    else:
        loss = criterion(out, data.y.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask=None):
    """
    Evaluate the model.
    
    Args:
        model: GNN model
        data: PyTorch Geometric Data object
        mask: Optional mask for evaluation nodes
    
    Returns:
        Accuracy
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        if mask is not None:
            pred = (out[mask] > 0.5).long()
            correct = (pred == data.y[mask]).sum().item()
            acc = correct / mask.sum().item()
        else:
            pred = (out > 0.5).long()
            correct = (pred == data.y).sum().item()
            acc = correct / data.y.shape[0]
    return acc


def train_with_early_stopping(model, data, optimizer, criterion, max_epochs=6000, patience=10, check_every=300, positive_indices_val=None, device='cpu'):
    """
    Train the model with early stopping based on positive validation accuracy.

    Args:
        model: GNN model
        data: PyTorch Geometric Data object with train_mask, val_mask
        optimizer: Optimizer
        criterion: Loss function
        max_epochs: Maximum number of epochs
        patience: Number of checks without improvement before stopping
        check_every: Check validation every this many epochs
        positive_indices_val: Indices of positive nodes in val_data for positive accuracy
        device: Device to train on ('cpu', 'cuda', etc.)

    Returns:
        model: Trained model
        losses: List of losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        final_pos_val_acc: Final positive validation accuracy
        test_acc: Test accuracy computed on appropriate mask
    """
    # Determine evaluation mask and positive indices for early stopping
    if data.val_mask.sum() > 0:
        eval_mask = data.val_mask
        positive_indices_eval = positive_indices_val
    else:
        eval_mask = data.train_mask
        positive_indices_eval = torch.where((data.train_mask) & (data.y == 1))[0]
    
    best_pos_val_acc = 0.0
    counter = 0
    losses = []
    train_accs = []
    val_accs = []
    best_model_state = None
    
    for epoch in range(max_epochs):
        loss = train(model, data, optimizer, criterion, mask=data.train_mask)
        losses.append(loss)
        
        if (epoch + 1) % check_every == 0:
            val_acc = evaluate(model, data, mask=eval_mask)
            train_acc = evaluate(model, data, mask=data.train_mask)
            val_accs.append(val_acc)
            train_accs.append(train_acc)  # Added to append train_acc
            
            # Compute positive validation accuracy if indices provided
            pos_val_acc = 0.0
            if positive_indices_eval is not None and positive_indices_eval.numel() > 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    preds = (out[eval_mask] > 0.5).float()
                    eval_y_masked = data.y[eval_mask]
                    # positive_indices_eval are indices in the full data, need to map to masked indices
                    eval_mask_indices = torch.where(eval_mask)[0]
                    pos_in_eval = [i for i, idx in enumerate(eval_mask_indices) if idx in positive_indices_eval]
                    if pos_in_eval:
                        correct_pos = (preds[pos_in_eval] == eval_y_masked[pos_in_eval]).sum().item()
                        pos_val_acc = correct_pos / len(pos_in_eval)
            
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Eval Acc: {val_acc:.4f} | Positive Eval Acc: {pos_val_acc:.4f}")
            
            # Early stopping based on positive validation accuracy
            if pos_val_acc > best_pos_val_acc:
                best_pos_val_acc = pos_val_acc
                counter = 0
                best_model_state = model.state_dict().copy()  # Save best model
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}: no improvement in positive validation accuracy for {patience} checks.")
                    break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with positive validation accuracy: {best_pos_val_acc:.4f}")
    else:
        print("No improvement found; using final trained model.")
    
    
    # Final positive validation accuracy
    final_pos_val_acc = 0.0
    if positive_indices_eval is not None and positive_indices_eval.numel() > 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = (out[eval_mask] > 0.5).float()
            eval_y_masked = data.y[eval_mask]
            # positive_indices_eval are indices in the full data, need to map to masked indices
            eval_mask_indices = torch.where(eval_mask)[0]
            pos_in_eval = [i for i, idx in enumerate(eval_mask_indices) if idx in positive_indices_eval]
            if pos_in_eval:
                correct_pos = (preds[pos_in_eval] == eval_y_masked[pos_in_eval]).sum().item()
                final_pos_val_acc = correct_pos / len(pos_in_eval)
    
    # Compute test accuracy
    if data.test_mask.sum() > 0:
        test_eval_mask = data.test_mask
    elif data.val_mask.sum() > 0:
        test_eval_mask = data.val_mask
    else:
        test_eval_mask = data.train_mask
    test_acc = evaluate(model, data, mask=test_eval_mask)
    
    return model, losses, train_accs, val_accs, final_pos_val_acc, test_acc


def visualize_model_weights(model, figsize=(10, 6), cmap='bwr', max_cols=3, normalize=True, save_path=None):
    """
    Visualize weight matrices and biases for a model's parameters.

    Args:
        model: torch.nn.Module
        figsize: tuple for figure size
        cmap: colormap for matrix plots
        max_cols: maximum number of subplot columns
        normalize: if True, use symmetric vmin/vmax around zero per matrix
        save_path: if provided, save the figure to this path

    Behavior:
        - Plots 2D parameters (weight matrices) as heatmaps.
        - Plots 1D parameters (bias vectors) as bar plots.
        - Uses model.named_parameters() so it works with GCNConv, Linear, etc.
    """
    # Gather parameters (skip zero-dim and non-numeric)
    params = [(name, p.detach().cpu().numpy()) for name, p in model.named_parameters() if p.numel() > 0]
    params = [ (n, v) for n, v in params if v.ndim in (1, 2) ]
    if len(params) == 0:
        print("No parameters found to visualize.")
        return

    n = len(params)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, (name, arr) in zip(axes, params):
        if arr.ndim == 2:
            if normalize:
                mx = np.max(np.abs(arr))
                vmin, vmax = -mx, mx
            else:
                vmin, vmax = None, None
            im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{name} {arr.shape}", fontsize=9)
            ax.set_xlabel("in", fontsize=8)
            ax.set_ylabel("out", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:  # 1D bias
            ax.bar(np.arange(arr.shape[0]), arr, color='k', alpha=0.7)
            ax.set_title(f"{name} {arr.shape}", fontsize=9)
            ax.set_xlabel("idx", fontsize=8)
            ax.set_ylabel("value", fontsize=8)

    # Hide unused axes
    for ax in axes[len(params):]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def visualize_forward_pass(model, data, node_idx):
    """
    Visualize the forward pass of the model for a specific node, showing intermediate values as tables.
    
    Args:
        model: The GNN model (assumes NeighborMeanLinear)
        data: PyTorch Geometric Data object
        node_idx: Index of the node to visualize
    """
    model.eval()
    
    # Compute aggregation exactly as the model does (MessagePassing.propagate uses incoming edges)
    # Ensure we run aggregation on the same device as the model parameters to avoid dtype/device mismatches
    model_device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')

    with torch.no_grad():
        x_dev = data.x.to(model_device)
        edge_index_dev = data.edge_index.to(model_device)

        # Full aggregation for all nodes (as the model does)
        # Use default edge_weight of ones for compatibility
        edge_weight_dev = torch.ones(edge_index_dev.shape[1], device=model_device)
        x_agg_all = model.propagate(edge_index_dev, x=x_dev, edge_weight=edge_weight_dev)

        # Compute degree for normalization (weighted sum of ones = count of edges)
        deg = model.propagate(
            edge_index_dev,
            x=torch.ones(x_dev.shape[0], 1, device=model_device),
            edge_weight=edge_weight_dev
        )
        deg = deg.clamp(min=1e-6)
        x_agg_all = x_agg_all / deg

        # Aggregated vector for the target node (mean of incoming neighbors)
        mean_neighbors = x_agg_all[node_idx].cpu().numpy()

        # Incoming neighbor indices (for display) - nodes that send messages to node_idx
        incoming_neighbors = data.edge_index[0][data.edge_index[1] == node_idx].cpu().tolist()

        # Node features (original tensor, shown on CPU)
        node_features = data.x[node_idx].cpu().numpy()

        # Neighbor features (for display) - fetch from original data for readability
        if len(incoming_neighbors) > 0:
            neighbor_features = data.x[incoming_neighbors].cpu().numpy()
        else:
            neighbor_features = np.array([]).reshape(0, data.x.shape[1])

        # Linear and sigmoid outputs computed from the aggregated vector (on model device)
        linear_input = x_agg_all[node_idx].unsqueeze(0)  # shape: [1, in_features]
        linear_out_tensor = model.linear(linear_input)
        linear_out = linear_out_tensor.cpu().item()
        output = torch.sigmoid(linear_out_tensor).cpu().item()
    
    # Display tables instead of plots
    print(f"\n{'='*60}")
    print(f"FORWARD PASS VISUALIZATION FOR NODE {node_idx}")
    print(f"{'='*60}")
    
    # 1. Node features table
    print(f"\n1. Node Features:")
    node_df = pd.DataFrame({
        'Feature Index': range(len(node_features)),
        'Value': node_features
    })
    print(node_df.to_string(index=False))
    
    # 2. Neighbor features table (incoming neighbors)
    print(f"\n2. Neighbor Features ({len(incoming_neighbors)} neighbors):")
    if len(neighbor_features) > 0:
        neighbor_df = pd.DataFrame(
            neighbor_features,
            columns=[f'Feature {i}' for i in range(neighbor_features.shape[1])]
        )
        neighbor_df.insert(0, 'Neighbor Index', incoming_neighbors)
        print(neighbor_df.to_string(index=False))
    else:
        print("No neighbors found.")
    
    # 3. Mean neighbor features table
    print(f"\n3. Mean Neighbor Features:")
    mean_df = pd.DataFrame({
        'Feature Index': range(len(mean_neighbors)),
        'Mean Value': mean_neighbors
    })
    print(mean_df.to_string(index=False))
    
    # 4. Output progression table
    print(f"\n4. Model Output Stages:")
    output_df = pd.DataFrame({
        'Stage': ['Linear Output', 'Sigmoid Output'],
        'Value': [linear_out, output]
    })
    print(output_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    
    # Print summary
    print(f"\nForward Pass Summary for Node {node_idx}:")
    print(f"Node features: {node_features}")
    print(f"Number of neighbors: {len(incoming_neighbors)}")
    if len(incoming_neighbors) > 0:
        print(f"Mean neighbor features: {mean_neighbors}")
    print(f"Linear output: {linear_out:.4f}")
    print(f"Final output (sigmoid): {output:.4f}")
    
    # Verify with full model call
    with torch.no_grad():
        full_model_output = model(data.x, data.edge_index)[node_idx].item()
    print(f"Full model call output for node {node_idx}: {full_model_output:.4f}")
    print(f"Difference: {abs(output - full_model_output):.6f}")


# ============================================================================
# CFF Explainer for Node Classification
# ============================================================================

class CFFExplainerNode(nn.Module):
    """
    CFF (Counterfactual and Factual) Explainer for node classification.
    Supports three modes: edge, feature, or edge+feature explanation.

    Based on: "Learning and Evaluating Graph Neural Network Explanations based on
    Counterfactual and Factual Reasoning" (WWW 2022)
    """
    def __init__(self, base_model, features, edge_index, target_node, mode="edge",
                 unchangeable_indices=None):
        super(CFFExplainerNode, self).__init__()
        self.base_model = base_model
        self.features = features
        self.edge_index = edge_index
        self.target_node = target_node
        self.num_edges = edge_index.shape[1]
        self.num_nodes = features.shape[0]
        self.feat_dim = features.shape[1]

        # Store unchangeable feature indices (these features should not be modified)
        self.unchangeable_indices = set(unchangeable_indices) if unchangeable_indices else set()

        # If no edges, force mode to "feature" only
        if self.num_edges == 0 and mode in ["edge", "edge+feature"]:
            mode = "feature"
        self.mode = mode

        # Initialize masks based on mode
        if mode in ["edge", "edge+feature"]:
            self.edge_mask = self._construct_edge_mask()
        if mode in ["feature", "edge+feature"]:
            self.feat_mask = self._construct_feat_mask()
    
    def _construct_edge_mask(self):
        """Initialize edge mask parameter."""
        mask = nn.Parameter(torch.FloatTensor(self.num_edges))
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / self.num_edges)
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask
    
    def _construct_feat_mask(self):
        """Initialize feature mask parameter (per node, per feature).

        Unchangeable features are initialized to a very negative value so that
        sigmoid(mask) ≈ 0, meaning they won't be considered important.
        """
        mask = nn.Parameter(torch.FloatTensor(self.num_nodes, self.feat_dim))
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (self.num_nodes + self.feat_dim))
        with torch.no_grad():
            mask.normal_(1.0, std)
            # Set unchangeable features to very negative value (sigmoid -> ~0)
            for idx in self.unchangeable_indices:
                if idx < self.feat_dim:
                    mask[:, idx] = -10.0  # sigmoid(-10) ≈ 0.00005
        return mask

    def _zero_unchangeable_gradients(self):
        """Zero out gradients for unchangeable features after backward pass."""
        if self.mode in ["feature", "edge+feature"] and len(self.unchangeable_indices) > 0:
            if self.feat_mask.grad is not None:
                for idx in self.unchangeable_indices:
                    if idx < self.feat_dim:
                        self.feat_mask.grad[:, idx] = 0.0
    
    def get_masked_edge_weight(self):
        """Get sigmoid-activated edge mask."""
        if self.mode in ["edge", "edge+feature"]:
            return torch.sigmoid(self.edge_mask)
        return None
    
    def get_masked_features(self):
        """Get sigmoid-activated feature mask applied to features."""
        if self.mode in ["feature", "edge+feature"]:
            feat_mask = torch.sigmoid(self.feat_mask)
            return self.features * feat_mask, feat_mask
        return self.features, None
    
    def forward(self):
        """Compute factual and counterfactual predictions."""
        # Get masked edge weights
        if self.mode in ["edge", "edge+feature"]:
            masked_edge_weight = self.get_masked_edge_weight()
            cf_edge_weight = 1 - masked_edge_weight
        else:
            masked_edge_weight = None
            cf_edge_weight = None
        
        # Get masked features
        if self.mode in ["feature", "edge+feature"]:
            masked_feat, feat_mask = self.get_masked_features()
            cf_feat = self.features * (1 - feat_mask)
        else:
            masked_feat = self.features
            cf_feat = self.features
        
        # Factual: prediction with masked subgraph/features
        pred_factual = self.base_model(
            masked_feat, 
            self.edge_index,
            edge_weight=masked_edge_weight
        )
        
        # Counterfactual: prediction with remaining subgraph/features
        pred_counterfactual = self.base_model(
            cf_feat,
            self.edge_index,
            edge_weight=cf_edge_weight
        )
        
        return pred_factual[self.target_node], pred_counterfactual[self.target_node]
    
    def loss(self, pred_factual, pred_counterfactual, orig_pred, gam, lam, alp):
        """
        Compute CFF loss for binary classification.
        
        Args:
            pred_factual: Factual prediction
            pred_counterfactual: Counterfactual prediction
            orig_pred: Original prediction (0 or 1)
            gam: Margin for BPR loss
            lam: Weight for prediction loss
            alp: Balance between factual and counterfactual (e.g., 0.6 = 60% factual)
        """
        relu = nn.ReLU()
        
        if orig_pred == 1:
            bpr_factual = relu(gam + 0.5 - pred_factual)
            bpr_counterfactual = relu(gam + pred_counterfactual - 0.5)
        else:
            bpr_factual = relu(gam + pred_factual - 0.5)
            bpr_counterfactual = relu(gam + 0.5 - pred_counterfactual)
        
        # L1 regularization
        L1 = torch.tensor(0.0)
        if self.mode in ["edge", "edge+feature"]:
            L1 = L1 + torch.norm(self.get_masked_edge_weight(), p=1)
        if self.mode in ["feature", "edge+feature"]:
            _, feat_mask = self.get_masked_features()
            L1 = L1 + torch.norm(feat_mask, p=1)
        
        # Total loss
        loss = L1 + lam * (alp * bpr_factual + (1 - alp) * bpr_counterfactual)
        
        return bpr_factual, bpr_counterfactual, L1, loss


def train_cff_explainer(base_model, features, edge_index, target_node, orig_pred,
                        mode="edge+feature", lr=0.01, epochs=500,
                        lam=500, alp=0.6, gam=0.5, verbose=False,
                        unchangeable_indices=None):
    """
    Train a CFF explainer for a single target node.

    Args:
        base_model: Pre-trained GNN model (frozen)
        features: Node features tensor
        edge_index: Edge index tensor
        target_node: Index of the node to explain
        orig_pred: Original prediction for the target node (0 or 1)
        mode: Explanation mode ("edge", "feature", or "edge+feature")
        lr: Learning rate
        epochs: Number of training epochs
        lam: Weight for prediction loss
        alp: Balance between factual and counterfactual
        gam: Margin for BPR loss
        verbose: Whether to print progress
        unchangeable_indices: List of feature indices that cannot be changed

    Returns:
        explainer: Trained CFF explainer
        result: Dictionary with explanation results
    """
    explainer = CFFExplainerNode(base_model, features, edge_index, target_node, mode=mode,
                                  unchangeable_indices=unchangeable_indices)
    optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)

    for epoch in range(epochs):
        explainer.train()
        optimizer.zero_grad()

        pred_factual, pred_counterfactual = explainer()
        bpr_factual, bpr_counterfactual, L1, loss = explainer.loss(
            pred_factual, pred_counterfactual, orig_pred, gam, lam, alp
        )

        loss.backward()

        # Zero out gradients for unchangeable features before optimizer step
        explainer._zero_unchangeable_gradients()

        optimizer.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    
    # Get final results
    explainer.eval()
    with torch.no_grad():
        pred_factual, pred_counterfactual = explainer()
        
    result = {
        'node_idx': target_node,
        'orig_pred': orig_pred,
        'factual_pred': int(pred_factual.item() > 0.5),
        'factual_score': pred_factual.item(),
        'counterfactual_pred': int(pred_counterfactual.item() > 0.5),
        'counterfactual_score': pred_counterfactual.item(),
        'factual_success': (int(pred_factual.item() > 0.5) == orig_pred),
        'counterfactual_success': (int(pred_counterfactual.item() > 0.5) != orig_pred),
        'mode': mode
    }
    
    # Extract masks
    if mode in ["edge", "edge+feature"]:
        edge_mask = explainer.get_masked_edge_weight().detach()
        result['edge_mask'] = edge_mask
        result['num_important_edges'] = (edge_mask > 0.5).sum().item()
        
    if mode in ["feature", "edge+feature"]:
        _, feat_mask = explainer.get_masked_features()
        feat_mask = feat_mask.detach()
        result['feat_mask'] = feat_mask
        result['node_feat_importance'] = feat_mask[target_node].cpu().numpy()
    
    return explainer, result


def generate_cff_explanations(base_model, features, edge_index, labels,
                              target_indices, mode="edge+feature",
                              lr=0.01, epochs=500, lam=500, alp=0.6, gam=0.5,
                              mask_threshold=0.5, verbose=True,
                              unchangeable_indices=None):
    """
    Generate CFF explanations for multiple target nodes.

    Args:
        base_model: Pre-trained GNN model (frozen)
        features: Node features tensor
        edge_index: Edge index tensor
        labels: Node labels tensor
        target_indices: Indices of nodes to explain
        mode: Explanation mode ("edge", "feature", or "edge+feature")
        lr: Learning rate
        epochs: Number of training epochs
        lam: Weight for prediction loss
        alp: Balance between factual and counterfactual
        gam: Margin for BPR loss
        mask_threshold: Threshold for counting important features/edges
        verbose: Whether to print progress
        unchangeable_indices: List of feature indices that cannot be changed

    Returns:
        cff_results: List of dictionaries with explanation results
    """
    # Freeze base model
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Get original predictions
    with torch.no_grad():
        predictions = base_model(features, edge_index)
    
    cff_results = []
    
    for i, node_idx in enumerate(target_indices):
        if isinstance(node_idx, torch.Tensor):
            node_idx = node_idx.item()
            
        true_label = labels[node_idx].item()
        orig_pred = int(predictions[node_idx].item() > 0.5)
        
        if verbose:
            print(f"\n[{i+1}/{len(target_indices)}] Node {node_idx} (True: {true_label}, Pred: {orig_pred})")
        
        _, result = train_cff_explainer(
            base_model, features, edge_index, node_idx, orig_pred,
            mode=mode, lr=lr, epochs=epochs, lam=lam, alp=alp, gam=gam,
            verbose=False, unchangeable_indices=unchangeable_indices
        )
        
        result['true_label'] = true_label
        
        if verbose:
            print(f"  Factual pred: {result['factual_pred']} ({result['factual_score']:.3f}) - {'OK' if result['factual_success'] else 'FAIL'}")
            print(f"  Counterfactual pred: {result['counterfactual_pred']} ({result['counterfactual_score']:.3f}) - {'OK' if result['counterfactual_success'] else 'FAIL'}")
            if 'num_important_edges' in result:
                print(f"  Important edges: {result['num_important_edges']}")
            if 'node_feat_importance' in result:
                print(f"  Feature importance: {result['node_feat_importance']}")
        
        cff_results.append(result)
    
    return cff_results
