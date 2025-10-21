"""
SAGoG: Similarity-Aware Graph of Graphs Neural Networks for Multivariate Time Series Classification

This implementation follows the architecture described in the SAGoG paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np


class GraphConstructor(nn.Module):
    """Constructs graphs from multivariate time series using similarity measures."""

    def __init__(self, input_dim, hidden_dim, method='adaptive'):
        super(GraphConstructor, self).__init__()
        self.method = method
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if method == 'adaptive':
            # Learnable adjacency matrix construction
            self.weight = nn.Parameter(torch.randn(input_dim, hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        """
        Construct graph from time series data.

        Args:
            x: Time series data [batch, num_variables, seq_len]

        Returns:
            edge_index: Graph edges
            edge_weight: Edge weights based on similarity
        """
        batch_size, num_vars, seq_len = x.shape

        if self.method == 'correlation':
            # Use Pearson correlation as similarity
            edge_index, edge_weight = self._correlation_graph(x)
        elif self.method == 'adaptive':
            # Learn adaptive adjacency matrix
            edge_index, edge_weight = self._adaptive_graph(x)
        elif self.method == 'dtw':
            # Dynamic Time Warping similarity
            edge_index, edge_weight = self._dtw_graph(x)
        else:
            raise ValueError(f"Unknown graph construction method: {self.method}")

        return edge_index, edge_weight

    def _correlation_graph(self, x):
        """Construct graph based on Pearson correlation."""
        batch_size, num_vars, seq_len = x.shape

        edge_indices = []
        edge_weights = []

        for b in range(batch_size):
            # Compute correlation matrix
            x_b = x[b].T  # [seq_len, num_vars]
            x_normalized = (x_b - x_b.mean(dim=0, keepdim=True)) / (x_b.std(dim=0, keepdim=True) + 1e-8)
            corr_matrix = torch.matmul(x_normalized.T, x_normalized) / seq_len

            # Threshold to create edges (keep top-k connections)
            k = min(5, num_vars - 1)  # Connect to top-5 similar nodes
            for i in range(num_vars):
                similarities = corr_matrix[i].clone()
                similarities[i] = -float('inf')  # Exclude self-loops
                top_k_indices = torch.topk(similarities, k).indices

                for j in top_k_indices:
                    edge_indices.append([i + b * num_vars, j.item() + b * num_vars])
                    edge_weights.append(corr_matrix[i, j].item())

        edge_index = torch.tensor(edge_indices, dtype=torch.long).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

        return edge_index, edge_weight

    def _adaptive_graph(self, x):
        """Learn adaptive adjacency matrix."""
        batch_size, num_vars, seq_len = x.shape

        # Project to hidden space
        x_mean = x.mean(dim=-1)  # [batch, num_vars]
        h = torch.matmul(x_mean, self.weight) + self.bias  # [batch, num_vars, hidden_dim]

        edge_indices = []
        edge_weights = []

        for b in range(batch_size):
            # Compute similarity in learned space
            h_b = h[b]  # [num_vars, hidden_dim]

            # Handle case where there's only 1 variable
            if num_vars == 1:
                continue  # No edges to create for single variable

            similarity = torch.matmul(h_b, h_b.T)  # [num_vars, num_vars]
            similarity = torch.sigmoid(similarity)

            # Ensure similarity is 2D (handle edge case where it might be squeezed)
            if similarity.dim() == 0:
                similarity = similarity.unsqueeze(0).unsqueeze(0)
            elif similarity.dim() == 1:
                similarity = similarity.unsqueeze(0)

            # Create edges based on similarity threshold
            k = min(5, num_vars - 1)
            if k <= 0:
                continue  # Skip if we can't create edges

            for i in range(num_vars):
                sim_i = similarity[i].clone()
                sim_i[i] = -float('inf')  # Mask self-connection

                # Ensure k doesn't exceed available elements
                actual_k = min(k, sim_i.numel() - 1)  # -1 because we masked self
                if actual_k <= 0:
                    continue

                top_k_indices = torch.topk(sim_i, actual_k).indices

                for j in top_k_indices:
                    edge_indices.append([i + b * num_vars, j.item() + b * num_vars])
                    edge_weights.append(similarity[i, j.item()].item())

        # Handle case where no edges were created
        if len(edge_indices) == 0:
            # Create minimal graph structure (self-loops for all nodes)
            total_nodes = batch_size * num_vars
            edge_indices = [[i, i] for i in range(total_nodes)]
            edge_weights = [1.0] * total_nodes

        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=x.device).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=x.device)

        return edge_index, edge_weight

    def _dtw_graph(self, x):
        """Construct graph using DTW distance (simplified version)."""
        # For efficiency, use Euclidean distance as proxy for DTW
        batch_size, num_vars, seq_len = x.shape

        edge_indices = []
        edge_weights = []

        for b in range(batch_size):
            x_b = x[b]  # [num_vars, seq_len]

            # Handle case where there's only 1 variable
            if num_vars == 1:
                continue  # No edges to create for single variable

            # Compute pairwise distances
            distances = torch.cdist(x_b, x_b, p=2)

            # Convert distance to similarity
            similarity = torch.exp(-distances / distances.std())

            k = min(5, num_vars - 1)
            for i in range(num_vars):
                sim_i = similarity[i].clone()
                sim_i[i] = -float('inf')

                # Ensure k doesn't exceed available elements
                actual_k = min(k, sim_i.numel() - 1)  # -1 because we masked self
                if actual_k <= 0:
                    continue

                top_k_indices = torch.topk(sim_i, actual_k).indices

                for j in top_k_indices:
                    edge_indices.append([i + b * num_vars, j.item() + b * num_vars])
                    edge_weights.append(similarity[i, j].item())

        # Handle case where no edges were created
        if len(edge_indices) == 0:
            # Create minimal graph structure (self-loops for all nodes)
            total_nodes = batch_size * num_vars
            edge_indices = [[i, i] for i in range(total_nodes)]
            edge_weights = [1.0] * total_nodes

        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=x.device).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=x.device)

        return edge_index, edge_weight


class GraphLevelEncoder(nn.Module):
    """Encodes individual graphs using GNN layers."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, gnn_type='gcn'):
        super(GraphLevelEncoder, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(in_dim, hidden_dim, heads=4, concat=False))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]
            edge_weight: Edge weights [num_edges]
            batch: Batch vector [num_nodes]
        """
        for i, conv in enumerate(self.convs):
            if self.gnn_type == 'gcn':
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)

            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        return x


class GraphOfGraphsLayer(nn.Module):
    """
    Graph of Graphs layer that constructs a meta-graph where each node
    represents an entire graph (time series window).
    """

    def __init__(self, graph_embed_dim, hidden_dim):
        super(GraphOfGraphsLayer, self).__init__()
        self.graph_embed_dim = graph_embed_dim
        self.hidden_dim = hidden_dim

        # Meta-graph construction
        self.similarity_network = nn.Sequential(
            nn.Linear(graph_embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # GNN on meta-graph
        self.meta_gcn = GCNConv(graph_embed_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, graph_embeddings):
        """
        Args:
            graph_embeddings: Embeddings of individual graphs [num_graphs, graph_embed_dim]

        Returns:
            meta_graph_features: Features from graph-of-graphs [num_graphs, hidden_dim]
        """
        num_graphs = graph_embeddings.shape[0]

        # Construct meta-graph edges based on similarity
        edge_indices = []
        edge_weights = []

        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                # Compute similarity between graph i and j
                concat_feat = torch.cat([graph_embeddings[i], graph_embeddings[j]], dim=-1)
                similarity = self.similarity_network(concat_feat).squeeze()

                # Add edge if similarity is above threshold
                if similarity > 0.5:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Undirected
                    edge_weights.append(similarity)
                    edge_weights.append(similarity)

        if len(edge_indices) == 0:
            # If no edges, create self-loops
            edge_indices = [[i, i] for i in range(num_graphs)]
            edge_weights = [1.0] * num_graphs

        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=graph_embeddings.device).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=graph_embeddings.device)

        # Apply GNN on meta-graph
        meta_features = self.meta_gcn(graph_embeddings, edge_index, edge_weight)
        meta_features = self.batch_norm(meta_features)
        meta_features = F.relu(meta_features)

        return meta_features


class SAGoG(nn.Module):
    """
    Similarity-Aware Graph of Graphs Neural Network for Multivariate Time Series Classification.
    """

    def __init__(self, num_variables, seq_len, num_classes, hidden_dim=64,
                 graph_hidden_dim=128, num_graph_layers=2, num_windows=5,
                 graph_construction='adaptive', gnn_type='gcn'):
        """
        Args:
            num_variables: Number of variables in multivariate time series
            seq_len: Length of time series sequence
            num_classes: Number of classes for classification
            hidden_dim: Hidden dimension for graph embeddings
            graph_hidden_dim: Hidden dimension for graph-level encoder
            num_graph_layers: Number of GNN layers
            num_windows: Number of windows to split time series into
            graph_construction: Method for graph construction ('correlation', 'adaptive', 'dtw')
            gnn_type: Type of GNN ('gcn', 'gat')
        """
        super(SAGoG, self).__init__()

        self.num_variables = num_variables
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_windows = num_windows
        self.window_size = seq_len // num_windows

        # Time series feature extraction
        self.temporal_encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_output_dim = hidden_dim * 2  # Bidirectional

        # Graph construction
        self.graph_constructor = GraphConstructor(
            num_variables, hidden_dim, method=graph_construction
        )

        # Graph-level encoder
        self.graph_encoder = GraphLevelEncoder(
            self.lstm_output_dim, graph_hidden_dim, num_graph_layers, gnn_type
        )

        # Graph-of-Graphs layer
        self.gog_layer = GraphOfGraphsLayer(graph_hidden_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input time series [batch_size, num_variables, seq_len]

        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        batch_size, num_vars, seq_len = x.shape

        # Split time series into windows
        windows = self._create_windows(x)  # [batch_size, num_windows, num_vars, window_size]

        all_graph_embeddings = []

        # Process each window
        for w in range(self.num_windows):
            window_data = windows[:, w, :, :]  # [batch_size, num_vars, window_size]

            # Extract temporal features for each variable
            node_features = self._extract_temporal_features(window_data)  # [batch_size * num_vars, lstm_output_dim]

            # Construct graph for this window
            edge_index, edge_weight = self.graph_constructor(window_data)

            # Create batch vector for graph operations
            batch_vector = torch.arange(batch_size, device=x.device).repeat_interleave(num_vars)

            # Encode graph
            graph_node_features = self.graph_encoder(
                node_features, edge_index, edge_weight, batch_vector
            )

            # Pool to get graph-level embedding
            graph_embedding = global_mean_pool(graph_node_features, batch_vector)  # [batch_size, graph_hidden_dim]
            all_graph_embeddings.append(graph_embedding)

        # Stack graph embeddings [batch_size, num_windows, graph_hidden_dim]
        graph_embeddings = torch.stack(all_graph_embeddings, dim=1)

        # Process each sample's graph-of-graphs
        batch_outputs = []
        for b in range(batch_size):
            sample_graphs = graph_embeddings[b]  # [num_windows, graph_hidden_dim]

            # Apply graph-of-graphs layer
            gog_features = self.gog_layer(sample_graphs)  # [num_windows, hidden_dim]

            # Aggregate across windows
            sample_output = torch.mean(gog_features, dim=0)  # [hidden_dim]
            batch_outputs.append(sample_output)

        batch_outputs = torch.stack(batch_outputs, dim=0)  # [batch_size, hidden_dim]

        # Classification
        logits = self.classifier(batch_outputs)

        return logits

    def _create_windows(self, x):
        """Split time series into windows."""
        batch_size, num_vars, seq_len = x.shape

        # Ensure seq_len is divisible by num_windows
        effective_len = self.window_size * self.num_windows
        if seq_len > effective_len:
            x = x[:, :, :effective_len]
        elif seq_len < effective_len:
            # Pad if necessary
            padding = effective_len - seq_len
            x = F.pad(x, (0, padding), mode='replicate')

        # Reshape into windows
        windows = x.reshape(batch_size, num_vars, self.num_windows, self.window_size)
        windows = windows.permute(0, 2, 1, 3)  # [batch, num_windows, num_vars, window_size]

        return windows

    def _extract_temporal_features(self, window_data):
        """Extract temporal features using LSTM."""
        batch_size, num_vars, window_size = window_data.shape

        # Reshape for LSTM: [batch * num_vars, window_size, 1]
        x_reshaped = window_data.reshape(batch_size * num_vars, window_size, 1)

        # Apply LSTM
        lstm_out, _ = self.temporal_encoder(x_reshaped)

        # Take last hidden state
        features = lstm_out[:, -1, :]  # [batch * num_vars, lstm_output_dim]

        return features


class SAGoGLite(nn.Module):
    """
    Lightweight version of SAGoG for faster training and inference.
    """

    def __init__(self, num_variables, seq_len, num_classes, hidden_dim=32,
                 graph_hidden_dim=64, num_windows=3):
        super(SAGoGLite, self).__init__()

        self.num_variables = num_variables
        self.seq_len = seq_len
        self.num_windows = num_windows
        self.window_size = seq_len // num_windows

        # Simple temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Graph construction
        self.graph_constructor = GraphConstructor(num_variables, hidden_dim, method='correlation')

        # Graph encoder
        self.graph_encoder = GraphLevelEncoder(hidden_dim, graph_hidden_dim, num_layers=1, gnn_type='gcn')

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(graph_hidden_dim * num_windows, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        batch_size, num_vars, seq_len = x.shape

        windows = self._create_windows(x)

        all_embeddings = []

        for w in range(self.num_windows):
            window_data = windows[:, w, :, :]

            # Extract features
            node_features = self._extract_features(window_data)

            # Build graph
            edge_index, edge_weight = self.graph_constructor(window_data)
            batch_vector = torch.arange(batch_size, device=x.device).repeat_interleave(num_vars)

            # Encode
            graph_features = self.graph_encoder(node_features, edge_index, edge_weight, batch_vector)
            graph_embedding = global_mean_pool(graph_features, batch_vector)

            all_embeddings.append(graph_embedding)

        # Concatenate all window embeddings
        combined = torch.cat(all_embeddings, dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def _create_windows(self, x):
        batch_size, num_vars, seq_len = x.shape
        effective_len = self.window_size * self.num_windows

        if seq_len != effective_len:
            x = F.interpolate(x, size=effective_len, mode='linear', align_corners=False)

        windows = x.reshape(batch_size, num_vars, self.num_windows, self.window_size)
        windows = windows.permute(0, 2, 1, 3)

        return windows

    def _extract_features(self, window_data):
        batch_size, num_vars, window_size = window_data.shape
        x_reshaped = window_data.reshape(batch_size * num_vars, 1, window_size)
        features = self.temporal_conv(x_reshaped).squeeze(-1)
        return features