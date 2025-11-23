import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LightGCN(nn.Module):
    """
    LightGCN model with optional content feature fusion.

    This model uses Graph Convolutional Networks to learn user and item embeddings
    by propagating information on the user-item bipartite graph. It can optionally
    fuse pre-computed content features (e.g., from text embeddings) into the item embeddings.
    """

    def __init__(self, num_users: int, num_items: int, graph: torch.sparse.FloatTensor,
                 feature_matrix: np.ndarray = None, embed_dim: int = 64, n_layers: int = 3, device: str = 'cuda'):
        """
        Initialize the LightGCN model.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            graph (torch.sparse.FloatTensor): Adjacency matrix of the user-item graph.
            feature_matrix (np.ndarray, optional): Pre-computed item feature matrix.
            embed_dim (int): Dimension of the embedding vectors.
            n_layers (int): Number of GCN layers.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.graph = graph.to(device)
        self.n_layers = n_layers
        self.device = device

        # ID Embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Content Signal (Qwen MRL)
        if feature_matrix is not None:
            self.use_features = True
            # Move to GPU once during init to prevent slow copy loop
            self.feat_matrix = torch.tensor(
                feature_matrix, dtype=torch.float32).to(device)
            self.feat_projector = nn.Linear(feature_matrix.shape[1], embed_dim)
        else:
            self.use_features = False

        self._init_weights()

    def _init_weights(self):
        """
        Initialize embeddings with Xavier Uniform distribution.
        """
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        if self.use_features:
            nn.init.normal_(self.feat_projector.weight, std=0.01)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate user and item embeddings.

        Returns:
            tuple: (user_embeddings, item_embeddings)
        """
        # Initial Embeddings
        u_e = self.user_embedding.weight
        i_e = self.item_embedding.weight

        # Fuse Text Features (Project & Add)
        if self.use_features:
            # Project 128d -> 64d
            feat_emb = self.feat_projector(self.feat_matrix)
            # Normalize and add to Item Embeddings
            i_e = i_e + F.normalize(feat_emb, dim=1)

        all_emb = torch.cat([u_e, i_e])
        embs = [all_emb]

        # Graph Propagation
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)

        # Aggregation
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)

        users, items = torch.split(
            final_embs, [self.num_users, self.num_items])
        return users, items
