import torch
import torch.nn as nn


class CMF(nn.Module):
    """
    Collaborative Matrix Factorization (CMF) model for cross-domain recommendation.

    This model learns shared user embeddings across domains while maintaining domain-specific
    item embeddings.
    """

    def __init__(self, num_users: int, num_src_items: int, num_tgt_items: int, embed_dim: int = 32):
        """
        Initialize the CMF model.

        Args:
            num_users (int): Total number of unique users (global).
            num_src_items (int): Number of items in the source domain.
            num_tgt_items (int): Number of items in the target domain.
            embed_dim (int): Dimension of the embedding vectors.
        """
        super(CMF, self).__init__()

        # SHARED User Embeddings (The Bridge)
        self.user_embedding = nn.Embedding(num_users, embed_dim)

        # Domain-Specific Item Embeddings
        self.src_item_embedding = nn.Embedding(num_src_items, embed_dim)
        self.tgt_item_embedding = nn.Embedding(num_tgt_items, embed_dim)

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """
        Initialize embeddings with normal distribution.
        """
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.src_item_embedding.weight, std=0.01)
        nn.init.normal_(self.tgt_item_embedding.weight, std=0.01)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, domain: str = 'target') -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            user_indices (torch.Tensor): Tensor of user indices.
            item_indices (torch.Tensor): Tensor of item indices.
            domain (str): Domain indicator ('source' or 'target').

        Returns:
            torch.Tensor: Predicted interaction scores (probabilities).
        """
        user_vec = self.user_embedding(user_indices)

        if domain == 'target':
            item_vec = self.tgt_item_embedding(item_indices)
        else:
            item_vec = self.src_item_embedding(item_indices)

        dot = (user_vec * item_vec).sum(dim=1)

        return self.sigmoid(dot)
