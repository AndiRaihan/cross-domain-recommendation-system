import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.

    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    to model non-linear user-item interactions.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32, layers: list = [64, 32, 16]):
        """
        Initialize the NCF model.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embed_dim (int): Dimension of the embedding vectors.
            layers (list): List of hidden layer sizes for the MLP.
        """
        super(NCF, self).__init__()

        # GMF Part
        self.gmf_user_embedding = nn.Embedding(num_users, embed_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embed_dim)

        # MLP Part
        self.mlp_user_embedding = nn.Embedding(num_users, embed_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embed_dim)

        mlp_modules = []
        # Input size is embed_dim * 2 (User + Item)
        input_size = embed_dim * 2
        for output_size in layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Final Prediction Layer
        predict_size = embed_dim + layers[-1]
        self.predict_layer = nn.Linear(predict_size, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with Xavier/He and Normal distributions.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            user_indices (torch.Tensor): Tensor of user indices.
            item_indices (torch.Tensor): Tensor of item indices.

        Returns:
            torch.Tensor: Predicted interaction probability.
        """
        # GMF
        gmf_u = self.gmf_user_embedding(user_indices)
        gmf_i = self.gmf_item_embedding(item_indices)
        gmf_vector = gmf_u * gmf_i

        # MLP
        mlp_u = self.mlp_user_embedding(user_indices)
        mlp_i = self.mlp_item_embedding(item_indices)
        mlp_vector = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)

        # Concat
        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)

        # Prediction
        logits = self.predict_layer(vector)
        return self.sigmoid(logits).view(-1)
