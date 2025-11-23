import torch
import torch.nn as nn


class PTUPCDR(nn.Module):
    """
    Personalized Transfer of User Preferences for Cross-Domain Recommendation (PTUPCDR).

    This model learns a mapping function to transfer user embeddings from a source domain
    to a target domain using a meta-network approach.
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 64, dropout: float = 0.2):
        """
        Initialize the PTUPCDR model.

        Args:
            input_dim (int): Dimension of User Content Profile (e.g., 128).
            latent_dim (int): Dimension of LightGCN Embedding (e.g., 64).
            dropout (float): Dropout rate.
        """
        super(PTUPCDR, self).__init__()

        # 1. Characteristic Encoder (Compresses History -> Latent State)
        self.char_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, latent_dim),
            nn.Tanh()  # Tanh gives us a bounded state (-1 to 1)
        )

        # 2. Meta Networks (HyperNetworks)
        # Predicts Diagonal Scaling Matrix (Vector)
        self.meta_scale = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()  # Scale is usually 0.0 to 1.0
        )

        # Predicts Bias/Shift Vector
        self.meta_shift = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, source_user_emb: torch.Tensor, user_characteristic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to map source user embeddings to target domain.

        Args:
            source_user_emb (torch.Tensor): User embeddings from the source domain.
            user_characteristic (torch.Tensor): User content profile features.

        Returns:
            torch.Tensor: Mapped user embeddings for the target domain.
        """
        # 1. Get User State
        state = self.char_encoder(user_characteristic)

        # 2. Generate Parameters
        scale = self.meta_scale(state)
        shift = self.meta_shift(state)

        # 3. Apply Affine Transformation
        # Target = Source * Scale + Shift
        mapped_emb = (source_user_emb * scale) + shift

        return mapped_emb
