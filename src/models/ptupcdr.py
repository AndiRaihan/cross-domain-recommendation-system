import torch
import torch.nn as nn

class PTUPCDR(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64, dropout=0.2):
        """
        Args:
            input_dim: Dimension of User Content Profile (128)
            latent_dim: Dimension of LightGCN Embedding (64)
        """
        super(PTUPCDR, self).__init__()
        
        # 1. Characteristic Encoder (Compresses History -> Latent State)
        self.char_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, latent_dim),
            nn.Tanh() # Tanh gives us a bounded state (-1 to 1)
        )
        
        # 2. Meta Networks (HyperNetworks)
        # Predicts Diagonal Scaling Matrix (Vector)
        self.meta_scale = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid() # Scale is usually 0.0 to 1.0
        )
        
        # Predicts Bias/Shift Vector
        self.meta_shift = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, source_user_emb, user_characteristic):
        # 1. Get User State
        state = self.char_encoder(user_characteristic)
        
        # 2. Generate Parameters
        scale = self.meta_scale(state)
        shift = self.meta_shift(state)
        
        # 3. Apply Affine Transformation
        # Target = Source * Scale + Shift
        mapped_emb = (source_user_emb * scale) + shift
        
        return mapped_emb