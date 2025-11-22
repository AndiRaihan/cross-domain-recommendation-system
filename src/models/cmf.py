import torch
import torch.nn as nn

class CMF(nn.Module):
    def __init__(self, num_users, num_src_items, num_tgt_items, embed_dim=32):
        super(CMF, self).__init__()
        
        # SHARED User Embeddings (The Bridge)
        # This matrix is updated by gradients from BOTH domains
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # Domain-Specific Item Embeddings
        self.src_item_embedding = nn.Embedding(num_src_items, embed_dim)
        self.tgt_item_embedding = nn.Embedding(num_tgt_items, embed_dim)
        
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
        
    def _init_weights(self):
        # Standard Normal initialization often works better for MF than Xavier
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.src_item_embedding.weight, std=0.01)
        nn.init.normal_(self.tgt_item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices, domain='target'):
        """
        domain: 'source' or 'target'
        """
        user_vec = self.user_embedding(user_indices)
        
        if domain == 'target':
            item_vec = self.tgt_item_embedding(item_indices)
        else:
            item_vec = self.src_item_embedding(item_indices)
            
        # Dot Product
        dot = (user_vec * item_vec).sum(dim=1)
        
        return self.sigmoid(dot)