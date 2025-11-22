import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, layers=[64, 32, 16]):
        super(NCF, self).__init__()
        
        # GMF Part (Generalized Matrix Factorization)
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
        # Concatenates GMF (embed_dim) + MLP (last_layer_size)
        predict_size = embed_dim + layers[-1]
        self.predict_layer = nn.Linear(predict_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize with Xavier/He
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        # GMF
        gmf_u = self.gmf_user_embedding(user_indices)
        gmf_i = self.gmf_item_embedding(item_indices)
        gmf_vector = gmf_u * gmf_i # Element-wise product
        
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