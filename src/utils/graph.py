import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os

def build_adjacency_matrix(data_dir, filename, num_users, num_items):
    """
    Builds the normalized adjacency matrix.
    Args:
        num_users: Global user count (including padding)
        num_items: Domain-specific item count (including padding)
    """
    print(f"   [Graph] Building Adjacency Matrix from {filename}...")
    
    # 1. Load Interactions
    df = pd.read_csv(os.path.join(data_dir, filename))
    src = df['user_id_idx'].values
    dst = df['item_id_idx'].values
    
    # 2. Create User-Item Bipartite Graph
    # User nodes: 0 to num_users-1
    # Item nodes: num_users to num_users+num_items-1
    dst = dst + num_users 
    
    # Create interactions (bidirectional)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.ones(len(row))
    
    # Coo Matrix
    num_nodes = num_users + num_items
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # 3. Normalize (D^-0.5 * A * D^-0.5)
    rowsum = np.array(adj.sum(1))
    
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    # 4. Convert to PyTorch Sparse Tensor
    coo = norm_adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    
    sparse_adj = torch.sparse_coo_tensor(indices, values, coo.shape)
    
    return sparse_adj