import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

def generate_user_profiles(data_dir, domain_prefix):
    print(f"--- Generating User Content Profiles for {domain_prefix} ---")
    
    # 1. Load Interactions (Training Data Only)
    df = pd.read_csv(os.path.join(data_dir, f'{domain_prefix}_train.csv'))
    
    # 2. Load Item Embeddings
    item_emb = np.load(os.path.join(data_dir, f'{domain_prefix}_domain_mrl.npy'))
    
    # 3. Load Global Mapping for Correct Matrix Size
    # We need the global count, not just the local max, to keep indices aligned
    with open(os.path.join(data_dir, 'user_mapping.json')) as f:
        user_map = json.load(f)
    
    # IDs are 1-based, so size is Count + 1 for padding index 0
    num_users_global = len(user_map) + 1 
    embed_dim = item_emb.shape[1]
    
    print(f"Global Users: {num_users_global}, Embedding Dim: {embed_dim}")
    
    # Initialize with zeros
    user_content_emb = np.zeros((num_users_global, embed_dim), dtype=np.float32)
    
    # 4. Aggregate (Mean Pooling)
    print("Aggregating item vectors per user...")
    grouped = df.groupby('user_id_idx')['item_id_idx'].apply(list)
    
    # Track how many users we actually found profiles for
    found_count = 0
    
    for user_id, item_ids in tqdm(grouped.items()):
        # Safety check: Ensure item IDs are within bounds of the item embedding matrix
        valid_items = [i for i in item_ids if i < len(item_emb)]
        
        if valid_items:
            vectors = item_emb[valid_items]
            # Mean pooling
            user_vector = np.mean(vectors, axis=0)
            user_content_emb[user_id] = user_vector
            found_count += 1
            
    # 5. L2 Normalization
    # This puts user vectors on the same scale as item vectors
    # We add a small epsilon to avoid division by zero for empty users
    norms = np.linalg.norm(user_content_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    user_content_emb = user_content_emb / norms
    
    # 6. Save
    out_path = os.path.join(data_dir, f'{domain_prefix}_user_content.npy')
    np.save(out_path, user_content_emb)
    print(f"Saved {domain_prefix} Matrix: {user_content_emb.shape} (Active Users: {found_count})")

if __name__ == "__main__":
    DATA_DIR = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\processed"
    
    generate_user_profiles(DATA_DIR, "source")
    generate_user_profiles(DATA_DIR, "target")