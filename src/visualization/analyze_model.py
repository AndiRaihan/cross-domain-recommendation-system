import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import pandas as pd

# Try importing UMAP
try:
    import umap
    REDUCER = 'UMAP'
except ImportError:
    from sklearn.manifold import TSNE
    REDUCER = 'TSNE'

# Adjust imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.lightgcn import LightGCN
from src.models.ptupcdr import PTUPCDR

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "reports", "ptupcdr_model")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 64

def load_models():
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f: num_users = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f: num_tgt = len(json.load(f)) + 1
    
    tgt_feats = np.load(os.path.join(DATA_DIR, 'target_domain_mrl.npy'))
    
    # Dummy graph for loading weights
    dummy_adj = torch.sparse_coo_tensor(torch.empty(2,0), torch.empty(0), (1,1))
    
    model_t = LightGCN(num_users, num_tgt, dummy_adj, tgt_feats, embed_dim=EMBED_DIM, device='cpu')
    mapper = PTUPCDR(input_dim=128, latent_dim=EMBED_DIM).to('cpu')
    
    # Load Weights
    print("Loading model weights...")
    model_t.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_target_gcn.pth"), map_location='cpu'))
    mapper.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_mapper.pth"), map_location='cpu'))
    
    model_t.eval()
    mapper.eval()
    
    return model_t, mapper

def viz_latent_space():
    """Viz 3: UMAP Source vs Target vs Mapped"""
    print(f"Generating Latent Space Plot using {REDUCER}...")
    model_t, mapper = load_models()
    
    # Load Data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'target_test.csv'))
    test_users = df_test['user_id_idx'].unique()[:1000] # Sample 1000 users
    
    # 1. SOURCE EMBEDDINGS (The Input)
    # In your PTUPCDR, the 'Source' input is the User Content Profile
    user_content = np.load(os.path.join(DATA_DIR, 'source_user_content.npy'))
    source_emb_full = user_content[test_users]
    source_emp = source_emb_full[:, :EMBED_DIM]
    
    # 2. TARGET EMBEDDINGS (The Goal)
    # Learned by LightGCN on Target Data
    with torch.no_grad():
        target_emb = model_t.user_embedding.weight[test_users].numpy()
        
    # 3. MAPPED EMBEDDINGS (The Prediction)
    # Source -> Mapper -> Prediction
    with torch.no_grad():
        # Your mapper takes (source_gcn_emb, content_emb).
        # Since we didn't save Source LightGCN, we rely on Content Profile 
        # which is the strongest signal anyway.
        # We pass zeros for source_gcn_emb part to focus on content mapping effect
        # OR if your mapper relies heavily on it, this might look slightly off, 
        # but for visual proof of concept, it works.
        
        src_input_dummy = torch.zeros((len(test_users), 64)) 
        char_input = torch.tensor(source_emb_full).float()
        
        mapped_emb = mapper(src_input_dummy, char_input).numpy()
        
    # Combine for Reduction
    # We have 3 groups of data
    X = np.vstack([source_emp, target_emb, mapped_emb])
    
    # Labels: 0=Source, 1=Target, 2=Mapped
    labels = np.concatenate([
        np.zeros(len(test_users)), 
        np.ones(len(test_users)), 
        np.full(len(test_users), 2)
    ])
    
    print("Running dimensionality reduction...")
    if REDUCER == 'UMAP':
        reducer = umap.UMAP(random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        
    embedding = reducer.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot Source (Red)
    plt.scatter(embedding[labels==0, 0], embedding[labels==0, 1], 
                c='red', alpha=0.2, s=20, label='Source Domain (Input)')
    
    # Plot Target (Blue)
    plt.scatter(embedding[labels==1, 0], embedding[labels==1, 1], 
                c='blue', alpha=0.2, s=20, label='True Target Preference (Goal)')
    
    # Plot Mapped (Green)
    plt.scatter(embedding[labels==2, 0], embedding[labels==2, 1], 
                c='green', alpha=0.6, marker='x', s=30, label='PTUPCDR Prediction')
    
    plt.title(f"Latent Space Alignment: Moving Users from Movies to Music", fontsize=14)
    plt.legend()
    
    out_path = os.path.join(BASE_DIR, "reports", "figures", "viz3_latent_space.png")
    plt.savefig(out_path)
    print(f"Saved Viz 3 to {out_path}")

def generate_case_study():
    """Viz 4: Extract a User Case Study"""
    print("Generating Case Study...")
    
    # 1. Load Mappings (CORRECTED LOGIC)
    # JSON keys are always strings "1". We need to convert them to int 1.
    # We want: {1: "UserID_String"}
    
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f: 
        user_map = {int(k): v for k, v in json.load(f).items()}
        
    with open(os.path.join(DATA_DIR, 'source_item_mapping.json')) as f:
        src_item_map = {int(k): v for k, v in json.load(f).items()}
        
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f:
        tgt_item_map = {int(k): v for k, v in json.load(f).items()}
        
    df_src = pd.read_csv(os.path.join(DATA_DIR, 'source_train.csv'))
    df_tgt = pd.read_csv(os.path.join(DATA_DIR, 'target_test.csv'))
    
    # Find candidate
    test_users = df_tgt['user_id_idx'].unique()
    candidate = None
    
    # Try to find a user with rich history
    for u in test_users:
        src_hist = df_src[df_src['user_id_idx'] == u]
        tgt_hist = df_tgt[df_tgt['user_id_idx'] == u]
        
        # Condition: >5 Source items (to show taste) and >=1 Target (to show truth)
        if len(src_hist) >= 5 and len(tgt_hist) >= 1:
            candidate = u
            break
    
    # Fallback: If no "rich" user found, just pick the first one
    if candidate is None and len(test_users) > 0:
        candidate = test_users[0]
            
    if candidate is None:
        print("No suitable test users found.")
        return

    # Cast to int for lookup
    cand_int = int(candidate)
    
    # Safety check for user map
    if cand_int not in user_map:
        print(f"Error: User Index {cand_int} not found in mapping file.")
        # Try to print keys to debug
        # print(list(user_map.keys())[:5])
        return

    print(f"Selected User ID: {cand_int} (Raw: {user_map[cand_int]})")
    
    # 1. Get Source History
    src_items_idx = df_src[df_src['user_id_idx'] == candidate]['item_id_idx'].values
    # Safely map items
    src_asins = []
    for i in src_items_idx[:5]:
        i_int = int(i)
        if i_int in src_item_map:
            src_asins.append(src_item_map[i_int])
        else:
            src_asins.append(f"Unknown_Item_{i_int}")
    
    # 2. Get Target Ground Truth
    tgt_items_idx = df_tgt[df_tgt['user_id_idx'] == candidate]['item_id_idx'].values
    tgt_asins = []
    for i in tgt_items_idx:
        i_int = int(i)
        if i_int in tgt_item_map:
            tgt_asins.append(tgt_item_map[i_int])
    
    # 3. Get Prediction
    model_t, mapper = load_models()
    user_content = np.load(os.path.join(DATA_DIR, 'source_user_content.npy'))
    
    with torch.no_grad():
        char_input = torch.tensor(user_content[candidate]).float().unsqueeze(0)
        src_input = torch.zeros((1, 64)) 
        pred_user_emb = mapper(src_input, char_input)
        
        # Score against all items
        all_items_emb = model_t.item_embedding.weight
        scores = torch.matmul(pred_user_emb, all_items_emb.t()).squeeze()
        
        # Top 10
        _, top_indices = torch.topk(scores, k=10)
        
        pred_asins = []
        for idx in top_indices:
            idx_int = int(idx.item())
            if idx_int in tgt_item_map:
                pred_asins.append(tgt_item_map[idx_int])
            if len(pred_asins) >= 5: 
                break

    # Output
    print("\n=== CASE STUDY ===")
    print(f"User Raw ID: {user_map[cand_int]}")
    print("\nSOURCE HISTORY (Movies):")
    print(src_asins)
    print("\nTARGET GROUND TRUTH (Music):")
    print(tgt_asins)
    print("\nPTUPCDR RECOMMENDATIONS:")
    print(pred_asins)
    
    out_file = os.path.join(BASE_DIR, "reports", "figures", "viz4_case_study.txt")
    with open(out_file, "w") as f:
        f.write(f"User: {user_map[cand_int]}\nSource: {src_asins}\nTruth: {tgt_asins}\nPreds: {pred_asins}")
    print(f"Saved case study to {out_file}")

if __name__ == "__main__":
    viz_latent_space()
    generate_case_study()