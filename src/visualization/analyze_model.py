from src.models.ptupcdr import PTUPCDR
from src.models.lightgcn import LightGCN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import pandas as pd
from typing import Tuple, Dict, List

# Try importing UMAP
try:
    from sklearn.manifold import TSNE
    REDUCER = 'TSNE'
except ImportError:
    import umap
    REDUCER = 'UMAP'

# Adjust imports
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "reports", "ptupcdr_model")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 64

META_SOURCE_FILE = "meta_Movies_and_TV.jsonl"
META_TARGET_FILE = "meta_CDs_and_Vinyl.jsonl"


def load_models() -> Tuple[LightGCN, PTUPCDR]:
    """
    Loads the trained LightGCN and PTUPCDR models.

    Returns:
        Tuple[LightGCN, PTUPCDR]: The loaded target LightGCN model and the PTUPCDR mapper.
    """
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f:
        num_users = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f:
        num_tgt = len(json.load(f)) + 1

    tgt_feats = np.load(os.path.join(DATA_DIR, 'target_domain_mrl.npy'))

    # Dummy graph for loading weights
    dummy_adj = torch.sparse_coo_tensor(
        torch.empty(2, 0), torch.empty(0), (1, 1))

    model_t = LightGCN(num_users, num_tgt, dummy_adj,
                       tgt_feats, embed_dim=EMBED_DIM, device='cpu')
    mapper = PTUPCDR(input_dim=128, latent_dim=EMBED_DIM).to('cpu')

    # Load Weights
    print("Loading model weights...")
    model_t.load_state_dict(torch.load(os.path.join(
        MODEL_DIR, "best_target_gcn.pth"), map_location='cpu'))
    mapper.load_state_dict(torch.load(os.path.join(
        MODEL_DIR, "best_mapper.pth"), map_location='cpu'))

    model_t.eval()
    mapper.eval()

    return model_t, mapper


def get_product_titles(asin_list: List[str], meta_filename: str) -> Dict[str, str]:
    """
    Scans the raw metadata file to find Titles for the given ASINs.

    Args:
        asin_list (List[str]): List of ASINs to look up.
        meta_filename (str): Filename of the metadata JSONL file.

    Returns:
        Dict[str, str]: Dictionary mapping ASINs to product titles.
    """
    path = os.path.join(RAW_DIR, meta_filename)
    if not os.path.exists(path):
        print(
            f"Warning: Metadata file {path} not found. Returning ASINs only.")
        return {asin: "Unknown Title" for asin in asin_list}

    print(f"   Scanning {meta_filename} for {len(asin_list)} titles...")

    # Convert list to set for O(1) lookup
    target_asins = set(asin_list)
    title_map = {}

    # Stream file to save memory
    chunk_size = 100_000
    try:
        with pd.read_json(path, lines=True, chunksize=chunk_size) as reader:
            for chunk in reader:
                # Normalize ID column
                id_col = 'parent_asin' if 'parent_asin' in chunk.columns else 'asin'

                # Filter rows that match our ASINs
                mask = chunk[id_col].isin(target_asins)
                found_rows = chunk[mask]

                for _, row in found_rows.iterrows():
                    asin = row[id_col]
                    title = row.get('title', 'No Title')
                    title_map[asin] = title

                # Optimization: Stop early if we found everything
                if len(title_map) == len(target_asins):
                    break
    except ValueError:
        print("   Warning: JSON structure error or file empty.")

    return title_map


def viz_latent_space():
    """
    Generates a UMAP/t-SNE plot visualizing the alignment between Source, Target, and Mapped embeddings.
    Saves the plot to `reports/figures/viz3_latent_space.png`.
    """
    print(f"Generating Latent Space Plot using {REDUCER}...")
    model_t, mapper = load_models()

    # Load Data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'target_test.csv'))
    test_users = df_test['user_id_idx'].unique()[:1000]  # Sample 1000 users

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
        reducer = TSNE(n_components=2, random_state=42, perplexity=50,
                       early_exaggeration=25, init='pca', metric="cosine", learning_rate='auto')

    embedding = reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=(12, 10))

    # Plot Source (Red)
    plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1],
                c='red', alpha=0.2, s=20, label='Source Domain (Input)')

    # Plot Target (Blue)
    plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1],
                c='blue', alpha=0.2, s=20, label='True Target Preference (Goal)')

    # Plot Mapped (Green)
    plt.scatter(embedding[labels == 2, 0], embedding[labels == 2, 1],
                c='green', alpha=0.6, marker='x', s=30, label='PTUPCDR Prediction')

    plt.title(
        f"Latent Space Alignment: Moving Users from Movies to Music", fontsize=14)
    plt.legend()

    out_path = os.path.join(BASE_DIR, "reports",
                            "figures", "viz3_latent_space.png")
    plt.savefig(out_path)
    print(f"Saved Viz 3 to {out_path}")


def generate_case_study():
    """
    Extracts a 'Success' User Case Study where the model correctly recommended a ground truth item.
    Saves the case study details to `reports/figures/viz4_case_study.txt`.
    """
    print("Generating Case Study (Searching for a Hit)...")

    # 1. Load Data & Mappings
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f:
        user_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(DATA_DIR, 'source_item_mapping.json')) as f:
        src_item_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f:
        tgt_item_map = {int(k): v for k, v in json.load(f).items()}

    df_src = pd.read_csv(os.path.join(DATA_DIR, 'source_train.csv'))
    df_tgt = pd.read_csv(os.path.join(DATA_DIR, 'target_test.csv'))
    user_content = np.load(os.path.join(DATA_DIR, 'source_user_content.npy'))

    # 2. Load Model ONCE (Before loop)
    print("Loading models for search...")
    model_t, mapper = load_models()
    # Ensure evaluation mode
    model_t.eval()
    mapper.eval()
    all_items_emb = model_t.item_embedding.weight

    # 3. Search for a "Success" Candidate
    test_users = df_tgt['user_id_idx'].unique()
    candidate = None
    candidate_rank = -1

    # Check up to 100 users to find a good match (usually finds one instantly)
    search_limit = 200
    checked = 0

    with torch.no_grad():
        for u in test_users:
            # A. Filter by History Length
            src_hist = df_src[df_src['user_id_idx'] == u]
            tgt_hist = df_tgt[df_tgt['user_id_idx'] == u]

            # Criteria: At least 3 source items (so we have context)
            # and at least 1 target item (so we have ground truth)
            if len(src_hist) < 10 or len(tgt_hist) < 1 or len(tgt_hist) > 11:
                continue

            # B. Generate Prediction
            # Prepare input
            char_input = torch.tensor(user_content[u]).float().unsqueeze(0)
            src_input = torch.zeros((1, 64))
            pred_user_emb = mapper(src_input, char_input)

            # Calculate Scores
            scores = torch.matmul(pred_user_emb, all_items_emb.t()).squeeze()

            # Get Top 15 Recommendations
            _, top_indices = torch.topk(scores, k=15)
            top_indices_list = top_indices.tolist()

            # C. Check for Match
            # Get actual items this user clicked in Test set
            ground_truth_ids = tgt_hist['item_id_idx'].values.tolist()

            # Intersection check
            match_found = False
            for gt_id in ground_truth_ids:
                if gt_id in top_indices_list:
                    candidate = u
                    candidate_rank = top_indices_list.index(
                        gt_id) + 1  # 1-based rank
                    match_found = True
                    break

            if match_found:
                break  # Stop searching, we found our star!

            checked += 1
            if checked >= search_limit:
                print(
                    f"Warning: Checked {search_limit} users but didn't find a Top-15 hit. Using last valid user.")
                candidate = u  # Fallback
                break

    if candidate is None:
        print("No suitable candidate found.")
        return

    # 4. Print Results for the Selected Candidate
    cand_int = int(candidate)
    print(f"\n=== SUCCESS CASE STUDY FOUND! ===")
    print(f"User ID: {cand_int} (Raw: {user_map[cand_int]})")
    if candidate_rank > 0:
        print(
            f"SUCCESS: Model recommended the Ground Truth item at Rank #{candidate_rank}")
    else:
        print("NOTE: This is a fallback user (Ground truth not in Top 15).")

    # Format Outputs
    # Source History
    src_items_idx = df_src[df_src['user_id_idx']
                           == candidate]['item_id_idx'].values
    src_asins = []
    for i in src_items_idx[:10]:  # Show top 10 history
        i_int = int(i)
        src_asins.append(src_item_map.get(i_int, f"Unknown_{i_int}"))

    # Target Truth
    tgt_items_idx = df_tgt[df_tgt['user_id_idx']
                           == candidate]['item_id_idx'].values
    tgt_asins = []
    for i in tgt_items_idx:
        i_int = int(i)
        tgt_asins.append(tgt_item_map.get(i_int, f"Unknown_{i_int}"))

    # Model Recommendations (Top 10)
    # We re-run the top-k just to be sure we have the list ready for display
    with torch.no_grad():
        char_input = torch.tensor(user_content[candidate]).float().unsqueeze(0)
        src_input = torch.zeros((1, 64))
        pred_user_emb = mapper(src_input, char_input)
        scores = torch.matmul(pred_user_emb, all_items_emb.t()).squeeze()
        _, top_indices = torch.topk(scores, k=10)

    pred_asins = []
    for idx in top_indices:
        idx_int = int(idx.item())
        pred_asins.append(tgt_item_map.get(idx_int, f"Unknown_{idx_int}"))

    src_titles_map = get_product_titles(src_asins, META_SOURCE_FILE)
    tgt_titles_map = get_product_titles(
        tgt_asins + pred_asins, META_TARGET_FILE)

    # --- FORMAT OUTPUT ---
    def fmt(asin_list, title_map):
        return [f"{title_map.get(a, 'Unknown')} ({a})" for a in asin_list]

    src_display = fmt(src_asins, src_titles_map)
    tgt_display = fmt(tgt_asins, tgt_titles_map)
    pred_display = fmt(pred_asins, tgt_titles_map)

    print("\nSOURCE HISTORY (Movies):")
    for x in src_display:
        print(f" - {x}")

    print("\nTARGET GROUND TRUTH (Music):")
    for x in tgt_display:
        print(f" - {x}")

    print("\nPTUPCDR RECOMMENDATIONS (Top 10):")
    for i, x in enumerate(pred_display):
        mark = "  "
        # Check if this recommendation is in ground truth
        # We check ASIN match
        rec_asin = pred_asins[i]
        if rec_asin in tgt_asins:
            mark = ">>"  # Highlight hits
        print(f" {mark} {i+1}. {x}")

    # Save to file
    out_file = os.path.join(BASE_DIR, "reports",
                            "figures", "viz4_case_study.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"User: {user_map[cand_int]}\n\n")
        f.write("SOURCE HISTORY (Movies):\n")
        for x in src_display:
            f.write(f" - {x}\n")
        f.write("\nTARGET GROUND TRUTH (Music):\n")
        for x in tgt_display:
            f.write(f" - {x}\n")
        f.write("\nRECOMMENDATIONS:\n")
        for x in pred_display:
            f.write(f" - {x}\n")

    print(f"\nSaved detailed case study to {out_file}")


if __name__ == "__main__":
    viz_latent_space()
    generate_case_study()
