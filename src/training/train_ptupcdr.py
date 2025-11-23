from src.evaluation.metrics import evaluate_model
from src.utils.graph import build_adjacency_matrix
from src.utils.datasets import CDDataset
from src.models.ptupcdr import PTUPCDR
from src.models.lightgcn import LightGCN
import torch
import torch.optim as optim
import numpy as np
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "ptupcdr_model")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBED_DIM = 64
BATCH_SIZE = 1024
EPOCHS_GCN = 20
EPOCHS_META = 30
LR_GCN = 0.001
LR_META = 0.005

# Toggle this if you want to force retraining even if models exist
FORCE_RETRAIN = False


def run_ptupcdr():
    """
    Executes the full training pipeline for the PTUPCDR model.

    The pipeline consists of three phases:
    1.  **Target Encoder Training**: Trains a LightGCN model on the target domain.
    2.  **Source Encoder Training**: Trains a LightGCN model on the source domain.
    3.  **Meta-Network Training**: Trains the PTUPCDR mapping network to transfer
        user preferences from source to target using bridge users.

    Finally, evaluates the model on the test set.
    """
    print(f"--- Running PTUPCDR Full Pipeline on {DEVICE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Metadata
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f:
        num_users = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'source_item_mapping.json')) as f:
        num_src_items = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f:
        num_tgt_items = len(json.load(f)) + 1

    # Load Features
    src_feats = np.load(os.path.join(DATA_DIR, 'source_domain_mrl.npy'))
    tgt_feats = np.load(os.path.join(DATA_DIR, 'target_domain_mrl.npy'))
    user_content = np.load(os.path.join(DATA_DIR, 'source_user_content.npy'))
    user_content_tensor = torch.tensor(user_content).float().to(DEVICE)

    history = {"gcn_loss": [], "gcn_val_recall": [],
               "meta_loss": [], "meta_val_recall": []}
    # ====================================================
    # PHASE 1: TRAIN TARGET ENCODER (LightGCN)
    # ====================================================
    print("\n[Phase 1] Target LightGCN...")

    adj_t = build_adjacency_matrix(
        DATA_DIR, 'target_train.csv', num_users, num_tgt_items)

    model_t = LightGCN(num_users, num_tgt_items, adj_t, tgt_feats,
                       embed_dim=EMBED_DIM, device=DEVICE).to(DEVICE)

    target_model_path = os.path.join(OUTPUT_DIR, "best_target_gcn.pth")

    if os.path.exists(target_model_path) and not FORCE_RETRAIN:
        print(f"   Found existing model at {target_model_path}. Loading...")
        model_t.load_state_dict(torch.load(target_model_path))
    else:
        print("   Training from scratch...")
        opt_t = optim.Adam(model_t.parameters(), lr=LR_GCN)
        loader_t = DataLoader(CDDataset(os.path.join(DATA_DIR, 'target_train.csv'),
                              num_tgt_items, num_negatives=1), batch_size=BATCH_SIZE, shuffle=True)
        valid_loader_t = DataLoader(CDDataset(os.path.join(
            DATA_DIR, 'target_valid.csv'), num_tgt_items, is_training=False), batch_size=128)

        best_gcn_hr = 0.0
        for epoch in range(EPOCHS_GCN):
            model_t.train()
            total_loss = 0
            loop = tqdm(
                loader_t, desc=f"Target Epoch {epoch+1}/{EPOCHS_GCN}", leave=False)
            for u, p, n in loop:
                u, p, n = u.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                u_e, i_e = model_t.forward()
                pos_scores = (u_e[u] * i_e[p]).sum(dim=1)
                neg_scores = (u_e[u].unsqueeze(1) * i_e[n]).sum(dim=2)
                loss = - \
                    torch.mean(torch.log(torch.sigmoid(
                        pos_scores.unsqueeze(1) - neg_scores) + 1e-10))
                opt_t.zero_grad()
                loss.backward()
                opt_t.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # Validation
            class GCNWrapper(torch.nn.Module):
                def __init__(self, model): super(
                ).__init__(); self.model = model
                def forward(self, u, i): ue, ie = self.model(); return (
                    ue[u] * ie[i]).sum(dim=1)

            metrics = evaluate_model(GCNWrapper(
                model_t), valid_loader_t, device=DEVICE)
            print(
                f"   Epoch {epoch+1}: Loss {total_loss:.2f} | Val Recall {metrics['recall']:.4f}")
            history["gcn_loss"].append(total_loss)
            history["gcn_val_recall"].append(metrics['recall'])

            if metrics['recall'] > best_gcn_hr:
                best_gcn_hr = metrics['recall']
                torch.save(model_t.state_dict(), target_model_path)

        torch.save(model_t.state_dict(), os.path.join(
            OUTPUT_DIR, "best_target_gcn.pth"))
        # Load best before moving on
        model_t.load_state_dict(torch.load(target_model_path))

    # ====================================================
    # PHASE 1.5: TRAIN SOURCE ENCODER
    # ====================================================
    print("\n[Phase 1.5] Source LightGCN (Teacher)...")

    adj_s = build_adjacency_matrix(
        DATA_DIR, 'source_train.csv', num_users, num_src_items)

    model_s = LightGCN(num_users, num_src_items, adj_s, src_feats,
                       embed_dim=EMBED_DIM, device=DEVICE).to(DEVICE)

    source_model_path = os.path.join(OUTPUT_DIR, "best_source_gcn.pth")

    if os.path.exists(source_model_path) and not FORCE_RETRAIN:
        print(f"   Found existing model at {source_model_path}. Loading...")
        model_s.load_state_dict(torch.load(source_model_path))
    else:
        print("   Training from scratch...")
        opt_s = optim.Adam(model_s.parameters(), lr=LR_GCN)
        loader_s = DataLoader(CDDataset(os.path.join(
            DATA_DIR, 'source_train.csv'), num_src_items), batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(5):
            model_s.train()
            total_loss = 0
            loop = tqdm(
                loader_s, desc=f"Source Epoch {epoch+1}/5", leave=False)
            for u, p, n in loop:
                u, p, n = u.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                u_e, i_e = model_s.forward()
                loss = -torch.mean(torch.log(torch.sigmoid((u_e[u]*i_e[p]).sum(
                    1).unsqueeze(1) - (u_e[u].unsqueeze(1)*i_e[n]).sum(2)) + 1e-10))
                opt_s.zero_grad()
                loss.backward()
                opt_s.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            print(f"   Source Epoch {epoch+1}: Loss {total_loss:.2f}")

        # Save Source Model so we can skip next time
        torch.save(model_s.state_dict(), source_model_path)

    # ====================================================
    # PHASE 2: TRAIN MAPPING (PTUPCDR)
    # ====================================================
    print("\n[Phase 2] Training Meta-Mapping Network...")

    model_s.eval()
    model_t.eval()

    # Get Frozen Embeddings
    with torch.no_grad():
        emb_s, _ = model_s.forward()
        emb_t, final_items_t = model_t.forward()

    mapper_model_path = os.path.join(OUTPUT_DIR, "best_mapper.pth")
    mapper = PTUPCDR(input_dim=128, latent_dim=EMBED_DIM).to(DEVICE)

    if os.path.exists(mapper_model_path) and not FORCE_RETRAIN:
        print(f"   Found existing mapper at {mapper_model_path}. Loading...")
        mapper.load_state_dict(torch.load(mapper_model_path))
    else:
        df_train = pd.read_csv(os.path.join(DATA_DIR, 'target_train.csv'))
        bridge_train = df_train[df_train['is_overlap']
                                == True]['user_id_idx'].unique()
        bridge_train_tensor = torch.tensor(bridge_train, dtype=torch.long)

        opt_m = optim.Adam(mapper.parameters(), lr=LR_META)
        map_loader = DataLoader(TensorDataset(
            bridge_train_tensor), batch_size=256, shuffle=True)

        df_val = pd.read_csv(os.path.join(DATA_DIR, 'target_valid.csv'))
        val_users_tensor = torch.tensor(
            df_val['user_id_idx'].unique(), dtype=torch.long).to(DEVICE)
        best_meta_recall = 0.0

        # Wrapper
        class HybridEvaluator(torch.nn.Module):
            def __init__(self, u_emb, i_emb): super().__init__(
            ); self.u_emb = u_emb; self.i_emb = i_emb
            def forward(self, u, i): return (
                self.u_emb[u] * self.i_emb[i]).sum(dim=1)

        valid_loader_t = DataLoader(CDDataset(os.path.join(
            DATA_DIR, 'target_valid.csv'), num_tgt_items, is_training=False), batch_size=128)

        for epoch in range(EPOCHS_META):
            mapper.train()
            total_loss = 0
            loop = tqdm(
                map_loader, desc=f"Meta Epoch {epoch+1}/{EPOCHS_META}", leave=False)

            for batch in loop:
                u_idx = batch[0].to(DEVICE)
                src_emb = emb_s[u_idx]
                tgt_emb = emb_t[u_idx]
                char_emb = user_content_tensor[u_idx]

                mapped_emb = mapper(src_emb, char_emb)
                loss = torch.nn.functional.mse_loss(mapped_emb, tgt_emb)

                opt_m.zero_grad()
                loss.backward()
                opt_m.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # Validation
            mapper.eval()
            with torch.no_grad():
                fake_tgt_val = mapper(
                    emb_s[val_users_tensor], user_content_tensor[val_users_tensor])
                hybrid_emb = emb_t.clone()
                hybrid_emb[val_users_tensor] = fake_tgt_val
                metrics = evaluate_model(HybridEvaluator(
                    hybrid_emb, final_items_t), valid_loader_t, device=DEVICE)

            print(
                f"   Epoch {epoch+1}: MSE {total_loss:.4f} | Val Recall {metrics['recall']:.4f}")
            history["meta_loss"].append(total_loss)
            history["meta_val_recall"].append(metrics['recall'])

            if metrics['recall'] > best_meta_recall:
                best_meta_recall = metrics['recall']
                torch.save(mapper.state_dict(), mapper_model_path)

        mapper.load_state_dict(torch.load(mapper_model_path))

    # ====================================================
    # PHASE 3: FINAL EVALUATION
    # ====================================================
    print("\n[Phase 3] Final Test Evaluation...")
    mapper.eval()

    df_test = pd.read_csv(os.path.join(DATA_DIR, 'target_test.csv'))
    test_users = df_test['user_id_idx'].unique()
    test_idx = torch.tensor(test_users, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        fake_test_emb = mapper(emb_s[test_idx], user_content_tensor[test_idx])
        final_user_emb = emb_t.clone()
        final_user_emb[test_idx] = fake_test_emb

        # Re-define Wrapper locally just in case
        class HybridEvaluator(torch.nn.Module):
            def __init__(self, u_emb, i_emb): super().__init__(
            ); self.u_emb = u_emb; self.i_emb = i_emb

            def forward(self, u, i): return (
                self.u_emb[u] * self.i_emb[i]).sum(dim=1)

        final_model = HybridEvaluator(final_user_emb, final_items_t)

        test_loader = DataLoader(CDDataset(os.path.join(
            DATA_DIR, 'target_test.csv'), num_tgt_items, is_training=False), batch_size=100)
        metrics = evaluate_model(final_model, test_loader, device=DEVICE)

    print(f"\nFINAL PTUPCDR RESULTS:")
    print(f"Recall@10:    {metrics['recall']:.4f}")
    print(f"NDCG@10:      {metrics['ndcg']:.4f}")
    print(f"MAP:          {metrics['map']:.4f}")
    print(f"Precision@10: {metrics['precision']:.4f}")
    print(f"F1@10:        {metrics['f1']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    run_ptupcdr()
