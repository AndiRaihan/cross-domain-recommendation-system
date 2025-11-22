import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import sys
import time
from tqdm import tqdm

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.cmf import CMF
from src.utils.datasets import CDDataset
from src.evaluation.metrics import evaluate_model

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "cmf_baseline")

BATCH_SIZE = 4096
LR = 0.001
EPOCHS = 30 
EMBED_DIM = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SOURCE_WEIGHT = 0.5 # How much we care about Source Loss (Alpha)

def run():
    print(f"--- Running CMF (Cross-Domain MF) Baseline on {DEVICE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Get Dimensions
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f:
        num_users = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'source_item_mapping.json')) as f:
        num_src_items = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f:
        num_tgt_items = len(json.load(f)) + 1
        
    print(f"Users: {num_users}, Source Items: {num_src_items}, Target Items: {num_tgt_items}")
    
    # 2. Load Data (WE NEED BOTH SOURCE AND TARGET TRAIN)
    print("Loading Datasets...")
    
    # Primary Loader (Target)
    tgt_train_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'target_train.csv'), num_tgt_items, num_negatives=4, is_training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    
    # Auxiliary Loader (Source) - Used to refine User Embeddings
    src_train_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'source_train.csv'), num_src_items, num_negatives=4, is_training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    
    # Validation Loader (Target Only)
    valid_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'target_valid.csv'), num_tgt_items, is_training=False),
        batch_size=128, shuffle=False
    )
    
    # 3. Init Model
    model = CMF(num_users, num_src_items, num_tgt_items, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()
    
    # History Tracking
    history = {
        "epoch": [], "loss": [], "val_recall": [], "val_ndcg": [], 
        "val_precision": [], "val_f1": [], "val_map": []
    }
    best_hr = 0.0
    
    # 4. Train Loop
    # We make the Source loader an iterator so we can cycle through it
    src_iter = iter(src_train_loader)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # We iterate based on Target Loader length (since that's our goal)
        train_loop = tqdm(tgt_train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for users_t, pos_t, neg_t in train_loop:
            users_t, pos_t, neg_t = users_t.to(DEVICE), pos_t.to(DEVICE), neg_t.to(DEVICE)
            
            # --- A. TARGET DOMAIN UPDATE ---
            pos_preds_t = model(users_t, pos_t, domain='target')
            pos_loss_t = criterion(pos_preds_t, torch.ones_like(pos_preds_t))
            
            # Negative Target
            neg_t_flat = neg_t.view(-1)
            users_t_exp = users_t.unsqueeze(1).repeat(1, neg_t.size(1)).view(-1)
            neg_preds_t = model(users_t_exp, neg_t_flat, domain='target')
            neg_loss_t = criterion(neg_preds_t, torch.zeros_like(neg_preds_t))
            
            loss_tgt = pos_loss_t + neg_loss_t

            # --- B. SOURCE DOMAIN UPDATE (The CMF Magic) ---
            # Try to get next source batch, restart if exhausted
            try:
                users_s, pos_s, neg_s = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_loader)
                users_s, pos_s, neg_s = next(src_iter)
            
            users_s, pos_s, neg_s = users_s.to(DEVICE), pos_s.to(DEVICE), neg_s.to(DEVICE)
            
            # Positive Source
            pos_preds_s = model(users_s, pos_s, domain='source')
            loss_src_pos = criterion(pos_preds_s, torch.ones_like(pos_preds_s))
            
            # Negative Source
            neg_s_flat = neg_s.view(-1)
            users_s_exp = users_s.unsqueeze(1).repeat(1, neg_s.size(1)).view(-1)
            neg_preds_s = model(users_s_exp, neg_s_flat, domain='source')
            loss_src_neg = criterion(neg_preds_s, torch.zeros_like(neg_preds_s))
            
            loss_src = loss_src_pos + loss_src_neg
            
            # --- C. COMBINED BACKPROP ---
            # We optimize Target Loss + (Weighted) Source Loss
            loss = loss_tgt + (SOURCE_WEIGHT * loss_src)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(tgt_train_loader)
        
        # 5. Evaluate per Epoch (Only on Target Valid)
        metrics = evaluate_model(model, valid_loader, device=DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Recall@{10}: {metrics['recall']:.4f} | Val NDCG@{10}: {metrics['ndcg']:.4f} | Val MAP: {metrics['map']:.4f}")
        
        history["epoch"].append(epoch + 1)
        history["loss"].append(total_loss)
        history["val_recall"].append(metrics['recall'])
        history["val_ndcg"].append(metrics['ndcg'])
        history["val_precision"].append(metrics['precision'])
        history["val_f1"].append(metrics['f1'])
        history["val_map"].append(metrics['map'])
        
        if metrics['recall'] > best_hr:
            best_hr = metrics['recall']
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_cmf_model.pth"))
            print("   -> New Best Model Saved!")

    # 6. Save Metrics
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # 7. Final Cold-Start Evaluation
    print("\n--- Final Evaluation on Cold-Start Test Set ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_cmf_model.pth")))
    
    test_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'target_test.csv'), num_tgt_items, is_training=False),
        batch_size=100, shuffle=False
    )
    
    metrics = evaluate_model(model, test_loader, device=DEVICE)
    test_recall = metrics['recall']
    test_ndcg = metrics['ndcg']
    test_precision = metrics["precision"]
    test_f1 = metrics["f1"]
    test_map = metrics["map"]
    
    print(f"COLD START TEST: R@10 = {test_recall:.4f}, NDCG@10 = {test_ndcg:.4f}, MAP = {test_map:.4f}")
    
    final_results = {"test_recall": test_recall, "test_ndcg": test_ndcg, "test_map": test_map, "test_precision": test_precision, "test_f1": test_f1}
    with open(os.path.join(OUTPUT_DIR, "final_test_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"Done! Artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run()