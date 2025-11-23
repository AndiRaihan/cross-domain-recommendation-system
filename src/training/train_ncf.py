from src.evaluation.metrics import evaluate_model
from src.utils.datasets import CDDataset
from src.models.ncf import NCF
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import sys
import time
from tqdm import tqdm

# Fix path so we can import from src
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


# --- CONFIG ---
# Adjust paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(__file__)))  # Goes up to Root
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "baseline")

BATCH_SIZE = 4096
LR = 0.001
EPOCHS = 30 # Increased slightly to see curves
EMBED_DIM = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run():
    """
    Runs the training pipeline for the NCF (Neural Collaborative Filtering) baseline model.

    Steps:
    1. Loads dataset dimensions and files.
    2. Initializes DataLoaders for training and validation.
    3. Initializes the NCF model, optimizer, and loss function.
    4. Runs the training loop using binary cross-entropy loss.
    5. Evaluates on validation set per epoch.
    6. Saves the best model and training history.
    7. Performs final evaluation on the cold-start test set.
    """
    print(f"--- Running NCF Baseline on {DEVICE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Get Dimensions
    with open(os.path.join(DATA_DIR, 'user_mapping.json')) as f:
        num_users = len(json.load(f)) + 1
    with open(os.path.join(DATA_DIR, 'target_item_mapping.json')) as f:
        num_items = len(json.load(f)) + 1

    # 2. Load Data
    print("Loading Datasets...")
    train_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'target_train.csv'),
                  num_items, num_negatives=4, is_training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    valid_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'target_valid.csv'),
                  num_items, is_training=False),
        batch_size=128, shuffle=False
    )

    # 3. Init Model
    model = NCF(num_users, num_items, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    # History Tracking
    history = {
        "epoch": [],
        "loss": [],
        "val_recall": [],
        "val_precision": [],
        "val_f1": [],
        "val_map": [],
        "val_ndcg": []
    }
    best_hr = 0.0
    print(f"Train Dataset Size: {len(train_loader.dataset)}")
    print(f"Train Loader Batches: {len(train_loader)}")

    # 4. Train Loop
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for users, pos_items, neg_items in train_loop:
            # users:, pos:, negs:
            users, pos_items, neg_items = users.to(
                DEVICE), pos_items.to(DEVICE), neg_items.to(DEVICE)

            # 1. Positive Loss (unchanged)
            pos_preds = model(users, pos_items)
            pos_loss = criterion(pos_preds, torch.ones_like(pos_preds))

            # 2. Negative Loss
            neg_items_flat = neg_items.view(-1)

            # Repeat users: [u1, u2] -> [u1, u1, u1, u1, u2, u2, u2, u2]
            users_expanded = users.unsqueeze(1).repeat(
                1, neg_items.size(1)).view(-1)

            neg_preds = model(users_expanded, neg_items_flat)
            neg_loss = criterion(neg_preds, torch.zeros_like(neg_preds))

            # 3. Combine
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val  # Accumulate total loss

            # Update the progress bar with the current batch loss
            train_loop.set_postfix(loss=loss_val)

        avg_loss = total_loss / len(train_loader)

        # 5. Evaluate per Epoch
        metrics = evaluate_model(model, valid_loader, device=DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Recall@{10}: {metrics['recall']:.4f} | Val NDCG@{10}: {metrics['ndcg']:.4f} | Val MAP: {metrics['map']:.4f}")

        # Update History
        history["epoch"].append(epoch + 1)
        history["loss"].append(total_loss)
        history["val_recall"].append(metrics['recall'])
        history["val_ndcg"].append(metrics['ndcg'])
        history["val_precision"].append(metrics['precision'])
        history["val_f1"].append(metrics['f1'])
        history["val_map"].append(metrics['map'])

        # Save Best Model
        if metrics['recall'] > best_hr:
            best_hr = metrics['recall']
            torch.save(model.state_dict(), os.path.join(
                OUTPUT_DIR, "best_ncf_model.pth"))
            print("   -> New Best Model Saved!")

    # 6. Save Metrics
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # 7. Final Cold-Start Evaluation (Load Best Model First)
    print("\n--- Final Evaluation on Cold-Start Test Set ---")
    model.load_state_dict(torch.load(
        os.path.join(OUTPUT_DIR, "best_ncf_model.pth")))

    test_loader = DataLoader(
        CDDataset(os.path.join(DATA_DIR, 'target_test.csv'),
                  num_items, is_training=False),
        batch_size=100, shuffle=False
    )

    metrics = evaluate_model(model, test_loader, device=DEVICE)
    test_recall = metrics['recall']
    test_ndcg = metrics['ndcg']
    test_precision = metrics["precision"]
    test_f1 = metrics["f1"]
    test_map = metrics["map"]
    print(
        f"COLD START TEST: R@10 = {test_recall:.4f}, P@10 = {test_precision:.4f}, f1@10 = {test_f1:.4f}, NDCG@10 = {test_ndcg:.4f}, MAP = {test_map:.4f}")

    # Save Final Result
    final_results = {"test_recall": test_recall, "test_precision": test_precision,
                     "test_f1": test_f1, "test_ndcg": test_ndcg, "test_map": test_map}
    with open(os.path.join(OUTPUT_DIR, "final_test_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"Done! Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
