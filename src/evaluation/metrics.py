import torch
import numpy as np

def get_metrics(rank_list, top_k=10):
    """
    Calculates all standard metrics for a single user's recommendation list.
    
    Args:
        rank_list (list): Binary list of length K. 1 = Relevant Item, 0 = Irrelevant.
                          In Leave-One-Out, there is only one '1' in the entire list.
        top_k (int): The cutoff (e.g., 10).
    
    Returns:
        dict: precision, recall, f1, ndcg, map
    """
    # Cut off at K
    r = rank_list[:top_k]
    
    # Variables
    is_hit = 1 in r
    num_relevant_total = 1 # In Leave-One-Out, we only have 1 ground truth item
    
    # 1. Recall@K (Equivalent to Hit Ratio in this setup)
    recall = 1.0 if is_hit else 0.0
    
    # 2. Precision@K
    # Formula: (Number of Hits) / K
    precision = sum(r) / top_k
    
    # 3. F1-Score@K
    # Formula: 2 * (P * R) / (P + R)
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    # 4. NDCG@K
    ndcg = 0.0
    if is_hit:
        # Found at index 'idx' (0-based)
        idx = r.index(1)
        # Formula: 1 / log2(rank + 1). Rank is idx+1. -> 1 / log2(idx + 2)
        ndcg = 1.0 / np.log2(idx + 2)
        
    # 5. MAP@K (Mean Average Precision)
    # In Leave-One-Out with 1 item, AP is simply (1 / Rank) if found.
    # Example: Found at index 0 (Rank 1) -> AP = 1/1 = 1.0
    # Example: Found at index 4 (Rank 5) -> AP = 1/5 = 0.2
    map_score = 0.0
    if is_hit:
        idx = r.index(1)
        map_score = 1.0 / (idx + 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": ndcg,
        "map": map_score
    }

def evaluate_model(model, test_loader, top_k=10, device='cuda'):
    """
    Evaluates the model and returns average metrics across all test users.
    """
    model.eval()
    
    # Accumulators
    metrics_sum = {
        "precision": [], "recall": [], "f1": [], "ndcg": [], "map": []
    }
    
    with torch.no_grad():
        for user, pos_item, negatives in test_loader:
            user = user.to(device)
            pos_item = pos_item.to(device)
            negatives = negatives.to(device)
            
            # Combine pos and neg items
            # Shape: [Batch_Size, 101]
            all_items = torch.cat([pos_item.unsqueeze(1), negatives], dim=1)
            
            # Repeat user for all items
            users_expanded = user.unsqueeze(1).repeat(1, all_items.size(1))
            
            # Flatten
            flat_users = users_expanded.view(-1)
            flat_items = all_items.view(-1)
            
            # Predict
            predictions = model(flat_users, flat_items)
            scores = predictions.view(-1, 101)
            
            # Sort descending
            _, indices = torch.topk(scores, k=top_k)
            indices = indices.cpu().numpy()
            
            for rank_indices in indices:
                # Create binary list: 1 if index is 0 (ground truth), else 0
                # Because we concatenated [pos, negs], the pos item is always at index 0 in the input
                rank_list = [1 if i == 0 else 0 for i in rank_indices]
                
                # Calculate metrics
                m = get_metrics(rank_list, top_k)
                
                for k, v in m.items():
                    metrics_sum[k].append(v)
                
    # Average out
    final_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    return final_metrics