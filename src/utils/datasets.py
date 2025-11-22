import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CDDataset(Dataset):
    def __init__(self, csv_path, num_items, num_negatives=1, is_training=True):
        """
        Args:
            csv_path: Path to target_train.csv or target_test.csv
            num_items: Total number of items (for negative sampling)
            num_negatives: How many negatives per positive (Train=1, Eval=100)
            is_training: Bool
        """
        self.df = pd.read_csv(csv_path)
        self.users = self.df['user_id_idx'].values
        self.items = self.df['item_id_idx'].values
        self.labels = self.df['label'].values if 'label' in self.df.columns else np.ones(len(self.df))
        
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.is_training = is_training
        self.mat = set(zip(self.users, self.items))

        # --- OPTIMIZED POPULARITY SAMPLING ---
        if self.is_training:
            print("   [Dataset] Pre-computing Popularity Pool...")
            
            # 1. Count frequencies
            item_counts = self.df['item_id_idx'].value_counts()
            
            # 2. Laplace Smoothing & Mapping
            counts = np.ones(num_items)
            valid_indices = item_counts.index[item_counts.index < num_items]
            counts[valid_indices] += item_counts[valid_indices].values
            counts[0] = 0 # Padding index
            
            # 3. Power Law (0.75)
            pow_counts = np.power(counts, 0.75)
            probs = pow_counts / pow_counts.sum()
            
            # 4. THE FIX: Generate a massive pool ONCE
            # We create a pool of 10 million weighted samples. 
            # Sampling from this pool uniformly is mathematically equivalent to 
            # sampling from the distribution, but instant.
            POOL_SIZE = 10_000_000 
            
            # This line takes ~2-5 seconds to run once on startup
            self.pop_pool = np.random.choice(
                np.arange(num_items), 
                size=POOL_SIZE, 
                p=probs
            )
            self.pool_size = POOL_SIZE
            print("   [Dataset] Pool Ready.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        
        if not self.is_training:
            # EVALUATION (Standard Uniform)
            rng = np.random.default_rng(seed=idx)
            negatives = []
            while len(negatives) < 100:
                neg = rng.integers(1, self.num_items)
                if (user, neg) not in self.mat:
                    negatives.append(neg)
            
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(negatives, dtype=torch.long)
            )
            
        else:
            # TRAINING (Popularity via Pool)
            negatives = []
            while len(negatives) < self.num_negatives:
                # FAST LOOKUP: Just pick a random index from the pre-weighted pool
                # np.random.randint is extremely fast
                pool_idx = np.random.randint(0, self.pool_size)
                neg = self.pop_pool[pool_idx]
                
                # Handle rare collision or padding
                if neg == 0: neg = 1
                
                if (user, neg) not in self.mat:
                    negatives.append(neg)
            
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(negatives, dtype=torch.long) 
            )