import pandas as pd
import numpy as np
import os
import json

class CrossDomainSplitter:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_data(self):
        print("[Splitter] Loading processed data...")
        self.df_source = pd.read_csv(os.path.join(self.data_dir, 'source_domain_processed.csv'))
        self.df_target = pd.read_csv(os.path.join(self.data_dir, 'target_domain_processed.csv'))
        
    def perform_cold_user_split(self, val_ratio=0.1, test_ratio=0.2):
        print("Performing Cold-Start User Split...")
        
        # 1. Identify Bridge Users (Overlap)
        source_users = set(self.df_source['user_id_idx'].unique())
        target_users = set(self.df_target['user_id_idx'].unique())
        bridge_users = list(source_users.intersection(target_users))
        
        print(f"Total Bridge Users: {len(bridge_users)}")
        
        # 2. Shuffle and Split Bridge Users (Train / Valid / Test)
        np.random.seed(42)
        np.random.shuffle(bridge_users)
        
        n_total = len(bridge_users)
        n_test = int(n_total * test_ratio)
        n_val  = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val
        
        test_bridge_users = set(bridge_users[:n_test])
        val_bridge_users  = set(bridge_users[n_test : n_test + n_val])
        train_bridge_users = set(bridge_users[n_test + n_val:])
        
        print(f"Bridge Users -> Train: {len(train_bridge_users)}, Val: {len(val_bridge_users)}, Test: {len(test_bridge_users)}")
        
        # 3. Create Dataframes
        
        # A. TARGET TRAIN:
        # Contains: All Non-Bridge Users + Train Bridge Users
        # We hide Valid and Test Bridge users completely from Training
        train_mask = self.df_target['user_id_idx'].isin(train_bridge_users) | \
                     (~self.df_target['is_overlap'])
        df_target_train = self.df_target[train_mask].copy()
        
        # B. TARGET VALIDATION (Cold Start Simulation):
        # Contains: Only Valid Bridge Users
        val_mask = self.df_target['user_id_idx'].isin(val_bridge_users)
        df_target_val = self.df_target[val_mask].copy()
        
        # C. TARGET TEST (Cold Start Simulation):
        # Contains: Only Test Bridge Users
        test_mask = self.df_target['user_id_idx'].isin(test_bridge_users)
        df_target_test = self.df_target[test_mask].copy()
        
        # 4. Save
        self.save_splits(self.df_source, df_target_train, df_target_val, df_target_test)

    def save_splits(self, src, tgt_train, tgt_val, tgt_test):
        print("Saving splits...")
        # Save Source (Full)
        src.to_csv(os.path.join(self.data_dir, 'source_train.csv'), index=False)
        
        # Save Target Splits
        tgt_train.to_csv(os.path.join(self.data_dir, 'target_train.csv'), index=False)
        tgt_val.to_csv(os.path.join(self.data_dir, 'target_valid.csv'), index=False)
        tgt_test.to_csv(os.path.join(self.data_dir, 'target_test.csv'), index=False)
        
        # Save Stats for Report
        stats = {
            "source_interactions": len(src),
            "target_train_interactions": len(tgt_train),
            "target_val_interactions": len(tgt_val),
            "target_test_interactions": len(tgt_test),
            "target_test_users": tgt_test['user_id_idx'].nunique(),
            "target_val_users": tgt_val['user_id_idx'].nunique()
        }
        
        stats_path = os.path.join(self.data_dir, 'split_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
        print(f"Splitting Complete. Statistics saved to {stats_path}")

if __name__ == "__main__":
    # Update Path
    DATA_DIR = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\processed"
    
    splitter = CrossDomainSplitter(DATA_DIR)
    splitter.load_data()
    # 10% Validation, 10% Test, 80% Train
    splitter.perform_cold_user_split(val_ratio=0.1, test_ratio=0.1)