import pandas as pd
import numpy as np
import os
import json
from typing import Tuple


class CrossDomainSplitter:
    """
    Handles the splitting of cross-domain data into training, validation, and test sets,
    specifically designed for cold-start user scenarios.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the CrossDomainSplitter.

        Args:
            data_dir (str): Directory containing the processed data files.
        """
        self.data_dir = data_dir
        self.df_source = None
        self.df_target = None

    def load_data(self):
        """
        Loads the processed source and target domain data from CSV files.
        """
        print("[Splitter] Loading processed data...")
        self.df_source = pd.read_csv(os.path.join(
            self.data_dir, 'source_domain_processed.csv'))
        self.df_target = pd.read_csv(os.path.join(
            self.data_dir, 'target_domain_processed.csv'))

    def perform_cold_user_split(self, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Performs a cold-start user split on the target domain.

        Identifies "bridge users" (users present in both domains) and splits them into
        train, validation, and test sets. Non-bridge users are always in the training set.

        Args:
            val_ratio (float): Proportion of bridge users to use for validation.
            test_ratio (float): Proportion of bridge users to use for testing.
        """
        print("Performing Cold-Start User Split...")

        if self.df_source is None or self.df_target is None:
            self.load_data()

        # Identify Bridge Users
        source_users = set(self.df_source['user_id_idx'].unique())
        target_users = set(self.df_target['user_id_idx'].unique())
        bridge_users = list(source_users.intersection(target_users))

        print(f"Total Bridge Users: {len(bridge_users)}")

        # Shuffle and Split
        np.random.seed(42)
        np.random.shuffle(bridge_users)

        n_total = len(bridge_users)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_bridge_users = set(bridge_users[:n_test])
        val_bridge_users = set(bridge_users[n_test: n_test + n_val])
        train_bridge_users = set(bridge_users[n_test + n_val:])

        print(
            f"Bridge Users -> Train: {len(train_bridge_users)}, Val: {len(val_bridge_users)}, Test: {len(test_bridge_users)}")

        # Create Dataframes

        # Target Train: Non-Bridge Users + Train Bridge Users
        train_mask = self.df_target['user_id_idx'].isin(train_bridge_users) | \
            (~self.df_target['is_overlap'])
        df_target_train = self.df_target[train_mask].copy()

        # Target Validation: Only Valid Bridge Users
        val_mask = self.df_target['user_id_idx'].isin(val_bridge_users)
        df_target_val = self.df_target[val_mask].copy()

        # Target Test: Only Test Bridge Users
        test_mask = self.df_target['user_id_idx'].isin(test_bridge_users)
        df_target_test = self.df_target[test_mask].copy()

        self.save_splits(self.df_source, df_target_train,
                         df_target_val, df_target_test)

    def save_splits(self, src: pd.DataFrame, tgt_train: pd.DataFrame, tgt_val: pd.DataFrame, tgt_test: pd.DataFrame):
        """
        Saves the generated splits to CSV files and exports statistics.

        Args:
            src (pd.DataFrame): Source domain dataframe (full).
            tgt_train (pd.DataFrame): Target domain training set.
            tgt_val (pd.DataFrame): Target domain validation set.
            tgt_test (pd.DataFrame): Target domain test set.
        """
        print("Saving splits...")
        src.to_csv(os.path.join(self.data_dir,
                   'source_train.csv'), index=False)

        tgt_train.to_csv(os.path.join(
            self.data_dir, 'target_train.csv'), index=False)
        tgt_val.to_csv(os.path.join(
            self.data_dir, 'target_valid.csv'), index=False)
        tgt_test.to_csv(os.path.join(
            self.data_dir, 'target_test.csv'), index=False)

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
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Split processed data into train/val/test.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to processed data directory")
    args = parser.parse_args()

    splitter = CrossDomainSplitter(args.data_dir)
    splitter.load_data()
    splitter.perform_cold_user_split(val_ratio=0.1, test_ratio=0.1)
