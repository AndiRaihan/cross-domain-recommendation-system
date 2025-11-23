# This file is for data loading, cleaning, and preprocessing scripts.
import pandas as pd
import json
import os
import gc
import numpy as np
from tqdm import tqdm

class Preprocessor:
    """
    Handles data loading, cleaning, preprocessing, and alignment for cross-domain recommendation.
    """
    def __init__(self, source_path: str, target_path: str, save_dir: str = './data'):
        """
        Initialize the Preprocessor.

        Args:
            source_path (str): Path to the source domain JSONL file.
            target_path (str): Path to the target domain JSONL file.
            save_dir (str): Directory to save processed data.
        """
        self.source_path = source_path
        self.target_path = target_path
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def stream_and_filter_columns(self, file_path: str, domain_name: str) -> pd.DataFrame:
        """
        Reads JSONL in chunks, filters columns, and performs initial cleaning.

        1. Drops heavy text/image columns immediately.
        2. Converts IDs to Categories (Integers) to save RAM.
        3. Drops duplicates.

        Args:
            file_path (str): Path to the JSONL file.
            domain_name (str): Name of the domain (for logging).

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        print(f"[{domain_name}] Streaming data from {file_path}...")
        
        # columns we actually need
        keep_cols = ['user_id', 'parent_asin', 'rating', 'timestamp']
        
        chunks = []
        # Chunk size: Adjust based on your RAM. 1M is usually safe for 16GB RAM.
        chunk_size = 1_000_000 
        
        try:
            # Use pandas read_json with lines=True and chunksize
            with pd.read_json(file_path, lines=True, chunksize=chunk_size) as reader:
                for i, chunk in enumerate(reader):
                    # 1. Filter Columns immediately
                    chunk = chunk[keep_cols]
                    
                    # 2. Optimization: Rename parent_asin to item_id for consistency
                    chunk = chunk.rename(columns={'parent_asin': 'item_id'})
                    
                    # 3. Optimization: Drop duplicates in chunk
                    # Keep last timestamp
                    chunk = chunk.sort_values('timestamp').drop_duplicates(subset=['user_id', 'item_id'], keep='last')
                    
                    chunks.append(chunk)
                    print(f"   Processed Chunk {i+1}...", end='\r')
                    
        except ValueError as e:
            print(f"\nError reading JSON. Make sure it's valid JSONL. {e}")
            return None

        print(f"\n[{domain_name}] Concatenating chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        # Free up memory from the list of chunks
        del chunks
        gc.collect()
        
        # Final Global Deduplication (in case duplicates spanned across chunks)
        df = df.sort_values('timestamp').drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        
        print(f"[{domain_name}] Initial Load: {len(df)} rows.")
        return df

    def recursive_k_core(self, df: pd.DataFrame, k: int, domain_name: str) -> pd.DataFrame:
        """
        Iteratively filters users and items with fewer than k interactions.

        Args:
            df (pd.DataFrame): Input dataframe.
            k (int): Minimum number of interactions.
            domain_name (str): Name of the domain (for logging).

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        print(f"[{domain_name}] Starting {k}-core filtering...")
        iteration = 0
        while True:
            start_len = len(df)
            
            # 1. Filter Users
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df['user_id'].isin(valid_users)]
            
            # 2. Filter Items
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= k].index
            df = df[df['item_id'].isin(valid_items)]
            
            end_len = len(df)
            
            if start_len == end_len:
                break
                
            iteration += 1
            print(f"   Iter {iteration}: Reduced to {end_len} rows")
            
        print(f"[{domain_name}] Converged. Final size: {len(df)}")
        return df

    def align_and_map(self, df_source: pd.DataFrame, df_target: pd.DataFrame):
        """
        Creates the RecBole compatible atomic files and mappings.
        Ensures global user mapping and local item mappings.

        Args:
            df_source (pd.DataFrame): Source domain dataframe.
            df_target (pd.DataFrame): Target domain dataframe.

        Returns:
            tuple: (df_source, df_target, user_map, item_map_s, item_map_t)
        """
        print("\n[Alignment] Mapping IDs...")
        
        # 1. Global User Mapping (Union of both domains)
        # This ensures User "A123" has ID 5 in both Source and Target files.
        unique_users = pd.concat([df_source['user_id'], df_target['user_id']]).unique()
        user_map = {u: i+1 for i, u in enumerate(unique_users)} # Start ID at 1 for RecBole
        
        # 2. Local Item Mappings (Items are distinct per domain)
        items_source = df_source['item_id'].unique()
        items_target = df_target['item_id'].unique()
        
        item_map_s = {i: idx+1 for idx, i in enumerate(items_source)}
        item_map_t = {i: idx+1 for idx, i in enumerate(items_target)}
        
        # 3. Apply Mapping
        print("[Alignment] Applying User/Item Integer IDs...")
        df_source['user_id_idx'] = df_source['user_id'].map(user_map)
        df_source['item_id_idx'] = df_source['item_id'].map(item_map_s)
        
        df_target['user_id_idx'] = df_target['user_id'].map(user_map)
        df_target['item_id_idx'] = df_target['item_id'].map(item_map_t)
        
        # Identify Overlap (Bridge Users)
        source_users_set = set(df_source['user_id'])
        target_users_set = set(df_target['user_id'])
        overlap = source_users_set.intersection(target_users_set)
        print(f"[Alignment] Bridge Users found: {len(overlap)}")
        
        # Mark overlap in Target for splitting
        df_target['is_overlap'] = df_target['user_id'].apply(lambda x: x in overlap)
        
        return df_source, df_target, user_map, item_map_s, item_map_t
    
    def generate_statistics(self, df_s: pd.DataFrame, df_t: pd.DataFrame):
        """
        Calculates and saves detailed statistics for the report.
        Essential for explaining Data Sparsity and Overlap.

        Args:
            df_s (pd.DataFrame): Source domain dataframe.
            df_t (pd.DataFrame): Target domain dataframe.
        """
        print("\n[Statistics] Calculating dataset metrics...")
        
        def get_domain_stats(df, name):
            n_users = df['user_id_idx'].nunique()
            n_items = df['item_id_idx'].nunique()
            n_inter = len(df)
            
            # SPARSITY: 1 - (Interactions / (Users * Items))
            # This is the #1 metric to justify Recommender Systems
            matrix_size = n_users * n_items
            sparsity = (1 - (n_inter / matrix_size)) * 100
            
            avg_user = n_inter / n_users
            avg_item = n_inter / n_items
            
            return {
                "domain": name,
                "users": int(n_users),
                "items": int(n_items),
                "interactions": int(n_inter),
                "sparsity_percent": round(sparsity, 4),
                "avg_interactions_per_user": round(avg_user, 2),
                "avg_interactions_per_item": round(avg_item, 2)
            }

        stats_s = get_domain_stats(df_s, "Source (Movies)")
        stats_t = get_domain_stats(df_t, "Target (Music)")
        
        # Cross-Domain Overlap Stats
        # Since user_id_idx is Global, we can just intersect the sets of IDs
        users_s = set(df_s['user_id_idx'])
        users_t = set(df_t['user_id_idx'])
        overlap_users = users_s.intersection(users_t)
        n_overlap = len(overlap_users)
        
        overlap_stats = {
            "overlap_users_count": n_overlap,
            "overlap_percentage_of_target": round((n_overlap / stats_t['users']) * 100, 2),
            "overlap_percentage_of_source": round((n_overlap / stats_s['users']) * 100, 2)
        }
        
        final_report = {
            "source_domain": stats_s,
            "target_domain": stats_t,
            "cross_domain": overlap_stats
        }
        
        # 1. Save as JSON (for programming use)
        with open(os.path.join(self.save_dir, 'dataset_statistics.json'), 'w') as f:
            json.dump(final_report, f, indent=4)
            
        # 2. Save as Readable Text (for Copy-Pasting into Report)
        with open(os.path.join(self.save_dir, 'dataset_statistics.txt'), 'w') as f:
            f.write("=== DATASET STATISTICS REPORT ===\n\n")
            
            f.write(f"--- {stats_s['domain']} ---\n")
            f.write(f"Users:        {stats_s['users']}\n")
            f.write(f"Items:        {stats_s['items']}\n")
            f.write(f"Interactions: {stats_s['interactions']}\n")
            f.write(f"Sparsity:     {stats_s['sparsity_percent']}%\n")
            f.write(f"Avg Actions/User: {stats_s['avg_interactions_per_user']}\n\n")
            
            f.write(f"--- {stats_t['domain']} ---\n")
            f.write(f"Users:        {stats_t['users']}\n")
            f.write(f"Items:        {stats_t['items']}\n")
            f.write(f"Interactions: {stats_t['interactions']}\n")
            f.write(f"Sparsity:     {stats_t['sparsity_percent']}%\n")
            f.write(f"Avg Actions/User: {stats_t['avg_interactions_per_user']}\n\n")
            
            f.write("--- CROSS-DOMAIN BRIDGE ---\n")
            f.write(f"Overlapping Users: {n_overlap}\n")
            f.write(f"Coverage in Target: {overlap_stats['overlap_percentage_of_target']}%\n")
            f.write("(This % represents the users we can help using Transfer Learning)\n")
            
        print("[Statistics] Report saved to dataset_statistics.txt")

    def export_data(self, df: pd.DataFrame, name: str, folder: str, implicit_threshold: int = 0):
        """
        Exports the processed data to CSV and RecBole compatible formats.

        Args:
            df (pd.DataFrame): Dataframe to export.
            name (str): Name of the dataset (e.g., 'source_domain').
            folder (str): Output folder.
            implicit_threshold (int): Threshold for implicit feedback.
        """
        # --- IMPLICIT FEEDBACK HANDLING ---
        # If rating >= threshold, keep it. Treat as 1 (positive).
        # If threshold is 0, we keep everything as positive interaction (View/Click)
        if implicit_threshold > 0:
            print(f"   [Implicit] Filtering {name} ratings < {implicit_threshold}...")
            df = df[df['rating'] >= implicit_threshold].copy()
        
        # For implicit, label is always 1.
        df['label'] = 1
        
        # --- 1. Standard CSV (For YOUR Custom PyTorch Model) ---
        # Use the MAPPED INTEGERS (user_id_idx)
        out_csv = os.path.join(folder, f"{name}_processed.csv")
        # We save the integer indices
        cols = ['user_id_idx', 'item_id_idx', 'label', 'timestamp', 'is_overlap']
        # If this is source domain, 'is_overlap' might not exist or be relevant, handle carefully
        if 'is_overlap' not in df.columns:
             df['is_overlap'] = False
        
        df[cols].to_csv(out_csv, index=False)
        
        # --- 2. RecBole Atomic Files (.inter) ---
        # Use the RAW STRINGS (user_id). RecBole handles its own indexing.
        # This ensures that if you use RecBole-CDR, it can match tokens "A123" in both files.
        recbole_df = pd.DataFrame()
        recbole_df['user_id:token'] = df['user_id'] # RAW STRING
        recbole_df['item_id:token'] = df['item_id'] # RAW STRING
        recbole_df['timestamp:float'] = df['timestamp']
        # For implicit, we don't usually need a rating column, RecBole infers positive.
        # But we can keep it as 'rating:float' if we want to use RecBole's threshold filters later.
        recbole_df['rating:float'] = df['rating'] 
        
        out_inter = os.path.join(folder, f"{name}.inter")
        recbole_df.to_csv(out_inter, sep='\t', index=False)
        print(f"   Exported {name} to {out_inter}")
        
    def save_mappings(self, user_map: dict, item_map_source: dict, item_map_target: dict):
        """
        Saves the mappings so we can link Integers back to Raw ASINs/Text/Images later.

        Args:
            user_map (dict): User ID mapping.
            item_map_source (dict): Source item ID mapping.
            item_map_target (dict): Target item ID mapping.
        """
        print("[Saving] Saving ID Mappings to JSON...")
        
        # Invert map for easy lookup later: {integer_id: raw_string_id}
        # Note: We cast keys to string because JSON keys must be strings
        
        # 1. Reverse User Map (Integer -> Raw ID)
        reverse_user_map = {str(v): k for k, v in user_map.items()}
        
        # 2. Reverse Item Maps
        reverse_item_s = {str(v): k for k, v in item_map_source.items()}
        reverse_item_t = {str(v): k for k, v in item_map_target.items()}
        
        # Save
        with open(os.path.join(self.save_dir, 'user_mapping.json'), 'w') as f:
            json.dump(reverse_user_map, f)
            
        with open(os.path.join(self.save_dir, 'source_item_mapping.json'), 'w') as f:
            json.dump(reverse_item_s, f)
            
        with open(os.path.join(self.save_dir, 'target_item_mapping.json'), 'w') as f:
            json.dump(reverse_item_t, f)

    def process(self):
        """
        Main execution flow for data processing.
        """
        # 1. Load
        df_s = self.stream_and_filter_columns(self.source_path, "Source")
        df_t = self.stream_and_filter_columns(self.target_path, "Target")
        
        # 2. Filter
        df_s = self.recursive_k_core(df_s, 10, "Source")
        df_t = self.recursive_k_core(df_t, 5, "Target")
        
        # 3. Align
        df_s, df_t, user_map, item_map_s, item_map_t = self.align_and_map(df_s, df_t)
        
        # 4. Save Mappings
        self.save_mappings(user_map, item_map_s, item_map_t)
        
        # 5. Generate Statistics (NEW STEP)
        self.generate_statistics(df_s, df_t)
        
        # 6. Export
        print("\n[Export] Writing files...")
        self.export_data(df_s, "source_domain", self.save_dir, implicit_threshold=0)
        self.export_data(df_t, "target_domain", self.save_dir, implicit_threshold=0)
        
        print("\nDone!")

if __name__ == "__main__":
    # CHANGE THESE TO YOUR ACTUAL JSONL PATHS
    SOURCE = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\raw\\Movies_and_TV.jsonl" 
    TARGET = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\data\\raw\\CDs_and_Vinyl.jsonl"
    
    runner = Preprocessor(SOURCE, TARGET, "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\processed")
    runner.process()