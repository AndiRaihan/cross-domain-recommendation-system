import json
import os
import numpy as np
import pandas as pd
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class FeatureEngineer:
    """
    Handles feature engineering, specifically generating text embeddings for items using a pre-trained transformer model.
    """

    def __init__(self, data_dir: str, model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
        """
        Initialize the FeatureEngineer.

        Args:
            data_dir (str): Directory containing the data files.
            model_name (str): Name of the pre-trained SentenceTransformer model to use.
        """
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] Loading model: {model_name} on {self.device}...")

        # Use float16 only if on GPU to save VRAM. CPU usually prefers float32.
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=self.device,
            model_kwargs={"torch_dtype": dtype}
        )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Init] Embedding Dimension: {self.embedding_dim}")

    def load_mapping(self, mapping_file: str) -> dict:
        """
        Loads the mapping file.
        File format on disk: {"1": "ASIN_XYZ", "2": "ASIN_ABC"}
        Returns: {"ASIN_XYZ": 1, "ASIN_ABC": 2}

        Args:
            mapping_file (str): Filename of the mapping JSON file.

        Returns:
            dict: Inverted mapping from item ID (string) to integer index.
        """
        path = os.path.join(self.data_dir, mapping_file)
        print(f"[Load] Loading mapping from {path}...")
        with open(path, 'r') as f:
            raw_map = json.load(f)

        # INVERT and CAST: String(Int) -> String(ASIN)  ===>  String(ASIN) -> Int
        inverted_map = {}
        for str_id, raw_asin in raw_map.items():
            inverted_map[raw_asin] = int(str_id)

        return inverted_map

    def clean_text_field(self, field) -> str:
        """
        Cleans and formats a text field.

        Args:
            field: The field value (can be str, list, or None).

        Returns:
            str: Cleaned text string.
        """
        if field is None or str(field).lower() == 'nan':
            return ""
        if isinstance(field, str):
            return field
        if isinstance(field, list):
            # Flatten list of lists
            flat_list = []
            for item in field:
                if isinstance(item, list):
                    flat_list.extend(item)
                else:
                    flat_list.append(item)
            # Deduplicate and join
            return ", ".join(sorted(list(set([str(x) for x in flat_list if x]))))
        return str(field)

    def parse_details(self, details_obj: dict) -> str:
        """
        Extracts specific high-signal keywords from details dict.

        Args:
            details_obj (dict): Dictionary containing item details.

        Returns:
            str: Extracted and formatted details string.
        """
        if not isinstance(details_obj, dict):
            return ""

        keep_keywords = ['content advisory', 'label', 'producer',
                         'director', 'starring', 'supporting actors']
        extracted_text = []
        for k, v in details_obj.items():
            k_lower = k.lower()
            if any(key in k_lower for key in keep_keywords):
                val_str = self.clean_text_field(v)
                extracted_text.append(f"{k}: {val_str}")
        return ". ".join(extracted_text)

    def construct_text(self, item: dict) -> str:
        """
        Constructs the semantic string for embedding.

        Args:
            item (dict): Dictionary containing item metadata.

        Returns:
            str: Constructed text representation of the item.
        """
        title = item.get('title', '')
        cat_str = self.clean_text_field(item.get('categories'))

        # Smart Artist/Creator Extraction
        author = str(item.get('author', ''))
        store_info = str(item.get('store', ''))
        creator_str = ""
        if author and author.lower() != 'nan':
            creator_str = f"Creator: {author}"
        elif "Artist" in store_info:
            creator_str = f"Info: {store_info}"

        # Description (Truncated)
        desc_str = self.clean_text_field(item.get('description'))
        if len(desc_str) > 500:
            desc_str = desc_str[:500]

        # Features
        feat_list = item.get('features', [])
        feat_str = ""
        if isinstance(feat_list, list) and len(feat_list) > 0:
            valid_feats = [str(f) for f in feat_list if f]
            feat_str = "Features: " + ", ".join(valid_feats)

        # Details
        details_str = self.parse_details(item.get('details'))

        # Combine
        parts = [title, creator_str, cat_str, feat_str, details_str, desc_str]
        text = ". ".join([p for p in parts if p])

        return text

    def process_domain(self, meta_file_path: str, mapping_file: str, output_name: str, batch_size: int = 64):
        """
        Processes a domain's metadata to generate item embeddings.

        Args:
            meta_file_path (str): Path to the metadata JSONL file.
            mapping_file (str): Filename of the mapping file.
            output_name (str): Name prefix for the output file.
            batch_size (int): Batch size for embedding generation.
        """
        print(f"\n--- Processing {output_name} ---")

        if (os.path.exists(os.path.join(self.data_dir, f"{output_name}_feat.npy"))):
            print(
                f"Features Already Exist, Skipping {os.path.join(self.data_dir, f'{output_name}_feat.npy')}")
            return

        # 1. Load Map (Corrected Inversion)
        item_map = self.load_mapping(mapping_file)
        valid_asins = set(item_map.keys())

        if not valid_asins:
            raise ValueError(
                f"Mapping file {mapping_file} is empty or invalid!")

        max_id = max(item_map.values())
        num_items = max_id + 1  # ID 0 is padding

        print(f"Target Matrix Size: ({num_items}, {self.embedding_dim})")
        print(f"Valid Items to Encode: {len(valid_asins)}")

        # 2. Pre-allocate Matrix (float16)
        embedding_matrix = np.zeros(
            (num_items, self.embedding_dim), dtype=np.float16)

        batch_texts = []
        batch_indices = []
        found_count = 0
        debug_printed = False  # To print the first text example

        # 3. Stream Metadata
        chunk_size = 100_000
        print(f"Streaming metadata from {meta_file_path}...")

        try:
            with pd.read_json(meta_file_path, lines=True, chunksize=chunk_size) as reader:
                for chunk in reader:
                    # Handle ID column variation
                    id_col = 'parent_asin' if 'parent_asin' in chunk.columns else 'asin'

                    # Filter relevant rows
                    mask = chunk[id_col].isin(valid_asins)
                    relevant_rows = chunk[mask]

                    for _, row in relevant_rows.iterrows():
                        raw_id = row[id_col]
                        int_id = item_map[raw_id]

                        text_body = self.construct_text(row)

                        # Debug: Print the first one to verify quality
                        if not debug_printed:
                            print(
                                f"\n[Sample Text] ID {int_id} ({raw_id}):\n{text_body}\n")
                            debug_printed = True

                        batch_texts.append(text_body)
                        batch_indices.append(int_id)

                        if len(batch_texts) >= batch_size:
                            self._encode_and_store(
                                batch_texts, batch_indices, embedding_matrix)
                            found_count += len(batch_texts)
                            batch_texts = []
                            batch_indices = []
                            print(f"Encoded {found_count} items...", end='\r')

                            if (found_count // batch_size) % 25 == 0:
                                torch.cuda.empty_cache()
                                gc.collect()

        except ValueError as e:
            print(f"\n[Error] Failed to read JSONL: {e}")
            return

        # Process remaining buffer
        if batch_texts:
            self._encode_and_store(
                batch_texts, batch_indices, embedding_matrix)
            found_count += len(batch_texts)

        print(
            f"\n[Complete] Encoded {found_count} items. (Expected ~{len(valid_asins)})")

        if found_count == 0:
            print(
                "[WARNING] Found 0 items! Check if 'parent_asin' vs 'asin' matches your metadata file.")

        # 4. Save
        out_path = os.path.join(self.data_dir, f"{output_name}_feat.npy")
        np.save(out_path, embedding_matrix)
        print(f"Saved feature matrix to {out_path}")

    def _encode_and_store(self, texts: list, indices: list, matrix: np.ndarray):
        """
        Encodes a batch of texts and stores them in the matrix.

        Args:
            texts (list): List of text strings to encode.
            indices (list): List of indices corresponding to the texts.
            matrix (np.ndarray): The embedding matrix to update.
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        matrix[indices] = embeddings.astype(np.float16)


if __name__ == "__main__":
    # --- UPDATE THESE PATHS BEFORE RUNNING ---
    DATA_DIR = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\processed"

    META_SOURCE = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\raw\\meta_Movies_and_TV.jsonl"
    META_TARGET = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\raw\\meta_CDs_and_Vinyl.jsonl"

    engineer = FeatureEngineer(
        DATA_DIR, model_name='Qwen/Qwen3-Embedding-0.6B')

    # Run Source
    engineer.process_domain(
        META_SOURCE, 'source_item_mapping.json', 'source_domain', batch_size=64)

    # Run Target
    engineer.process_domain(
        META_TARGET, 'target_item_mapping.json', 'target_domain', batch_size=64)
