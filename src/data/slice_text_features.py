import numpy as np
import os
from sklearn.preprocessing import normalize


class FeatureSlicer:
    """
    Slices feature matrices to a target dimension using Matryoshka Representation Learning (MRL) concepts.
    """

    def __init__(self, data_dir: str, target_dim: int = 128):
        """
        Initialize the FeatureSlicer.

        Args:
            data_dir (str): Directory containing the feature files.
            target_dim (int): Target dimension for the features.
        """
        self.data_dir = data_dir
        self.target_dim = target_dim

    def slice_and_save(self, filename: str, output_name: str):
        """
        Slices the input feature file and saves the result.

        Args:
            filename (str): Name of the input feature file (npy).
            output_name (str): Name of the output file (without extension).
        """
        input_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (not found)")
            return

        print(f"--- Slicing {filename} ---")

        # Load with mmap so we don't explode RAM
        data = np.load(input_path, mmap_mode='r')
        print(f"Original Shape: {data.shape}")

        # MRL Magic: Just take the first N columns
        sliced_data = data[:, :self.target_dim]

        sliced_data = normalize(sliced_data, norm='l2', axis=1)

        # Verify we didn't lose data (copy to memory)
        sliced_data = np.array(sliced_data)
        print(f"New Shape:      {sliced_data.shape}")

        out_path = os.path.join(self.data_dir, f"{output_name}.npy")
        np.save(out_path, sliced_data)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    DATA_DIR = "C:\\Tugas Raihan\\Latihan Python\\cross-domain-recommendation-system\\data\\processed"

    # 128 is the standard dimension for NCF / LightGCN models
    slicer = FeatureSlicer(DATA_DIR, target_dim=128)

    slicer.slice_and_save("source_domain_feat.npy", "source_domain_mrl")
    slicer.slice_and_save("target_domain_feat.npy", "target_domain_mrl")

    print("\nDone! Use *_mrl.npy files for training.")
