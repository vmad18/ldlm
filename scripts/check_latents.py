import numpy as np
import os
import argparse

def check_latents(latent_path):
    """
    Loads precomputed latents from a .npy file and checks for NaN values.
    """
    # The dataloader expects a file named 'latents.npy' inside the directory
    latent_file = os.path.join(latent_path, 'latents.npy')

    if not os.path.exists(latent_file):
        print(f"Error: Latent file not found at {latent_file}")
        return

    print(f"Loading latents from {latent_file}...")
    try:
        # Use mmap_mode for memory-efficient reading of large files
        latents = np.load(latent_file, mmap_mode='r')
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        return

    # Check the whole array for NaNs, which is efficient
    has_nan = np.isnan(latents).any()

    if has_nan:
        print("\n" + "="*20)
        print("!!! FAILURE: Precomputed latents contain NaN values. !!!")
        print("="*20 + "\n")

        # Find where the NaNs are
        nan_indices = np.where(np.isnan(latents))
        num_nans = len(nan_indices[0])
        print(f"Found {num_nans} NaN values out of {latents.size} total values ({num_nans / latents.size:.4%}).")

        # Print info about the first few NaNs
        print("First 5 NaN locations (sample_idx, token_idx, feature_idx):")
        for i in range(min(5, num_nans)):
            idx = (nan_indices[0][i], nan_indices[1][i], nan_indices[2][i])
            print(f"  - {idx}")

    else:
        print("\n" + "="*20)
        print("Success! No NaN values found in the precomputed latents.")
        print("="*20 + "\n")

    # Print some general stats about the latents
    print("--- Latent Array Stats ---")
    print(f"Shape: {latents.shape}")
    print(f"Dtype: {latents.dtype}")
    print(f"Min value: {np.nanmin(latents)}")
    print(f"Max value: {np.nanmax(latents)}")
    print(f"Mean value: {np.nanmean(latents)}")
    print(f"Std dev: {np.nanstd(latents)}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check precomputed latents for NaN values.")
    parser.add_argument("latent_path", type=str, help="Path to the directory containing the latents.npy file.")
    args = parser.parse_args()
    
    check_latents(args.latent_path)
