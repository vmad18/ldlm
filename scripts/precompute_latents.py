import sys
import os
import json
# Add the project root to the Python path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset_util.dataset_helper import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True

def precompute_latents(args):
    """
    Pre-computes and saves the T5 encoder latents for a given dataset.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load T5 model and tokenizer
    print(f"Loading T5 model: {args.model_name}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    
    print("Compiling the T5 encoder with torch.compile...")
    # model.encoder = torch.compile(model.get_encoder(), mode="max-autotune-no-cudagraphs")
    model.eval()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} with shard size: {args.shard_size}")
    dataset = get_dataset(args.dataset_name, shard_size=args.shard_size)['train']
    num_samples = len(dataset)
    print(f"Dataset loaded with {num_samples} samples.")

    # Create a simple dataloader
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=args.max_seq_len)
        return inputs

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    # --- Warm-up Pass ---
    if num_samples == 0:
        print("Dataset is empty. No latents to precompute.")
        return

    print("Starting warm-up pass to trigger torch.compile...")
    with torch.no_grad():
        warmup_batch = next(iter(dataloader))
        warmup_batch = {k: v.to(device) for k, v in warmup_batch.items()}
        _ = model.encoder(**warmup_batch)
    print("Warm-up pass complete.")
    # --- End Warm-up Pass ---

    # --- Pre-computation ---
    output_dir = os.path.join(args.save_path, f"{args.model_name.replace('/', '__')}_latents")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving latents to: {output_dir}")

    latent_dim = model.config.d_model
    latent_file_path = os.path.join(output_dir, "latents.npy")
    
    # Use numpy memmap to avoid storing all latents in RAM
    print(f"Creating memory-mapped file for {num_samples} latents...")
    latents_shape = (num_samples, args.max_seq_len, latent_dim)
    all_latents = np.memmap(latent_file_path, dtype=np.float16, mode='w+', shape=latents_shape)

    processed_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Pre-computing latents")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get encoder's last hidden state
            encoder_outputs = model.encoder(**batch)
            last_hidden_state = encoder_outputs.last_hidden_state.cpu().numpy().astype(np.float16)
            
            batch_size = last_hidden_state.shape[0]
            start_idx = processed_samples
            end_idx = start_idx + batch_size

            if end_idx > num_samples:
                print(f"Warning: Batch size would exceed total samples. Truncating.")
                batch_size = num_samples - start_idx
                last_hidden_state = last_hidden_state[:batch_size]
                end_idx = num_samples

            all_latents[start_idx:end_idx] = last_hidden_state
            processed_samples += batch_size

    # Flush memory map to disk
    all_latents.flush()

    if processed_samples != num_samples:
        print(f"Warning: Processed {processed_samples} samples, but dataset contains {num_samples}. The latent file may be incomplete or padded.")
    
    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "num_samples": processed_samples,
        "latent_dim": latent_dim,
        "seq_len": args.max_seq_len,
        "dtype": "float16"
    }
    metadata_file_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Successfully pre-computed and saved {processed_samples} latents to {latent_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute T5 latents for VAE training.")
    parser.add_argument("--model_name", type=str, default="bigscience/T0pp", help="The T5 model to use for encoding.")
    parser.add_argument("--dataset_name", type=str, default="fineweb-edu_10b", help="The dataset to process.")
    parser.add_argument("--shard_size", type=int, default=100000, help="Size of the dataset shard to process for testing. Use None for the full dataset.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--save_path", type=str, default="./precomputed_latents", help="Directory to save the latent files.")
    
    args = parser.parse_args()
    precompute_latents(args) 