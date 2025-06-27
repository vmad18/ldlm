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
    model.encoder = torch.compile(model.get_encoder(), mode="max-autotune-no-cudagraphs")
    model.eval()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} with shard size: {args.shard_size}")
    dataset = get_dataset(args.dataset_name, shard_size=args.shard_size)['train']

    # Create a simple dataloader
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=args.max_seq_len)
        return inputs

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    # --- Warm-up Pass ---
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

    all_latents = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Pre-computing latents")):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get encoder's last hidden state
            encoder_outputs = model.encoder(**batch)
            last_hidden_state = encoder_outputs.last_hidden_state.cpu().numpy() # Move to CPU and convert to numpy
            all_latents.append(last_hidden_state)

    # Concatenate all latents and save to a single file
    final_latents = np.concatenate(all_latents, axis=0)
    latent_file_path = os.path.join(output_dir, "latents.npy")
    np.save(latent_file_path, final_latents)

    # Save metadata
    metadata = {
        "num_samples": final_latents.shape[0],
        "latent_dim": final_latents.shape[2],
        "seq_len": final_latents.shape[1]
    }
    metadata_file_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Successfully pre-computed and saved {metadata['num_samples']} latents to {latent_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute T5 latents for VAE training.")
    parser.add_argument("--model_name", type=str, default="t5-base", help="The T5 model to use for encoding.")
    parser.add_argument("--dataset_name", type=str, default="fineweb-edu_10b", help="The dataset to process.")
    parser.add_argument("--shard_size", type=int, default=10000, help="Size of the dataset shard to process for testing. Use None for the full dataset.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    parser.add_argument("--save_path", type=str, default="./precomputed_latents", help="Directory to save the latent files.")
    
    args = parser.parse_args()
    precompute_latents(args) 