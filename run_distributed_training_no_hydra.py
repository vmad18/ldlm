#!/usr/bin/env python3
"""
Simple distributed training launcher for LVAE without Hydra.
This script manually constructs the training configuration and launches distributed training.

Usage:
    # For single node with 8 GPUs:
    torchrun --nproc_per_node=8 run_distributed_training.py

    # For multi-node (example with 2 nodes, 8 GPUs each):
    # On node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="MASTER_IP" --master_port=29500 run_distributed_training.py
    # On node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="MASTER_IP" --master_port=29500 run_distributed_training.py
"""

import os
import sys
import argparse
from pathlib import Path
from omegaconf import DictConfig

# Import the training function and config
from autoencoder.train_lvae_distributed import main, TrainingConfig


def create_model_config():
    """Create the model configuration manually."""
    return DictConfig({
        "max_seq_len": 512,
        "d_model": 768,
        "latent_dim": 1536,
        "num_latents": 64,
        "dim_head": 128,
        "num_layers": 8,
        "tokenizer_name": "gpt2",
        "vocab_size": 50257,
    })


def create_training_config(
    # Training params
    train_num_steps: int = 10000,
    train_bs: int = 8,
    eval_bs: int = 8,
    eval_every: int = 1000,
    grad_accumulate: int = 1,
    
    # Data params
    train_bin_pattern: str = "/scratch/gpfs/ashwinee/new/modded-nanogpt/data/fineweb100B/fineweb_train_*.bin",
    val_bin_pattern: str = "/scratch/gpfs/ashwinee/new/modded-nanogpt/data/fineweb100B/fineweb_val_*.bin",
    total_tokens: int = 100_000_000_000,
    
    # Optimization params
    learning_rate: float = 1e-4,
    adam_betas: tuple = (0.9, 0.95),
    adam_weight_decay: float = 0.1,
    adam_eps: float = 1e-8,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    
    # VAE params
    kld_weight: float = 1e-4,
    kld_annealing_steps: int = 2000,
    
    # Logging params
    seed: int = 42,
    output_dir: str = "outputs",
    wandb_name: str = None,
    save_checkpoint: bool = True,
    
    # Resume params
    resume_from: str = None,
    
    # Model config overrides
    max_seq_len: int = None,
    d_model: int = None,
    latent_dim: int = None,
    num_latents: int = None,
    dim_head: int = None,
    num_layers: int = None,
):
    """Create the full training configuration."""
    
    # Create base model config
    model_config = create_model_config()
    
    # Override model config if specified
    if max_seq_len is not None:
        model_config.max_seq_len = max_seq_len
    if d_model is not None:
        model_config.d_model = d_model
    if latent_dim is not None:
        model_config.latent_dim = latent_dim
    if num_latents is not None:
        model_config.num_latents = num_latents
    if dim_head is not None:
        model_config.dim_head = dim_head
    if num_layers is not None:
        model_config.num_layers = num_layers
    
    # Create training config
    return TrainingConfig(
        model_config=model_config,
        train_num_steps=train_num_steps,
        train_bs=train_bs,
        eval_bs=eval_bs,
        eval_every=eval_every,
        grad_accumulate=grad_accumulate,
        train_bin_pattern=train_bin_pattern,
        val_bin_pattern=val_bin_pattern,
        total_tokens=total_tokens,
        learning_rate=learning_rate,
        adam_betas=adam_betas,
        adam_weight_decay=adam_weight_decay,
        adam_eps=adam_eps,
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
        kld_weight=kld_weight,
        kld_annealing_steps=kld_annealing_steps,
        seed=seed,
        output_dir=output_dir,
        wandb_name=wandb_name,
        save_checkpoint=save_checkpoint,
        resume_from=resume_from,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run distributed LVAE training")
    
    # Training arguments
    parser.add_argument("--train_num_steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--train_bs", type=int, default=8,
                        help="Training batch size per GPU")
    parser.add_argument("--eval_bs", type=int, default=8,
                        help="Evaluation batch size per GPU")
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--grad_accumulate", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Data arguments
    parser.add_argument("--train_bin_pattern", type=str, default="/scratch/gpfs/ashwinee/new/modded-nanogpt/data/fineweb100B/fineweb_train_*.bin",
                        help="Pattern for training data files")
    parser.add_argument("--val_bin_pattern", type=str, default="/scratch/gpfs/ashwinee/new/modded-nanogpt/data/fineweb100B/fineweb_val_*.bin",
                        help="Pattern for validation data files")
    parser.add_argument("--total_tokens", type=int, default=100_000_000_000,
                        help="Total number of tokens in dataset")
    
    # Optimization arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for AdamW")
    parser.add_argument("--adam_weight_decay", type=float, default=0.1,
                        help="Weight decay for AdamW")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Learning rate for Muon optimizer")
    parser.add_argument("--muon_momentum", type=float, default=0.95,
                        help="Momentum for Muon optimizer")
    
    # VAE arguments
    parser.add_argument("--kld_weight", type=float, default=1e-4,
                        help="KLD loss weight")
    parser.add_argument("--kld_annealing_steps", type=int, default=2000,
                        help="Steps for KLD weight annealing")
    
    # Model arguments
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model dimension")
    parser.add_argument("--latent_dim", type=int, default=1536,
                        help="Latent dimension")
    parser.add_argument("--num_latents", type=int, default=64,
                        help="Number of latent tokens")
    parser.add_argument("--dim_head", type=int, default=128,
                        help="Attention head dimension")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of transformer layers")
    
    # Logging arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--no_save_checkpoint", action="store_true",
                        help="Disable checkpoint saving")
    
    # Resume arguments
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main_entry():
    """Main entry point."""
    args = parse_args()
    
    # Create training configuration
    cfg = create_training_config(
        # Training params
        train_num_steps=args.train_num_steps,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        eval_every=args.eval_every,
        grad_accumulate=args.grad_accumulate,
        
        # Data params
        train_bin_pattern=args.train_bin_pattern,
        val_bin_pattern=args.val_bin_pattern,
        total_tokens=args.total_tokens,
        
        # Optimization params
        learning_rate=args.learning_rate,
        adam_weight_decay=args.adam_weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        
        # VAE params
        kld_weight=args.kld_weight,
        kld_annealing_steps=args.kld_annealing_steps,
        
        # Model params
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        dim_head=args.dim_head,
        num_layers=args.num_layers,
        
        # Logging params
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_name=args.wandb_name,
        save_checkpoint=not args.no_save_checkpoint,
        
        # Resume params
        resume_from=args.resume_from,
    )
    
    # Print configuration for verification
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        print("=" * 50)
        print("Training Configuration:")
        print("=" * 50)
        print(f"Model config: {cfg.model_config}")
        print(f"Training steps: {cfg.train_num_steps}")
        print(f"Batch size: {cfg.train_bs}")
        print(f"Learning rate: {cfg.learning_rate}")
        print(f"Output dir: {cfg.output_dir}")
        if cfg.resume_from:
            print(f"Resuming from: {cfg.resume_from}")
        print("=" * 50)
    
    # Launch training
    main(cfg)


if __name__ == "__main__":
    main_entry() 