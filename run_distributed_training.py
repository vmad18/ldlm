#!/usr/bin/env python3
"""
Script to run distributed LVAE training with torchrun and Hydra.

Usage:
    python run_distributed_training.py --config-path conf --config-name train_lvae_dist
    
    or
    
    torchrun --standalone --nproc_per_node=8 run_distributed_training.py --config-path conf --config-name train_lvae_dist
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from autoencoder.train_lvae_distributed import main, TrainingConfig

def create_training_config(cfg: DictConfig) -> TrainingConfig:
    """Create TrainingConfig from Hydra configuration"""
    
    # Extract model config
    model_cfg = cfg.model
    
    # Create training configuration
    training_cfg = TrainingConfig(
        model_config=model_cfg,
        train_num_steps=cfg.train_num_steps,
        train_bs=cfg.train_bs,
        eval_bs=cfg.eval_bs,
        eval_every=cfg.eval_every,
        grad_accumulate=cfg.grad_accumulate,
        train_bin_pattern=cfg.train_bin_pattern,
        val_bin_pattern=cfg.val_bin_pattern,
        total_tokens=cfg.get('total_tokens', 100_000_000_000),
        learning_rate=cfg.learning_rate,
        adam_betas=(cfg.adam_beta1, cfg.adam_beta2),
        adam_weight_decay=cfg.adam_weight_decay,
        adam_eps=cfg.get('adam_eps', 1e-8),
        muon_lr=cfg.muon_lr,
        muon_momentum=cfg.muon_momentum,
        kld_weight=cfg.kld_weight,
        kld_annealing_steps=cfg.get('kld_annealing_steps', 2000),
        seed=cfg.get('seed', 42),
        output_dir=cfg.get('output_dir', 'outputs'),
        wandb_name=cfg.get('wandb_name', None),
        save_checkpoint=cfg.get('save_checkpoint', True),
        resume_from=cfg.get('resume_from', None)
    )
    
    return training_cfg

@hydra.main(version_base=None, config_path="conf", config_name="train_lvae_dist")
def main_script(cfg: DictConfig) -> None:
    """Main script entry point with Hydra configuration"""
    
    # Check if we're in a distributed environment
    if "RANK" not in os.environ:
        print("Not in distributed environment. Running with torchrun...")
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available!")
            sys.exit(1)
        
        print(f"Found {num_gpus} GPUs. Starting distributed training...")
        
        # Get the original command line arguments
        original_args = sys.argv[1:]
        
        # Run with torchrun
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            __file__
        ] + original_args
        
        # Run the command
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    
    # We're in distributed environment, run training
    print("Running distributed training...")
    
    # Create training configuration
    training_cfg = create_training_config(cfg)
    
    # Run training
    best_val_loss = main(training_cfg)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main_script() 