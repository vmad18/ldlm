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

from autoencoder.train_distributed import main

print(f"Importing complete in run_distributed_training.py", flush=True)

@hydra.main(version_base=None, config_path="conf", config_name="train_lvae_dist")
def main_script(cfg: DictConfig) -> None:
    """Main script entry point with Hydra configuration"""

    print(f"Top of main in run_distributed_training.py", flush=True)
    
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
    
    # Run training directly with Hydra config
    best_val_loss = main(cfg)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main_script() 