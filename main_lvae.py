import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import sys

from autoencoder.train_lvae import Trainer 

@hydra.main(version_base="1.3", config_path="conf", config_name="train_lvae_llnl")
def main(cfg: DictConfig):
    """
    Main training script for Latent VAE.
    
    To resume training from a checkpoint, use:
    uv run main_lvae.py model.latent_model_path=/path/to/checkpoint/directory
    
    The checkpoint directory should contain:
    - config.yaml (model architecture will be loaded from this)
    - model_best.pt or model.pt (checkpoint file)
    
    When resuming, the model architecture from the checkpoint config will be used,
    but other training parameters (learning rate, batch size, etc.) from the 
    current config will be preserved.
    """
    output_dir = HydraConfig.get().run.dir
    print(f"Output directory: {output_dir}")
    trainer = Trainer(cfg, output_dir)
    trainer.train() 

if __name__ == "__main__":
    main() 
