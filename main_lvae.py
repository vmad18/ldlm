import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import sys

from autoencoder.train_lvae import Trainer 

@hydra.main(version_base="1.3", config_path="conf", config_name="train_lvae_llnl")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir
    print(f"Output directory: {output_dir}")
    trainer = Trainer(cfg, output_dir)
    trainer.train() 

if __name__ == "__main__":
    main() 
